"""
trainer for auxnet
"""
import time
import numpy as np

import torch.autograd
import torch.nn as nn
from torch.autograd import Variable

import utils
from models.official.PreResNet import PreBasicBlock
from models.official.ResNet import BasicBlock, Bottleneck
from aux_classifier import AuxClassifier_A as AuxClassifier


class View(nn.Module):
    """
    view data from 4 dimension to 2 dimension
    """

    def forward(self, x):
        """
        forward
        """
        assert x.dim() == 2 or x.dim() == 4, "invalid dimension of input %d" % (x.dim())
        if x.dim() == 4:
            out = x.view(x.size(0), -1)
        else:
            out = x
        return out


class AuxTrainer(object):
    """
    trainer for training network, use SGD
    """

    def __init__(self, model, lr_master, train_loader,
                 test_loader, settings,
                 logger=None):
        self.model = model
        self.settings = settings
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_master = lr_master
        self.logger = logger
        self.criterion = nn.CrossEntropyLoss().cuda()

        base_weight = 0
        self.lr_weight = torch.zeros(len(self.settings.pivotSet))
        self.pivot_weight = self.lr_weight.clone()
        for i in range(len(self.settings.pivotSet) - 1, -1, -1):
            if self.settings.netType == "PreResNet" or (
                    self.settings.netType == "ResNet" and self.settings.depth < 50):
                num_layers = 2
            elif self.settings.netType == "ResNet" and self.settings.depth >= 50:
                num_layers = 3
            temp_weight = max(pow(float(
                self.settings.pivotSet[i] * num_layers + 1) / (
                    self.settings.pivotSet[-1] * num_layers + 1), 2), 0.01)

            base_weight += temp_weight
            self.pivot_weight[i] = temp_weight
            self.lr_weight[i] = base_weight
        # print self.pivot_weight
        # print self.lr_weight
        # assert False
        self.segments = []
        self.seg_optimizer = []
        self.auxfc = []
        self.fc_optimizer = []

        self.run_count = 0

        # run pre-processing
        self._network_split()
        # assert False

    def resume(self, aux_fc_state, seg_opt_state=None, fc_opt_state=None):
        seg_num = len(aux_fc_state)
        for i in range(seg_num):
            if seg_opt_state is not None:
                self.seg_optimizer[i].load_state_dict(seg_opt_state[i])
            if fc_opt_state is not None:
                self.fc_optimizer[i].load_state_dict(fc_opt_state[i])
            self.auxfc[i].eval()
            self.auxfc[i].load_state_dict(aux_fc_state[i])
            self.auxfc[i].train()

    def _network_split(self):
        r"""
        1. split the network into several segments with pre-define pivot set
        2. create auxiliary classifiers
        3. create optimizers for network segments and fcs
        """
        if self.settings.netType in ["PreResNet", "ResNet"]:
            if self.settings.netType == "PreResNet":
                shallow_model = nn.Sequential(self.model.conv)
            else:
                shallow_model = nn.Sequential(
                    self.model.conv1,
                    self.model.bn1,
                    self.model.relu,
                    self.model.maxpool,)
            print "init shallow head done!"
        elif self.settings.netType == "VGG":
            shallow_model = None

        else:
            assert False, "unsupported netType: %s" % self.settings.netType

        block_count = 0
        if self.settings.netType in ["ResNet", "PreResNet"]:
            for module in self.model.modules():
                if isinstance(module, (PreBasicBlock, Bottleneck, BasicBlock)):
                    # print "enter block: ", type(module)
                    # copy blocks
                    if shallow_model is not None:
                        shallow_model.add_module(
                            str(len(shallow_model)), module)
                    else:
                        shallow_model = nn.Sequential(module)
                    block_count += 1

                    # if block_count is equals to pivot_num, then create new segment
                    if block_count in self.settings.pivotSet:
                        self.segments.append(shallow_model)
                        shallow_model = None
                        # print self.segments[-1]

        elif self.settings.netType == "VGG":
            for module in self.model.features.modules():
                if isinstance(module, nn.ReLU):
                    if shallow_model is not None:
                        shallow_model.add_module(
                            str(len(shallow_model)), module
                        )
                        block_count += 1
                    else:
                        assert False, "shallow model is None"
                    if block_count in self.settings.pivotSet:
                        self.segments.append(shallow_model)
                        shallow_model = None
                elif isinstance(module, (nn.Conv2d, nn.MaxPool2d)):
                    if shallow_model is None:
                        shallow_model = nn.Sequential(module)
                    else:
                        shallow_model.add_module(
                            str(len(shallow_model)), module
                        )
        print self.segments

        self.segments.append(shallow_model)

        # create auxiliary classifier
        num_classes = self.settings.nClasses
        for i in range(len(self.segments) - 1):
            if isinstance(self.segments[i][-1], (PreBasicBlock, BasicBlock)):
                in_channels = self.segments[i][-1].conv2.out_channels
            elif isinstance(self.segments[i][-1], Bottleneck):
                in_channels = self.segments[i][-1].conv3.out_channels
            elif isinstance(self.segments[i][-1], nn.ReLU) and self.settings.netType == "VGG":
                in_channels = self.segments[i][-2].out_channels
            
            self.auxfc.append(AuxClassifier(
                in_channels=in_channels,
                num_classes=num_classes),)

        if self.settings.netType == "PreResNet":
            final_fc = nn.Sequential(
                self.model.bn,
                self.model.relu,
                self.model.avg_pool,
                View(),
                self.model.fc,)
        elif self.settings.netType == "ResNet":
            final_fc = nn.Sequential(
                self.model.avgpool,
                View(),
                self.model.fc,)
        elif self.settings.netType == "VGG":
            final_fc = self.model.classifier

        self.auxfc.append(final_fc)

        # model parallel
        self.segments = utils.data_parallel(model=self.segments,
                                            ngpus=self.settings.nGPU,
                                            gpu0=self.settings.GPU)
        self.auxfc = utils.data_parallel(model=self.auxfc,
                                         ngpus=1,
                                         gpu0=self.settings.GPU)

        # create optimizers
        for i in range(len(self.segments)):
            temp_optim = []
            for j in range(i + 1):
                # add parameters in segmenets into optimizer
                # from the i-th optimizer contains [0:i] segments
                if self.settings.optimizerAlgorithm == "SGD":
                    temp_optim.append({'params': self.segments[j].parameters(),
                                       'lr': self.lr_master.lr})
                else:
                    temp_optim.append(
                        {'params': self.segments[j].parameters()})

            # optimizer for segments and fc
            if self.settings.optimizerAlgorithm == "SGD":
                temp_seg_optim = torch.optim.SGD(
                    temp_optim,
                    momentum=self.settings.momentum,
                    weight_decay=self.settings.weightDecay,
                    nesterov=True,)

                temp_fc_optim = torch.optim.SGD(
                    params=self.auxfc[i].parameters(),
                    lr=self.lr_master.lr,
                    momentum=self.settings.momentum,
                    weight_decay=self.settings.weightDecay,
                    nesterov=True,)

            elif self.settings.optimizerAlgorithm == "AdaDelta":
                temp_seg_optim = torch.optim.Adadelta(
                    params=temp_optim,
                    weight_decay=self.settings.weightDecay)
                temp_fc_optim = torch.optim.Adadelta(
                    params=self.auxfc[i].parameters(),
                    weight_decay=self.settings.weightDecay)

            elif self.settings.optimizerAlgorithm == "RMSprop":
                temp_seg_optim = torch.optim.RMSprop(
                    params=temp_optim,
                    weight_decay=self.settings.weightDecay,
                    eps=1e-6,
                    lr=self.settings.lr,)

                temp_fc_optim = torch.optim.RMSprop(
                    params=self.auxfc[i].parameters(),
                    weight_decay=self.settings.weightDecay,
                    eps=1e-6,
                    lr=self.settings.lr,)

            elif self.settings.optimizerAlgorithm == "Adam":
                temp_seg_optim = torch.optim.Adam(
                    params=temp_optim,
                    weight_decay=self.settings.weightDecay,
                    eps=1e-6,
                    lr=self.settings.lr,)

                temp_fc_optim = torch.optim.Adam(
                    params=self.auxfc[i].parameters(),
                    weight_decay=self.settings.weightDecay,
                    eps=1e-6,
                    lr=self.settings.lr,)

            else:
                assert False, "invalid optimizer alogrithm: %s" % self.settings.optimizerAlgorithm

            self.seg_optimizer.append(temp_seg_optim)
            self.fc_optimizer.append(temp_fc_optim)
        # print self.segments
        # print self.auxfc

    @staticmethod
    def adjustweight(current_loss, final_loss, lr_weight=1.0):
        """
        adjust weight according to loss
        """
        """
        if current_loss / final_loss > 2:
            # print final_loss * 1.0 / current_loss
            return final_loss / current_loss
        # elif current_loss / final_loss > 100:
        #     return 0
        else:
            return 1.0 / lr_weight
        """
        return 1.0 / lr_weight

    def forward(self, images, labels=None):
        """
        forward propagation
        """
        outputs = []
        temp_input = images
        losses = []
        for i in range(len(self.segments)):
            # forward
            temp_output = self.segments[i](temp_input)
            fcs_output = self.auxfc[i](temp_output)
            outputs.append(fcs_output)
            if labels is not None:
                losses.append(self.criterion(fcs_output, labels))
            temp_input = temp_output
        return outputs, losses

    @staticmethod
    def _correct_nan(grad):
        grad[grad.ne(grad)] = 0
        return grad

    def backward(self, losses):
        """
        backward propagation
        """
        # two different backward method: from last segment or from first segment
        for i in range(len(self.seg_optimizer)):
            self.seg_optimizer[i].zero_grad()
            self.fc_optimizer[i].zero_grad()

            # backward
            # set params of auxiliary fc
            if i < len(self.segments) - 1:
                if self.settings.optimizerAlgorithm == "SGD":
                    for param_group in self.seg_optimizer[i].param_groups:
                        param_group['lr'] = self.lr_master.lr * self.adjustweight(
                            losses[i].data[0],
                            losses[-1].data[0],
                            self.lr_weight[i],)

                losses[i].backward(retain_graph=True)
                for param_group in self.seg_optimizer[i].param_groups:
                    for p in param_group['params']:
                        if p.grad is None:
                            continue
                        p.grad.data.mul_(self.pivot_weight[i])

            else:
                # set params of final fc
                losses[i].backward(retain_graph=True)

            # correct NaN values
            for param_group in self.seg_optimizer[i].param_groups:
                for p in param_group['params']:
                    if p.grad is None:
                        continue
                    grad = self._correct_nan(p.grad.data)
                    p.grad.data.copy_(grad)
                    # torch.nn.utils.clip_grad_norm(p, 2.)

            for param_group in self.fc_optimizer[i].param_groups:
                for p in param_group['params']:
                    if p.grad is None:
                        continue
                    grad = self._correct_nan(p.grad.data)
                    # torch.nn.utils.clip_grad_norm(p, 2.)

            self.fc_optimizer[i].step()
            self.seg_optimizer[i].step()

    def update_lr(self, epoch):
        """
        update learning rate of optimizers
        """
        if self.settings.optimizerAlgorithm != "SGD":
            return

        lr = self.lr_master.get_lr(epoch)

        for i in range(len(self.seg_optimizer)):
            for param_group in self.seg_optimizer[i].param_groups:
                param_group['lr'] = lr

            for param_group in self.fc_optimizer[i].param_groups:
                param_group['lr'] = lr

    @staticmethod
    def _convert_results(top1_error, top1_loss, top5_error):
        assert isinstance(top1_error, list), "input should be a list"
        length = len(top1_error)
        top1_error_list = []
        top5_error_list = []
        top1_loss_list = []
        for i in range(length):
            top1_error_list.append(top1_error[i].avg)
            top5_error_list.append(top5_error[i].avg)
            top1_loss_list.append(top1_loss[i].avg)
        top1_error_list = np.array(top1_error_list)
        top5_error_list = np.array(top5_error_list)
        top1_loss_list = np.array(top1_loss_list)
        return top1_error_list, top1_loss_list, top5_error_list

    def train(self, epoch):
        """
        training
        """
        iters = len(self.train_loader)
        self.update_lr(epoch)

        top1_error = []
        top5_error = []
        top1_loss = []
        num_segments = len(self.segments)
        for i in range(num_segments):
            self.segments[i].train()
            self.auxfc[i].train()
            top1_error.append(utils.AverageMeter())
            top5_error.append(utils.AverageMeter())
            top1_loss.append(utils.AverageMeter())

        start_time = time.time()
        end_time = start_time
        for i, (images, labels) in enumerate(self.train_loader):
            start_time = time.time()
            data_time = start_time - end_time
            # if we use multi-gpu, its more efficient to send input to different gpu,
            # instead of send it to the master gpu.
            if self.settings.nGPU == 1:
                images = images.cuda()
            labels = labels.cuda()
            images_var = Variable(images)
            labels_var = Variable(labels)

            # forward
            outputs, losses = self.forward(images_var, labels_var)
            # backward
            self.backward(losses)

            # compute loss and error rate
            single_error, single_loss, single5_error = utils.compute_singlecrop(
                outputs=outputs, labels=labels_var,
                loss=losses, top5_flag=True, mean_flag=True)

            for j in range(num_segments):
                top1_error[j].update(single_error[j], images.size(0))
                top5_error[j].update(single5_error[j], images.size(0))
                top1_loss[j].update(single_loss[j], images.size(0))

            end_time = time.time()
            iter_time = end_time - start_time

            utils.print_result(epoch, self.settings.nEpochs, i + 1,
                               iters, self.lr_master.lr, data_time, iter_time,
                               single_error,
                               single_loss,
                               mode="Train",)

        """
        warning: for better comparison, we inverse the index of data
        """
        top1_error_list, top1_loss_list, top5_error_list = self._convert_results(
            top1_error=top1_error, top1_loss=top1_loss, top5_error=top5_error)
        if self.logger is not None:
            length = num_segments - 1
            for i in range(num_segments):
                self.logger.scalar_summary(
                    "train_top1_error_%d" % (length - i), top1_error[i].avg, self.run_count)
                self.logger.scalar_summary(
                    "train_top5_error_%d" % (length - i), top5_error[i].avg, self.run_count)
                self.logger.scalar_summary(
                    "train_loss_%d" % (length - i), top1_loss[i].avg, self.run_count)

        print "|===>Training Error: %.4f /t%.4f, Loss: %.4f" % (
            top1_error[-1].avg, top5_error[-1].avg, top1_loss[-1].avg)
        return top1_error_list, top1_loss_list, top5_error_list

    def test(self, epoch):
        """
        testing
        """
        top1_error = []
        top5_error = []
        top1_loss = []
        num_segments = len(self.segments)
        for i in range(num_segments):
            self.segments[i].eval()
            self.auxfc[i].eval()
            top1_error.append(utils.AverageMeter())
            top5_error.append(utils.AverageMeter())
            top1_loss.append(utils.AverageMeter())

        iters = len(self.test_loader)

        start_time = time.time()
        end_time = start_time
        for i, (images, labels) in enumerate(self.test_loader):
            start_time = time.time()
            data_time = start_time - end_time

            # if we use multi-gpu, its more efficient to send input to different gpu,
            # instead of send it to the master gpu.
            if self.settings.nGPU == 1:
                images = images.cuda()
            images_var = Variable(images, volatile=True)
            labels = labels.cuda()
            labels_var = Variable(labels, volatile=True)

            # forward
            outputs, losses = self.forward(images_var, labels_var)
            # print len(outputs), len(losses)

            # compute loss and error rate
            single_error, single_loss, single5_error = utils.compute_singlecrop(
                outputs=outputs, labels=labels_var,
                loss=losses, top5_flag=True, mean_flag=True)

            for j in range(num_segments):
                top1_error[j].update(single_error[j], images.size(0))
                top5_error[j].update(single5_error[j], images.size(0))
                top1_loss[j].update(single_loss[j], images.size(0))

            end_time = time.time()
            iter_time = end_time - start_time

            utils.print_result(epoch, self.settings.nEpochs, i + 1,
                               iters, self.lr_master.lr, data_time, iter_time,
                               single_error,
                               single_loss,
                               mode="Test",)

        """
        warning: for better comparison, we inverse the index of data
        """
        top1_error_list, top1_loss_list, top5_error_list = self._convert_results(
            top1_error=top1_error, top1_loss=top1_loss, top5_error=top5_error)
        if self.logger is not None:
            length = num_segments - 1
            for i in range(num_segments):
                self.logger.scalar_summary(
                    "test_top1_error_%d" % (length - i), top1_error[i].avg, self.run_count)
                self.logger.scalar_summary(
                    "test_top5_error_%d" % (length - i), top5_error[i].avg, self.run_count)
                self.logger.scalar_summary(
                    "test_loss_%d" % (length - i), top1_loss[i].avg, self.run_count)
        self.run_count += 1

        print "|===>Testing Error: %.4f/%.4f, Loss: %.4f" % (
            top1_error[-1].avg, top5_error[-1].avg, top1_loss[-1].avg)
        return top1_error_list, top1_loss_list, top5_error_list
