"""
trainer for auxnet
"""
import time
import numpy as np
from collections import OrderedDict

import torch.autograd
import torch.nn as nn
from torch.autograd import Variable

import utils
from models.official.PreResNet import PreBasicBlock
from models.official.CifarResNeXt import ResNeXtBottleneck
from models.official.DARTSNet import Cell
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
            if self.settings.netType == "PreResNet" or self.settings.netType == "CifarResNeXt" \
                    or self.settings.netType == "DARTSNet" or (
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

        self.segments = []
        self.seg_optimizer = []
        self.auxfc = []
        self.fc_optimizer = []
        self.output_cache = OrderedDict()

        self.run_count = 0

        # run pre-processing
        self._network_split()
        self._reset_output_cache()
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

    def _forward_hook(self, module, input, output):
        gpu_id = str(output.get_device())
        block_id = str(module.block_index)

        if self.output_cache[block_id][gpu_id] is None:
            self.output_cache[block_id][gpu_id] = output

    def _reset_output_cache(self):
        self.output_cache = OrderedDict()
        for block_index in self.settings.pivotSet:
            self.output_cache[str(block_index)] = OrderedDict()
            for gpu_id in range(self.settings.nGPU):
                self.output_cache[str(block_index)][str(gpu_id)] = None

    @staticmethod
    def _concat_gpu_data(data):
        data_cat = data["0"]
        for i in range(1, len(data)):
            data_cat = torch.cat((data_cat, data[str(i)].cuda(0)))
        return data_cat

    def _network_split(self):
        r"""
        1. split the network into several segments with pre-define pivot set
        2. create auxiliary classifiers
        3. create optimizers for network segments and fcs
        """
        # register forward hook
        block_count = 0
        if self.settings.netType in ["ResNet", "PreResNet","CifarResNeXt","DARTSNet"]:
            i=0
            for module in self.model.modules():
                i+=1
                if isinstance(module, (BasicBlock, Bottleneck, PreBasicBlock,ResNeXtBottleneck,Cell)):
                    block_count += 1
                    module.block_index = block_count
                    if block_count in self.settings.pivotSet:
                        module.register_forward_hook(self._forward_hook)

        if self.settings.netType in ["PreResNet", "ResNet","CifarResNeXt","DARTSNet"]:
            if self.settings.netType == "DARTSNet":
                shallow_model = nn.Sequential(
                            nn.Conv2d(3, 3*self.settings.init_channels, 3, padding=1, bias=False),
                            nn.BatchNorm2d(3*self.settings.init_channels)
                        )
            elif self.settings.netType == "PreResNet":
                shallow_model = nn.Sequential(self.model.conv)
            elif self.settings.netType == "CifarResNeXt":
                shallow_model = nn.Sequential(
                    self.model.conv_1_3x3,
                    self.model.bn_1,
                    self.model.relu,)
            else:
                shallow_model = nn.Sequential(
                    self.model.conv1,
                    self.model.bn1,
                    self.model.relu,
                    self.model.maxpool,)
            print "init shallow head done!"
        else:
            assert False, "unsupported netType: %s" % self.settings.netType

        block_count = 0
        for module in self.model.modules():
            if isinstance(module, (PreBasicBlock, Bottleneck, BasicBlock,ResNeXtBottleneck,Cell)):
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
            else:
                pass
        self.segments.append(shallow_model)

        # create auxiliary classifier
        num_classes = self.settings.nClasses
        for i in range(len(self.segments) - 1):
            if isinstance(self.segments[i][-1], (Cell)):
                in_channels = self.segments[i][-1].preprocess1.conv21.in_channels
            elif isinstance(self.segments[i][-1], (ResNeXtBottleneck)):
                in_channels = self.segments[i][-1].conv_expand.out_channels
            elif isinstance(self.segments[i][-1], (PreBasicBlock, BasicBlock)):
                in_channels = self.segments[i][-1].conv2.out_channels
            elif isinstance(self.segments[i][-1], Bottleneck):
                in_channels = self.segments[i][-1].conv3.out_channels

            self.auxfc.append(AuxClassifier(
                in_channels=in_channels,
                num_classes=num_classes),)
        if self.settings.netType == "DARTSNet":
            final_fc = nn.Sequential(
                self.model.auxiliary_head,
                self.model.global_pooling,
                View(),
                self.model.classifier, )
        elif self.settings.netType == "PreResNet":
            final_fc = nn.Sequential(
                self.model.bn,
                self.model.relu,
                self.model.avg_pool,
                View(),
                self.model.fc,)
        elif self.settings.netType == "CifarResNeXt":
            final_fc = nn.Sequential(
                self.model.avg_pool,
                View(),
                self.model.classifier,)
        elif self.settings.netType == "ResNet":
            final_fc = nn.Sequential(
                self.model.avgpool,
                View(),
                self.model.fc,)

        self.auxfc.append(final_fc)

        # model parallel
        """
        self.segments = utils.data_parallel(model=self.segments,
                                            ngpus=self.settings.nGPU,
                                            gpu0=self.settings.GPU)
        """
        self.model = utils.data_parallel(model=self.model,
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
                temp_optim.append({'params': self.segments[j].parameters(),
                                   'lr': self.lr_master.lr})

            # optimizer for segments and fc

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
        losses = []
        if self.settings.netType == "DARTSNet":
            final_output,final_output_aux = self.model(images)
        else:
            final_output = self.model(images)
        for i, k in enumerate(self.settings.pivotSet):
            temp_output = self.output_cache[str(k)]
            temp_output = self._concat_gpu_data(temp_output)
            temp_output = self.auxfc[i](temp_output)
            outputs.append(temp_output)
            if labels is not None:
                losses.append(self.criterion(temp_output, labels))
        outputs.append(final_output)
        if self.settings.netType == "DARTSNet":
            outputs.append(final_output_aux)
        if labels is not None:
            losses.append(self.criterion(final_output, labels))
            if self.settings.netType == "DARTSNet":
                losses.append(self.criterion(final_output_aux, labels))
        self._reset_output_cache()
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

                for param_group in self.seg_optimizer[i].param_groups:
                    param_group['lr'] = self.lr_master.lr * self.adjustweight(
                        # losses[i].data[0],
                        # losses[-1].data[0],
                        losses[i].item,
                        losses[-1].item,
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
                top1_loss[j].update(single_loss[j](), images.size(0))

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
                    "train_top1_error_%d" % (length - i), top1_error[i].avg.item(), self.run_count)
                self.logger.scalar_summary(
                    "train_top5_error_%d" % (length - i), top5_error[i].avg.item(), self.run_count)
                self.logger.scalar_summary(
                    "train_loss_%d" % (length - i), top1_loss[i].avg, self.run_count)

        print "|===>Training Error: %.4f /%.4f, Loss: %.4f" % (
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
                top1_loss[j].update(single_loss[j](), images.size(0))

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
                    "test_top1_error_%d" % (length - i), top1_error[i].avg.item(), self.run_count)
                self.logger.scalar_summary(
                    "test_top5_error_%d" % (length - i), top5_error[i].avg.item(), self.run_count)
                self.logger.scalar_summary(
                    "test_loss_%d" % (length - i), top1_loss[i].avg, self.run_count)
        self.run_count += 1

        print "|===>Testing Error: %.4f/%.4f, Loss: %.4f" % (
            top1_error[-1].avg, top5_error[-1].avg, top1_loss[-1].avg)
        return top1_error_list, top1_loss_list, top5_error_list
