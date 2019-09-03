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


class AuxClassifier(nn.Module):
    """
    define auxiliary classifier
    """

    def __init__(self, in_channels, num_classes):
        super(AuxClassifier, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_channels, num_classes)

        # init params
        self.fc.bias.data.zero_()

    def forward(self, x):
        """
        forward propagation
        out = global_average_pooling(x)
        out = fully_connected(out)
        """
        out = self.bn(x)
        out = self.relu(out)
        out = out.mean(2).mean(2)
        out = self.fc(out)
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
            temp_weight = max(pow(float(
                self.settings.pivotSet[i] * 2 + 1) / (
                    self.settings.pivotSet[-1] * 2 + 1), 2), 0.01)
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
        else:
            assert False, "unsupported netType: %s" % self.settings.netType

        block_count = 0
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
            else:
                pass
                # print type(module)
        # print shallow_model
        self.segments.append(shallow_model)

        # create auxiliary classifier
        num_classes = self.settings.nClasses
        for i in range(len(self.segments) - 1):
            if isinstance(self.segments[i][-1], (PreBasicBlock, BasicBlock)):
                in_channels = self.segments[i][-1].conv2.out_channels
            elif isinstance(self.segments[i][-1], Bottleneck):
                in_channels = self.segments[i][-1].conv3.out_channels

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
                self.model.avg_pool,
                View(),
                self.model.fc,)

        self.auxfc.append(final_fc)

        # model parallel
        self.segments = utils.data_parallel(model=self.segments,
                                            ngpus=self.settings.nGPU,
                                            gpu0=self.settings.GPU)
        self.auxfc = utils.data_parallel(model=self.auxfc,
                                         ngpus=1,
                                         gpu0=self.settings.GPU)

        # create optimizers
        temp_optim = []
        for i in range(len(self.segments)):
            # add parameters in segmenets into optimizer
            # from the i-th optimizer contains [0:i] segments
            temp_optim.append({'params': self.segments[i].parameters(),
                               'lr': self.lr_master.lr})

        # optimizer for segments and fc
        temp_seg_optim = torch.optim.SGD(
            temp_optim,
            momentum=self.settings.momentum,
            weight_decay=self.settings.weightDecay,
            nesterov=True,)
        self.seg_optimizer.append(temp_seg_optim)

        for i in range(len(self.segments)):
            temp_fc_optim = torch.optim.SGD(
                params=self.auxfc[i].parameters(),
                lr=self.lr_master.lr,
                momentum=self.settings.momentum,
                weight_decay=self.settings.weightDecay,
                nesterov=True,)
            self.fc_optimizer.append(temp_fc_optim)
        # print self.segments
        # print self.auxfc

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
        self.seg_optimizer[0].zero_grad()
        loss = None
        for i in range(len(losses)):
            self.fc_optimizer[i].zero_grad()
            # backward
            """
            if loss is None:
                if i < len(self.segments) - 1:
                    loss = losses[i] * self.pivot_weight[i]
                else:
                    loss = losses[i]
            else:
                if i < len(self.segments) - 1:
                    loss += losses[i] * self.pivot_weight[i]
                else:
                    loss += losses[i]
            """
            if loss is None:
                loss = losses[i]
            else:
                loss += losses[i]
        loss.backward()

        # correct NaN values
        for param_group in self.seg_optimizer[0].param_groups:
            for p in param_group['params']:
                if p.grad is None:
                    continue
                grad = self._correct_nan(p.grad.data)
                p.grad.data.copy_(grad)

        for i in range(len(self.fc_optimizer)):
            for param_group in self.fc_optimizer[i].param_groups:
                for p in param_group['params']:
                    if p.grad is None:
                        continue
                    grad = self._correct_nan(p.grad.data)
            self.fc_optimizer[i].step()
        self.seg_optimizer[0].step()

    def update_lr(self, epoch):
        """
        update learning rate of optimizers
        """

        lr = self.lr_master.get_lr(epoch)

        for i in range(len(self.seg_optimizer)):
            for param_group in self.seg_optimizer[i].param_groups:
                param_group['lr'] = lr

        for i in range(len(self.fc_optimizer)):
            for param_group in self.fc_optimizer[i].param_groups:
                param_group['lr'] = lr

    def train(self, epoch):
        """
        training
        """
        top1_error = np.zeros(len(self.segments))
        top5_error = np.zeros(len(self.segments))
        top1_loss = np.zeros(len(self.segments))
        iters = len(self.train_loader)

        self.update_lr(epoch)

        for i in range(len(self.segments)):
            self.segments[i].train()
            self.auxfc[i].train()

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

            top1_loss += single_loss
            top1_error += single_error
            top5_error += single5_error

            end_time = time.time()
            iter_time = end_time - start_time

            utils.print_result(epoch, self.settings.nEpochs, i + 1,
                               iters, self.lr_master.lr, data_time, iter_time,
                               single_error,
                               single_loss,
                               mode="Train",)

        top1_loss /= iters
        top1_error /= iters
        top5_error /= iters

        """
        warning: for better comparison, we inverse the index of data
        """
        if self.logger is not None:
            length = len(top1_error) - 1
            for i, item in enumerate(top1_error):
                self.logger.scalar_summary(
                    "train_top1_error_%d" % (length - i), item, self.run_count)
                self.logger.scalar_summary(
                    "train_top5_error_%d" % (length - i), top5_error[i], self.run_count)
                self.logger.scalar_summary(
                    "train_loss_%d" % (length - i), top1_loss[i], self.run_count)

        print "|===>Training Error: %.4f Loss: %.4f" % (top1_error[-1], top1_loss[-1])
        return top1_error, top1_loss, top5_error

    def test(self, epoch):
        """
        testing
        """
        top1_error = np.zeros(len(self.segments))
        top5_error = np.zeros(len(self.segments))
        top1_loss = np.zeros(len(self.segments))

        for i in range(len(self.segments)):
            self.segments[i].eval()
            self.auxfc[i].eval()
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

            top1_loss += single_loss
            top1_error += single_error
            top5_error += single5_error

            end_time = time.time()
            iter_time = end_time - start_time

            utils.print_result(epoch, self.settings.nEpochs, i + 1,
                               iters, self.lr_master.lr, data_time, iter_time,
                               single_error,
                               single_loss,
                               mode="Test",)

        top1_loss /= iters
        top1_error /= iters
        top5_error /= iters

        """
        warning: for better comparison, we inverse the index of data
        """
        if self.logger is not None:
            length = len(top1_error) - 1
            for i, item in enumerate(top1_error):
                self.logger.scalar_summary(
                    "test_top1_error_%d" % (length - i), item, self.run_count)
                self.logger.scalar_summary(
                    "test_top5_error_%d" % (length - i), top5_error[i], self.run_count)
                self.logger.scalar_summary(
                    "test_loss_%d" % (length - i), top1_loss[i], self.run_count)
        self.run_count += 1

        print "|===>Testing Error: %.4f Loss: %.4f" % (top1_error[-1], top1_loss[-1])
        return top1_error, top1_loss, top5_error
