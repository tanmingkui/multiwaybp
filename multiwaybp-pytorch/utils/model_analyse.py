import timeit

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import model_transform as mt
from models import depth_search as ds
from prune import wcConv2d, wcLinear
from quantization import twnConv2d, twnLinear

__all__ = ["ModelAnalyse", "LayerAnalyse"]


class LayerAnalyse:
    def __init__(self, model, visual_tool):
        self.model = mt.list2sequential(model)
        self.visual_tool = visual_tool

    def channels_count(self):
        layer_count = 0
        remove_num = []
        select_num = []
        self.visual_tool.write_readme(
            "number of violated channels------------------------\n")
        for layer in self.model.modules():
            if isinstance(layer, (wcConv2d, wcLinear)):
                remove_channels = layer.binary_weight.data.eq(0).sum()
                select_channels = layer.binary_weight.data.eq(1).sum()
                remove_num.append(remove_channels)
                select_num.append(select_channels)

                repo_str = "|===>layer %d, type: %s, #removed: %d, #selected: %d\n" % (layer_count,
                                                                                       str(type(layer)).split(
                                                                                           '.')[-1],
                                                                                       remove_channels,
                                                                                       select_channels)
                self.visual_tool.write_readme(repo_str)
                layer_count += 1
                print repo_str
        if len(remove_num) == 0:
            remove_num = [0.0]
            select_num = [1e-8]

        return np.array(remove_num), np.array(select_num)

    def params_count(self):
        weight_num = []
        bias_num = []
        layer_count = 0
        self.visual_tool.write_readme(
            "number of parameters------------------------\n")
        for layer in self.model.modules():
            if isinstance(layer, (twnConv2d, twnLinear)):
                weight_num.append(layer.weight_ternary.data.view(-1).size(0))
                if layer.bias is not None:
                    bias_num.append(layer.bias.data.view(-1).size(0))
                else:
                    bias_num.append(0)

                repo_str = "|===>layer %d, type: %s, #weight: %d, #bias: %d\n" % (
                    layer_count,
                    str(type(layer)).split(
                        '.')[-1],
                    weight_num[-1],
                    bias_num[-1])

                self.visual_tool.write_readme(repo_str)
                print repo_str
                layer_count += 1

            elif isinstance(layer, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.PReLU)):
                # print type(layer)
                weight_num.append(layer.weight.data.view(-1).size(0))
                if not isinstance(layer, nn.PReLU) and layer.bias is not None:
                    bias_num.append(layer.bias.data.view(-1).size(0))
                else:
                    bias_num.append(0)

                repo_str = "|===>layer %d, type: %s, #weight: %d, #bias: %d\n" % (
                    layer_count,
                    str(type(layer)).split(
                        '.')[-1],
                    weight_num[-1],
                    bias_num[-1])

                self.visual_tool.write_readme(repo_str)
                print repo_str
                layer_count += 1
        if len(weight_num) == 0:
            weight_num = [0]
            bias_num = [0]
        return np.array(weight_num), np.array(bias_num)

    def zero_count(self):
        weight_zero = []
        bias_zero = []
        layer_count = 0

        self.visual_tool.write_readme(
            "number of zeros in parameters------------------------\n")
        for layer in self.model.modules():
            if isinstance(layer, (wcConv2d, wcLinear)):
                if isinstance(layer, wcConv2d):
                    new_weight = layer.binary_weight.unsqueeze(0).unsqueeze(
                        2).unsqueeze(3).expand_as(layer.weight) * layer.weight
                elif isinstance(layer, wcLinear):
                    new_weight = layer.binary_weight.unsqueeze(
                        0).expand_as(layer.weight) * layer.weight

                weight_zero.append(new_weight.data.eq(0).sum())
                if layer.bias is not None:
                    bias_zero.append(layer.bias.data.eq(0).sum())
                else:
                    bias_zero.append(0)

                repo_str = "|===>layer %d, type: %s, #weight: %d, #bias: %d" % (layer_count,
                                                                                str(type(layer)).split(
                                                                                    '.')[-1],
                                                                                weight_zero[-1],
                                                                                bias_zero[-1])
                self.visual_tool.write_readme(repo_str)
                print repo_str
                layer_count += 1

            elif isinstance(layer, (twnConv2d, twnLinear)):
                weight_zero.append(layer.weight_ternary.data.eq(0).sum())
                if layer.bias is not None:
                    bias_zero.append(layer.bias.data.eq(0).sum())
                else:
                    bias_zero.append(0)

                repo_str = "|===>layer %d, type: %s, #weight: %d, #bias: %d" % (layer_count,
                                                                                str(type(layer)).split(
                                                                                    '.')[-1],
                                                                                weight_zero[-1],
                                                                                bias_zero[-1])
                self.visual_tool.write_readme(repo_str)
                print repo_str
                layer_count += 1

            elif isinstance(layer, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.PReLU)):
                weight_zero.append(layer.weight.data.eq(0).sum())

                if not isinstance(layer, nn.PReLU) and layer.bias is not None:
                    bias_zero.append(layer.bias.data.eq(0).sum())
                else:
                    bias_zero.append(0)

                repo_str = "|===>layer %d, type: %s, #weight: %d, #bias: %d" % (
                    layer_count,
                    str(type(layer)).split(
                        '.')[-1],
                    weight_zero[-1],
                    bias_zero[-1])
                self.visual_tool.write_readme(repo_str)
                print repo_str
                layer_count += 1

        # assert False
        return np.array(weight_zero), np.array(bias_zero)


class ModelAnalyse(object):
    def __init__(self, model, visual_tool):
        self.model = mt.list2sequential(model)
        self.layer_analyse = LayerAnalyse(self.model, visual_tool)
        self.weight_num = None
        self.bias_num = None
        self.weight_zero = None
        self.bias_zero = None
        self.visual_tool = visual_tool
        self.flops = []
        self.madds = []

    def params_count(self):
        weight_num, bias_num = self.layer_analyse.params_count()
        params_num = weight_num.sum() + bias_num.sum()

        print "|===>Number of parameters is: ", params_num
        return params_num

    def zero_count(self):
        weight_zero, bias_zero = self.layer_analyse.zero_count()

        zero_num = weight_zero.sum() + bias_zero.sum()

        print "|===>Number of zeros is: ", zero_num
        return zero_num

    def zero_rate(self):
        ratio = self.weight_zero / self.weight_num
        self.visual_tool.write_readme(
            "zero rate-------------------------------\n")
        for r in ratio:
            self.visual_tool.write_readme("zero rate: %f\n" % r)

        print "zero rate:"
        print self.weight_zero / self.weight_num

    def prune_rate(self):
        remove_num, select_num = self.layer_analyse.channels_count()
        self.visual_tool.write_readme(
            "prune rate-------------------------------\n")
        for i in range(remove_num.shape[0]):
            self.visual_tool.write_readme(
                "layer-%d: prune rate:%f" % (i, remove_num[i] * 1.0 / (remove_num[i] + select_num[i])))

        self.visual_tool.write_readme("overall: prune rate:%f" % (
            remove_num.sum() * 1.0 / (remove_num.sum() + select_num.sum())))
        print remove_num * 1.0 / (remove_num + select_num)
        print remove_num.sum() * 1.0 / (remove_num.sum() + select_num.sum())

    def structure(self, net_type, depth):
        d = torch.Tensor(ds.extract_dp(self.model))
        if net_type == "PreResNet":
            seg = (depth - 2) / 6 * np.arange(4)
            seg_1_sum = d[seg[0]:seg[1]].ne(0).sum()
            seg_2_sum = d[seg[1]:seg[2]].ne(0).sum()
            seg_3_sum = d[seg[2]:seg[3]].ne(0).sum()
            print "|===>structure: [%d, %d, %d]" % (seg_1_sum, seg_2_sum, seg_3_sum)
            repo_str = "%s structure: [%d, %d, %d]\n" % (
                net_type + str(depth), seg_1_sum, seg_2_sum, seg_3_sum)
            self.visual_tool.write_readme(repo_str)
            self.visual_tool.write_readme(str(d.numpy()) + "\n")

        elif net_type == "VGG":
            if depth == 19:
                seg_1_sum = d[0:2].ne(0).sum()
                seg_2_sum = d[2:4].ne(0).sum()
                seg_3_sum = d[4:8].ne(0).sum()
                seg_4_sum = d[8:12].ne(0).sum()
                seg_5_sum = d[12:16].ne(0).sum()
            elif depth == 16:
                seg_1_sum = d[0:2].ne(0).sum()
                seg_2_sum = d[2:4].ne(0).sum()
                seg_3_sum = d[4:7].ne(0).sum()
                seg_4_sum = d[7:10].ne(0).sum()
                seg_5_sum = d[10:13].ne(0).sum()
            print "|===>structure: [%d, %d, %d, %d, %d]" % (seg_1_sum, seg_2_sum, seg_3_sum, seg_4_sum, seg_5_sum)
            repo_str = "%s structure: [%d, %d, %d, %d, %d]\n" % (
                net_type + str(depth), seg_1_sum, seg_2_sum, seg_3_sum, seg_4_sum, seg_5_sum)
            self.visual_tool.write_readme(repo_str)
            self.visual_tool.write_readme(str(d.numpy()) + "\n")

        elif net_type == "ResNet":
            if depth == 152:
                seg_1_sum = d[0:3].ne(0).sum()
                seg_2_sum = d[3:11].ne(0).sum()
                seg_3_sum = d[11:47].ne(0).sum()
                seg_4_sum = d[47:50].ne(0).sum()
                print "|===>structure: [%d, %d, %d, %d]" % (seg_1_sum, seg_2_sum, seg_3_sum, seg_4_sum)
                repo_str = "%s structure: [%d, %d, %d, %d]\n" % (
                    net_type + str(depth), seg_1_sum, seg_2_sum, seg_3_sum, seg_4_sum)
                self.visual_tool.write_readme(repo_str)
                self.visual_tool.write_readme(str(d.numpy()) + "\n")

        return repo_str

    def _flops_conv_hook(self, layer, x, out):
        # layer_flops = 2 * x[0].size(2) * x[0].size(3) * (layer.weight.size(1) * layer.weight.size(2) * layer.weight.size(3) + 1) * layer.weight.size(0)
        # compute number of multiply-add
        layer_flops = out.size(2) * out.size(3) * (2. * layer.weight.size(
            1) * layer.weight.size(2) * layer.weight.size(3) - 1.) * layer.weight.size(0)
        self.flops.append(layer_flops)
        # if we only care about multipy operation, use following equation instead
        """
        layer_flops = out.size(2)*out.size(3)*layer.weight.size(1)*layer.weight.size(2)*layer.weight.size(0)
        """

    def _flops_linear_hook(self, layer, x, out):
        # compute number of flops
        layer_flops = (2 * layer.weight.size(1) - 1) * layer.weight.size(0)
        # if we only care about multipy operation, use following equation instead
        """
        layer_flops = layer.weight.size(1)*layer.weight.size(0)
        """
        self.flops.append(layer_flops)

    def _madds_conv_hook(self, layer, x, out):
        # compute number of multiply-add
        layer_madds = out.size(2) * out.size(3) * layer.weight.size(
            1) * layer.weight.size(2) * layer.weight.size(3) * layer.weight.size(0)

        if layer.bias is not None:
            layer_madds += out.view(-1).size(0)

        self.madds.append(layer_madds)

    def _madds_linear_hook(self, layer, x, out):
        # compute number of multiply-add
        layer_madds = layer.weight.size(0) * layer.weight.size(1)

        if layer.bias is not None:
            layer_madds += layer.weight.size(0)

        self.madds.append(layer_madds)

    def time_benchmark(self, x, batch_size=32, loop=1000, ratio=0.8, mode="gpu"):
        """
        compute time on CPU (CPU mode) or single GPU (GPU mode)
        """
        x_expand = x.expand(batch_size, x.size(1), x.size(2), x.size(3))
        label = Variable(torch.zeros(batch_size).long().cuda())

        if mode == "gpu":
            self.model.eval()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            forward_time = []
            backward_time = []
            for i in range(loop):
                torch.cuda.synchronize()
                start_time = timeit.default_timer()
                y = F.cross_entropy(self.model(x_expand), label)
                torch.cuda.synchronize()
                forward_time.append(timeit.default_timer() - start_time)

                torch.cuda.synchronize()
                start_time = timeit.default_timer()
                y.backward()
                torch.cuda.synchronize()
                backward_time.append(timeit.default_timer() - start_time)
        elif mode == "cpu":
            x_expand = x_expand.cpu()
            label = label.cpu()
            model = self.model.cpu().eval()
            forward_time = []
            backward_time = []
            for i in range(loop):
                start_time = timeit.default_timer()
                y = F.cross_entropy(model(x_expand), label)
                forward_time.append(timeit.default_timer() - start_time)
                start_time = timeit.default_timer()
                y.backward()
                backward_time.append(timeit.default_timer() - start_time)
        else:
            assert False, "invalid mode:%s"%mode
        forward_time = np.array(forward_time)
        avg_forward_time = forward_time[np.argsort(
            forward_time)[:int(loop * ratio)]].mean()

        backward_time = np.array(backward_time)
        avg_backward_time = backward_time[np.argsort(
            backward_time)[:int(loop * ratio)]].mean()

        # write results to readme file
        t_string = "### Benchmark Time ====\n"
        t_string += "loop: %d, ratio: %f, batchsize: %d\n"%(loop, ratio, batch_size)
        t_string += "avg_forward time: %f, avg_backward time: %f\n" % (
            avg_forward_time, avg_backward_time)
        t_string += "detailed forward time: %s \n" % (str(forward_time))
        t_string += "detailed backward time: %s \n" % (str(backward_time))

        self.visual_tool.write_readme(t_string)
        print t_string
        return forward_time, avg_forward_time, backward_time, avg_backward_time

    def madds_compute(self, x):
        """
        compute number of multiply-adds of the model
        """
        hook_list = []
        self.madds = []
        for layer in self.model.modules():
            if isinstance(layer, nn.Conv2d):
                hook_list.append(
                    layer.register_forward_hook(self._madds_conv_hook)
                )
            elif isinstance(layer, nn.Linear):
                hook_list.append(
                    layer.register_forward_hook(self._madds_linear_hook)
                )
        # run forward for computing FLOPs
        self.model(x)

        madds_np = np.array(self.madds)
        madds_sum = float(madds_np.sum())
        percentage = madds_np / madds_sum
        for i in range(len(self.madds)):
            repo_str = "|===>MAdds of layer [%d]: %e, %f" % (
                i, madds_np[i], percentage[i])
            print repo_str
            self.visual_tool.write_readme(repo_str)
        repo_str = "### Total MAdds: %e" % (madds_sum)
        print repo_str
        self.visual_tool.write_readme(repo_str)

        for hook in hook_list:
            hook.remove()

        return madds_np

    def flops_compute(self, x):
        """
        compute number of flops of the model
        """
        hook_list = []
        self.flops = []
        for layer in self.model.modules():
            if isinstance(layer, (nn.Conv2d, )):
                hook_list.append(
                    layer.register_forward_hook(self._flops_conv_hook))
            elif isinstance(layer, (nn.Linear,)):
                hook_list.append(layer.register_forward_hook(
                    self._flops_linear_hook))

        # run forward for computing FLOPs
        self.model(x)

        flops_np = np.array(self.flops)
        flops_sum = float(flops_np.sum())
        percentage = flops_np / flops_sum
        for i in range(len(self.flops)):
            repo_str = "|===>FLOPs of layer [%d]: %e, %f" % (
                i, flops_np[i], percentage[i])
            print repo_str
            self.visual_tool.write_readme(repo_str)
        repo_str = "### Total FLOPs: %e" % (flops_sum)
        print repo_str
        self.visual_tool.write_readme(repo_str)

        for hook in hook_list:
            hook.remove()

        return flops_np
