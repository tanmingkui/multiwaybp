import numpy as np
import math

__all__ = ["compute_tencrop", "compute_singlecrop", "AverageMeter"]


def compute_tencrop(outputs, labels):
    output_size = outputs.size()
    outputs = outputs.view(output_size[0] / 10, 10, output_size[1])
    outputs = outputs.sum(1).squeeze(1)
    # compute top1
    _, pred = outputs.topk(1, 1, True, True)
    pred = pred.t()
    top1_count = pred.eq(labels.data.view(
        1, -1).expand_as(pred)).view(-1).float().sum(0)
    top1_error = 100.0 - 100.0 * top1_count / labels.size(0)
    top1_error = float(top1_error.cpu().numpy())

    # compute top5
    _, pred = outputs.topk(5, 1, True, True)
    pred = pred.t()
    top5_count = pred.eq(labels.data.view(
        1, -1).expand_as(pred)).view(-1).float().sum(0)
    top5_error = 100.0 - 100.0 * top5_count / labels.size(0)
    top5_error = float(top5_error.cpu().numpy())
    return top1_error, 0, top5_error


def compute_singlecrop(outputs, labels, loss, top5_flag=False, mean_flag=False):
    if isinstance(outputs, list):
        top1_loss = []
        top1_error = []
        top5_error = []
        for i in range(len(outputs)):
            # get index of the max log-probability
            predicted = outputs[i].data.max(1)[1]
            top1_count = predicted.ne(labels.data).float().cpu().sum()
            top1_error.append(100.0 * top1_count)
            # top1_loss.append(loss[i].data[0])
            top1_loss.append(loss[i].item)
            if top5_flag:
                _, pred = outputs[i].data.topk(5, 1, True, True)
                pred = pred.t()
                top5_count = pred.eq(labels.data.view(
                    1, -1).expand_as(pred)).float().sum()
                single_top5 = 100.0 * (labels.size(0) - top5_count)
                top5_error.append(single_top5)

        top1_error = np.array(top1_error)
        top5_error = np.array(top5_error)

    else:
        # get index of the max log-probability
        predicted = outputs.data.max(1)[1]
        top1_count = predicted.ne(labels.data).float().sum()

        top1_error = top1_count * 100.0
        top1_loss = loss.data[0]

        if top5_flag:
            _, pred = outputs.data.topk(5, 1, True, True)
            pred = pred.t()
            top5_count = pred.eq(labels.data.view(1, -1).expand_as(pred)).float().sum()
            top5_error = (labels.size(0) - top5_count) * 100.0

    if mean_flag:
        scale = labels.size(0)
    else:
        scale = 1.0

    if top5_flag:
        return top1_error/scale, top1_loss, top5_error/scale
    else:
        return top1_error/scale, top1_loss

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        """
        reset all parameters
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        update parameters
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count