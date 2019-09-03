from termcolor import colored
import numpy as np
import datetime


__all__ = ["print_result", "print_weight", "print_grad"]


single_train_time = 0
single_test_time = 0
single_train_iters = 0
single_test_iters = 0


def print_result(epoch, nEpochs, count, iters, lr, data_time, iter_time, error, loss, top5error=None, mode="Train"):
    global single_train_time, single_test_time
    global single_train_iters, single_test_iters

    # log_str = ">>> %s [%.3d|%.3d], Iter[%.3d|%.3d], LR:%.4f, DataTime: %.4f, IterTime: %.4f" \
    #           % (mode, epoch + 1, nEpochs, count, iters, lr, data_time, iter_time)

    log_str = colored(">>> %s: " % mode, "white") + colored("[%.3d|%.3d], " % (epoch + 1, nEpochs), "magenta") \
        + "Iter: " + colored("[%.3d|%.3d], " % (count, iters), "magenta") \
        + "LR: " + colored("%.6f, " % lr, "magenta") \
        + "DataTime: " + colored("%.4f, " % data_time, "blue") \
        + "IterTime: " + colored("%.4f, " % iter_time, "blue")
    if isinstance(error, list) or isinstance(error, np.ndarray):
        for i in range(len(error)):
            # log_str += ", Error_%d: %.4f, Loss_%d: %.4f" % (i, error[i], i, loss[i])
            log_str += "Error_%d: " % i + colored("%.4f, " % error[i], "cyan") \
                       + "Loss_%d: " % i + colored("%.4f, " % loss[i](), "cyan")
    else:
        # log_str += ", Error: %.4f, Loss: %.4f" % (error, loss)
        log_str += "Error: " + colored("%.4f, " % error, "cyan") \
                   + "Loss: " + colored("%.4f, " % loss, "cyan")

    if top5error is not None:
        if isinstance(top5error, list) or isinstance(top5error, np.ndarray):
            for i in range(len(top5error)):
                # log_str += ", Top5_Error_%d: %.4f" % (i, top5error[i])
                log_str += " Top5_Error_%d:" % i + \
                    colored("%.4f, " % top5error[i], "cyan")
        else:
            # log_str += ", Top5_Error: %.4f" % top5error
            log_str += "Top5_Error: " + colored("%.4f, " % top5error, "cyan")

    # compute cost time
    if mode == "Train":
        single_train_time = single_train_time * \
            0.95 + 0.05 * (data_time + iter_time)
        # single_train_time = data_time + iter_time
        single_train_iters = iters
        train_left_iter = single_train_iters - count + \
            (nEpochs - epoch - 1) * single_train_iters
        # print "train_left_iters", train_left_iter
        test_left_iter = (nEpochs - epoch) * single_test_iters
    else:
        single_test_time = single_test_time * \
            0.95 + 0.05 * (data_time + iter_time)
        # single_test_time = data_time+iter_time
        single_test_iters = iters
        train_left_iter = (nEpochs - epoch - 1) * single_train_iters
        test_left_iter = single_test_iters - count + \
            (nEpochs - epoch - 1) * single_test_iters

    left_time = single_train_time * train_left_iter + \
        single_test_time * test_left_iter
    total_time = (single_train_time * single_train_iters +
                  single_test_time * single_test_iters) * nEpochs
    # time_str = ",Total Time: %s, Remaining Time: %s" % (str(datetime.timedelta(seconds=total_time)),
    #                                                     str(datetime.timedelta(seconds=left_time)))
    time_str = "Total Time: " + colored("%s, " % str(datetime.timedelta(seconds=total_time)), "red") \
               + "Remaining Time: " + \
        colored("%s" % str(datetime.timedelta(seconds=left_time)), "red")

    print log_str + time_str

    return total_time, left_time


def print_weight(layers):
    if isinstance(layers, MD.qConv2d):
        print layers.weight
    elif isinstance(layers, MD.qLinear):
        print layers.weight
        print layers.weight_mask
    print "------------------------------------"


def print_grad(m):
    if isinstance(m, MD.qLinear):
        print m.weight.data
