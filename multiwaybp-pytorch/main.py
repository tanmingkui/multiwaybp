
import datetime
import time
import os
from termcolor import colored
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from dataloader import DataLoader
from checkpoint import CheckPoint
import visualization as vs
from aux_trainer import AuxTrainer as Trainer
from options import Option
import models as md
import utils

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"


class ExperimentDesign(object):
    """
    run experiments with pre-defined pipeline
    """

    def __init__(self):
        self.settings = Option()
        self.checkpoint = None
        self.data_loader = None
        self.model = None

        self.trainer = None
        self.seg_opt_state = None
        self.fc_opt_state = None
        self.aux_fc_state = None
        self.start_epoch = 0

        self.model_analyse = None

        self.visualize = vs.Visualization(self.settings.save_path)
        self.logger = vs.Logger(self.settings.save_path)

        self.prepare()

    def prepare(self):
        """
        preparing experiments
        """
        self._set_gpu()
        self._set_dataloader()
        self._set_model()
        self._set_checkpoint()
        self._set_trainer()

    def _set_gpu(self):
        # set torch seed
        # init random seed
        torch.manual_seed(self.settings.manualSeed)
        torch.cuda.manual_seed(self.settings.manualSeed)
        assert self.settings.GPU <= torch.cuda.device_count() - 1, "Invalid GPU ID"
        torch.cuda.set_device(self.settings.GPU)
        cudnn.benchmark = True

    def _set_dataloader(self):
        # create data loader
        self.data_loader = DataLoader(dataset=self.settings.dataset,
                                      batch_size=self.settings.batchSize,
                                      data_path=self.settings.dataPath,
                                      n_threads=self.settings.nThreads,
                                      ten_crop=self.settings.tenCrop)

    def _set_checkpoint(self):
        assert self.model is not None, "please create model first"

        self.checkpoint = CheckPoint(self.settings.save_path)
        if self.settings.retrain is not None:
            model_state = self.checkpoint.load_model(self.settings.retrain)
            self.model = self.checkpoint.load_state(self.model, model_state)

        if self.settings.resume is not None:
            check_point_params = torch.load(self.settings.resume)
            model_state = check_point_params["model"]
            self.seg_opt_state = check_point_params["seg_opt"]
            self.fc_opt_state = check_point_params["fc_opt"]
            self.aux_fc_state = check_point_params["aux_fc"]
            self.model = self.checkpoint.load_state(self.model, model_state)
            self.start_epoch = 90

    def _set_model(self):
        print("netType:",self.settings.netType)
        if self.settings.dataset in ["cifar10", "cifar100"]:
            if self.settings.netType == "DARTSNet":
                genotype = md.genotypes.DARTS
                self.model = md.DARTSNet(self.settings.init_channels, self.settings.nClasses, self.settings.layers, self.settings.auxiliary, genotype)
            elif self.settings.netType == "PreResNet":
                self.model = md.PreResNet(depth=self.settings.depth,
                                          num_classes=self.settings.nClasses,
                                          wide_factor=self.settings.wideFactor)
            elif self.settings.netType == "CifarResNeXt":
                self.model = md.CifarResNeXt(self.settings.cardinality, self.settings.depth, self.settings.nClasses, 
                                          self.settings.base_width, self.settings.widen_factor)

            elif self.settings.netType == "ResNet":
                self.model = md.ResNet(
                    self.settings.depth, self.settings.nClasses)
            else:
                assert False, "use %s data while network is %s" % (
                    self.settings.dataset, self.settings.netType)
        else:
            assert False, "unsupported data set: " + self.settings.dataset

    def _set_trainer(self):
        # set lr master
        lr_master = utils.LRPolicy(self.settings.lr,
                                   self.settings.nEpochs,
                                   self.settings.lrPolicy)
        params_dict = {
            'power': self.settings.power,
            'step': self.settings.step,
            'end_lr': self.settings.endlr,
            'decay_rate': self.settings.decayRate
        }

        lr_master.set_params(params_dict=params_dict)
        # set trainer
        train_loader, test_loader = self.data_loader.getloader()
        self.trainer = Trainer(model=self.model,
                               train_loader=train_loader,
                               test_loader=test_loader,
                               lr_master=lr_master,
                               settings=self.settings,
                               logger=self.logger)
        if self.seg_opt_state is not None:
            self.trainer.resume(aux_fc_state=self.aux_fc_state,
                                seg_opt_state=self.seg_opt_state,
                                fc_opt_state=self.fc_opt_state)

    def _save_best_model(self, model, aux_fc):

        check_point_params = {}
        if isinstance(model, nn.DataParallel):
            check_point_params["model"] = model.module.state_dict()
        else:
            check_point_params["model"] = model.state_dict()
        aux_fc_state = []
        for i in range(len(aux_fc)):
            if isinstance(aux_fc[i], nn.DataParallel):
                aux_fc_state.append(aux_fc[i].module.state_dict())
            else:
                aux_fc_state.append(aux_fc[i].state_dict())
        check_point_params["aux_fc"] = aux_fc_state
        torch.save(check_point_params, os.path.join(
            self.checkpoint.save_path, "best_model_with_AuxFC.pth"))

    def _save_checkpoint(self, model, seg_optimizers, fc_optimizers, aux_fc, index=0):
        check_point_params = {}
        if isinstance(model, nn.DataParallel):
            check_point_params["model"] = model.module.state_dict()
        else:
            check_point_params["model"] = model.state_dict()
        seg_opt_state = []
        fc_opt_state = []
        aux_fc_state = []
        for i in range(len(seg_optimizers)):
            seg_opt_state.append(seg_optimizers[i].state_dict())
        for i in range(len(fc_optimizers)):
            fc_opt_state.append(fc_optimizers[i].state_dict())
            if isinstance(aux_fc[i], nn.DataParallel):
                aux_fc_state.append(aux_fc[i].module.state_dict())
            else:
                aux_fc_state.append(aux_fc[i].state_dict())

        check_point_params["seg_opt"] = seg_opt_state
        check_point_params["fc_opt"] = fc_opt_state
        check_point_params["aux_fc"] = aux_fc_state

        torch.save(check_point_params, os.path.join(
            self.checkpoint.save_path, "checkpoint_%03d.pth" % index))

    def run(self, run_count=0):
        """
        training and testing
        """

        best_top1 = 100
        best_top5 = 100
        start_time = time.time()
        if self.settings.resume is not None or self.settings.retrain is not None:
            self.trainer.test(0)
        for epoch in range(self.start_epoch, self.settings.nEpochs):
            self.start_epoch = 0

            train_error, train_loss, train5_error = self.trainer.train(
                epoch=epoch)
            test_error, test_loss, test5_error = self.trainer.test(
                epoch=epoch)

            # write and print result
            if isinstance(train_error, np.ndarray):
                log_str = "%d\t" % (epoch)
                for i in range(len(train_error)):
                    log_str += "%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t" % (
                        train_error[i], train_loss[i], test_error[i],
                        test_loss[i], train5_error[i], test5_error[i],
                    )

                best_flag = False
                if best_top1 >= test_error[-1]:
                    best_top1 = test_error[-1]
                    best_top5 = test5_error[-1]
                    best_flag = True

            else:
                log_str = "%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t" % (
                    epoch,
                    train_error, train_loss, test_error,
                    test_loss, train5_error, test5_error
                )
                best_flag = False
                if best_top1 >= test_error:
                    best_top1 = test_error
                    best_top5 = test5_error
                    best_flag = True

            self.visualize.write_log(log_str)

            if best_flag:
                self.checkpoint.save_model(
                    self.trainer.model, best_flag=best_flag)
                self._save_best_model(self.trainer.model, self.trainer.auxfc)

                print colored("# %d ==>Best Result is: Top1 Error: %f, Top5 Error: %f\n" % (
                    run_count, best_top1, best_top5), "red")
            else:
                print colored("# %d ==>Best Result is: Top1 Error: %f, Top5 Error: %f\n" % (
                    run_count, best_top1, best_top5), "blue")

            self._save_checkpoint(self.trainer.model, self.trainer.seg_optimizer,
                                  self.trainer.fc_optimizer, self.trainer.auxfc)



        end_time = time.time()

        # compute cost time
        time_interval = end_time - start_time
        t_string = "Running Time is: " + \
            str(datetime.timedelta(seconds=time_interval)) + "\n"
        print t_string
        # write cost time to file
        self.visualize.write_readme(t_string)

        # save experimental results
        self.visualize.write_readme(
            "Best Result of all is: Top1 Error: %f, Top5 Error: %f\n" % (best_top1, best_top5))


def main():
    experiment = ExperimentDesign()
    experiment.run()


if __name__ == '__main__':
    main()
