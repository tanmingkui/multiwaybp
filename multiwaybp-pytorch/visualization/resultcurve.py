import numpy as np
import matplotlib.pyplot as plt
import os
if os.environ.get('DISPLAY', '') == '':
    plt.switch_backend('agg')

__all__ = ["DrawCurves", "DrawHistogram"]
# other useful params:
# marker, markeredgecolor(mec), markeredgewidth(mew), markerfacecolor(mfc)
# markersize, markerevery(means mark/data_nodes)
# example: mew=2, ms=4, markevery=10


class DrawCurves:
    def __init__(self, file_path, fig_path=""):
        self.fig_params = {"figure_path": "./",
                           "figure_name": "Test-Acc",
                           "label": "ResNet20",
                           "xlabel": "epoch",
                           "ylabel": "testing error (%)",
                           "title": "",
                           "line_width": 2,
                           "line_style": "-",
                           "xlim": [],
                           "ylim": [],
                           "inverse": True,
                           "figure_format": "pdf"}
        self.file_path = file_path
        self.fig_path = fig_path
        split_str = self.fig_path.split("/")
        label_str = split_str[-2]
        split_label_str = label_str.split("_")
        label_str = ""
        for i in range(len(split_label_str)-2):
            label_str += "-" + split_label_str[i+1]
        self.fig_params["label"] = label_str

    @staticmethod
    # log format epoch, train_error, train_loss, test_error, test_loss
    def logparse(file_path):
        log_data = {"train_error": [],
                    "train_loss": [],
                    "test_error": [],
                    "test_loss": []}

        log_file = open(file_path)
        content = log_file.readline()

        while content:
            inf = content.split('\t')
            log_data["train_error"].append(round(float(inf[1]), 2))
            log_data["train_loss"].append(round(float(inf[2]), 2))
            log_data["test_error"].append(round(float(inf[3]), 2))
            log_data["test_loss"].append(round(float(inf[4]), 2))
            content = log_file.readline()
        return log_data

    def draw(self, target="train_error"):
        if target == "train_error":
            self.fig_params["ylabel"] = "training error (%)"
        elif target == "test_error":
            self.fig_params["ylabel"] = "testing error (%)"
        elif target == "train_loss":
            self.fig_params["ylabel"] = "training loss"
        elif target == "test_loss":
            self.fig_params["ylabel"] = "testing loss"
        else:
            print "invalid curve target"
            return False

        if not os.path.isfile(self.file_path):
            print "File not exist!"
            return False
        plt.figure()
        log_data = self.logparse(file_path=self.file_path)
        input_data = log_data[target]

        data_len = len(input_data)
        self.fig_params["label"] = self.fig_params["label"] + "," + str(min(input_data))
        x = np.linspace(1, data_len, data_len)
        plt.plot(x, input_data, self.fig_params["line_style"], linewidth=self.fig_params["line_width"],
                 label=self.fig_params["label"])

        plt.xlabel(self.fig_params["xlabel"])
        plt.ylabel(self.fig_params["ylabel"])
        if not self.fig_params["title"] == "":
            plt.title(self.fig_params["title"])
        if len(self.fig_params["xlim"]) > 0:
            plt.xlim(self.fig_params["xlim"])
        if len(self.fig_params["ylim"]) > 0:
            plt.ylim(self.fig_params["ylim"])
        if not self.fig_path == "":
            self.fig_params["figure_path"] = self.fig_path

        self.fig_params["figure_name"] = target
        plt.grid()
        plt.legend(loc=0, fontsize="small")
        plt.savefig(self.fig_params["figure_path"]+self.fig_params["figure_name"]+"."+self.fig_params["figure_format"],
                    format=self.fig_params["figure_format"])
        plt.close()


class DrawHistogram(object):
    def __init__(self, txt_folder, fig_folder):
        self.txt_folder = txt_folder
        self.fig_folder = fig_folder

    @staticmethod
    # log format epoch, train_error, train_loss, test_error, test_loss
    def logparse(file_path):
        log_file = open(file_path)
        content = log_file.readline()
        log_data_list = []
        while content:
            inf = content.split('\t')
            log_data = {}
            log_data["epoch"] = int(inf[0])
            log_data["block"] = int(inf[1])
            log_data["layer"] = int(inf[2])
            log_data["data"] = np.array([float(x) for x in inf[3:-1]])
            log_data_list.append(log_data)
            content = log_file.readline()
        return log_data_list

    def draw(self):
        if not os.path.isdir(self.txt_folder):
            print "Folder not exist!"
            return False

        txt_file_list = os.listdir(self.txt_folder)
        for i in range(len(txt_file_list)):
            log_data_list = self.logparse(self.txt_folder + txt_file_list[i])
            for j in range(len(log_data_list)):
                plt.figure()
                input_data = log_data_list[j]["data"]
                plt.hist(input_data)
                plt.grid()
                title_str = "epoch:%d, block:%d, layer:%d" % (log_data_list[j]["epoch"],
                                                              log_data_list[j]["block"],
                                                              log_data_list[j]["layer"])
                plt.title(title_str)
                save_path = self.fig_folder + "epoch_%d_block_%d_layer_%d.png" % (log_data_list[j]["epoch"],
                                                                                  log_data_list[j]["block"],
                                                                                  log_data_list[j]["layer"])
                plt.savefig(save_path, format="png")
                plt.close()
