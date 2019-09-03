"""
write readme.md
draw figures
save log
copy code
and so on
"""
import os
import shutil
from resultcurve import DrawCurves
from graphgen import Graph


__all__ = ["Visualization"]

class Visualization(object):
    """
    create package of experimental results
    """
    def __init__(self, save_path):

        self.save_path = save_path

        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        # define path of specific files
        self.log_file = os.path.join(self.save_path, "log.txt")
        self.readme = os.path.join(self.save_path, "README.md")
        self.settings_file = os.path.join(self.save_path, "settings.log")
        self.weight_file = os.path.join(self.save_path, "weight_file.npy")


        # if file is created, remove it
        if os.path.isfile(self.log_file):
            os.remove(self.log_file)
        if os.path.isfile(self.readme):
            os.remove(self.readme)


        print "|===>Result will be saved at", self.save_path



    def write_settings(self, settings):
        """
        save expriment settings to a file
        """
        with open(self.settings_file, "w") as f:
            for k, v in settings.__dict__.items():
                f.write(str(k)+": "+str(v)+"\n")

    def write_log(self, input_data):
        """
        save log
        """
        txt_file = open(self.log_file, 'a+')
        txt_file.write(str(input_data) + "\n")
        txt_file.close()

    def write_readme(self, input_data):
        """
        save extra information to readme.md
        """
        txt_file = open(self.readme, 'a+')
        txt_file.write(str(input_data) + "\n")
        txt_file.close()

    def draw_curves(self):
        """
        draw experimental curves
        """
        drawer = DrawCurves(file_path=self.log_file, fig_path=self.save_path)
        drawer.draw(target="test_error")
        drawer.draw(target="train_error")

    def save_network(self, var):
        """
        draw network structure
        """
        graph = Graph()
        graph.draw(var=var)
        graph.save(file_name=self.save_path+"network.svg")
