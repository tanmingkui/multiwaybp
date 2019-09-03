from __future__ import division
import torch
import math
import random
from PIL import Image, ImageOps
import numpy as np
import numbers
import types
import torchvision.transforms as transforms

__all__ = ["TenCrop", "Lighting", "RandomOrder", "Grayscale",
           "Brightness", "Contrast", "Saturation", "ColorJitter"]


class TenCrop(object):
    """
    get ten crop from single image
    ten crop means: top-<left, right> corner, bottom-<left, right> corner, center, with/without flip
    """

    def __init__(self, size, normalize=None):
        self.size = size
        self.normalize = normalize

    def __call__(self, img):
        img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
        center_crop = transforms.CenterCrop(self.size)
        img_list = []
        w, h = img.size
        for image in [img, img_flip]:
            img_list.append(center_crop(image))
            img_list.append(image.crop((0, 0, self.size, self.size)))
            img_list.append(image.crop((w - self.size, 0, w, self.size)))
            img_list.append(image.crop((0, h - self.size, self.size, h)))
            img_list.append(image.crop((w - self.size, h - self.size, w, h)))
        imgs = None
        to_tensor = transforms.ToTensor()
        for image in img_list:
            if imgs is None:
                temp_img = to_tensor(image)
                imgs = self.normalize(temp_img)
            else:
                temp_img = to_tensor(image)
                temp_img = self.normalize(temp_img)
                imgs = torch.cat((imgs, temp_img))
        return imgs


# ---------------------------------------------------------------------------------------------
# code is from: https://github.com/eladhoffer/convNet.pytorch/blob/master/preprocess.py
class Lighting(object):
    """
    Lighting noise(AlexNet - style PCA - based noise)
    """

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class RandomOrder(object):
    """ 
    Composes several transforms together in random order.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        if self.transforms is None:
            return img
        order = torch.randperm(len(self.transforms))
        for i in order:
            img = self.transforms[i](img)
        return img


class Grayscale(object):
    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Brightness(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Contrast(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Saturation(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class ColorJitter(RandomOrder):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.transforms = []
        if brightness != 0:
            self.transforms.append(Brightness(brightness))
        if contrast != 0:
            self.transforms.append(Contrast(contrast))
        if saturation != 0:
            self.transforms.append(Saturation(saturation))
# ---------------------------------------------------------------------------------------------
