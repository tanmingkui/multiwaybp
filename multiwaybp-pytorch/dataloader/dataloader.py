"""
data loder for loading data
"""
import os
import math
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import transform as auxtransform


__all__ = ["DataLoader", "PartDataLoader"]


class DataLoader(object):
    """
    data loader for CV data sets
    """

    def __init__(self, dataset, batch_size, n_threads=4,
                 ten_crop=False, data_path='/home/dataset/'):
        """
        create data loader for specific data set
        :params n_treads: number of threads to load data, default: 4
        :params ten_crop: use ten crop for testing, default: False
        :params data_path: path to data set, default: /home/dataset/
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_threads = n_threads
        self.ten_crop = ten_crop
        self.data_path = data_path

        print "|===>Creating data loader for " + self.dataset

        if self.dataset in ["cifar10", "cifar100"]:
            self.train_loader, self.test_loader = self.cifar(
                dataset=self.dataset)

        elif self.dataset in ["imagenet", "imagenet100", "thi_imgnet"]:
            self.train_loader, self.test_loader = self.imagenet(
                dataset=self.dataset)

        elif self.dataset in ["imagenet_tf", "imagenet100_tf"]:
            self.train_loader, self.test_loader = self.imagenet_tf(
                dataset=self.dataset)

        elif self.dataset == "mnist":
            self.train_loader, self.test_loader = self.mnist()

        elif self.dataset == "sphere":
            self.train_loader = self.casia()
            self.test_loader = self.lfw()

        elif self.dataset == "sphere_large":
            self.train_loader = self.msface()
            self.test_loader = self.lfw()
        elif self.dataset =="lfw_validation":
            self.train_loader = self.lfw()
            self.test_loader = self.lfw_list()

        else:
            assert False, "invalid data set"

    def getloader(self):
        """
        get train_loader and test_loader
        """
        return self.train_loader, self.test_loader

    def msface(self):
        """
        dataset: MS-Celb-1M
        """
        img_source = os.path.join(
            self.data_path, 'Face/FaceRecognition/MS-Celb-1M/MS_20000_clean_list.txt')
        # img_source = os.path.join(self.data_path,
        # 'CASIA_WEBFACE/training/CASIA-WebFace-112X96.txt')
        train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.5, 0.5, 0.5],
                                                                    [0.50196, 0.50196, 0.50196])])
        train_dataset = CASIAWebFaceDataset(
            img_source, transform=train_transforms)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.n_threads,
            pin_memory=True)
        return train_loader

    def casia(self):
        """
        dataset: CASIA-WebFace-112X96
        """
        img_source = os.path.join(
            self.data_path,
            'Face/FaceRecognition/CASIA_WEBFACE/training/CASIA-WebFace-112X96-de-duplication.txt')
            # 'Face/FaceRecognition/CASIA_WEBFACE/training/CASIA-WebFace-112X96.txt')
        # img_source = os.path.join(self.data_path,
        # 'CASIA_WEBFACE/training/CASIA-WebFace-112X96.txt')
        train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.5, 0.5, 0.5],
                                                                    [0.50196, 0.50196, 0.50196])])
        train_dataset = CASIAWebFaceDataset(
            img_source, transform=train_transforms)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.n_threads,
            pin_memory=True)

        return train_loader

    def lfw(self):
        """
        dataset: LFW
        """
        lfw_pairs = os.path.join(
            self.data_path, 'Face/FaceRecognition/CASIA_WEBFACE/testing/pairs.txt')
        lfw_dir = os.path.join(
            self.data_path, 'Face/FaceRecognition/CASIA_WEBFACE/testing/lfw-112X96')
        # lfw_pairs = os.path.join(self.data_path, 'CASIA_WEBFACE/testing/pairs.txt')
        # lfw_dir = os.path.join(self.data_path, 'CASIA_WEBFACE/testing/lfw-112X96')
        test_transforms = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.5, 0.5, 0.5],
                                  [0.50196, 0.50196, 0.50196])])
        test_dataset = LFWDataset(
            lfw_dir, lfw_pairs, transform=test_transforms)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=100,
            shuffle=False, num_workers=self.n_threads,
            pin_memory=False)
        return test_loader

    def lfw_list(self):

        dataset_path = os.path.join(
            self.data_path, "Face/FaceRecognition/CASIA_WEBFACE/testing/lfw-112X96")

        test_transforms = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.5, 0.5, 0.5],
                                  [0.50196, 0.50196, 0.50196])])

        test_loader = torch.utils.data.DataLoader(
            dsets.ImageFolder(dataset_path, test_transforms),
            batch_size=100,
            shuffle=False,
            num_workers=self.n_threads,
            pin_memory=False)

        return test_loader

    def mnist(self):
        """
        dataset: mnist
        """
        root_path = os.path.join(self.data_path, "mnist")
        norm_mean = [0.1307]
        norm_std = [0.3081]
        train_loader = torch.utils.data.DataLoader(
            dsets.MNIST(root_path, train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(norm_mean, norm_std)
                        ])),
            batch_size=self.batch_size, shuffle=True,
            num_workers=self.n_threads,
            pin_memory=False
        )
        test_loader = torch.utils.data.DataLoader(
            dsets.MNIST(root_path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)
            ])),
            batch_size=self.batch_size, shuffle=True,
            num_workers=self.n_threads,
            pin_memory=False
        )
        return train_loader, test_loader

    def imagenet(self, dataset="imagenet"):
        if dataset == "imagenet":
            dataset_path = os.path.join(self.data_path, "imagenet")
        elif dataset == "imagenet100":
            dataset_path = os.path.join(self.data_path, "imagenet100")
        elif dataset == "thi_imgnet":
            dataset_path = os.path.join(self.data_path, "thi_imgnet")

        traindir = os.path.join(dataset_path, "train")
        if dataset == "thi_imgnet":
            testdir = os.path.join(self.data_path, "imagenet", "val")
        else:
            testdir = os.path.join(dataset_path, "val")

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_loader = torch.utils.data.DataLoader(
            dsets.ImageFolder(traindir, transforms.Compose([
                # transforms.RandomResizedCrop(224),
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_threads,
            pin_memory=True)

        if self.ten_crop:
            test_transform = transforms.Compose([
                # transforms.Resize(256),
                transforms.Scale(256),
                normalize
            ])
        else:
            test_transform = transforms.Compose([
                # transforms.Resize(256),
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
        test_loader = torch.utils.data.DataLoader(
            dsets.ImageFolder(testdir, test_transform),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_threads,
            pin_memory=False)
        return train_loader, test_loader

    def imagenet_tf(self, dataset="imagenet_tf"):
        """
        imagenet dataset with transform: color_jitter and lighting
        """
        if dataset == "imagenet_tf":
            dataset_path = os.path.join(self.data_path, "imagenet")
        else:
            dataset_path = os.path.join(self.data_path, "imagenet100")

        traindir = os.path.join(dataset_path, "train")
        testdir = os.path.join(dataset_path, "val")

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        imagenet_pca = {
            'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
            'eigvec': torch.Tensor([
                [-0.5675, 0.7192, 0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948, 0.4203],
            ])
        }
        train_loader = torch.utils.data.DataLoader(
            dsets.ImageFolder(traindir, transforms.Compose([
                # transforms.RandomResizedCrop(224),
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                auxtransform.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4),
                auxtransform.Lighting(
                    alphastd=0.1, eigval=imagenet_pca['eigval'], eigvec=imagenet_pca['eigvec']),
                normalize,
            ])),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_threads,
            pin_memory=False)

        if self.ten_crop:
            test_transform = transforms.Compose([
                # transforms.Resize(256),
                transforms.Scale(256),
                auxtransform.TenCrop(224, normalize)
            ])
        else:
            test_transform = transforms.Compose([
                # transforms.Resize(256),
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
        test_loader = torch.utils.data.DataLoader(
            dsets.ImageFolder(testdir, test_transform),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_threads,
            pin_memory=False)
        return train_loader, test_loader

    def cifar(self, dataset="cifar10"):
        """
        dataset: cifar
        """
        if dataset == "cifar10":
            norm_mean = [0.49139968, 0.48215827, 0.44653124]
            norm_std = [0.24703233, 0.24348505, 0.26158768]
        elif dataset == "cifar100":
            norm_mean = [0.50705882, 0.48666667, 0.44078431]
            norm_std = [0.26745098, 0.25568627, 0.27607843]

        else:
            assert False, "Invalid cifar dataset"
        # data_root = "data/cifar"
        data_root = os.path.join(self.data_path, "cifar")
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)])

        if self.ten_crop:
            test_transform = transforms.Compose([
                auxtransform.TenCrop(28, transforms.Normalize(norm_mean, norm_std))])
            print "use TenCrop()"
        else:
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)])

        # cifar10 data set
        if self.dataset == "cifar10":
            train_dataset = dsets.CIFAR10(root=data_root,
                                          train=True,
                                          transform=train_transform,
                                          download=True)

            test_dataset = dsets.CIFAR10(root=data_root,
                                         train=False,
                                         transform=test_transform)
        elif self.dataset == "cifar100":
            train_dataset = dsets.CIFAR100(root=data_root,
                                           train=True,
                                           transform=train_transform,
                                           download=True)

            test_dataset = dsets.CIFAR100(root=data_root,
                                          train=False,
                                          transform=test_transform)
        else:
            assert False, "invalid data set"

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=self.n_threads)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=self.batch_size,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=self.n_threads)
        return train_loader, test_loader


class PartDataLoader(DataLoader):
    """
    dataloader for loading part of data set
    """

    def __init__(self, dataset, batch_size, n_threads=4,
                 ten_crop=False, data_path='/home/dataset/', data_ratio=1.0):
        super(PartDataLoader, self).__init__(
            dataset, batch_size, n_threads, ten_crop, data_path)
        """
        create data loader for specific data set, load part of the data only
        :params dataset
        :params batch_size
        :params n_treads: number of threads to load data, default: 4
        :params ten_crop: use ten crop for testing, default: False
        :params data_path: path to data set, default: /home/dataset/
        :params data_ratio: ratio of loaded data, default: 1.0
        """
        self.data_ratio = data_ratio
        print "|===>Creating data loader for" + self.dataset

        if self.dataset in ["cifar10", "cifar100"]:
            self.train_loader, self.test_loader = self.cifar(
                dataset=self.dataset)

    def cifar(self, dataset="cifar10"):
        if dataset == "cifar10":
            norm_mean = [0.49139968, 0.48215827, 0.44653124]
            norm_std = [0.24703233, 0.24348505, 0.26158768]
        elif dataset == "cifar100":
            norm_mean = [0.50705882, 0.48666667, 0.44078431]
            norm_std = [0.26745098, 0.25568627, 0.27607843]
        else:
            assert False, "Invalid cifar dataset"

        data_root = os.path.join(self.data_path, "cifar")
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)])

        if self.ten_crop:
            test_transform = transforms.Compose([
                auxtransform.TenCrop(28, transforms.Normalize(norm_mean, norm_std))])
            print "use TenCrop()"
        else:
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)])

        # cifar10 data set
        if self.dataset == "cifar10":
            train_dataset = PartCifar10(root=data_root,
                                        train=True,
                                        transform=train_transform,
                                        download=True,
                                        data_ratio=self.data_ratio)

            test_dataset = dsets.CIFAR10(root=data_root,
                                         train=False,
                                         transform=test_transform)
        elif self.dataset == "cifar100":
            train_dataset = PartCifar100(root=data_root,
                                         train=True,
                                         transform=train_transform,
                                         download=True,
                                         data_ratio=self.data_ratio)

            test_dataset = dsets.CIFAR100(root=data_root,
                                          train=False,
                                          transform=test_transform)
        else:
            assert False, "invalid data set"

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=self.n_threads)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=self.batch_size,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=self.n_threads)
        return train_loader, test_loader


class PartCifar10(dsets.CIFAR10):
    """
    part of cifar10
    """

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, data_ratio=1.0):
        super(PartCifar10, self).__init__(root=root, train=train, transform=transform,
                                          target_transform=target_transform, download=download)
        if self.train and data_ratio > 0.0:
            self.train_data = self.train_data[:int(50000 * data_ratio - 1)]

    def __len__(self):
        if self.train:
            return self.train_data.shape[0]
        else:
            return 10000


class PartCifar100(dsets.CIFAR100):
    """
    part of cifar100
    """

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, data_ratio=1.0):
        super(PartCifar100, self).__init__(root=root, train=train, transform=transform,
                                           target_transform=target_transform, download=download)
        if self.train and data_ratio > 0.0:
            self.train_data = self.train_data[:int(50000 * data_ratio - 1)]

    def __len__(self):
        if self.train:
            return self.train_data.shape[0]
        else:
            return 10000

# ---------------------------------------------------------------------------------------
# this code is writen by liu jing


class CASIAWebFaceDataset(data.Dataset):
    """Face dataset class

    Attributes
        img_source: A txt file containing image list.
                    Each line in this file contains image path and target.
                    For example, '/home/xxxx/data/xxxxx/0001.jpg 0'.
        transform: Preprocessing of image.
        target_transform: Preprocessing of target.
        img_path_list: List contains img_paths
        target_list: List contains targets
    """

    def __init__(self, img_source, transform=None, target_transform=None):
        """
        Inits CASIAWebFaceDataset with img_source file and transformation.
        """
        self.img_source = img_source
        self.transform = transform
        self.target_transform = target_transform
        self.img_path_list = []
        self.target_list = []

        with open(img_source, 'r') as f:
            file_list = f.readlines()
            for file_name in file_list:
                file_name = file_name.split()
                self.img_path_list.append(file_name[0])
                self.target_list.append(int(file_name[1]))

    def __getitem__(self, index):
        """
        Get item with index.
        """
        img_path = self.img_path_list[index]
        target = self.target_list[index]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """
        Return the number of image.
        """
        return len(self.img_path_list)


class LFWDataset(data.Dataset):
    """LFW dataset

    Attributes:
        lfw_dir: Path to the lfw dataset root directory.
        lfw_pairs_path: Path to lfw pairs.txt
        pairs: List contains lfw pairs.
               Each item in lfw pairs is a dict contating file_l, file_r, target and fold
        transform: Preprocessing of image.
        target_transform: Preprocessing of target.
    """

    def __init__(self, lfw_dir, lfw_pairs_path, transform=None, target_transform=None):
        self.lfw_dir = lfw_dir
        self.lfw_pairs_path = lfw_pairs_path
        self.pairs = self.parse_list(lfw_dir, lfw_pairs_path)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Get item with index.
        """
        pair = self.pairs[index]
        file_l = pair['file_l']
        file_r = pair['file_r']
        target = pair['target']
        fold = pair['fold']

        img_l = Image.open(file_l).convert('RGB')
        img_r = Image.open(file_r).convert('RGB')

        if self.transform is not None:
            img_l = self.transform(img_l)
            img_r = self.transform(img_r)

        if self.target_transform is not None:
            target = self.target_transform(target)
            fold = self.target_transform(target)

        return img_l, img_r, target, fold

    def __len__(self):
        """
        Return the number of image.
        """
        return len(self.pairs)

    def parse_list(self, lfw_dir, lfw_pairs_path):
        """
        Get lfw image pairs and parse it to list
        """
        pairs = []
        with open(lfw_pairs_path, 'r') as f:
            for i, line in enumerate(f.readlines()[1:]):
                pair = {}
                strings = line.strip().split()
                if len(strings) == 3:
                    pair['file_l'] = os.path.join(
                        lfw_dir, strings[0], strings[0] + '_' + '%04d' % int(strings[1]) + '.jpg')
                    pair['file_r'] = os.path.join(
                        lfw_dir, strings[0], strings[0] + '_' + '%04d' % int(strings[2]) + '.jpg')
                    pair['fold'] = math.floor(i / 600.0)
                    pair['target'] = 1
                elif len(strings) == 4:
                    pair['file_l'] = os.path.join(
                        lfw_dir, strings[0], strings[0] + '_' + '%04d' % int(strings[1]) + '.jpg')
                    pair['file_r'] = os.path.join(
                        lfw_dir, strings[2], strings[2] + '_' + '%04d' % int(strings[3]) + '.jpg')
                    pair['fold'] = math.floor(i / 600.0)
                    pair['target'] = -1
                else:
                    print 'Lfw pairs.txt file error!'
                    exit(-1)
                pairs.append(pair)
        return pairs
