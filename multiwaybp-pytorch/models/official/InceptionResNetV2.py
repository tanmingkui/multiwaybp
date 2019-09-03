import torch.nn as nn


# ------------------------------------------------------------------------
# code from: https://github.com/Cadene/tensorflow-model-zoo.torch/blob/master/inceptionv4/pytorch_load.py
# inception-resnet-v2 stem
class BasicConv2d(nn.Module):
    def __init__(self, in_plane, out_plane, kernel_size, stride, padding=0):
        """
        :param in_plane: size of input plane
        :param out_plane: size of output plane
        :param kernel_size: kernel size of layer
        :param stride: stride of layer
        :param padding: padding with 0, default 0
        """
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_plane, out_plane, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_plane)
        self.relu = nn.ReLU(inplace=True)

        self.apply(initweights)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class Mixed_3a(nn.Module):
    def __init__(self):
        super(Mixed_3a, self).__init__()
        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv = BasicConv2d(in_plane=64, out_plane=96, kernel_size=3, stride=2)

        self.apply(initweights)

    def forward(self, x):
        out_1 = self.max_pooling(x)
        out_2 = self.conv(x)
        out = torch.cat((out_1, out_2), 1)
        return out


class Mixed_4a(nn.Module):
    def __init__(self):
        super(Mixed_4a, self).__init__()
        self.branch_1 = nn.Sequential(
            BasicConv2d(in_plane=160, out_plane=64, kernel_size=1, stride=1),
            BasicConv2d(in_plane=64, out_plane=96, kernel_size=3, stride=1)
        )
        self.branch_2 = nn.Sequential(
            BasicConv2d(in_plane=160, out_plane=64, kernel_size=1, stride=1),
            BasicConv2d(in_plane=64, out_plane=64, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            BasicConv2d(in_plane=64, out_plane=64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(in_plane=64, out_plane=96, kernel_size=3, stride=1)
        )
        self.apply(initweights)

    def forward(self, x):
        out_1 = self.branch_1(x)
        out_2 = self.branch_2(x)
        out = torch.cat((out_1, out_2), 1)
        return out


class Mixed_5a(nn.Module):
    def __init__(self):
        super(Mixed_5a, self).__init__()
        self.conv = BasicConv2d(in_plane=192, out_plane=192, kernel_size=3, stride=2)
        self.max_poiling = nn.MaxPool2d(kernel_size=3, stride=2)

        self.apply(initweights)

    def forward(self, x):
        out_1 = self.conv(x)
        out_2 = self.max_poiling(x)
        out = torch.cat((out_1, out_2), 1)
        return out


class InceptionStem(nn.Module):
    def __init__(self):
        super(InceptionStem, self).__init__()
        self.stem = nn.Sequential(
            BasicConv2d(in_plane=3, out_plane=32, kernel_size=3, stride=2),
            BasicConv2d(in_plane=32, out_plane=32, kernel_size=3, stride=1),
            BasicConv2d(in_plane=32, out_plane=64, kernel_size=3, stride=1, padding=1),
            Mixed_3a(),
            Mixed_4a(),
            Mixed_5a()
        )

        self.apply(initweights)

    def forward(self, x):
        out = self.stem(x)
        return out


class InceptionResNetA(nn.Module):
    def __init__(self, scale=1.0):
        super(InceptionResNetA, self).__init__()
        self.branch_1 = BasicConv2d(in_plane=384, out_plane=32, kernel_size=1, stride=1)
        self.branch_2 = nn.Sequential(
            BasicConv2d(in_plane=384, out_plane=32, kernel_size=1, stride=1),
            BasicConv2d(in_plane=32, out_plane=32, kernel_size=3, stride=1, padding=1)
        )
        self.branch_3 = nn.Sequential(
            BasicConv2d(in_plane=384, out_plane=32, kernel_size=1, stride=1),
            BasicConv2d(in_plane=32, out_plane=48, kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_plane=48, out_plane=64, kernel_size=3, stride=1, padding=1)
        )
        self.ensemble = nn.Conv2d(128, 384, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.scale = scale

        self.apply(initweights)

    def forward(self, x):
        residual = x
        out_1 = self.branch_1(x)
        out_2 = self.branch_2(x)
        out_3 = self.branch_3(x)
        out = torch.cat((out_1, out_2, out_3), 1)
        out = residual + self.scale*self.ensemble(out)
        out = self.relu(out)
        return out


class ReductionA(nn.Module):
    def __init__(self):
        super(ReductionA, self).__init__()
        self.branch_1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.branch_2 = BasicConv2d(in_plane=384, out_plane=384, kernel_size=3, stride=2)
        self.branch_3 = nn.Sequential(
            BasicConv2d(in_plane=384, out_plane=256, kernel_size=1, stride=1),
            BasicConv2d(in_plane=256, out_plane=256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_plane=256, out_plane=384, kernel_size=3, stride=2)
        )

        self.apply(initweights)

    def forward(self, x):
        out_1 = self.branch_1(x)
        out_2 = self.branch_2(x)
        out_3 = self.branch_3(x)
        out = torch.cat((out_1, out_2, out_3), 1)
        return out


class InceptionResNetB(nn.Module):
    def __init__(self, scale=1.0):
        super(InceptionResNetB, self).__init__()
        self.scale = scale
        self.branch_1 = BasicConv2d(in_plane=1152, out_plane=192, kernel_size=1, stride=1)
        self.branch_2 = nn.Sequential(
            BasicConv2d(in_plane=1152, out_plane=128, kernel_size=1, stride=1),
            BasicConv2d(in_plane=128, out_plane=160, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(in_plane=160, out_plane=192, kernel_size=(7, 1), stride=1, padding=(3, 0))
        )
        self.ensemble = nn.Conv2d(384, 1152, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

        self.apply(initweights)

    def forward(self, x):
        residual = x
        out_1 = self.branch_1(x)
        out_2 = self.branch_2(x)
        out = torch.cat((out_1, out_2), 1)
        out = self.ensemble(out)
        out = out*self.scale + residual
        out = self.relu(out)
        return out


class ReductionB(nn.Module):
    def __init__(self):
        super(ReductionB, self).__init__()
        self.branch_1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.branch_2 = nn.Sequential(
            BasicConv2d(in_plane=1152, out_plane=256, kernel_size=1, stride=1),
            BasicConv2d(in_plane=256, out_plane=384, kernel_size=3, stride=2)
        )
        self.branch_3 = nn.Sequential(
            BasicConv2d(in_plane=1152, out_plane=256, kernel_size=1, stride=1),
            BasicConv2d(in_plane=256, out_plane=288, kernel_size=3, stride=2)
        )
        self.branch_4 = nn.Sequential(
            BasicConv2d(in_plane=1152, out_plane=256, kernel_size=1, stride=1),
            BasicConv2d(in_plane=256, out_plane=288, kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_plane=288, out_plane=320, kernel_size=3, stride=2)
        )

        self.apply(initweights)

    def forward(self, x):
        out_1 = self.branch_1(x)
        out_2 = self.branch_2(x)
        out_3 = self.branch_3(x)
        out_4 = self.branch_4(x)
        out = torch.cat((out_1, out_2, out_3, out_4), 1)
        return out


class InceptionResNetC(nn.Module):
    def __init__(self, scale):
        super(InceptionResNetC, self).__init__()
        self.scale = scale
        self.branch_1 = BasicConv2d(in_plane=2144, out_plane=192, kernel_size=1, stride=1)
        self.branch_2 = nn.Sequential(
            BasicConv2d(in_plane=2144, out_plane=192, kernel_size=1, stride=1),
            BasicConv2d(in_plane=192, out_plane=224, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv2d(in_plane=224, out_plane=256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        )
        self.ensemble = BasicConv2d(in_plane=448, out_plane=2144, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

        self.apply(initweights)

    def forward(self, x):
        residual = x
        out_1 = self.branch_1(x)
        out_2 = self.branch_2(x)
        out = torch.cat((out_1, out_2), 1)
        out = self.ensemble(out)
        out = self.scale*out + residual
        out = self.relu(out)
        return out


class InceptionClassifier(nn.Module):
    def __init__(self, num_classes=1000):
        super(InceptionClassifier, self).__init__()
        self.num_classes = num_classes
        self.avg_pooling = nn.AvgPool2d(kernel_size=8, stride=1)
        self.drop_out = nn.Dropout(p=0.8, inplace=True)
        self.fc = nn.Linear(2144, self.num_classes)

        self.apply(initweights)

    def forward(self, x):
        out = self.avg_pooling(x)
        out = self.drop_out(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class InceptionResnetV2(object):
    def __init__(self, num_classes=1000, scale_a=0.17, scale_b=0.10, scale_c=0.20, stage=0):
        self.num_classes = num_classes
        self.scale_a = scale_a
        self.scale_b = scale_b
        self.scale_c = scale_c
        self.stage = stage

    def create(self):
        model = nn.Sequential(
            InceptionStem(),
            InceptionResNetA(scale=self.scale_a),
            InceptionResNetA(scale=self.scale_a),
            ReductionA(),
            InceptionResNetB(scale=self.scale_b),
            InceptionResNetB(scale=self.scale_b),
            InceptionResNetB(scale=self.scale_b),
            InceptionResNetB(scale=self.scale_b),
            ReductionB(),
            InceptionResNetC(scale=self.scale_c),
            InceptionResNetC(scale=self.scale_c),
            InceptionClassifier(num_classes=self.num_classes)
        )
        for i in range(self.stage):
            model = self.insertBlock(model)
            print "insert blocks"

        return model

    def insertBlock(self, model):
        model = model2list(model)

        model_len = len(model)
        insert_flag = True
        if model_len == 12:
            for i in range(2):
                model.insert(3, InceptionResNetA(self.scale_a))
                model.insert(8+i+1, InceptionResNetB(self.scale_b))
                model.insert(8+i+1, InceptionResNetB(self.scale_b))
                model.insert(11+2*(i+1), InceptionResNetC(self.scale_c))
        elif model_len == 20:
            model.insert(3, InceptionResNetA(self.scale_a))
            model.insert(11, InceptionResNetB(self.scale_b))
            model.insert(17, InceptionResNetC(self.scale_c))
        else:
            print "no block inserted with model length:", model_len
            insert_flag = False

        return nn.Sequential(*model), insert_flag

# ------------------------------------------------------------------------
