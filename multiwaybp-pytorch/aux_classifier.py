import torch.nn as nn


class AuxClassifier_A(nn.Module):
    """
    define auxiliary classifier
    type-A:
    BN->RELU->AVGPOOLING->FC
    """

    def __init__(self, in_channels, num_classes):
        super(AuxClassifier_A, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_channels, num_classes)

        # init params
        self.fc.bias.data.zero_()

    def forward(self, x):
        """
        forward propagation
        """
        out = self.bn(x)
        out = self.relu(out)
        out = out.mean(2).mean(2)
        out = self.fc(out)
        return out


class AuxClassifier_B(nn.Module):
    """
    define auxiliary classifier
    type-B:
    BN->RELU->FC
    """

    def __init__(self, in_channels, num_classes, feature_size):
        super(AuxClassifier_B, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_channels * feature_size, num_classes)

        # init params
        self.fc.bias.data.zero_()

    def forward(self, x):
        """
        forward propagation
        """
        out = self.bn(x)
        out = self.relu(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class AuxClassifier_C(nn.Module):
    """
    define auxiliary classifier
    type-C:
    AVGPOOLING->FC
    """

    def __init__(self, in_channels, num_classes):
        super(AuxClassifier_C, self).__init__()
        self.fc = nn.Linear(in_channels, num_classes)

        # init params
        self.fc.bias.data.zero_()

    def forward(self, x):
        """
        forward propagation
        """
        out = x.mean(2).mean(2)
        out = self.fc(out)
        return out

class AuxClassifier_D(nn.Module):
    """
    define auxiliary classifier
    type-D:
    RELU->AVGPOOLING->FC
    """

    def __init__(self, in_channels, num_classes):
        super(AuxClassifier_D, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_channels, num_classes)

        # init params
        self.fc.bias.data.zero_()

    def forward(self, x):
        """
        forward propagation
        """
        x = self.relu(x)
        out = x.mean(2).mean(2)
        out = self.fc(out)
        return out
