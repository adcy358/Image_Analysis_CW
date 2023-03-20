import torch.nn as nn
import torch
import pickle


class Convlayer(nn.Module):
    """
    ** CNN block **
    Conv + BatchNorm + LeakyReLU
    """
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding):
        super(Convlayer, self).__init__()
        self.in_channels = input_channels
        self.out_channels = output_channels
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(output_channels)
        self.lrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.lrelu(out)
        return out

    
class ResidualBlock(nn.Module):
    """
    *** Residual block***
    """
    def __init__(self, layers):
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        residual_input = x
        out = self.layers(x)
        out += residual_input
        return out


class YOLOv1(nn.Module):
    """
    *YOLO v1*
    Implementation of YOLO v1 architecture
    """
    def __init__(self, input_channels, arch=None, S=7, B=2, C=4):
        """
        :param input_channels: number of input channels
        :param arch: description of the network architecture
        :param S: size of the squares
        :param B: number of boxes per square
        :param C: number of classes
        """
        super(YOLOv1, self).__init__()
        if arch is not None:
            self.arch = arch
        else:
            self.arch = pickle.load(open("Architectures/default_arch.pkl", "rb"))
        self.in_channels = input_channels
        self.darknet = self._get_darknet()
        self.fc = nn.Sequential(
            nn.Linear(1024 * S * S, 4096),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (B * 5 + C))
        )

    def forward(self, x):
        out = self.darknet(x)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)
        return out

    def _get_darknet(self):
        """
        Creates the darknet layers
        """
        n_in = self.in_channels
        darknet = []
        for layer in self.arch:
            block, n_in = self._make_block(n_in, *layer)
            darknet.append(block)
        return nn.Sequential(*darknet)

    def _make_block(self, n_int, convs, maxpool=False):
        """
        Creates a block of ConvLayer, given their parameters (convs).
        If maxpool = True, it adds a final MaxPool2d layer.
        """
        layers = []
        n_input = n_int
        for c in convs:
            layers.append(Convlayer(n_input, *c))
            n_input = c[0]
        n_out = layers[-1].out_channels
        if maxpool:
            layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        return nn.Sequential(*layers), n_out


    
class YOLO(nn.Module):
    """
    *YOLO v1*
    Implementation of YOLO v1 architecture
    """
    def __init__(self, input_channels, arch="darknetv1", S=7, B=2, C=4):
        """
        :param input_channels: number of input channels
        :param arch: description of the network architecture
        :param S: size of the squares
        :param B: number of boxes per square
        :param C: number of classes
        """
        super(YOLO, self).__init__()
        self.architecture = pickle.load(open(f"Architectures/{arch}.pkl", "rb"))
        self.in_channels = input_channels
        self.darknet = self._create_darknet()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, S * S * (B * 5 + C))

    def forward(self, x):
        out = self.darknet(x)
        out = self.avgpool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)
        return out

    def _create_darknet(self):
        """
        Creates the darknet layers
        """
        n_in = self.in_channels
        darknet = []
        for layer in self.architecture:
            block, n_in = self._make_block(n_in, *layer)
            darknet.append(block)
        return nn.Sequential(*darknet)

    def _make_block(self, n_int, convs, type=None):
        """
        Creates a block of ConvLayer, given their parameters (convs).
        If maxpool = True, it adds a final MaxPool2d layer.
        """
        layers = []
        n_input = n_int
        for c in convs:
            layers.append(Convlayer(n_input, *c))
            n_input = c[0]
        n_out = layers[-1].out_channels
        if type == 'maxpool':
            layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
            return nn.Sequential(*layers), n_out
        elif type == 'residual':
            return ResidualBlock(layers), n_out
        return nn.Sequential(*layers), n_out

class YOLO2(nn.Module):
    """
    *YOLO v1*
    Implementation of YOLO v1 architecture
    """
    def __init__(self, input_channels, arch="darknetv1", S=7, B=2, C=4):
        """
        :param input_channels: number of input channels
        :param arch: description of the network architecture
        :param S: size of the squares
        :param B: number of boxes per square
        :param C: number of classes
        """
        super(YOLO2, self).__init__()
        self.architecture = pickle.load(open(f"Architectures/{arch}.pkl", "rb"))
        self.in_channels = input_channels
        self.darknet = self._create_darknet()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(1024, 4096)  
        self.LeakyReLU = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(4096, S*S*(B*5+C))
        self.S = S
        self.B = B 
        self.C = C

    def forward(self, x):
        out = self.darknet(x)
        out = self.avgpool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc1(out)
        out = self.LeakyReLU(out)
        out = self.fc2(out)
        
        return out

    def _create_darknet(self):
        """
        Creates the darknet layers
        """
        n_in = self.in_channels
        darknet = []
        for layer in self.architecture:
            block, n_in = self._make_block(n_in, *layer)
            darknet.append(block)
        return nn.Sequential(*darknet)

    def _make_block(self, n_int, convs, type=None):
        """
        Creates a block of ConvLayer, given their parameters (convs).
        If maxpool = True, it adds a final MaxPool2d layer.
        """
        layers = []
        n_input = n_int
        for c in convs:
            layers.append(Convlayer(n_input, *c))
            n_input = c[0]
        n_out = layers[-1].out_channels
        if type == 'maxpool':
            layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
            return nn.Sequential(*layers), n_out
        elif type == 'residual':
            return ResidualBlock(layers), n_out
        return nn.Sequential(*layers), n_out