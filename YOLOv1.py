import torch.nn as nn
import torch


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


class YOLOv1(nn.Module):
    """
    *YOLO v1*
    Implementation of YOLO v1 architecture
    """

    def __init__(self, input_channels, arch, S, B, C):
        """
        :param input_channels: number of input channels
        :param arch: description of the network architecture
        :param S: size of the squares
        :param B: number of boxes per square
        :param C: number of classes
        """
        super(YOLOv1, self).__init__()
        self.in_channels = input_channels
        self.arch = arch
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

