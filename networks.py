from torch import nn
from torch.autograd import Variable
import torch
import numpy as np
from constants import NUM_CLASSES, IMG_SIZE
from modules.SpatialTransformerNetwork import SpatialTransformerNetwork
from modules.FullyConnected import FullyConnected
from modules.SoftMaxClassifier import SoftMaxClassifier
from modules.Classifier import Classifier
from modules.ConvNet import ConvNet
from modules.utils import utils


class GeneralNetwork(nn.Module):
    def __init__(self, opt):
        super(GeneralNetwork, self).__init__()

        if not opt.cnn:
            opt.cnn = '100, 150, 250, 350'
        self.kernel_sizes = [5, 3, 1]
        conv_params = list(map(int, opt.cnn.split(",")))

        self.conv1 = ConvNet(1, conv_params[0], kernel_size=self.kernel_sizes[0],
                             padding_size=0)
        self.conv2 = ConvNet(conv_params[0], conv_params[1],
                             kernel_size=self.kernel_sizes[1],
                             padding_size=0)

        conv_output_size, _ = utils.get_convnet_output_size([self.conv1, self.conv2])

        self.fc = FullyConnected(conv_output_size, conv_params[2])
        self.classifier = SoftMaxClassifier(conv_params[2], NUM_CLASSES)

        self.locnet_1 = None
        if opt.st and opt.locnet:
            params = list(map(int, opt.locnet.split(",")))
            self.locnet_1 = SpatialTransformerNetwork(params,
                                                      kernel_sizes=[7, 5])

        self.locnet_2 = None
        if opt.st and opt.locnet2:
            params = list(map(int, opt.locnet2.split(",")))
            _, current_size = utils.get_convnet_output_size([self.conv1])
            self.locnet_2 = SpatialTransformerNetwork(params,
                                                      [5, 3],
                                                      current_size,
                                                      conv_params[0])
        self.dropout = nn.Dropout2d()

    def forward(self, x):
        if self.locnet_1:
            x = self.locnet_1(x)

        x = self.conv1(x)

        if self.locnet_2:
            x = self.locnet_2(x)

        return self.classifier(self.fc(self.dropout(self.conv2(x))))


class IDSIANetwork(GeneralNetwork):
    def __init__(self, opt):
        super().__init__(opt)
        conv_params = list(map(int, opt.cnn.split(",")))

        self.conv3 = ConvNet(conv_params[1], conv_params[2], kernel_size=self.kernel_sizes[2],
                             padding_size=0)
        conv_output_size, _ = utils.get_convnet_output_size([self.conv1,
                                                             self.conv2,
                                                             self.conv3])
        self.fc = FullyConnected(conv_output_size, conv_params[3])
        self.classifier = SoftMaxClassifier(conv_params[3], NUM_CLASSES)

        self.locnet_3 = None
        if opt.st and opt.locnet3:
            params = list(map(int, opt.locnet3.split(",")))
            _, current_size = utils.get_convnet_output_size([self.conv1, self.conv2])
            self.locnet_3 = SpatialTransformerNetwork(params,
                                                      [3, 3],
                                                      current_size,
                                                      conv_params[1])

    def forward(self, x):
        if self.locnet_1:
            x = self.locnet_1(x)

        x = self.conv1(x)
        x = self.dropout(x)

        if self.locnet_2:
            x = self.locnet_2(x)

        x = self.conv2(x)
        x = self.dropout(x)

        if self.locnet_3:
            x = self.locnet_3(x)

        x = self.conv3(x)
        x = self.dropout(x)

        x = x.view(x.size()[0], -1)
        return self.classifier(self.fc(x))
