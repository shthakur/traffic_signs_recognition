from .FullyConnected import FullyConnected
from .ConvNet import ConvNet
from .utils import get_convnet_output_size
from .Classifier import Classifier

from torch import nn
import torch


class LocalizationNetwork(nn.Module):
    nbr_params = 6
    init_bias = torch.Tensor([1, 0, 0, 0, 1, 0])

    def __init__(self, conv_params, kernel_sizes,
                 input_size, input_channels=3):
        super(LocalizationNetwork, self).__init__()

        if not kernel_sizes:
            kernel_sizes = [5, 5]

        if len(kernel_sizes) != 2:
            raise Exception("Number of kernel sizes != 2")

        self.conv1 = ConvNet(input_channels, conv_params[0],
                             kernel_size=kernel_sizes[0])
        self.conv2 = ConvNet(conv_params[0], conv_params[1],
                             kernel_size=kernel_sizes[1])
        conv_output_size, _ = get_convnet_output_size([self.conv1, self.conv2],
                                                      input_size)

        self.fc = FullyConnected(conv_output_size, conv_params[2])
        self.classifier = Classifier(conv_params[2], self.nbr_params)

        self.classifier.lin.weight.data.fill_(0)
        self.classifier.lin.bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])
        self.droput = nn.Dropout2d()

    def forward(self, input):
        conv_output = self.dropout(self.conv2(self.conv1(input)))
        conv_output = conv_output.view(conv_output.size()[0], -1)
        return self.classifier(self.fc(conv_output))
