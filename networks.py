from torch import nn
from torch.autograd import Variable
import torch
import numpy as np

BASE_INPUT_SIZE = 32
NUM_CLASSES = 43


def new_conv(input_c, out_c, filter_size=5, padding_size=2,
             pooling_size=2):
    model = nn.Sequential(
        nn.Conv2d(input_c, out_c, filter_size, padding=padding_size),
        nn.ReLU(),
        nn.MaxPool2d(pooling_size)
    )

    return model


def new_fc(input_nbr, out_nbr):
    model = nn.Sequential(
        nn.Linear(input_nbr, out_nbr),
        nn.ReLU()
    )

    return model


def new_classifier(input_nbr, out_nbr):
    model = nn.Sequential(
        nn.Linear(input_nbr, out_nbr)
    )

    return model


def get_convs_output_size(network, input_size=BASE_INPUT_SIZE):
    input_size = input_size or BASE_INPUT_SIZE

    if type(network) != list:
        network = [network]

    in_channels = network[0][0].in_channels

    output = Variable(torch.Tensor(1, in_channels, input_size, input_size))
    output.require_grad = False

    for conv in network:
        output = conv.forward(output)

    return np.prod(output.data.shape), output.data.size()[2]


class LocalizationNetwork(nn.Module):
    nbr_params = 6
    init_bias = torch.Tensor([1, 0, 0, 0, 1, 0])

    def __init__(self, conv_params, input_size, input_channels=3):
        super(LocalizationNetwork, self).__init__()

        self.conv1 = new_conv(input_channels, conv_params[0])
        self.conv2 = new_conv(conv_params[0], conv_params[1])
        conv_output_size, _ = get_convs_output_size([self.conv1, self.conv2],
                                                    input_size / 2)
        self.fc = new_fc(conv_output_size, conv_params[2])
        self.classifier = new_classifier(conv_params[2], nbr_params)
        self.classifier.weight.data.zero_()
        self.classifier.bias.data.copy_(init_bias)

    def forward(self, input):
        conv_output = self.conv2(self.conv1(input))
        conv_output = conv_output.view(conv_output.size()[0], -1)

        return self.classifier(self.fc(conv_output))


class SpatialTransformer(nn.Module):
    def __init__(self, params, input_size=BASE_INPUT_SIZE
                 input_channels=3):
        super(SpatialTransformer, self).__init__()
        conv_params = list(map(int, params.split(",")))
        self.localization_network = LocalizationNetwork(conv_params,
                                                        input_size,
                                                        input_channels)

    def forward(self, input):
        out = self.localization_network(input)
        out = out.view(out.size()[0], 2, 3)
        grid = nn.functional.affine_grid(out, input.size())
        return nn.functional.grid_sample(input, grid)


class GeneralNetwork(nn.Module):
    def __init__(opt):
        super(Network, self).__init__()
        conv_params = list(map(int, opt.cnn.split(",")))
        self.conv1 = new_conv(3, conv_params[0])
        self.conv2 = new_conv(conv_params[0], conv_params[1])
        conv_output_size, _ = get_convs_output_size([self.conv1, self.conv2])
        self.fc = new_fc(conv_output_size, conv_params[2])
        self.classifier = new_classifier(conv_params[2], NUM_CLASSES)
        self.locnet_1 = None
        if opt.st and opt.locnet:
            self.locnet_1 = SpatialTransformer(opt.locnet)

        self.locnet_2 = None
        if opt.st and opt.locnet2:
            _, current_size = get_convs_output_size([self.conv1])
            self.locnet_2 = SpatialTransformer(opt.locnet2,
                                               current_size,
                                               conv_params[0])

    def forward(self, input):
        if self.locnet_1:
            input = self.locnet_1(input)

        input = self.conv1(input)

        if self.locnet_2:
            input = self.locnet_2(input)

        return self.classifier(self.fc(self.conv2(input)))


class IDSIANetwork(GeneralNetwork):
    def __init__(opt):
        super().__init__(opt)
        conv_params = list(map(int, opt.cnn.split(",")))
        self.conv3 = new_conv(conv_params[1], conv_params[2])
        conv_output_size, _ = get_convs_output_size([self.conv1,
                                                     self.conv2,
                                                     self.conv3])
        self.fc = new_fc(conv_output_size, conv_params[3])
        self.classifier = new_classifier(conv_params[3], NUM_CLASSES)

        self.locnet_3 = None
        if opt.st and opt.locnet3:
            _, current_size = get_convs_output_size([self.conv1, self.conv2])
            self.locnet_3 = SpatialTransformer(opt.locnet3,
                                               current_size,
                                               conv_params[1])

    def forward(self, input):
        if self.locnet_1:
            input = self.locnet_1(input)

        input = self.conv1(input)

        if self.locnet_2:
            input = self.locnet_2(input)

        input = self.conv2(input)

        if self.locnet_3:
            input = self.locnet_3(input)
        return self.classifier(self.fc(self.conv3(input)))
