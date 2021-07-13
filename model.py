import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.utils import utils
from modules.ConvNet import ConvNet
from constants import IMG_SIZE, NUM_CLASSES


# This is not the main file anymore, check networks.py
# I implemented this initially and achieved accuracy of 97.5%


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = ConvNet(3, 150, kernel_size=5, padding_size=0)
        self.conv2 = ConvNet(150, 200, kernel_size=3, padding_size=0)
        self.conv3 = ConvNet(200, 300, kernel_size=3, padding_size=0)

        inp, _ = utils.get_convnet_output_size([self.conv1,
                                                self.conv2,
                                                self.conv3],
                                               IMG_SIZE)
        self.fc1 = nn.Linear(inp, 50)
        self.fc2 = nn.Linear(50, NUM_CLASSES)

    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
