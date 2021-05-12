from constants import IMG_SIZE
from torch.autograd import Variable
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils


def get_convnet_output_size(network, input_size=IMG_SIZE):
    input_size = input_size or IMG_SIZE

    if type(network) != list:
        network = [network]

    in_channels = network[0].conv.in_channels

    output = Variable(torch.ones(1, in_channels, input_size, input_size))
    output.require_grad = False
    for conv in network:
        output = conv.forward(output)

    return np.asscalar(np.prod(output.data.shape)), output.data.size()[2]


def imshow(img):
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def plot_images(trainloader):
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(utils.make_grid(images))
