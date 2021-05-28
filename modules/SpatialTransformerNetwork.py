from torch import nn
from .LocalizationNetwork import LocalizationNetwork
from constants import IMG_SIZE


class SpatialTransformerNetwork(nn.Module):
    def __init__(self, params, kernel_sizes, input_size=IMG_SIZE,
                 input_channels=1):
        super(SpatialTransformerNetwork, self).__init__()
        self.localization_network = LocalizationNetwork(params,
                                                        kernel_sizes,
                                                        input_size,
                                                        input_channels)

    def forward(self, input):
        out = self.localization_network(input)
        out = out.view(out.size()[0], 2, 3)
        grid = nn.functional.affine_grid(out, input.size())
        return nn.functional.grid_sample(input, grid)
