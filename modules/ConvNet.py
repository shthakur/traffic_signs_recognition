from torch import nn
from transforms import gaussian_blur


class ConvNet(nn.Module):

    def __init__(self, in_c, out_c,
                 kernel_size,
                 padding_size='same',
                 pool_stride=2,
                 batch_norm=True):
        super().__init__()

        if padding_size == 'same':
            padding_size = kernel_size // 2
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, padding=padding_size)
        self.max_pool2d = nn.MaxPool2d(pool_stride, stride=pool_stride)
        self.batch_norm = batch_norm
        self.batch_norm_2d = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = self.max_pool2d(nn.functional.leaky_relu(self.conv(x)))

        if self.batch_norm:
            return self.batch_norm_2d(x)
        else:
            return x
