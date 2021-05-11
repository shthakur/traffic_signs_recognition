from torch import nn


class ConvNet(nn.Module):

    def __init__(self, in_c, out_c,
                 kernel_size,
                 padding_size='same',
                 pool_stride=2):
        super().__init__()

        if padding_size == 'same':
            padding_size = kernel_size // 2
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, padding=padding_size)
        self.max_pool2d = nn.MaxPool2d(pool_stride, stride=pool_stride)

    def forward(self, x):
        return self.max_pool2d(nn.functional.relu(self.conv(x)))
