from torch import nn
from .Classifier import Classifier


class SoftMaxClassifier(Classifier):
    def __init__(self, in_len, out_len):
        super().__init__(in_len, out_len)

    def forward(self, x):
        x = super().forward(x)
        return nn.functional.log_softmax(x)
