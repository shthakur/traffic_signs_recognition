from torch import nn


class FullyConnected(nn.Module):
    def __init__(self, input_nbr, out_nbr):
        super(FullyConnected, self).__init__()
        self.input_nbr = input_nbr
        self.lin = nn.Linear(input_nbr, out_nbr)
        self.rel = nn.ReLU()

    def forward(self, input):
        return self.rel(self.lin(input))
