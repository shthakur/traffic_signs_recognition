from torch import nn


class FullyConnected(nn.Module):
    def __init__(self, input_nbr, out_nbr):
        super(FullyConnected, self).__init__()
        self.input_nbr = input_nbr
        self.lin = nn.Linear(input_nbr, out_nbr)
        self.rel = nn.LeakyReLU()
        self.dropout = nn.Dropout()

    def forward(self, input):
        return self.dropout(self.rel(self.lin(input)))
