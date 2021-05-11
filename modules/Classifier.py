from torch import nn


class Classifier(nn.Module):
    def __init__(self, input_nbr, out_nbr):
        super(Classifier, self).__init__()
        self.input_nbr = input_nbr
        self.lin = nn.Linear(input_nbr, out_nbr)

    def forward(self, x):
        return self.lin(x)
