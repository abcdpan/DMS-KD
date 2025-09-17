from torch import nn


class Reg(nn.Module):
    """Linear regressor"""
    def __init__(self, dim_in=1024, dim_out=1024):
        super(Reg, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = self.linear(x)
        return x