import torch
from torch.nn import functional as F
from torch import nn

class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return F.conv2d(x, self.weight.unsqueeze(0).unsqueeze(0), self.bias)

    