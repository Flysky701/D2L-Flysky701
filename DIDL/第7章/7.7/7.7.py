import torch
from torch import nn
from d2l import torch as d2l

def conv_block(in_channels, out_channels, kernel_size, strides=1, padding=0):
    """卷积块：卷积层+批量归一化层+ReLU激活函数"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )


class DenseBlock(nn.Module):
    """稠密块"""
    def __init__(self, num_convs, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential()
        for i in range(num_convs):
            # 每个卷积层的输入通道数为 in_channels + i * out_channels
            self.net.add_module(f'conv_{i}', conv_block(
                in_channels + i * out_channels, out_channels, kernel_size=3, padding=1))

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 在通道维上将输入和输出连结
            X = torch.cat((X, Y), dim=1)
        return X

print("DenseBlock 测试")
blk = DenseBlock(2, 3, 10)
X = torch.rand(4, 3, 8, 8)
Y = blk(X)


def transition_block(in_channels, out_channels):
    """过渡块：卷积层+平均池化层"""
    return nn.Sequential(
        nn.BatchNorm2d(in_channels), nn.Relu(),
        conv_block(in_channels, out_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )

blk = transition_block(23, 10)
blk(Y).shape
