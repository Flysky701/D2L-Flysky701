"""6.3 填充和步幅：观察输出空间尺寸。"""

import torch
from torch import nn


def comp_conv2d(conv2d, X):
    """计算卷积层的输出。"""
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])

conv2d = nn.Conv2d(1, 1, kernel_size=3)
X = torch.rand(size=(8, 8))
print("输入形状:", tuple(X.shape))
Y = comp_conv2d(conv2d, X)
print("输出形状:", tuple(Y.shape))




def main():
    X = torch.zeros(1, 1, 8, 8)
    print("输入形状:", tuple(X.shape))
    for padding, stride in ((0, 1), (1, 1), (1, 2)):
        conv = nn.Conv2d(1, 1, kernel_size=3, padding=padding, stride=stride)
        print(f"padding={padding}, stride={stride}: {tuple(conv(X).shape)}")

    


    

    # TODO：先用尺寸公式预测结果，再尝试非方形卷积核和不同方向的步幅。


if __name__ == "__main__":
    main()
