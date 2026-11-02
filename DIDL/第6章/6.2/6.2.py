"""6.2 图像卷积：先用 PyTorch 运行一个二维互相关例子。"""

import torch
from torch.nn import functional as F
from torch import nn

from importlib import import_module
Conv2D = import_module("class").Conv2D


def main():
    X = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    # conv2d 接收四维张量，实际计算的是互相关，不翻转卷积核。
    Y = F.conv2d(X.reshape(1, 1, 3, 3), K.reshape(1, 1, 2, 2))
    # print("输入 X:\n", X)
    # print("卷积核 K:\n", K)
    # print("输出 Y:\n", Y[0, 0])

    def corr2d(X, K):
        h, w = K.shape
        Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
        return Y
    Y_manual = corr2d(X, K)
    print("手动计算输出 Y_manual:\n", Y_manual)

    X = torch.ones((6, 8))
    X[:, 1:2] = 0
    X[:, 5:6] = 0

    print("输入 X:\n", X)

    K = torch.tensor([[1.0, -1.0]])
    Y = corr2d(X, K)
    # print("卷积核 K:\n", K)
    # print("输出 Y:\n", Y)
    # Y2 = corr2d(X.t(), K)
    # print("输出 Y2:\n", Y2)


    conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=True)
    X = X.reshape(1, 1, 6, 8)
    Y = Y.reshape(1, 1, 6, 7)
    # Y = [0, 1, 0, 0, 0, 1, 0] X N
    # Y 是卷积层的输出，目标是让卷积层学会卷积核 K。
    lr = 3e-2


    for i in range(10):
        Y = conv2d(X)
        l = (Y - torch.ones_like(Y)).pow(2).sum()
        l.backward()
        print(f"迭代 {i + 1}, loss {l.item()}")
        conv2d.weight.data[:] -= lr * conv2d.weight.grad
        conv2d.bias.data[:] -= lr * conv2d.bias.grad
        conv2d.weight.grad.zero_()
        conv2d.bias.grad.zero_()

    print("学习后的卷积核:\n", conv2d.weight.data.reshape((1, 2)))







if __name__ == "__main__":
    main()
