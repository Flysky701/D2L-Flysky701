import torch
from torch import nn


# 尝试写一个Inception 模块
# PDF版本
class Inception(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, pool_proj):
        super(Inception, self).__init__()
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1),
            nn.ReLU()
        )
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels, red_3x3, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(red_3x3, out_3x3, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.path3 = nn.Sequential(
            nn.Conv2d(in_channels, red_5x5, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(red_5x5, out_5x5, kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.path4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.ReLU()
        )

    # 简单介绍下
    # Path1： 简单的1X1 卷积
    # Path2： 1X1 卷积 + 3X3 卷积
    # Path3： 1X1 卷积 + 5X5 卷积
    # Path4： 最大池化 + 1X1 卷积

    def forward(self, x):
        out1 = self.path1(x)
        out2 = self.path2(x)
        out3 = self.path3(x)
        out4 = self.path4(x)
        # 将四条路径的输出在通道维度上拼接起来
        return torch.cat((out1, out2, out3, out4), dim=1)


def build_googlenet():
    # b1、b2：先从灰度图片中提取特征，并缩小高和宽。
    b1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )
    b2 = nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
        nn.Conv2d(64, 192, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )

    # Inception 参数依次为：输入通道、路径 1 输出、路径 2 压缩/输出、
    # 路径 3 压缩/输出、路径 4 输出。四路的最终输出通道数相加。
    b3 = nn.Sequential(
        Inception(192, 64, 96, 128, 16, 32, 32),       # 输出 256 通道
        Inception(256, 128, 128, 192, 32, 96, 64),     # 输出 480 通道
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )
    b4 = nn.Sequential(
        Inception(480, 192, 96, 208, 16, 48, 64),     # 输出 512 通道
        Inception(512, 160, 112, 224, 24, 64, 64),    # 输出 512 通道
        Inception(512, 128, 128, 256, 24, 64, 64),    # 输出 512 通道
        Inception(512, 112, 144, 288, 32, 64, 64),    # 输出 528 通道
        Inception(528, 256, 160, 320, 32, 128, 128),   # 输出 832 通道
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )
    b5 = nn.Sequential(
        Inception(832, 256, 160, 320, 32, 128, 128),   # 输出 832 通道
        Inception(832, 384, 192, 384, 48, 128, 128),   # 输出 1024 通道
        nn.AdaptiveAvgPool2d((1, 1)),                # 每个通道取一个平均值
        nn.Flatten(),                               # (N, 1024, 1, 1) → (N, 1024)
    )

    # b5 输出 1024 个特征，最后映射到 Fashion-MNIST 的 10 个类别。
    return nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))


if __name__ == "__main__":
    # 测试Inception模块
    x = torch.randn(1, 3, 224, 224)  # 假设输入是一个批次的图像，大小为224x224，通道数为3
    model = Inception(in_channels=3, out_1x1=64, red_3x3=128, out_3x3=128, red_5x5=32, out_5x5=64, pool_proj=32)
    output = model(x)
    print(output.shape)  # 输出的形状应该是(1, 64+128+64+32, 224, 224) = (1, 288, 224, 224)

    # 教材使用 96×96 灰度图片；这里用随机输入检查完整网络的数据流。
    net = build_googlenet()
    net.eval()
    x = torch.randn(1, 1, 96, 96)
    print("GoogLeNet input:", tuple(x.shape))
    with torch.no_grad():
        for name, layer in zip(("b1", "b2", "b3", "b4", "b5", "classifier"), net):
            x = layer(x)
            print(f"{name} output:", tuple(x.shape))
