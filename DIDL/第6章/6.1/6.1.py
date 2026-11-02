"""6.1 从全连接层到卷积：参数数量的入门对比。"""

import torch
from torch import nn


def main():
    # 同一张单通道 8×8 图像；两种层的输出结构不同。
    X = torch.zeros(1, 1, 8, 8)  # (批量, 通道, 高, 宽)
    linear = nn.Linear(8 * 8, 6 * 6)
    conv = nn.Conv2d(1, 1, kernel_size=3)

    print("输入形状:", tuple(X.shape))
    for name, layer, inputs in (
        ("全连接", linear, X.flatten(start_dim=1)),
        ("卷积", conv, X),
    ):
        count = sum(param.numel() for param in layer.parameters())
        print(f"{name}: 输出 {tuple(layer(inputs).shape)}, 参数数量 {count}")

    X[0, 0, 0:3, 1] = 1
    X[0, 0, 4:7, 5] = 1
    kernel = torch.tensor([[-1., 1., 0.],
                           [-1., 1., 0.],
                           [-1., 1., 0.]])
    with torch.no_grad():
        conv.weight.copy_(kernel[None, None])
        conv.bias.zero_()
        Y = conv(X)

    print("\n图像（0 为背景，1 为亮像素；两处相同的竖线）:\n", X[0, 0])
    print("共享的 3×3 卷积核:\n", kernel)
    print("扫描整张图像得到的特征图:\n", Y[0, 0])
    for row, col in ((0, 0), (4, 4)):
        patch = X[0, 0, row:row + 3, col:col + 3]
        response = (patch * kernel).sum().item()
        print(f"\n输出[{row}, {col}] 只连接图像"
              f"[{row}:{row + 3}, {col}:{col + 3}]:\n{patch}")
        print(f"局部像素与同一个卷积核逐项相乘再求和: "
              f"{response:g} = {Y[0, 0, row, col].item():g}")
    print("两处竖线的位置不同，但使用相同权重，得到相同响应。")


if __name__ == "__main__":
    main()
