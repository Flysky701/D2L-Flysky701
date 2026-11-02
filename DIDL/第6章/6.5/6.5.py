"""6.5 汇聚层：窗口计算、步幅与填充、各通道独立池化。"""

import argparse

import torch
from torch import nn


def pool2d(X, pool_size, mode="max"):
    """教材的二维手写版本：步幅固定为 1，无填充，输入为浮点张量。"""
    if X.ndim != 2 or mode not in ("max", "avg"):
        raise ValueError("需要二维输入，mode 只能是 max 或 avg")
    p_h, p_w = pool_size
    if not (0 < p_h <= X.shape[0] and 0 < p_w <= X.shape[1]):
        raise ValueError("窗口必须为正，且不超过输入高宽")
    Y = X.new_zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            window = X[i:i + p_h, j:j + p_w]
            Y[i, j] = window.max() if mode == "max" else window.mean()
    return Y


def demo_basic():
    print("\n6.5.1 最大池化与平均池化：手写步幅为 1")
    X = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    print("输入:\n", X)
    print("左上角窗口:\n", X[:2, :2])
    for mode, layer in (("max", nn.MaxPool2d(2, stride=1)),
                        ("avg", nn.AvgPool2d(2, stride=1))):
        Y = pool2d(X, (2, 2), mode)
        Y_torch = layer(X.reshape(1, 1, 3, 3))[0, 0]
        print(f"{mode} 手写输出:\n{Y}")
        torch.testing.assert_close(Y, Y_torch)
    print("最大/平均池化：手写与 PyTorch 一致")


def demo_stride_padding():
    print("\n6.5.2 步幅、默认步幅和填充")
    X = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    print("输入:\n", X[0, 0])
    for name, cls in (("最大汇聚", nn.MaxPool2d), ("平均汇聚", nn.AvgPool2d)):
        for stride in (1, 2):
            pool = cls(kernel_size=2, stride=stride)
            Y = pool(X)
            print(f"{name} kernel=2 stride={stride}: {tuple(Y.shape)}\n{Y[0, 0]}")
        # 保留原来的 4×4、窗口2、步幅2示例，并核对省略步幅的情况。
        default_pool = cls(kernel_size=2)
        torch.testing.assert_close(default_pool(X), cls(2, stride=2)(X))
        print(f"{name} 省略 stride 时，实际 stride={default_pool.stride}")

    for kernel, stride, padding in ((3, 2, 1), ((2, 3), (2, 3), (0, 1))):
        pool = nn.MaxPool2d(kernel, stride=stride, padding=padding)
        Y = pool(X)
        print(f"最大汇聚 kernel={kernel} stride={stride} padding={padding}:")
        print(tuple(Y.shape), "\n", Y[0, 0])
    # MaxPool2d 的隐式填充值为 -inf，不能把边界当成普通零。
    negative = -torch.arange(1.0, 10.0).reshape(1, 1, 3, 3)
    edge = nn.MaxPool2d(3, stride=1, padding=1)(negative)
    torch.testing.assert_close(edge[0, 0, 0, 0], torch.tensor(-1.0))
    print("全负输入的左上角池化结果:", edge[0, 0, 0, 0].item())


def demo_channels():
    print("\n6.5.3 多通道：各自池化，不做通道求和")
    X = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    X_multi = torch.cat((X, X + 1), dim=1)  # 沿已有通道轴拼接，不新增轴。
    pool = nn.MaxPool2d(3, stride=2, padding=1)
    Y = pool(X_multi)
    print("输入形状:", tuple(X_multi.shape))
    print("输出形状:", tuple(Y.shape))
    print("两个输出通道:\n", Y[0])
    # 与两个通道分别调用池化逐一比较，验证没有跨通道混合。
    for c in range(X_multi.shape[1]):
        torch.testing.assert_close(Y[:, c:c + 1], pool(X_multi[:, c:c + 1]))
    print("池化层可训练参数数量:", sum(p.numel() for p in pool.parameters()))
    print("逐通道独立计算核对通过")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", default="all",
                        choices=("all", "6.5.1", "6.5.2", "6.5.3"))
    parser.add_argument("--question", action="store_true",
                        help="只显示复习题，不计算答案")
    args = parser.parse_args()
    if args.question:
        print("复习题：单通道输入为 [[1,3,2],[4,0,5],[2,6,1]]。")
        print("窗口2×2、步幅1、无填充，分别手算最大池化和平均池化。")
        print("输入形状若为 (2,3,4,4)，窗口2×2，无填充，步幅1和2时输出各是什么形状？")
        print("这些题保留为复习任务；下一主线是 6.6 LeNet。")
        return
    demos = {"6.5.1": demo_basic, "6.5.2": demo_stride_padding,
             "6.5.3": demo_channels}
    if args.section == "all":
        for demo in demos.values():
            demo()
    else:
        demos[args.section]()


if __name__ == "__main__":
    main()
