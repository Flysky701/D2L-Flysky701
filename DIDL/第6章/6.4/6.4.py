"""6.4.1—6.4.3：通道求和、输出堆叠与 1×1 卷积的矩阵实现。"""

import argparse

import torch
from torch import nn
from torch.nn import functional as F


def corr2d(X, K):
    """单通道二维互相关，沿用 6.2 的窗口计算；不填充、步幅为 1。"""
    h, w = K.shape
    Y = X.new_zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


def corr2d_multi_in(X, K):
    """X: (输入通道, 高, 宽)，K: (输入通道, 核高, 核宽)。"""
    # zip 会在较短的一方结束，所以先检查通道数，避免漏算通道。
    if X.ndim != 3 or K.ndim != 3:
        raise ValueError("X 和 K 都应是三维张量")
    if X.shape[0] != K.shape[0] or X.shape[0] == 0:
        raise ValueError("X 和 K 的输入通道数必须相同且大于 0")
    # 遍历第 0 维，每次取一张二维输入和它对应的二维核。
    # sum 对各张结果逐位置相加，不会把输出的高、宽加没。
    return sum(corr2d(x, k) for x, k in zip(X, K))


def corr2d_multi_in_out(X, K):
    """X: (c_i,H,W)，K: (c_o,c_i,k_h,k_w)，返回 (c_o,H_out,W_out)。"""
    if K.ndim != 4 or K.shape[0] == 0:
        raise ValueError("K 应是四维张量，且至少有一个输出通道")
    # 一个 k 覆盖全部输入通道，corr2d_multi_in 得到一张二维结果。
    # stack 新增第 0 维，分别保留每个输出通道，不把它们相加。
    return torch.stack([corr2d_multi_in(X, k) for k in K], dim=0)


def corr2d_multi_in_out_1x1(X, K):
    """仅用于核为 1×1、步幅 1、无填充、无偏置的普通卷积。"""
    if X.ndim != 3 or K.ndim != 4 or K.shape[2:] != (1, 1):
        raise ValueError("需要 X=(c_i,H,W)、K=(c_o,c_i,1,1)")
    c_i, h, w = X.shape
    c_o = K.shape[0]
    if c_i == 0 or c_o == 0 or K.shape[1] != c_i:
        raise ValueError("输入通道数必须匹配，输入与输出通道数必须大于 0")
    X_flat = X.reshape(c_i, h * w)  # 每一列是同一位置的全部通道值。
    K_flat = K.reshape(c_o, c_i)   # 每一行是一个输出通道的全部权重。
    Y_flat = torch.matmul(K_flat, X_flat)  # (c_o,c_i) @ (c_i,H*W)
    return Y_flat.reshape(c_o, h, w)       # 把位置轴还原成高、宽。


def example_multi_in():
    """6.4.1 与 6.4.2 共用的小矩阵。"""
    # 一张输入、两个通道；各通道是 3×3，不是两张独立图片。
    X = torch.tensor([[[0.0, 1.0, 2.0],
                       [3.0, 4.0, 5.0],
                       [6.0, 7.0, 8.0]],
                      [[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0],
                       [7.0, 8.0, 9.0]]])
    # 一个输出通道需要一组核，组内有两个 2×2 核切片。
    K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]],
                      [[1.0, 2.0], [3.0, 4.0]]])
    return X, K


def demo_multi_in():
    print("\n6.4.1 多输入通道：分别计算，再求和")
    X, K = example_multi_in()
    print("输入 X 形状:", tuple(X.shape))
    print("卷积核 K 形状:", tuple(K.shape))
    for c, (x, k) in enumerate(zip(X, K)):
        print(f"输入通道 {c} 的互相关结果:\n{corr2d(x, k)}")

    Y = corr2d_multi_in(X, K)
    print("逐通道结果相加 Y:\n", Y)
    print("输出 Y 形状:", tuple(Y.shape))

    # 对照同一个计算：补上批量维和输出通道维，使用相同权重。
    # bias=False 对应上面的纯互相关求和；这里只做前向计算。
    conv = nn.Conv2d(in_channels=2, out_channels=1,
                     kernel_size=2, bias=False)
    with torch.no_grad():
        conv.weight.copy_(K.unsqueeze(0))  # (2,2,2) → (1,2,2,2)
        Y_torch = conv(X.unsqueeze(0))    # (2,3,3) → (1,2,3,3)
    print("PyTorch 权重形状:", tuple(conv.weight.shape))
    print("PyTorch 输出形状:", tuple(Y_torch.shape))
    print("PyTorch 输出的二维切片:\n", Y_torch[0, 0])
    torch.testing.assert_close(Y, Y_torch[0, 0])
    print("手写计算与 PyTorch 一致")


def demo_multi_out():
    print("\n6.4.2 多输出通道：一组完整核得到一个输出通道")
    X, K = example_multi_in()
    K_multi = torch.stack((K, K + 1, K + 2), dim=0)
    # 三组核各有两张 2×2 切片；K+1 是所有权重加 1，不是偏置。
    Y = corr2d_multi_in_out(X, K_multi)
    print("输入形状:", tuple(X.shape))
    print("完整权重形状:", tuple(K_multi.shape))
    print("一个输出通道的核形状:", tuple(K_multi[0].shape))
    print("输出形状:", tuple(Y.shape))
    print("三个输出通道:\n", Y)
    print("权重数量:", K_multi.numel(), "（无偏置）")
    Y_torch = F.conv2d(X.unsqueeze(0), K_multi)[0]
    torch.testing.assert_close(Y, Y_torch)
    print("sum + stack 与 PyTorch 一致")


def demo_1x1():
    print("\n6.4.3 1×1 卷积：在同一位置组合通道")
    X = torch.arange(1.0, 13.0).reshape(3, 2, 2)
    # 对每个位置：输出0 = 通道0 + 通道2；输出1 = 2*通道1 - 通道2。
    K = torch.tensor([[1.0, 0.0, 1.0],
                      [0.0, 2.0, -1.0]]).reshape(2, 3, 1, 1)
    print("输入 X:\n", X)
    print("左上角的三个输入值:", X[:, 0, 0])
    print("权重矩阵:\n", K.reshape(2, 3))
    print("X 展开 (3,2,2) → (3,4):\n", X.reshape(3, 4))
    print("矩阵形状: (2,3) @ (3,4) → (2,4) → (2,2,2)")
    Y_matrix = corr2d_multi_in_out_1x1(X, K)
    Y_general = corr2d_multi_in_out(X, K)
    Y_torch = F.conv2d(X.unsqueeze(0), K)[0]
    print("两个输出通道:\n", Y_matrix)
    print("左上角的两个输出值:", Y_matrix[:, 0, 0])
    torch.testing.assert_close(Y_matrix, Y_general)
    torch.testing.assert_close(Y_matrix, Y_torch)
    print("固定例子：矩阵实现、通用手写实现、PyTorch 一致")

    # 保留教材的随机输入对照；固定种子和 float64 方便重复核对。
    rng = torch.Generator().manual_seed(64)
    X_random = torch.randn(3, 3, 3, generator=rng, dtype=torch.float64)
    K_random = torch.randn(2, 3, 1, 1, generator=rng, dtype=torch.float64)
    Y1 = corr2d_multi_in_out_1x1(X_random, K_random)
    Y2 = corr2d_multi_in_out(X_random, K_random)
    Y3 = F.conv2d(X_random.unsqueeze(0), K_random)[0]
    torch.testing.assert_close(Y1, Y2)
    torch.testing.assert_close(Y1, Y3)
    print("随机例子输出形状:", tuple(Y1.shape))
    print("矩阵与通用实现的最大绝对误差:", (Y1 - Y2).abs().max().item())
    print("随机例子：三种实现数值对照通过（允许浮点误差）")


def preview_multi_out_shapes():
    """保留最初的形状示例；用 --section shapes 单独运行。"""
    X = torch.zeros(2, 3, 8, 8)  # (批量, 输入通道, 高, 宽)
    conv = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3)
    print("输入形状:", tuple(X.shape))
    print("权重形状:", tuple(conv.weight.shape))  # (输出通道, 输入通道, 核高, 核宽)
    print("偏置形状:", tuple(conv.bias.shape))
    print("输出形状:", tuple(conv(X).shape))

    # 练习：解释每个输出通道怎样对所有输入通道的互相关结果求和。
    # 练习：把 kernel_size 改为 1，先预测形状与参数数量，再运行。


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", default="all",
                        choices=("all", "6.4.1", "6.4.2", "6.4.3", "shapes"))
    parser.add_argument("--question", action="store_true",
                        help="只显示保留的独立复习题，不计算答案")
    args = parser.parse_args()
    if args.question:
        print("复习任务：1×1 卷积独立验证（无偏置、步幅1、无填充）")
        print("某位置的三个输入值为 [2,-1,3]；两组权重为 [1,2,0]、[-1,0,2]。")
        print("先手算该位置的两个输出。若完整输入 X=(3,2,2)，写出 K、Y 的形状。")
        print("再解释 X.reshape(3,4) 的每列含义，以及为什么用 K @ X。")
        return
    demos = {"6.4.1": demo_multi_in, "6.4.2": demo_multi_out,
             "6.4.3": demo_1x1, "shapes": preview_multi_out_shapes}
    if args.section == "all":
        for name in ("6.4.1", "6.4.2", "6.4.3"):
            demos[name]()
    else:
        demos[args.section]()


if __name__ == "__main__":
    main()
