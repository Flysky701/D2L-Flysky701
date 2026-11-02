# 7.3 的阅读（待核对）

NiN（Network in Network）：网络中的网络。

方案：在每个像素的通道上分别使用多层感知机。

## NiN 块

同样从块开始，将每个块视为完整的网络？

其实是类似对每个像素的通道（如 RGB）使用类似的网络。

两个 1×1 的卷积层，充当带有 ReLU 激活函数的逐像素全连接层。

下面以 3×3 卷积开头，后接两个 1×1 卷积层，简写一个 NiN 块。

```python
from torch import nn


def nin_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
    )
```

## NiN 网络

同理，和 VGG 类似，由块组成一个网络。

没什么新东西。

沿用上面的 `nin_block`，下面缩略展示 10 分类的网络结构。末尾使用全局平均汇聚，将每个通道变成一个数，再展平成 `(批量大小, 10)`。

```python
net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, stride=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    nin_block(384, 10),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
)
```

本章的 PY 代码略。

直接不求甚解地跳过啊。
