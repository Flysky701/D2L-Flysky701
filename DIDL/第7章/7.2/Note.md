# 7.2 的阅读

本节主要介绍 VGG。

为什么需要 VGG？因为 AlexNet 没有通用的模板。

VGG 相当于一个统一的模板。

## VGG 块

VGG 块是神经网络块，类似于一个常用的组合序列：

卷积 + 激活函数 ReLU + 汇聚层。

助手补充：下面以两个卷积层的块为例，缩略展示结构。`in_channels` 和 `out_channels` 表示输入、输出通道数。

```python
block = nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
)
```

## VGG 网络

类似于更加规范的 AlexNet，主要由 VGG 块组合而成。

沿用 [7.2.py](7.2.py) 中的函数和变量，把网络结构缩写成下面这样：

```python
conv = nn.Sequential(*[
    vgg_block(num_convs, in_channels, out_channels)
    for num_convs, in_channels, out_channels in conv_arch
])

net = nn.Sequential(
    conv,
    # CONV * N, 可能由好几个块组成
    nn.Flatten(),
    nn.Linear(fc_features, 4096), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(4096, 10),
)
```

最后组合成网络

简单的训练， 复用之前的函数即可

大致思路类似类的复用，可以组成更加复杂的网络，没啥可说的。
