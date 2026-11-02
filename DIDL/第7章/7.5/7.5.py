import torch
from torch import nn
from d2l import torch as d2l

def batch_norm_(
    X, gamma, beta, moving_mean, moving_var,
    eps=1e-5, momentum=0.1, *, training=True
):
    """手写 BN；返回输出、新的累计均值和方差，调用方需要保存后两者。"""
    # X: 全连接输入 [N, C]，或卷积输入 [N, C, H, W]。
    # gamma、beta、moving_mean、moving_var: 每个特征/通道一个值。
    # 参数既可以是 [C]，也可以是 [1, C] / [1, C, 1, 1]。
    # momentum: 新批次统计量的占比，与 PyTorch 内置 BN 的含义一致。
    # training=False 时使用累计统计量；no_grad() 只负责关闭梯度记录。
    if X.dim() not in (2, 4):
        raise ValueError("输入数据必须是 2D 或 4D 张量")

    param_shape = (1, X.shape[1]) if X.dim() == 2 else (1, X.shape[1], 1, 1)
    gamma = gamma.reshape(param_shape)
    beta = beta.reshape(param_shape)

    if training:
        # 全连接层沿样本维统计；卷积层沿样本、高、宽三个维度统计。
        dims = (0,) if X.dim() == 2 else (0, 2, 3)
        count = X.shape[0] if X.dim() == 2 else X.shape[0] * X.shape[2] * X.shape[3]
        if count <= 1:
            raise ValueError("训练时每个特征/通道至少需要两个统计元素")
        mean = X.mean(dim=dims, keepdim=True)
        var = ((X - mean) ** 2).mean(dim=dims, keepdim=True)

        # 累计统计量不参与反向传播，返回时保留传入统计量的形状。
        with torch.no_grad():
            new_mean = (1 - momentum) * moving_mean + momentum * mean.reshape_as(moving_mean)
            # 与 PyTorch 一致：前向用有偏方差，累计方差使用无偏估计。
            unbiased_var = var * count / (count - 1)
            new_var = (1 - momentum) * moving_var + momentum * unbiased_var.reshape_as(moving_var)
    else:
        mean = moving_mean.reshape(param_shape)
        var = moving_var.reshape(param_shape)
        # 预测只读取累计统计量，不更新它们。
        new_mean, new_var = moving_mean.detach(), moving_var.detach()

    X_hat = (X - mean) / torch.sqrt(var + eps)
    Y = gamma * X_hat + beta
    return Y, new_mean, new_var




class BatchNorm(nn.Module):
    """手写 BN；返回输出、新的累计均值和方差，调用方需要保存后两者。"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        # 可学习参数
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        # 累计统计量，非可学习参数
        self.register_buffer("moving_mean", torch.zeros(num_features))
        self.register_buffer("moving_var", torch.ones(num_features))

        self.moving_mean = torch.zeros(num_features)
        self.moving_var = torch.ones(num_features)

    def forward(self, X):
        Y, new_mean, new_var = batch_norm_(
            X, self.gamma, self.beta,
            self.moving_mean, self.moving_var,
            self.eps, self.momentum, training=self.training
        )
        # 更新累计统计量
        if self.training:
            with torch.no_grad():
                self.moving_mean.copy_(new_mean)
                self.moving_var.copy_(new_var)
        return Y

# 备注：这个 BatchNorm 类可以直接用于神经网络中，类似于 PyTorch 内置的 nn.BatchNorm1d 或 nn.BatchNorm2d.


# LeNet 中直接使用NN.BatchNorm2d，或者使用自定义的 BatchNorm 类都可以。
# 示例

if __name__ == "__main__":
    net = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 4 * 4, 120), BatchNorm(120), nn.Sigmoid(),
        nn.Linear(120, 84), BatchNorm(84), nn.Sigmoid(),
        nn.Linear(84, 10)
    )

    # 上面这个网络可以直接用于训练和测试，BatchNorm 层会在训练时更新累计均值和方差，在测试时使用这些累计统计量进行归一化。
    lr, num_epochs, batch_size = 0.9, 10, 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

    d2l.plt.show()
