"""模型构建：明确每个隐藏层的宽度、激活函数和学习率。"""
import torch
from torch import nn

ACTIVATIONS = {'relu': nn.ReLU, 'leaky_relu': lambda: nn.LeakyReLU(0.01),
               'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid}


def build_model(hidden_sizes=(256,), activations=('relu',), initialization='normal'):
    """一个激活名应用于所有隐藏层；多个名称必须与隐藏层逐层对应。"""
    hidden_sizes, activations = list(hidden_sizes), list(activations)
    if not hidden_sizes or any(w < 1 for w in hidden_sizes):
        raise ValueError('Hidden widths must be positive')
    if len(activations) == 1:
        activations *= len(hidden_sizes)
    if len(activations) != len(hidden_sizes):
        raise ValueError('Use one activation, or one activation per hidden layer')
    if initialization not in ('normal', 'xavier'):
        raise ValueError('Unknown initialization')
    layers, inputs = [nn.Flatten()], 784
    for width, name in zip(hidden_sizes, activations):
        # 前层输出自动成为后层输入，避免 64 → 256 这样的维度错误。
        layers.extend([nn.Linear(inputs, width), ACTIVATIONS[name]()])
        inputs = width
    layers.append(nn.Linear(inputs, 10))
    net = nn.Sequential(*layers)

    def initialize(layer):
        if isinstance(layer, nn.Linear):
            if initialization == 'normal':
                nn.init.normal_(layer.weight, mean=0, std=0.01)
            else:
                # 比较所有激活时统一 Xavier gain=1，避免额外改变初始化规则。
                nn.init.xavier_uniform_(layer.weight, gain=1)
            nn.init.zeros_(layer.bias)
    net.apply(initialize)
    return net


def make_optimizer(net, lr=0.1, layer_lrs=None):
    """layer_lrs 可为每个 Linear 设置学习率，列表也包含输出层。"""
    if lr <= 0:
        raise ValueError('Learning rate must be positive')
    if layer_lrs is None:
        return torch.optim.SGD(net.parameters(), lr=lr)
    layers = [layer for layer in net if isinstance(layer, nn.Linear)]
    if len(layer_lrs) != len(layers) or any(rate <= 0 for rate in layer_lrs):
        raise ValueError('Use one positive learning rate per Linear layer, including output')
    return torch.optim.SGD([{'params': layer.parameters(), 'lr': rate}
                            for layer, rate in zip(layers, layer_lrs)], lr=lr)
