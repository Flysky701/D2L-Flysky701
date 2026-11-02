import torch
from torch import nn
from d2l import torch as d2l
from pathlib import Path
import sys

# 复用项目中的训练辅助函数；路径不依赖 Code Runner 的工作目录。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / '4.2'))
from chapter3 import load_data_fashion_mnist, train_V1, plot_loss

def dropout_layer(x, dropout):
    """在训练期间随机丢弃一些元素。"""
    assert 0 <= dropout <= 1
    # 在测试期间不丢弃任何元素。
    if dropout == 1:
        return torch.zeros_like(x)
    if dropout == 0:
        return x
    
    mask = (torch.rand(x.shape, device=x.device) > dropout).float()
    # 在训练期间丢弃一些元素，并缩放剩余元素，mask值为1/(1-dropout)
    return mask * x / (1.0 - dropout)

# X = torch.arange(16, dtype=torch.float32).reshape((2, 8))
# print(dropout_layer(X, 0.))
# print(dropout_layer(X, 0.2))
# print(dropout_layer(X, 0.5))
# print(dropout_layer(X, 1.))

#上面展示了dropout_layer的raw实现，下面是一个完整的神经网络实现

#省略了从0实现的版本

#高级API版本
net = nn.Sequential(nn.Flatten(), 
                    nn.Linear(784, 256), 
                    nn.Dropout(0.2),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.Dropout(0.2),
                    nn.ReLU(),
                    nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


if __name__ == '__main__':
    net.apply(init_weights)
    batch_size, lr, num_epochs = 256, 0.5, 10
    loss = nn.CrossEntropyLoss(reduction='none')
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    trainer = torch.optim.SGD(net.parameters(), lr = lr)
  
    history = train_V1(net, train_iter, test_iter, loss, num_epochs, trainer)
    output_path = Path(__file__).resolve().parent.parent / 'outputs' / 'dropout' / 'loss_curve.png'
    plot_loss(history, output_path)




