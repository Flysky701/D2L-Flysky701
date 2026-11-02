import torch
from torch import nn
from d2l import torch as d2l
from chapter3 import load_data_fashion_mnist, train_ch3, predict_ch3


# 每批读取 256 张图片。训练集用于学习参数，测试集用于检查泛化能力。
# Fashion-MNIST 的图片为 28×28 的单通道灰度图，共有 10 个服饰类别。
batch_size = 256


# 将图片展平成 28×28=784 个输入特征；隐藏层有 256 个神经元。
num_inputs = 784
num_outputs = 10 
num_hiddens = 256 

# 第一层：784 个输入 → 256 个隐藏单元；第二层：256 个隐藏单元 → 10 个输出。
# W 是权重矩阵，b 是偏置。小随机权重让各神经元从不同的初始状态开始学习。
# requires_grad=True 让 PyTorch 记录梯度，后续训练时用来更新这些参数。
W1 = torch.normal(0, 0.01, size=(num_inputs, num_hiddens), requires_grad=True)
b1 = torch.zeros(num_hiddens, requires_grad=True) 
W2 = torch.normal(0, 0.01, size=(num_hiddens, num_outputs), requires_grad=True)
b2 = torch.zeros(num_outputs, requires_grad=True) 

# 收集待训练的参数，后续可交给优化器；不要命名为 print，否则会覆盖打印函数。
params = [W1, b1, W2, b2]


def relu(X):
    # 手写 ReLU：创建同形状的零张量，再逐元素取最大值，保留正数并将负数置零。
    a = torch.zeros_like(X)
    return torch.max(X, a)

def net(X):
    # -1 自动推算批大小：把 [批大小, 1, 28, 28] 转成 [批大小, 784]。
    X = X.reshape((-1, num_inputs))  # 将每张图片展平为长度为 784 的向量
    # @ 表示矩阵乘法；隐藏层输出形状为 [批大小, 256]，ReLU 引入非线性。
    H = relu(X@W1 + b1)
    # 返回 [批大小, 10] 的原始类别分数（logits），供交叉熵损失函数使用。
    return H@W2 + b2  # 输出层不使用激活函数 

# 交叉熵接收原始分数和整数类别标签，内部会处理 softmax，无需在 net 中再调用 softmax。
loss = nn.CrossEntropyLoss(reduction='none')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    display = parser.add_mutually_exclusive_group()
    display.add_argument('--show', action='store_true', help='Open the prediction window after training')
    display.add_argument('--no-show', action='store_true', help='Save predictions without opening a window (default)')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be at least 1')
    # Windows 的数据加载子进程会导入此文件，入口保护避免子进程重复开始训练。
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    num_epochs, lr = args.epochs, 0.1

    def updater(current_batch_size):
        # sgd 更新参数并清零梯度；最后一批可能不足 256 张，使用实际批大小。
        d2l.sgd(params, lr, current_batch_size)

    train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
    predict_ch3(net, test_iter, show=args.show)
    print('Training and prediction complete.', flush=True)

