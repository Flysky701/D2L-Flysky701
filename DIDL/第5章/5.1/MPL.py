import torch
from torch import nn


# 对照教材 5.1「层和块」（选择 PyTorch）：
# https://zh.d2l.ai/chapter_deep-learning-computation/model-construction.html
# 块可以是一层，也可以把多层组合起来；这里三个类都是 nn.Module 的子类。

# 5.1.1 自定义块：__init__ 定义有哪些层，forward 定义数据怎样经过这些层。
class MLP(nn.Module):
    def __init__(self):
        # 初始化父类，让 PyTorch 能管理后面赋给 self 的子层及其参数。
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 每个样本：20 个输入特征 -> 256 个隐藏特征
        self.act = nn.ReLU()             # 负数变成 0，如 [-2, 3] -> [0, 3]；形状不变
        self.output = nn.Linear(256, 10) # 256 个隐藏特征 -> 10 个输出，无输出激活函数

    def forward(self, X):
        # net = MLP() 创建模型；net(X) 会通过 nn.Module 的调用机制执行 forward。
        # 从内向外读：hidden -> act -> output；(2, 20) -> (2, 256) -> (2, 10)。
        return self.output(self.act(self.hidden(X)))
    

# 5.1.2 顺序块：自己实现一个容器，按传入的顺序串联各个子块。
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        # *args 收集传入的多个层；_modules 是 nn.Module 管理已注册子块的字典。
        for idx, block in enumerate(args):
            # 用 '0'、'1'、'2' 作键：即使有两个 Linear，也不会相互覆盖。
            self._modules[str(idx)] = block
    def forward(self, X):
        # 按保存顺序执行：上一层的输出，成为下一层的输入。
        for block in self._modules.values():
            X = block(X)
        return X

# 5.1.3 在前向传播函数中执行代码：块内也能有矩阵运算、循环和分支。
# 教材中的类名是 FixedHiddenMLP；这里保留你写的 FixHiddenMLP。
class  FixHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 创建时随机生成，之后保持固定；它不是可训练参数，不计算它的梯度。
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)  # 这一层的权重和偏置是可训练参数
        
    def forward(self, X):
        X = self.linear(X)  # 若输入是 (2, 20)，以下运算直到求和前都保持这个形状
        # (2, 20) @ (20, 20) -> (2, 20)；+1 给每个元素加 1。
        X = torch.mm(X, self.rand_weight) + 1
        X = torch.relu(X)
        # 再次调用同一个 linear，复用同一套参数，并非新建一层。
        X = self.linear(X)
        # 对所有元素取绝对值再求和；例如总和为 3，则减半为 1.5、0.75 后退出。
        while X.abs().sum() > 1:
            X /= 2
        # 这是你额外加的分支（教材没有）：总和 < 0.8 时放大 10 倍，可能再次 > 1。
        if X.abs().sum() < 0.8:
            X *= 10
        return X.sum()  # 对整个批次的所有元素求和，返回一个标量（零维张量）

#5.1.6 Q2

#平行块
class ParallelBlock(nn.Module):
    def __init__(self, net1, net2):
        super().__init__()
        self.net1 = net1
        self.net2 = net2
    def forward(self, X):
        Y1 = self.net1(X)
        Y2 = self.net2(X)
        return torch.cat((Y1, Y2), dim=1)
    