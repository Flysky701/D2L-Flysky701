import torch
from torch import nn
from MPL import MLP, MySequential, FixHiddenMLP

# 5.1.1 自定义块：__init__ 定义有哪些层，forward 定义数据怎样经过这些层。
net1 = MLP()
X = torch.rand(size=(2, 20))
net1(X)  # (2, 20) -> (2, 256) -> (2, 10)

# 5.1.2 顺序块：自己实现一个容器，按传入的顺序串联各个子块。
net2 = MySequential(
    nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10)
)
net2(X)  # (2, 20) -> (2, 256) -> (2, 10)

# 5.1.3 在前向传播函数中执行代码：块内也能有矩阵运算、循环和分支。
net3 = FixHiddenMLP()
net3(X)  # 中间保持 (2, 20)，最后 X.sum() 将所有元素相加，返回一个标量

# print(net) 打印模型结构；要看计算结果，可写 print(net(X))。
print("net1:", net1(X))
print("net2:", net2(X))
print("net3:", net3(X))




