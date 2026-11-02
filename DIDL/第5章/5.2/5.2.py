import torch
from torch import nn


net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
net(X)

#5.2.1
print("5.2.1")
print(net[2].state_dict())  # 查看第 3 层的参数字典

#5.2.1.1
print("5.2.1.1")
print(type(net[2].bias))  # 查看第 3 层的偏置参数
print(net[2].bias)  # 查看第 3 层的偏置参数
print(net[2].bias.shape)  # 查看第 3 层的偏置参数形状
print(net[2].bias.data)  # 查看第 3 层的偏置参数数据

#5.2.1.2
#递归看子块的每个参数
print("5.2.1.2")
print(*[(name, param.shape) for name, param in net[2].named_parameters()])  # 查看第 3 层的参数名和形状
print(*[(name, param.shape) for name, param in net.named_parameters()])  # 查看整个网络的参数名和形状

#5.2.1.3

def block1():
    return nn.Sequential(nn.Linear(4, 8), 
                         nn.ReLU(),
                         nn.Linear(8, 4), 
                         nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)




#主要是查看参数， 跳过了
