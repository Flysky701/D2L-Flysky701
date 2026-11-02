import torch 
from torch import nn


# AlexNet
#
net = nn.Sequential( nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2),
                     nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2),
                     nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
                     nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
                     nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2),
                     nn.Flatten(), 
                     nn.Linear(256 * 5 * 5, 4096), nn.ReLU(), nn.Dropout(p=0.5),
                     nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
                     nn.Linear(4096, 10)
                     )
# 相对于LeNet的主要改进是使用了更深的网络结构和更大的卷积核，同时引入了ReLU激活函数和Dropout正则化方法，以提高模型的表达能力和防止过拟合。

