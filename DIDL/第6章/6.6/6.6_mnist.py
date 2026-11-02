"""6.6 LeNet + MNIST：识别手写数字 0—9，和服饰分类分开试验。"""

from torch import nn

from lenet_training import load_data, run, train

#
#这个数据集输入为 (N,1,28,28)，输出为 (N,10)；可在这里独立修改数字模型。
#注意：MNIST 数据集的标签是 0-9，对应 10 个类别。

def build_net():
    """输入 (N,1,28,28)，输出 (N,10)；可在这里独立修改数字模型。"""
    return nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2),
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5),
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120),
        nn.Sigmoid(),
        nn.Linear(120, 84),
        nn.Sigmoid(),
        nn.Linear(84, 10),
    )


def main():
    run(build_net, dataset_name="mnist")


if __name__ == "__main__":
    main()
