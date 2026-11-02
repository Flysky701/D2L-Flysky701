"""6.6 LeNet + Fashion-MNIST：默认查看数据，加 --train 开始训练。"""

from torch import nn

from lenet_training import load_data, run, train


def build_net():
    """适配单通道 28×28 图像，输出 10 类的 logits。"""
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
    # 数据加载与训练的完整中文说明位于同目录 lenet_training.py。
    run(build_net, dataset_name="fashion-mnist")


if __name__ == "__main__":
    main()
