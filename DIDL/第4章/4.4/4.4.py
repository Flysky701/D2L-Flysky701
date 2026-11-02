import math
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # 直接保存图片，适用于运行 .py 的终端。
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

# 固定随机种子，重新运行时使用相同的数据。
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

max_degree = 20
n_train, n_test = 100, 100
true_w = np.zeros(max_degree)
true_w[0:4] = [5, 1.2, -3.4, 5.6]

features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)

# 这个函数将features的每一列提升到0到max_degree-1次幂，并将结果存储在poly_features中
poly_features = np.power(
    features, 
    np.arange(max_degree).reshape(1, -1)
)

# 循环将poly_features的每一列除以相应的阶乘值，以便在计算多项式回归时更稳定
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)  # gamma(n)=(n-1)!
# 这行代码计算了多项式特征与真实权重的点积，得到标签labels。labels的维度为(n_train+n_test,)，表示每个样本的标签值。
labels = np.dot(poly_features, true_w)

#labels的维度:(n_train+n_test,)
#这行代码将高斯噪声添加到标签中，以模拟现实世界中的数据噪声。噪声的标准差为0.1，大小与标签的形状相同。
labels += np.random.normal(scale=0.1, size=labels.shape)

# NumPy ndarray转换为tensor
true_w, features, poly_features, labels = [
    torch.tensor(x, dtype = torch.float32) 
    for x in [true_w, features, poly_features, labels]]


# features[:2], poly_features[:2, :], labels[:2]

# print("features[:2]:", features[:2])
# print("poly_features[:2, :]:", poly_features[:2, :]) 
# print("labels[:2]:", labels[:2])


@torch.no_grad()  # 评估时只计算预测和损失，不需要记录梯度。
def evaluate_loss(net, data_iter, loss):  #@save
    """评估给定数据集上模型的损失"""
    metric = d2l.Accumulator(2)  # 损失的总和,样本数量
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]

def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400, lr=0.01, batch_size=10):
    torch.manual_seed(seed)
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    
    batch_size = min(batch_size, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1, 1)), batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1, 1)), batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    # 记录 epoch=0（尚未训练）的损失，后面用这些记录画静态图。
    history = {
        'degree': input_shape - 1,
        'epochs': [0],
        'train_loss': [evaluate_loss(net, train_iter, loss)],
        'test_loss': [evaluate_loss(net, test_iter, loss)],
        'lr': lr,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'n_train': train_labels.shape[0],
        'n_test': test_labels.shape[0],
    }
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y.reshape(-1, 1))
            l.mean().backward()
            trainer.step()
        # 第一轮、每 20 轮和最后一轮记录一次，保证最后的结果会显示。
        if epoch == 0 or (epoch + 1) % 20 == 0 or epoch + 1 == num_epochs:
            history['epochs'].append(epoch + 1)
            history['train_loss'].append(evaluate_loss(net, train_iter, loss))
            history['test_loss'].append(evaluate_loss(net, test_iter, loss))
    history['weights'] = net[0].weight.detach().numpy().copy()
    print(f"degree={history['degree']}, epochs={num_epochs}, lr={lr}, "
          f"batch_size={batch_size}, train_loss={history['train_loss'][-1]:.6f}, "
          f"test_loss={history['test_loss'][-1]:.6f}")
    print('weight:', history['weights'])
    return history


def plot_comparison(histories, output_dir):
    """画出每个模型的学习曲线，以及不同模型之间的损失对比。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = ['#2563eb', '#059669', '#d97706']

    # 图一：每个模型单独一个子图，训练/测试损失使用相同的纵轴范围。
    fig, axes = plt.subplots(1, len(histories), figsize=(15, 5.2),
                             sharex=True, sharey=True, constrained_layout=True)
    for ax, history in zip(axes, histories):
        ax.plot(history['epochs'], history['train_loss'],
                color='#2563eb', label='Train', linewidth=2)
        ax.plot(history['epochs'], history['test_loss'],
                color='#e76f51', label='Test', linewidth=2, linestyle='--')
        ax.set_title(f"Degree {history['degree']} ({history['degree'] + 1} weights)")
        ax.set_xlabel('Epoch')
        ax.set_yscale('log')
        ax.grid(True, which='both', alpha=0.2)
        ax.legend(loc='upper right')
        ax.text(0.04, 0.05,
                f"Epochs: {history['num_epochs']}\n"
                f"Learning rate: {history['lr']}\n"
                f"Batch size: {history['batch_size']}\n"
                f"Final train MSE: {history['train_loss'][-1]:.5f}\n"
                f"Final test MSE: {history['test_loss'][-1]:.5f}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='#dddddd'))
    axes[0].set_ylabel('Mean squared error (log scale)')
    fig.suptitle(f"Polynomial regression | same data: "
                 f"{histories[0]['n_train']} train / {histories[0]['n_test']} test | seed={seed}",
                 fontsize=14)
    model_path = output_dir / 'loss_by_model.png'
    fig.savefig(model_path, dpi=160)
    plt.close(fig)

    # 图二：把三种模型叠加，左边比较训练损失，右边比较测试损失。
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True,
                             constrained_layout=True)
    for ax, key, title in zip(axes, ['train_loss', 'test_loss'],
                             ['Training loss comparison', 'Test loss comparison']):
        for history, color in zip(histories, colors):
            ax.plot(history['epochs'], history[key], color=color, linewidth=2,
                    label=f"Degree {history['degree']}")
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_yscale('log')
        ax.grid(True, which='both', alpha=0.2)
        ax.legend()
    axes[0].set_ylabel('Mean squared error (log scale)')
    fig.suptitle(f"SGD | epochs={histories[0]['num_epochs']} | "
                 f"lr={histories[0]['lr']} | batch size={histories[0]['batch_size']}")
    comparison_path = output_dir / 'loss_comparison.png'
    fig.savefig(comparison_path, dpi=160)
    plt.close(fig)

    # 保留曲线背后的数值，方便核对具体 epoch 对应的 loss。
    with (output_dir / 'loss_history.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['degree', 'epoch', 'train_loss', 'test_loss',
                         'lr', 'batch_size', 'seed'])
        for history in histories:
            for epoch, train_loss, test_loss in zip(
                    history['epochs'], history['train_loss'], history['test_loss']):
                writer.writerow([history['degree'], epoch, train_loss, test_loss,
                                 history['lr'], history['batch_size'], seed])
    print('Saved:', model_path)
    print('Saved:', comparison_path)

if __name__ == '__main__':
    # 共用相同数据、轮数、学习率和批大小，只改变多项式的阶数。
    num_epochs, lr, batch_size = 400, 0.01, 10
    histories = []
    for num_features in [2, 4, max_degree]:  # 对应一阶、三阶、十九阶。
        history = train(poly_features[:n_train, :num_features],
                        poly_features[n_train:, :num_features],
                        labels[:n_train], labels[n_train:],
                        num_epochs=num_epochs, lr=lr, batch_size=batch_size)
        histories.append(history)
    # 路径基于脚本位置，Code Runner 切换工作目录也能找到输出。
    output_dir = Path(__file__).resolve().parent.parent / 'outputs' / 'polynomial_regression'
    plot_comparison(histories, output_dir)
    
