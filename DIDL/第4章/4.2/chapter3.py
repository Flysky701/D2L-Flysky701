"""供本项目使用的第三章训练与预测辅助函数。"""

import torch
from pathlib import Path
from torchvision import datasets, transforms
from d2l import torch as d2l


def load_data_fashion_mnist(batch_size):
    """数据固定存放在项目目录，不受终端工作目录影响。"""
    root = Path(__file__).resolve().parent.parent / 'data'
    print(f'Loading Fashion-MNIST: {root}', flush=True)
    # 使用数据集发布者的 HTTPS 地址，torchvision 会校验压缩文件的 MD5。
    datasets.FashionMNIST.mirrors = [
        'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/'
    ]
    train = datasets.FashionMNIST(root=str(root), train=True,
                                  transform=transforms.ToTensor(), download=True)
    test = datasets.FashionMNIST(root=str(root), train=False,
                                 transform=transforms.ToTensor(), download=True)
    print(f'Data ready: train={len(train)}, test={len(test)}', flush=True)
    # 学习脚本使用单进程加载，避免 Windows 子进程启动开销及编辑器兼容问题。
    return (torch.utils.data.DataLoader(train, batch_size, shuffle=True, num_workers=0),
            torch.utils.data.DataLoader(test, batch_size, shuffle=False, num_workers=0))


def train_V1(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练 nn.Module；loss 返回每个样本的损失，并返回逐轮记录。"""
    history = {'epoch': [], 'train_loss': [], 'test_loss': []}
    for epoch in range(num_epochs):
        net.train()
        print(f'Starting epoch {epoch + 1}/{num_epochs}', flush=True)
        loss_sum, correct, count = 0.0, 0, 0
        for batch, (X, y) in enumerate(train_iter, start=1):
            updater.zero_grad()
            scores = net(X)
            losses = loss(scores, y)
            losses.mean().backward()
            updater.step()
            loss_sum += float(losses.detach().sum())
            correct += int((scores.detach().argmax(dim=1) == y).sum())
            count += y.numel()
            if batch % 50 == 0:
                print(f'  batch {batch}/{len(train_iter)}', flush=True)

        # 每轮训练结束后评估一次；eval() 关闭 Dropout。
        test_loss_sum, test_correct, test_count = 0.0, 0, 0
        net.eval()
        with torch.no_grad():
            for X, y in test_iter:
                scores = net(X)
                test_loss_sum += float(loss(scores, y).sum())
                test_correct += int((scores.argmax(dim=1) == y).sum())
                test_count += y.numel()
                

        # 按样本数量求平均，最后一个较小的批次也能正确统计。
        train_loss = loss_sum / count
        test_loss = test_loss_sum / test_count
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        print(f'epoch {epoch + 1}: train loss {train_loss:.4f}, '
              f'test loss {test_loss:.4f}, '
              f'train acc {correct / count:.3f}, '
              f'test acc {test_correct / test_count:.3f}', flush=True)
              # :.3f 保留三位小数
    return history


def plot_loss(history, output_path):
    """使用训练函数返回的逐轮记录，保存 train_loss / test_loss 对比图。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = d2l.plt.subplots(figsize=(7, 4.5))
    ax.plot(history['epoch'], history['train_loss'], marker='o',
            label='train_loss (train mode)', color='#2563eb')
    ax.plot(history['epoch'], history['test_loss'], marker='o',
            label='test_loss (eval mode)', color='#e76f51')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average cross-entropy loss')
    ax.set_title('Training and test loss')
    ax.set_xticks(history['epoch'])
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    d2l.plt.close(fig)
    print(f'Loss plot saved: {output_path}', flush=True)


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """支持手写参数更新函数（4.2）和 PyTorch 优化器（4.3）。"""
    for epoch in range(num_epochs):
        if isinstance(net, torch.nn.Module):
            net.train()
        print(f'Starting epoch {epoch + 1}/{num_epochs}', flush=True)
        loss_sum, correct, count = 0.0, 0, 0
        for batch, (X, y) in enumerate(train_iter, start=1):
            if isinstance(updater, torch.optim.Optimizer):
                updater.zero_grad()
            scores = net(X)
            losses = loss(scores, y)
            if isinstance(updater, torch.optim.Optimizer):
                # 4.3：优化器不再除以批大小，因此对平均损失求梯度。
                losses.mean().backward()
                updater.step()
            else:
                # 4.2：手写 d2l.sgd 内部会除以批大小，因此先对损失求和。
                losses.sum().backward()
                updater(y.numel())
            loss_sum += float(losses.detach().sum())
            correct += int((scores.detach().argmax(dim=1) == y).sum())
            count += y.numel()
            if batch % 50 == 0:
                print(f'  batch {batch}/{len(train_iter)}', flush=True)

        test_correct, test_count = 0, 0
        if isinstance(net, torch.nn.Module):
            net.eval()
        # 测试只计算预测结果，不记录梯度，也不更新参数。
        with torch.no_grad():
            for X, y in test_iter:
                test_correct += int((net(X).argmax(dim=1) == y).sum())
                test_count += y.numel()
        print(f'epoch {epoch + 1}: loss {loss_sum / count:.4f}, '
              f'train acc {correct / count:.3f}, '
              f'test acc {test_correct / test_count:.3f}', flush=True)


def predict_ch3(net, test_iter, n=6, show=False):
    """显示第一批中若干图片，标题上行为真实类别，下行为预测类别。"""
    if not show:
        # 保存图片无需图形窗口；使用非交互后端，程序不会等待关闭窗口。
        d2l.plt.switch_backend('Agg')
    X, y = next(iter(test_iter))
    n = min(n, len(y))
    with torch.no_grad():
        predictions = net(X).argmax(dim=1)
    trues = d2l.get_fashion_mnist_labels(y[:n])
    preds = d2l.get_fashion_mnist_labels(predictions[:n])
    titles = [f'{true}\n{pred}' for true, pred in zip(trues, preds)]
    d2l.show_images(X[:n].reshape(n, 28, 28), 1, n, titles=titles)
    output = Path(__file__).resolve().parent.parent / 'outputs' / 'predictions.png'
    output.parent.mkdir(exist_ok=True)
    d2l.plt.savefig(output, bbox_inches='tight')
    print(f'Predictions saved: {output}', flush=True)
    if show:
        print('Close the prediction window to finish the program.', flush=True)
        d2l.plt.show()
    d2l.plt.close()
