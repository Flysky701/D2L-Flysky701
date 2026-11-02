import argparse
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# 简单解释 卷积数量，出入通道
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

# 接下来是VGG网络的实现
class VGG(nn.Module):
    def __init__(self, conv_arch, fc_features, fc_hidden_units=4096):
    #   conv_arch: 卷积层的架构(几层)， fc_features: 全连接层的输入特征数，
    #   fc_hidden_units: 全连接层隐藏单元数
        super().__init__()
        self.conv = nn.Sequential()
        for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
            self.conv.add_module(f'vgg_block_{i+1}',
                                 vgg_block(num_convs, in_channels, out_channels))
        # 模板的简单实现
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_features, fc_hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fc_hidden_units, fc_hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fc_hidden_units, 10)
        )
    #  前向传播函数，x是输入数据
    def forward(self, x):
        feature = self.conv(x)
        output = self.fc(feature)
        return output


@torch.no_grad()
def evaluate_accuracy(net, data_iter, device):
    net.eval()
    correct, count = 0, 0
    for X, y in data_iter:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            logits = net(X)
        correct += (logits.argmax(dim=1) == y).sum().item()
        count += y.numel()
    return correct / count


def train(net, train_iter, test_iter, num_epochs, lr, device, output_dir):
    """在普通 Python 窗口打印进度，每轮保存指标和曲线。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(device)
    print(f'训练设备: {device}；训练批次: {len(train_iter)}；轮数: {num_epochs}', flush=True)
    print(f'结果目录: {output_dir}', flush=True)
    use_amp = device.type == 'cuda'
    if use_amp:
        torch.backends.cudnn.benchmark = True
        print(f'GPU: {torch.cuda.get_device_name(device)}；混合精度: 开启', flush=True)

    def init_weights(layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(layer.weight)

    net.to(device)
    net.apply(init_weights)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, num_epochs + 1):
        net.train()
        if use_amp:
            torch.cuda.synchronize(device)
        start = last_report = time.perf_counter()
        loss_sum, correct, count = 0.0, 0, 0
        for batch, (X, y) in enumerate(train_iter, 1):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if epoch == 1 and batch == 1:
                print(f'实际位置: 模型={next(net.parameters()).device}, '
                      f'输入={X.device}, 标签={y.device}', flush=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = net(X)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * y.numel()
            correct += (logits.detach().argmax(dim=1) == y).sum().item()
            count += y.numel()
            now = time.perf_counter()
            if batch == 1 or now - last_report >= 10 or batch == len(train_iter):
                print(f'epoch {epoch}/{num_epochs}, batch {batch}/{len(train_iter)}: '
                      f'loss {loss_sum / count:.4f}, train acc {correct / count:.2%}', flush=True)
                last_report = now
        print(f'epoch {epoch}/{num_epochs}: 正在评估测试集…', flush=True)
        test_acc = evaluate_accuracy(net, test_iter, device)
        if use_amp:
            torch.cuda.synchronize(device)
        row = dict(epoch=epoch, train_loss=loss_sum / count,
                   train_acc=correct / count, test_acc=test_acc,
                   seconds=time.perf_counter() - start)
        history.append(row)
        with (output_dir / 'history.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)

        epochs = [item['epoch'] for item in history]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(epochs, [item['train_loss'] for item in history], label='train loss')
        axes[1].plot(epochs, [item['train_acc'] for item in history], label='train acc')
        axes[1].plot(epochs, [item['test_acc'] for item in history], label='test acc')
        for ax, title in zip(axes, ('Cross entropy', 'Accuracy')):
            ax.set(xlabel='Epoch', title=title)
            ax.legend()
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / 'curves.png', dpi=150)
        plt.close(fig)
        print(f'epoch {epoch}/{num_epochs}: loss {row["train_loss"]:.4f}, '
              f'train acc {row["train_acc"]:.2%}, test acc {test_acc:.2%}, '
              f'{row["seconds"]:.1f}s；指标和曲线已保存', flush=True)
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VGG / Fashion-MNIST GPU 训练')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--smoke-test', action='store_true', help='仅运行一个训练批次和一个测试批次')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('--epochs 和 --batch-size 必须大于 0')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA 不可用，请使用 DIDL/.venv-d2l-gpu/Scripts/python.exe；CPU 运行需显式加 --device cpu')
    device = torch.device(args.device)
    # 定义卷积层的架构
    conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
    fc_features = 512 * 7 * 7
    net = VGG(conv_arch, fc_features)
    X = torch.randn(1, 1, 224, 224)
    # T = input("Do you want to see the output shape of each layer? (Yes/No): ")
    T = "No"
    if T == "Yes":
        for name, blk in net.named_children():
            X = blk(X)
            print(name, 'output shape:\t', X.shape)

    # 训练VGG模型
    # 定义参数
    lr, num_epochs, batch_size = 0.05, args.epochs, args.batch_size
    data_root = Path(__file__).absolute().parents[2] / '第4章' / 'data'
    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    train_data = datasets.FashionMNIST(data_root, train=True, transform=transform, download=True)
    test_data = datasets.FashionMNIST(data_root, train=False, transform=transform, download=True)
    if args.smoke_test:
        num_epochs = 1
        train_data = Subset(train_data, range(min(batch_size, len(train_data))))
        test_data = Subset(test_data, range(min(batch_size, len(test_data))))
        print('短验证：仅一个训练批次和一个测试批次，准确率不代表完整训练结果。', flush=True)
    workers = 2 if device.type == 'cuda' and not args.smoke_test else 0
    loader_options = dict(batch_size=batch_size, num_workers=workers,
                          pin_memory=device.type == 'cuda', persistent_workers=workers > 0)
    train_iter = DataLoader(train_data, shuffle=True, **loader_options)
    test_iter = DataLoader(test_data, shuffle=False, **loader_options)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d-%H%M%S-%f')
    if args.smoke_test:
        stamp = 'smoke-' + stamp
    output_dir = Path(__file__).absolute().parents[1] / 'outputs' / 'vgg' / stamp
    train(net, train_iter, test_iter, num_epochs, lr, device, output_dir)
