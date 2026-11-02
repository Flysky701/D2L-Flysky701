"""固定验证集、多种子比较；保留原始指标和未经平滑的曲线。"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torchvision.datasets import FashionMNIST
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mlp import build_model, make_optimizer

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = [
    ('depth1', [256], ['relu'], 0.1),
    ('depth2', [256, 256], ['relu'], 0.1),
    ('depth3', [256, 256, 256], ['relu'], 0.1),
    ('lr003', [256, 256], ['relu'], 0.03),
    ('lr03', [256, 256], ['relu'], 0.3),
    ('leaky_relu', [256, 256], ['leaky_relu'], 0.1),
    ('tanh', [256, 256], ['tanh'], 0.1),
    ('sigmoid', [256, 256], ['sigmoid'], 0.1),
    ('mixed', [256, 256], ['relu', 'tanh'], 0.1),
]


def load_tensors():
    train = FashionMNIST(str(ROOT / 'data'), train=True, download=False)
    test = FashionMNIST(str(ROOT / 'data'), train=False, download=False)
    X = train.data.float().reshape(-1, 784) / 255
    # 单独固定划分种子：50000 训练、10000 验证。测试集不参与配置选择。
    order = torch.randperm(len(X), generator=torch.Generator().manual_seed(2026))
    return (X[order[:50000]], train.targets[order[:50000]],
            X[order[50000:]], train.targets[order[50000:]],
            test.data.float().reshape(-1, 784) / 255, test.targets)


def evaluate(net, X, y):
    net.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(y), 1024):
            scores = net(X[start:start + 1024])
            labels = y[start:start + 1024]
            total_loss += nn.functional.cross_entropy(scores, labels, reduction='sum').item()
            correct += (scores.argmax(1) == labels).sum().item()
    return total_loss / len(y), correct / len(y)


def fit(net, X, y, X_val, y_val, lr, epochs, seed):
    trainer = make_optimizer(net, lr)
    # 顺序随机数独立于初始化，同种子下不同结构使用相同的批次顺序。
    generator = torch.Generator().manual_seed(10000 + seed)
    history = []
    for epoch in range(epochs):
        net.train()
        order = torch.randperm(len(y), generator=generator)
        running_loss, correct = 0.0, 0
        for indices in order.split(256):
            trainer.zero_grad()
            scores = net(X[indices])
            labels = y[indices]
            loss = nn.functional.cross_entropy(scores, labels)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss')
            loss.backward()
            trainer.step()
            running_loss += loss.item() * len(indices)
            correct += (scores.detach().argmax(1) == labels).sum().item()
        val_loss, val_acc = evaluate(net, X_val, y_val)
        history.append(dict(epoch=epoch + 1, train_loss=running_loss / len(y),
                            train_acc=correct / len(y), val_loss=val_loss, val_acc=val_acc))
    return history


def plot_results(results, destination):
    groups = [(['depth1', 'depth2', 'depth3'], 'Hidden layers'),
              (['lr003', 'depth2', 'lr03'], 'Learning rate'),
              (['depth2', 'leaky_relu', 'tanh', 'sigmoid', 'mixed'], 'Activation')]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for column, (names, title) in enumerate(groups):
        for name in names:
            runs = [r for r in results if r['name'] == name]
            for row, metric in enumerate(['train_loss', 'val_acc']):
                values = np.array([[h[metric] for h in r['history']] for r in runs])
                mean, std = values.mean(0), values.std(0)
                epochs = np.arange(1, len(mean) + 1)
                line, = axes[row, column].plot(epochs, mean, label=name)
                axes[row, column].fill_between(epochs, mean - std, mean + std,
                                               color=line.get_color(), alpha=0.15)
                axes[row, column].set_xlabel('Epoch')
                axes[row, column].set_ylabel('Train cross entropy' if row == 0 else 'Validation accuracy')
                axes[row, column].grid(alpha=0.2)
        axes[0, column].set_title(title)
        axes[1, column].legend(fontsize=9)
    fig.suptitle('Fashion-MNIST | CPU float32 | Xavier gain=1 | mean +/- 1 SD across seeds')
    fig.savefig(destination, dpi=150)
    plt.close(fig)

    # 明确指定 seed=0，展示单次原始曲线，不按表现挑选随机种子。
    means = {name: np.mean([r['history'][-1]['val_acc'] for r in results if r['name'] == name])
             for name, *_ in CONFIGS}
    winner = max(means, key=means.get)
    names = list(dict.fromkeys(['depth1', 'depth3', 'sigmoid', winner]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for name in names:
        run = next(r for r in results if r['name'] == name and r['seed'] == results[0]['seed'])
        for ax, metric in zip(axes, ['train_loss', 'val_acc']):
            ax.plot([h['epoch'] for h in run['history']],
                    [h[metric] for h in run['history']], marker='o', label=name)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            ax.grid(alpha=0.2)
            ax.legend()
    fig.suptitle(f'Representative raw runs | seed={results[0]["seed"]} | no smoothing')
    fig.savefig(destination.with_name('representative.png'), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    from mlp import ACTIVATIONS
    for name, factory in ACTIVATIONS.items():
        x = torch.linspace(-5, 5, 501, requires_grad=True)
        values = factory()(x)
        gradient, = torch.autograd.grad(values.sum(), x)
        axes[0].plot(x.detach(), values.detach(), label=name)
        axes[1].plot(x.detach(), gradient, label=name)
    for ax, title in zip(axes, ['Activation f(x)', 'Local gradient df/dx']):
        ax.set_xlabel('Input x')
        ax.set_title(title)
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle('ReLU / LeakyReLU(0.01) / Tanh / Sigmoid')
    fig.savefig(destination.with_name('activations.png'), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    args = parser.parse_args()
    if args.epochs < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Use positive epochs and unique seeds')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    output = ROOT / 'outputs' / 'experiments'
    output.mkdir(parents=True, exist_ok=True)
    X, y, X_val, y_val, X_test, y_test = load_tensors()
    results = []
    metadata = dict(torch=str(torch.__version__), device='cpu', dtype='float32', threads=4,
                    epochs=args.epochs, seeds=args.seeds, split_seed=2026,
                    train_size=len(y), val_size=len(y_val), test_size=len(y_test), batch_size=256,
                    initialization='xavier_uniform gain=1',
                    selection='mean final-epoch validation accuracy')
    for name, hidden, activation, lr in CONFIGS:
        for seed in args.seeds:
            torch.manual_seed(seed)
            net = build_model(hidden, activation, 'xavier')
            start = time.perf_counter()
            history = fit(net, X, y, X_val, y_val, lr, args.epochs, seed)
            result = dict(name=name, hidden=hidden, activation=activation, lr=lr, seed=seed,
                          parameters=sum(p.numel() for p in net.parameters()),
                          seconds=time.perf_counter() - start, history=history)
            results.append(result)
            (output / 'results.json').write_text(json.dumps(dict(metadata=metadata, runs=results),
                                                           indent=2), encoding='utf-8')
            print(f'{name} seed={seed}: val={history[-1]["val_acc"]:.4f}, '
                  f'loss={history[-1]["train_loss"]:.4f}, time={result["seconds"]:.1f}s', flush=True)
    summary = []
    for name, hidden, activation, lr in CONFIGS:
        runs = [r for r in results if r['name'] == name]
        values = [r['history'][-1]['val_acc'] for r in runs]
        summary.append(dict(name=name, hidden=hidden, activation=activation, lr=lr,
                            val_mean=float(np.mean(values)), val_std=float(np.std(values)),
                            parameters=runs[0]['parameters']))
    summary.sort(key=lambda r: r['val_mean'], reverse=True)
    best = summary[0]
    tests = []
    # 选定配置后重现训练，对保留的测试集评估，不用测试结果再改变配置。
    for seed in args.seeds:
        torch.manual_seed(seed)
        net = build_model(best['hidden'], best['activation'], 'xavier')
        fit(net, X, y, X_val, y_val, best['lr'], args.epochs, seed)
        test_loss, test_acc = evaluate(net, X_test, y_test)
        tests.append(dict(seed=seed, loss=test_loss, accuracy=test_acc))
    payload = dict(metadata=metadata, summary=summary, selected=best, test_results=tests)
    (output / 'summary.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    plot_results(results, output / 'curves.png')
    print('Selected:', best, flush=True)
    print('Held-out test:', tests, flush=True)
    print('Saved:', output, flush=True)


if __name__ == '__main__':
    main()
