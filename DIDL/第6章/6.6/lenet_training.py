"""两个 LeNet 入口共用的数据加载、训练、评估和结果保存。"""

import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).absolute().parents[2]
CHAPTER_ROOT = Path(__file__).absolute().parents[1]


def load_data(batch_size=256, dataset_name="fashion-mnist", download=False, seed=0):
    """返回训练和测试迭代器；Windows 使用 num_workers=0。"""
    if dataset_name == "fashion-mnist":
        dataset_class = datasets.FashionMNIST
        root = PROJECT_ROOT / "第4章" / "data"  # 复用已有数据，不复制。
        if download:
            dataset_class.mirrors = [
                "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
            ]
    elif dataset_name == "mnist":
        dataset_class = datasets.MNIST
        root = CHAPTER_ROOT / "data"  # 独立保存到 第6章/data/MNIST。
        if download:
            # 使用 torchvision 自带的 HTTPS 镜像，下载时会校验 MD5。
            dataset_class.mirrors = [u for u in dataset_class.mirrors if u.startswith("https://")]
    else:
        raise ValueError("dataset_name 应为 fashion-mnist 或 mnist")
    transform = transforms.ToTensor()  # 灰度图 → float32 (1,28,28)，数值 /255。
    try:
        training = dataset_class(str(root), train=True, transform=transform, download=download)
        testing = dataset_class(str(root), train=False, transform=transform, download=download)
    except RuntimeError as exc:
        raise RuntimeError(f"无法加载 {dataset_name}，数据目录：{root}。缺少数据时加 --download。原错误：{exc}") from exc
    print(f"数据目录: {root}\n训练集: {len(training)}，测试集: {len(testing)}", flush=True)
    # 打乱训练样本；测试只用于评估，不打乱，也不参与反向传播。
    rng = torch.Generator().manual_seed(seed)
    return (DataLoader(training, batch_size=batch_size, shuffle=True, num_workers=0, generator=rng),
            DataLoader(testing, batch_size=batch_size, shuffle=False, num_workers=0))


def init_weights(layer):
    """与教材一样，对卷积层和全连接层的权重使用 Xavier 初始化。"""
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(layer.weight)


@torch.no_grad()
def evaluate(net, data_iter, device):
    """关闭梯度，只统计测试损失与准确率，不更新参数。"""
    net.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    loss_sum, correct, count = 0.0, 0, 0
    for X, y in data_iter:
        X, y = X.to(device), y.to(device)
        logits = net(X)
        loss_sum += loss_fn(logits, y).item()
        correct += (logits.argmax(dim=1) == y).sum().item()
        count += y.numel()
    return loss_sum / count, correct / count


def train(net, train_iter, test_iter, num_epochs, lr, device):
    """初始化后从头训练，返回逐轮记录；保留原有的六个参数。"""
    device = torch.device(device)
    net.apply(init_weights)
    
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()  # 输入 logits (N,10)，标签 int64 (N,)。
    history = []
    print(f"训练设备: {device}；epochs={num_epochs}；lr={lr}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}；CUDA {torch.version.cuda}", flush=True)
    for epoch in range(1, num_epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        net.train()
        loss_sum, correct, count = 0.0, 0, 0
        for batch, (X, y) in enumerate(train_iter, 1):
            X, y = X.to(device), y.to(device)
            if epoch == 1 and batch == 1:
                print(f"实际位置: 模型={next(net.parameters()).device}, "
                      f"输入={X.device}, 标签={y.device}", flush=True)
            optimizer.zero_grad()  # 1. 清除上一次的梯度。
            logits = net(X)         # 2. 前向传播，不额外加 softmax。
            loss = loss_fn(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError("损失出现 NaN/Inf，停止训练；请检查学习率和输入。")
            loss.backward()         # 3. 反向传播，计算参数梯度。
            optimizer.step()        # 4. SGD 更新参数。
            loss_sum += loss.item() * y.numel()
            correct += (logits.detach().argmax(dim=1) == y).sum().item()
            count += y.numel()
            if batch % 50 == 0:
                print(f"  epoch {epoch}/{num_epochs}, batch {batch}/{len(train_iter)}", flush=True)
        test_loss, test_acc = evaluate(net, test_iter, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        # 按样本数加权，避免最后一个小批次被当成完整批次统计。
        row = dict(epoch=epoch, train_loss=loss_sum / count, train_acc=correct / count,
                   test_loss=test_loss, test_acc=test_acc,
                   seconds=time.perf_counter() - start)
        history.append(row)
        print(f"epoch {epoch}/{num_epochs}: train loss {row['train_loss']:.4f}, "
              f"train acc {row['train_acc']:.2%}, test loss {test_loss:.4f}, "
              f"test acc {test_acc:.2%}, {row['seconds']:.1f}s", flush=True)
    return history


def save_results(net, history, config, dataset_name):
    """每次训练建独立目录，保存数值、曲线和最后一轮模型，不覆盖旧结果。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S-%f")
    output = CHAPTER_ROOT / "outputs" / "lenet" / dataset_name / stamp
    output.mkdir(parents=True, exist_ok=False)
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    (output / "results.json").write_text(
        json.dumps({"config": config, "history": history}, indent=2, ensure_ascii=False), encoding="utf-8")
    # 传入 Python 文件对象，兼容本机旧版 PyTorch 在中文路径下的保存。
    with (output / "model.pt").open("wb") as f:
        torch.save({"model_state_dict": net.state_dict(), "config": config}, f)
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for prefix in ("train", "test"):
        axes[0].plot(epochs, [row[f"{prefix}_loss"] for row in history], marker="o", label=prefix)
        axes[1].plot(epochs, [row[f"{prefix}_acc"] for row in history], marker="o", label=prefix)
    for ax, title in zip(axes, ("Cross entropy", "Accuracy")):
        ax.set(xlabel="Epoch", title=title, xticks=epochs)
        ax.legend()
        ax.grid(alpha=0.3)
    axes[1].set_ylim(0, 1)
    fig.suptitle(f"LeNet / {dataset_name}")
    fig.tight_layout()
    fig.savefig(output / "curves.png", dpi=150)
    plt.close(fig)
    print(f"训练记录与模型已保存: {output}", flush=True)
    return output


def run(net_factory, dataset_name):
    parser = argparse.ArgumentParser(description=f"LeNet / {dataset_name}")
    parser.add_argument("--train", action="store_true", help="开始训练；默认只查看形状和数据")
    parser.add_argument("--download", action="store_true", help="首次使用时下载缺少的数据")
    parser.add_argument("--shapes-only", action="store_true", help="只看网络形状，不加载数据")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.lr < float("inf"):
        parser.error("epochs、batch-size、lr 必须为正数，lr 必须有限")
    if args.train and args.shapes_only:
        parser.error("--train 与 --shapes-only 不能同时使用")
    torch.manual_seed(args.seed)
    net = net_factory()
    net.eval()
    X = torch.zeros(1, 1, 28, 28)
    print(f"数据集: {dataset_name}\n输入形状: {tuple(X.shape)}", flush=True)
    with torch.no_grad():
        for layer in net:
            X = layer(X)
            print(f"{layer.__class__.__name__:12s} -> {tuple(X.shape)}")
    if args.shapes_only:
        return
    train_iter, test_iter = load_data(args.batch_size, dataset_name, args.download, args.seed)
    # 从数据集直接取一个样本，避免提前消耗训练迭代器的随机顺序。
    image, label = train_iter.dataset[0]
    print(f"单张图片: {tuple(image.shape)}, {image.dtype}, "
          f"范围 [{image.min().item():.1f},{image.max().item():.1f}]")
    print(f"标签示例: {label} ({train_iter.dataset.classes[label]})")
    print(f"批次形状: X=(最多{args.batch_size},1,28,28), y=(最多{args.batch_size},)，标签为 int64")
    if not args.train:
        print("数据已就绪。加 --train 开始训练；加 --train --epochs 1 先试一轮。")
        return
    use_cuda = torch.cuda.is_available() if args.device == "auto" else args.device == "cuda"
    if use_cuda and not torch.cuda.is_available():
        parser.error("当前 PyTorch 无可用 CUDA，请使用 --device cpu 或 auto")
    device = torch.device("cuda" if use_cuda else "cpu")
    history = train(net, train_iter, test_iter, args.epochs, args.lr, device)
    config = dict(dataset=dataset_name, train_samples=len(train_iter.dataset),
                  test_samples=len(test_iter.dataset), epochs=args.epochs, batch_size=args.batch_size,
                  lr=args.lr, seed=args.seed, device=str(device), torch_version=str(torch.__version__),
                  cuda_version=torch.version.cuda,
                  gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                  optimizer="SGD", initialization="Xavier uniform weights; default biases",
                  classes=list(train_iter.dataset.classes))
    save_results(net, history, config, dataset_name)
