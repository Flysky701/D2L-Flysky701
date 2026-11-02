import torch
from torch import nn
from pathlib import Path
import sys
from mlp import ACTIVATIONS, build_model, make_optimizer

# 复用 4.2 中的辅助函数，路径以此文件为基准，不依赖终端当前目录。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / '4.2'))
from chapter3 import load_data_fashion_mnist, train_ch3, predict_ch3

# build_model 在 mlp.py 中按列表构建 Sequential 并初始化所有 Linear 层。
# 默认宽度 [256, 256, 64] 对应 784 → 256 → 256 → 64 → 10。

# 保留各样本的损失，训练函数内部取平均用于反向传播，求和用于统计。
loss = nn.CrossEntropyLoss(reduction='none')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--hidden', type=int, nargs='+', default=[256, 256, 64])
    parser.add_argument('--activation', nargs='+', choices=list(ACTIVATIONS), default=['relu'])
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--layer-lrs', type=float, nargs='+')
    # 多隐藏层的小正态初始化可能学习很慢，默认使用 Xavier；normal 保留用于对照。
    parser.add_argument('--init', choices=['normal', 'xavier'], default='xavier')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--show', action='store_true', help='Open prediction window')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be at least 1')

    batch_size, lr, num_epochs = 256, args.lr, args.epochs
    torch.manual_seed(args.seed)
    try:
        # 一个激活名称应用于所有隐藏层，多个名称与隐藏层逐层对应。
        net = build_model(args.hidden, args.activation, args.init)
        trainer = make_optimizer(net, lr, args.layer_lrs)
    except ValueError as error:
        parser.error(str(error))
    print(net, flush=True)
    # 加载数据后开始训练；优化器已包含所有 Linear 层的权重与偏置。
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    # 当前 d2l 1.0.3 没有 train_ch3，使用项目内兼容两种更新方式的函数。
    train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
    predict_ch3(net, test_iter, show=args.show)
    print('Training complete.', flush=True)


