"""8.5 循环神经网络的从零开始实现：前向计算、训练与预测。"""

import math
import runpy
from pathlib import Path

import torch
from torch.nn import functional as F
from d2l import torch as d2l


def load_data_time_machine(batch_size, num_steps,
                           use_random_iter=False, max_tokens=10000):
    """复用 8.3 的数据加载器和 8.2 的缓存，支持多轮迭代。"""
    # 8.3.py 的文件名含点，按绝对路径加载；不会执行它的 main() 演示。
    data_path = Path(__file__).resolve().parent.parent / "8.3" / "8.3.py"
    data_module = runpy.run_path(str(data_path))
    return data_module["load_data_time_machine"](
        batch_size, num_steps, use_random_iter, max_tokens
    )


def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 记录梯度，供后续训练使用。
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_rnn_state(batch_size, num_hiddens, device):
    """每条序列用一个全零向量作为初始隐状态。"""
    return (torch.zeros((batch_size, num_hiddens), device=device),)


def rnn(inputs, state, params):
    """inputs 为 (时间步数, 批量大小, 词表大小)，输出未经 softmax 的分数。"""
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)


class RNNModelScratch:
    """从零开始实现的循环神经网络模型。"""

    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        # (批量大小, 时间步数) → (时间步数, 批量大小, 词表大小)。
        X = F.one_hot(X.T.long(), self.vocab_size)
        X = X.to(device=self.params[0].device, dtype=self.params[0].dtype)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


@torch.no_grad()
def predict_ch8(prefix, num_preds, net, vocab, device):
    """在 prefix 后面接 num_preds 个字符。"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:  # 预热期
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 预测 num_preds 步
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return "".join([vocab.idx_to_token[i] for i in outputs])


def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """训练模型；采样标志应与创建 train_iter 时的设置一致。"""
    if num_epochs <= 0:
        raise ValueError("num_epochs 必须为正整数")
    loss = torch.nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[1, max(2, num_epochs)])
    updater = torch.optim.SGD(net.params, lr=lr)
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        state = None
        metric = d2l.Accumulator(2)  # 训练损失总和, 词元数量
        for X, Y in train_iter:
            if state is None or use_random_iter:
                state = net.begin_state(batch_size=X.shape[0], device=device)
            else:
                # 顺序分区保留隐状态的数值，截断跨批次的计算图。
                for s in state:
                    s.detach_()
            y = Y.T.reshape(-1)
            X, y = X.to(device), y.to(device)
            updater.zero_grad()
            y_hat, state = net(X, state)
            l = loss(y_hat, y.long())  # 默认对本批次的所有词元取平均。
            l.backward()
            d2l.grad_clipping(net, 1)
            updater.step()
            metric.add(l.detach().item() * y.numel(), y.numel())
        if metric[1] == 0:
            raise ValueError("没有可训练的批次，请检查语料长度、batch_size 和 num_steps")
        ppl = math.exp(metric[0] / metric[1])
        if (epoch + 1) % 10 == 0 or epoch + 1 == num_epochs:
            animator.add(epoch + 1, (ppl,))
    print(f'困惑度 {ppl:.1f}, '
          f'在 {str(device)} 上训练 {num_epochs} epoch，'
          f'耗时 {timer.stop():.1f} sec')
    print(predict_ch8('time traveller', 50, net, vocab, device))
    print(predict_ch8('traveller', 50, net, vocab, device))


def main():
    batch_size, num_steps = 32, 35
    use_random_iter = True
    train_iter, vocab = load_data_time_machine(
        batch_size, num_steps, use_random_iter=use_random_iter
    )
    device = d2l.try_gpu()

    num_hiddens = 256
    net = RNNModelScratch(
        len(vocab), num_hiddens, device, get_params, init_rnn_state, rnn
    )
    train_ch8(net, train_iter, vocab, lr=1, num_epochs=500, device=device,
              use_random_iter=use_random_iter)
    d2l.plt.show()


if __name__ == "__main__":
    main()
