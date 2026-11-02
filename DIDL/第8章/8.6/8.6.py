"""8.6 循环神经网络的简洁实现：定义模型，查看循环层和完整模型的输出。"""

import runpy
from pathlib import Path

import torch
from torch import nn


def load_data_time_machine(batch_size, num_steps,
                           use_random_iter=False, max_tokens=10000):
    """复用 8.3 的数据加载器和 8.2 的缓存。"""
    data_path = Path(__file__).resolve().parent.parent / "8.3" / "8.3.py"
    data_module = runpy.run_path(str(data_path))
    return data_module["load_data_time_machine"](
        batch_size, num_steps, use_random_iter, max_tokens
    )


class RNNModel(nn.Module):
    """循环神经网络模型。"""

    def __init__(self, rnn_layer, vocab_size):
        super().__init__()
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = rnn_layer.hidden_size
        self.num_directions = 2 if rnn_layer.bidirectional else 1
        # 双向模型会拼接两个方向的特征，输出层的输入宽度也要加倍。
        self.dense = nn.Linear(
            self.num_hiddens * self.num_directions, vocab_size
        )

    def forward(self, inputs, state):
        # 将输入转置后再做 one-hot 编码，得到形状为 (时间步数, 批量大小, 词表大小) 的输入。
        X = nn.functional.one_hot(inputs.T.long(), self.vocab_size).to(
            dtype=self.dense.weight.dtype
        )
        # Y 保存最后一层每个时间步的隐状态；state 保存各层的最终状态。
        Y, state = self.rnn(X, state)
        # 将 Y 变形为 (时间步数 * 批量大小, 方向数 * 隐藏单元数)，
        # 它的输出形状是 (时间步数 * 批量大小, 词表大小)。
        output = self.dense(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, batch_size, device):
        """为每层、每个方向准备初始记忆；LSTM 还需要单独的记忆单元状态。"""
        shape = (self.rnn.num_layers * self.num_directions,
                 batch_size, self.num_hiddens)
        state = torch.zeros(shape, device=device, dtype=self.dense.weight.dtype)
        if isinstance(self.rnn, nn.LSTM):
            # 后续章节的 LSTM 使用 (H, C) 两个状态，这里分别初始化为零。
            return state, torch.zeros_like(state)

        return state  # 普通 RNN、GRU 使用一个状态张量。


def main():
    batch_size, num_steps = 32, 35
    train_iter, vocab = load_data_time_machine(batch_size, num_steps)
    num_hiddens = 256
    rnn_layer = nn.RNN(len(vocab), num_hiddens)

    # 隐状态：(层数, 批量大小, 隐藏单元数)。
    state = torch.zeros((1, batch_size, num_hiddens))

    # nn.RNN 默认按时间步在前的 (T, B, V) 接收输入。
    X = torch.rand(size=(num_steps, batch_size, len(vocab)))
    Y, state_new = rnn_layer(X, state)
    print("循环层输入：", X.shape)
    print("循环层输出（每步隐状态）：", Y.shape)
    print("循环层最终隐状态：", state_new.shape)

    # 完整模型接收真实字符编号，在循环层后接输出层，得到字符分数。
    net = RNNModel(rnn_layer, len(vocab))
    inputs, targets = next(iter(train_iter))
    state = net.begin_state(batch_size=inputs.shape[0], device=inputs.device)
    output, state_new = net(inputs, state)
    labels = targets.T.reshape(-1)  # 与输出的时间步、批量展开顺序一致。
    print("字符编号输入：", inputs.shape)
    print("完整模型输出（字符分数）：", output.shape)
    print("展开后的标签：", labels.shape)
    print("完整模型最终隐状态：", state_new.shape)


if __name__ == "__main__":
    main()
