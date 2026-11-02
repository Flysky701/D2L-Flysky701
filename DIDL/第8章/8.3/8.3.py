"""8.3 语言模型和数据集：词频统计、序列采样与数据加载。

直接运行：观察两种采样方式；--demo stats：词频图；--demo data：真实语料批次。
"""

import argparse
import random
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from d2l import torch as d2l


# 复用 8.2 的数据缓存，路径不受运行时工作目录影响。
DATA_DIR = Path(__file__).resolve().parent.parent / "8.2" / "data"


def read_time_machine():
    """下载或复用缓存，返回清理后的文本行列表。"""
    filename = d2l.download(
        "https://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt",
        folder=str(DATA_DIR),
        sha1_hash="090b5e7e70c295757f55df93cb0a180b9691891a",
    )
    with open(filename, "r", encoding="utf-8") as f:
        return [re.sub(r"[^A-Za-z]+", " ", line).strip().lower() for line in f]


def demo_word_statistics():
    """8.3.3：按单词统计一元、二元和三元词组的频率。"""
    lines = read_time_machine()
    tokens = d2l.tokenize(lines, token="word")

    # 因为每个文本行不一定是一个句子或一个段落，因此把所有文本行拼接到一起。
    corpus = [word for line in tokens for word in line]
    vocab = d2l.Vocab(corpus)

    for word, freq in vocab.token_freqs[:10]:
        print(f"{word:<12} {freq}")

    # 用相邻的两个、三个词组成词组；这里只统计频率，不额外建立词表。
    bigram_counts = Counter(zip(corpus, corpus[1:]))
    trigram_counts = Counter(zip(corpus, corpus[1:], corpus[2:]))

    freqs = [freq for _, freq in vocab.token_freqs]
    bigram_freqs = sorted(bigram_counts.values(), reverse=True)
    trigram_freqs = sorted(trigram_counts.values(), reverse=True)
    frequencies = [freqs, bigram_freqs, trigram_freqs]

    # 横轴是按频率降序排列后的名次，从 1 开始，避免对数坐标出现 0。
    fig, ax = plt.subplots(figsize=(6, 4))
    for values, label, style in zip(
        frequencies,
        ["unigram", "bigram", "trigram"],
        ["-", "m--", "g-."],
    ):
        ranks = range(1, len(values) + 1)
        ax.loglog(ranks, values, style, label=label)
    ax.set_xlabel("frequency rank")
    ax.set_ylabel("frequency: n(x)")
    ax.legend()
    fig.tight_layout()
    plt.show()


def load_corpus_time_machine(max_tokens=-1):
    """读取字符词元，返回编号语料和词表；max_tokens <= 0 时保留全部语料。"""
    # 词频图使用单词；这里按教材的后续训练任务，改用字符作为词元。
    tokens = d2l.tokenize(read_time_machine(), token="char")
    vocab = d2l.Vocab(tokens)
    corpus = [vocab[char] for line in tokens for char in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


def seq_data_iter_random(corpus, batch_size, num_steps):
    """随机取用子序列，生成形状均为 (batch_size, num_steps) 的 X、Y。"""
    if batch_size <= 0 or num_steps <= 0:
        raise ValueError("batch_size 和 num_steps 必须为正整数")

    # 改变切分起点；最后一个输入词元还需要一个后继词元作为标签。
    offset = random.randint(0, num_steps - 1)
    corpus = corpus[offset:]
    num_subseqs = max(0, (len(corpus) - 1) // num_steps)
    starts = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(starts)  # 只打乱片段的取用顺序，片段内部的顺序不变。

    # 只输出完整批次，余下不足 batch_size 条的子序列不使用。
    num_batches = num_subseqs // batch_size
    for i in range(num_batches):
        batch_starts = starts[i * batch_size:(i + 1) * batch_size]
        X = [corpus[start:start + num_steps] for start in batch_starts]
        Y = [corpus[start + 1:start + num_steps + 1] for start in batch_starts]
        yield torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)


def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """按行连续读取子序列，保证相邻批次中同一行的内容能够接续。"""
    if batch_size <= 0 or num_steps <= 0:
        raise ValueError("batch_size 和 num_steps 必须为正整数")

    offset = random.randint(0, num_steps)
    # 先为标签保留一个词元，再截取能够均分为 batch_size 行的部分。
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    if num_tokens <= 0:
        return

    Xs = torch.tensor(corpus[offset:offset + num_tokens], dtype=torch.long)
    Ys = torch.tensor(corpus[offset + 1:offset + num_tokens + 1], dtype=torch.long)
    Xs = Xs.reshape(batch_size, -1)
    Ys = Ys.reshape(batch_size, -1)

    # 沿每一行每次向后取 num_steps 个词元，末尾不足长度的部分不使用。
    num_batches = Xs.shape[1] // num_steps
    for i in range(num_batches):
        start = i * num_steps
        yield Xs[:, start:start + num_steps], Ys[:, start:start + num_steps]


class SeqDataLoader:
    """保存语料与采样设置，每轮迭代重新生成 X、Y 批次。"""

    def __init__(self, batch_size, num_steps, use_random_iter=False, max_tokens=10000):
        if batch_size <= 0 or num_steps <= 0:
            raise ValueError("batch_size 和 num_steps 必须为正整数")
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.data_iter_fn = (
            seq_data_iter_random if use_random_iter else seq_data_iter_sequential
        )

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    """后续训练的统一入口，返回可反复迭代的数据加载器和字符词表。"""
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


def demo_sampling():
    """用 0～34 观察标签偏移和两种采样方式。"""
    corpus = list(range(35))
    for name, sample_fn in (
        ("随机采样", seq_data_iter_random),
        ("顺序分区", seq_data_iter_sequential),
    ):
        print(f"\n{name}（batch_size=2, num_steps=5）：")
        for batch, (X, Y) in enumerate(sample_fn(corpus, batch_size=2, num_steps=5), 1):
            print(f"第 {batch} 批，X.shape={tuple(X.shape)}")
            print("X:", X)
            print("Y:", Y)


def demo_data_loading():
    """从真实语料取一个批次，并把第一行编号还原为字符。"""
    data_iter, vocab = load_data_time_machine(batch_size=2, num_steps=5)
    X, Y = next(iter(data_iter))
    print(f"字符词表大小（含 <unk>）：{len(vocab)}")
    print(f"X.shape={tuple(X.shape)}, Y.shape={tuple(Y.shape)}")
    print("X:", X)
    print("Y:", Y)
    print("X[0] 对应文本：", "".join(vocab.to_tokens(X[0].tolist())))
    print("Y[0] 对应文本：", "".join(vocab.to_tokens(Y[0].tolist())))


def main(argv=None):
    demos = {
        "sampling": demo_sampling,
        "stats": demo_word_statistics,
        "data": demo_data_loading,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", choices=demos, default="sampling", help="选择运行的演示")
    args = parser.parse_args(argv)
    demos[args.demo]()


if __name__ == "__main__":
    main()
