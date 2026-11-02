"""8.2 文本预处理：读取并清理《时光机器》数据集。"""

import re
from collections import Counter
from pathlib import Path

from d2l import torch as d2l


# 数据保存在本节目录中，不受运行时工作目录影响。
DATA_DIR = Path(__file__).resolve().parent / "data"
d2l.DATA_HUB["time_machine"] = (
    "https://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt",
    "090b5e7e70c295757f55df93cb0a180b9691891a",
)


def read_time_machine():
    """下载或复用缓存，返回清理后的文本行列表。"""
    filename = d2l.download("time_machine", folder=str(DATA_DIR))
    with open(filename, "r", encoding="utf-8") as f:
        # 将非英文字母替换为空格，去掉首尾空白，再转成小写。
        return [re.sub(r"[^A-Za-z]+", " ", line).strip().lower() for line in f]


def tokenize(lines, token="word"):
    """将文本行拆分为单词或字符词元。"""
    if token == "word":
        return [line.split() for line in lines]
    elif token == "char":
        return [list(line) for line in lines]
    else:
        print("错误：未知词元类型：" + token)

def count_corpus(tokens):
    """统计词元的频率。"""
    # 将嵌套列表展平成一个列表。
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return Counter(tokens)

class Vocab:
    """文本词表。"""

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []

        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        # 未知词元索引为0
        self.idx_to_token = ["<unk>"] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

if __name__ == "__main__":
    lines = read_time_machine()
    tokens = tokenize(lines, "char")
    vocab = Vocab(tokens)

# 上面是教材代码，一份简单的文本预处理代码，主要包括读取文本、清理文本、分词、统计词频和构建词表等功能。
