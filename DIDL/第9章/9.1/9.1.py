import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l



batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

def get_params(vocab_size, num_hiddens, device):
    """返回模型参数。"""
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xh, W_hh, b_h = three()  # 隐状态参数
    W_xr, W_hr, b_r = three()  # 重置门参数
    W_xz, W_hz, b_z = three()  # 更新门参数
    W_hq = normal((num_hiddens, num_outputs))  # 输出层参数
    b_q = torch.zeros(num_outputs, device=device)
    # 附上梯度

    params = [W_xh, W_hh, b_h, W_xr, W_hr, b_r, W_xz, W_hz, b_z, W_hq, b_q]

    for param in params:
        param.requires_grad_(True)
    return params
