# 8.7 通过时间反向传播

对梯度相关的内容进行进一步的诠释

首先，通过时间进行反向传播效率不高
一个长度为n的序列， 最前面的Token可以对后面的Token进行很远的影响
由于时间和内存的限制， 这种计算是显然不划算的


始终围绕一件事：权重改变一点，经过这些计算，最后的损失会怎样变化

## 8.7.1 循环神经网络的梯度分析

以下符号与公式推导为助手补充。

### 符号约定

- $T$：序列长度，也就是前文的 $n$；$t=1,\ldots,T$ 表示时间步。
- $\mathbf{x}_t$、$\mathbf{h}_t$、$\mathbf{o}_t$、$\mathbf{y}_t$：第 $t$ 步的输入、隐状态、输出和真实标签。
- $\mathbf{w}_h$：隐状态更新所用的共享参数，包含输入权重、循环权重和偏置；$\mathbf{w}_o$：输出层的共享参数。
- $\ell_t=\ell(\mathbf{o}_t,\mathbf{y}_t)$：第 $t$ 步的损失；$L$：整个序列的平均损失。

为明确矩阵乘法的顺序，下面将状态、输出和参数都视为列向量，矩阵参数先展平。$\partial\mathbf{a}/\partial\mathbf{b}$ 表示雅可比矩阵，其第 $(i,j)$ 项为 $\partial a_i/\partial b_j$；标量损失的梯度 $\nabla$ 也写成列向量。初始状态 $\mathbf{h}_0$ 固定，不依赖这些参数。

### 前向传播

每个时间步的隐状态和输出：

$$
\mathbf{h}_t=f(\mathbf{x}_t,\mathbf{h}_{t-1},\mathbf{w}_h),
\qquad
\mathbf{o}_t=g(\mathbf{h}_t,\mathbf{w}_o).
$$

前向传播是比较简单的
可以得到一个比较简单的公式

$$
L=\frac{1}{T}\sum_{t=1}^{T}\ell_t
=\frac{1}{T}\sum_{t=1}^{T}\ell(\mathbf{o}_t,\mathbf{y}_t).
$$

### 反向传播

但是反向传播，需要求的是损失对共享参数的梯度 $\nabla_{\mathbf{w}_h}L$，而不是 $L/\mathbf{w}_h$。

每一步的损失都依赖同一组参数，因此要将各步的贡献相加。沿着 $\mathbf{w}_h\to\mathbf{h}_t\to\mathbf{o}_t\to\ell_t$ 使用链式法则：

$$
\nabla_{\mathbf{w}_h}L
=\frac{1}{T}\sum_{t=1}^{T}
\left(\frac{\partial\mathbf{h}_t}{\partial\mathbf{w}_h}\right)^{\!\top}
\left(\frac{\partial\mathbf{o}_t}{\partial\mathbf{h}_t}\right)^{\!\top}
\nabla_{\mathbf{o}_t}\ell_t.
$$

难点在于 $\mathbf{w}_h$ 既直接参与当前状态的计算，也通过 $\mathbf{h}_{t-1}$ 间接影响当前状态。记

$$
D_t=\frac{\partial\mathbf{h}_t}{\partial\mathbf{w}_h},
\qquad
A_t=\frac{\partial f(\mathbf{x}_t,\mathbf{h}_{t-1},\mathbf{w}_h)}{\partial\mathbf{h}_{t-1}},
\qquad
B_t=\frac{\partial f(\mathbf{x}_t,\mathbf{h}_{t-1},\mathbf{w}_h)}{\partial\mathbf{w}_h}.
$$

其中 $A_t$ 和 $B_t$ 都是当前一步的局部偏导；计算 $B_t$ 时将 $\mathbf{h}_{t-1}$ 视为常量，而 $D_t$ 包含所有历史路径的影响。若隐状态维度为 $d$、隐状态更新参数总数为 $p$，则 $A_t\in\mathbb{R}^{d\times d}$，$B_t,D_t\in\mathbb{R}^{d\times p}$。

再次使用链式法则，得到递推式：

$$
D_t=B_t+A_tD_{t-1},\qquad D_0=0.
$$

逐步代入就能看到这个链怎样变长：

$$
\begin{aligned}
D_1&=B_1,\\
D_2&=B_2+A_2B_1,\\
D_3&=B_3+A_3B_2+A_3A_2B_1,\\
D_t&=\sum_{i=1}^{t}\left(A_tA_{t-1}\cdots A_{i+1}\right)B_i.
\end{aligned}
$$

当 $i=t$ 时，括号中没有因子，约定为空乘积 $I_d$，对应当前步的直接影响 $B_t$。当 $i<t$ 时，$B_i$ 表示参数在第 $i$ 步的直接影响，左侧的雅可比矩阵连乘表示这个影响如何沿隐状态传播到第 $t$ 步；矩阵顺序不能交换。

将这个展开式代回损失梯度：

$$
\nabla_{\mathbf{w}_h}L
=\frac{1}{T}\sum_{t=1}^{T}\sum_{i=1}^{t}
B_i^{\top}\left(A_tA_{t-1}\cdots A_{i+1}\right)^{\!\top}
\left(\frac{\partial\mathbf{o}_t}{\partial\mathbf{h}_t}\right)^{\!\top}
\nabla_{\mathbf{o}_t}\ell_t.
$$

输出层参数不参与隐状态递推，所以它的梯度可以直接按时间步相加：

$$
\nabla_{\mathbf{w}_o}L
=\frac{1}{T}\sum_{t=1}^{T}
\left(\frac{\partial g(\mathbf{h}_t,\mathbf{w}_o)}{\partial\mathbf{w}_o}\right)^{\!\top}
\nabla_{\mathbf{o}_t}\ell_t.
$$

通过很长的推导， 导致需要循环计算
虽然我们可以使用链式法则递归地计算 $\partial\mathbf{h}_t/\partial\mathbf{w}_h$，但当 $t$ 很大时这个链就会变得很长。我们需要想想办法来处理这一问题


- 完全计算
就是直接计算，
