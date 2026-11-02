# 6.6 卷积神经网络（LeNet）



- 代码入口：[6.6.py](6.6.py)
- 手写数字入口：[6.6_mnist.py](6.6_mnist.py)，可独立修改数字模型的网络结构。
- 共用训练函数：[lenet_training.py](lenet_training.py)，两个入口共用数据加载、初始化、训练、评估和保存逻辑。
- 待记录：卷积、激活、汇聚和全连接层如何组合？
- 待验证：说明 `nn.Linear(16 * 5 * 5, 120)` 的输入维度从何而来。
- 已补充：Fashion-MNIST 与 MNIST 数据加载、训练循环和结果记录，使用方法如下。

## 两个数据集怎样选？

| 入口 | 任务 | 标签 | 本地数据目录 |
| --- | --- | --- | --- |
| `6.6.py` | Fashion-MNIST 服饰分类 | 10 类服饰，数字标签 0—9 代表类别 | 复用项目 `第4章/data/FashionMNIST` |
| `6.6_mnist.py` | MNIST 手写数字识别 | 0—9 十个数字 | 项目 `第6章/data/MNIST` |

两个数据集均为 60,000 张训练图、10,000 张测试图，图片都是单通道 `28×28`。数据集格式相同，识别的内容不同，所以可以从相同的 LeNet 结构开始比较。

两个入口各自保留 `build_net()`，便于分别修改结构，训练函数放在共用文件中。数据需要自行准备或使用脚本的下载选项。

## 数据怎样进入网络？

`load_data()` 使用 `transforms.ToTensor()` 将灰度图片转为 `float32`，并把像素值从 0—255 转到 0—1；没有额外做标准化或数据增强。

```text
单张图片：(1,28,28)
一批图片 X：(N,1,28,28)，默认 N=256，最后一批可以更小
一批标签 y：(N,)，类型 int64，值为 0—9
网络输出 logits：(N,10)
预测类别：logits.argmax(dim=1)，形状 (N,)
```

训练集 `shuffle=True`；测试集 `shuffle=False`。Windows 使用 `num_workers=0`。数据目录由 `Path(__file__)` 定位，所以从项目根目录或脚本目录运行都能找到相同文件。

## 怎样运行？

直接运行显示网络形状、数据数量和样本信息；加 `--train` 开始训练，默认 10 轮。缺少数据时使用 `--download` 下载。

```powershell
# Fashion-MNIST：先检查数据，再训练
.\.venv-d2l\Scripts\python.exe .\第6章\6.6\6.6.py
.\.venv-d2l\Scripts\python.exe .\第6章\6.6\6.6.py --train

# MNIST：独立的手写数字试验入口
.\.venv-d2l\Scripts\python.exe .\第6章\6.6\6.6_mnist.py
.\.venv-d2l\Scripts\python.exe .\第6章\6.6\6.6_mnist.py --train

# 只试一轮；也可调整 --batch-size、--lr、--seed、--device
.\.venv-d2l\Scripts\python.exe .\第6章\6.6\6.6_mnist.py --train --epochs 1

# 只查看网络形状，不加载数据
.\.venv-d2l\Scripts\python.exe .\第6章\6.6\6.6_mnist.py --shapes-only
```

VS Code 的 Run Code 没有附加参数时，也只做形状与数据检查。正式训练用上面的终端命令。

## 训练参数与每批数据的流程

默认使用教材的基本配置：Xavier 初始化卷积/全连接权重、SGD、小批量 256、学习率 0.9、10 轮。偏置保留 PyTorch 默认初始化；随机种子默认为 0。

`--device auto` 在有 CUDA 时使用 GPU，否则使用 CPU。

```text
optimizer.zero_grad() → net(X) → CrossEntropyLoss(logits,y)
                     → loss.backward() → optimizer.step()
```

- `train()` 保留原来预留的六个参数：网络、训练迭代器、测试迭代器、轮数、学习率、设备。实现位于 `lenet_training.py`，调用会初始化权重并从头训练。
- 交叉熵直接接收 logits，不要在最后一层再加 softmax。
- 每轮先 `net.train()`；测试时使用 `net.eval()` 和 `torch.no_grad()`，不计算梯度、不更新参数。
- 每轮记录训练/测试损失与准确率。损失按样本数量加权，最后一个较小批次也正确统计。
- 训练指标在参数逐批更新时累计；测试指标使用该轮结束后的模型，两者计算时的模型状态不同。
- 按教材每轮查看测试集指标。本例没有额外划出验证集；以后如果反复调参比较模型，应另留验证集。

## 结果保存到哪里？

每次训练完成会在以下目录新建带时间的子目录，不覆盖前一次结果：

```text
第6章/outputs/lenet/fashion-mnist/<时间>/
第6章/outputs/lenet/mnist/<时间>/
```

| 文件 | 内容 |
| --- | --- |
| `history.csv` | 逐轮训练/测试损失、准确率及用时 |
| `results.json` | 数据集、样本数、超参数、版本与逐轮指标 |
| `curves.png` | 损失和准确率的静态图，不需要弹窗 |
| `model.pt` | 最后一轮模型的 `state_dict` 与配置；不是完整网络定义，也不含优化器续训状态 |

已用 Python 文件对象保存模型，以兼容本机 PyTorch 2.0 在中文路径下的保存行为。

## GPU 环境与运行方式

GPU 运行需要安装与设备兼容的 PyTorch 版本，并确认 CUDA 可用。

对应版本为 Python 3.9、PyTorch `2.0.1+cu118`、torchvision `0.15.2+cu118`，沿用教材依赖版本。CUDA 11.8 是该 PyTorch 包对应的运行时版本，与 `nvidia-smi` 显示的驱动支持版本不是同一个含义；实际可用性以 PyTorch 的 CUDA 检查和运行结果为准。

在项目根目录使用 GPU 环境的解释器，显式指定 `--device cuda`：

```powershell
# MNIST 手写数字，先跑5轮
.\.venv-d2l-gpu\Scripts\python.exe .\第6章\6.6\6.6_mnist.py --train --epochs 5 --device cuda

# Fashion-MNIST，默认10轮
.\.venv-d2l-gpu\Scripts\python.exe .\第6章\6.6\6.6.py --train --device cuda
```

上述命令需要先准备包含 CUDA 版 PyTorch 的 `.venv-d2l-gpu` 虚拟环境。

训练日志会显示 GPU 型号，以及模型、输入、标签的实际设备。结果配置额外记录 `cuda_version` 和 `gpu_name`；每轮计时在 GPU 同步后读取，包含该轮训练、数据加载与测试评估，不含安装和数据下载。学习率仍为固定的 0.9，本次没有同时加入衰减。

环境版本依据：[PyTorch 官方历史版本安装说明](https://pytorch.org/get-started/previous-versions/)。

