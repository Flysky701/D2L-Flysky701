# 5.1 层和块


从第 4 章的 `nn.Sequential` 接着看：一个模型怎样由层和块组成？自己编写模型时，`nn.Module` 和 `forward` 分别做什么？

## 代码笔记

- `MLP` 写具体网络；`MySequential` 写通用的顺序执行规则，两者都属于 `nn.Module`。
- 子块应保存为不同名字；当前代码用编号作键，避免两层 `Linear` 相互覆盖。
- `FixHiddenMLP` 可在 `forward` 中包含循环和分支，最后 `X.sum()` 返回标量。
- 已写 `ParallelBlock`：两条支路同时接收 `X`，再沿特征维度拼接。两路各输出 `(2, 10)` 时，合并形状为 `(2, 20)`。

