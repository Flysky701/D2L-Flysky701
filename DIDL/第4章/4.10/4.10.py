r"""D2L 4.10：房价预测，从数据处理到 5 折验证和提交文件。

在项目根目录运行：
    .\.venv-d2l\Scripts\python.exe .\第4章\4.10\4.10.py
教材：https://zh.d2l.ai/chapter_multilayer-perceptrons/kaggle-house-price.html
"""

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import urlopen

import matplotlib

matplotlib.use("Agg")  # 保存曲线，不弹出会阻塞程序的窗口。
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "house_price"
FILES = {
    "train": ("kaggle_house_pred_train.csv", "585e9cc93e70b39160e7921475f9bcd7d31219ce"),
    "test": ("kaggle_house_pred_test.csv", "fa19780a7b011d9b009e8bff8e99922a8ee2eb90"),
}


def download_data(kind):
    """下载教材提供的真实数据，缓存到项目内，并检查文件哈希。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filename, expected_hash = FILES[kind]
    path = DATA_DIR / filename
    if path.exists() and hashlib.sha1(path.read_bytes()).hexdigest() == expected_hash:
        return path
    for host in ("d2l-data.s3-accelerate.amazonaws.com", "d2l-data.s3.amazonaws.com"):
        url = f"https://{host}/{filename}"
        print(f"下载 {url}", flush=True)
        try:
            with urlopen(url, timeout=30) as response:
                content = response.read()
            if hashlib.sha1(content).hexdigest() != expected_hash:
                raise ValueError(f"{filename} 的 SHA-1 与教材记录不一致")
            path.write_bytes(content)
            return path
        except (OSError, ValueError) as error:
            print(f"下载未成功：{error}", flush=True)
    raise RuntimeError(f"无法取得 {filename}；请检查网络后重新运行。")


def fit_preprocessor(frame):
    """只用训练部分确定均值、标准差和 one-hot 列，避免验证数据泄漏。"""
    numeric = frame.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in frame.columns if column not in numeric]
    means = frame[numeric].mean().fillna(0)
    stds = frame[numeric].std().replace(0, 1).fillna(1)
    dummy_columns = pd.get_dummies(frame[categorical], dummy_na=True).columns.tolist()
    return {
        "numeric": numeric,
        "categorical": categorical,
        "means": means.to_dict(),
        "stds": stds.to_dict(),
        "dummy_columns": dummy_columns,
    }


def transform_features(frame, state):
    """原始表格 -> float32 张量，形状为 [样本数, 处理后的特征数]。"""
    # 数值：先标准化，再把缺失值填为 0（相当于填训练均值）。
    numeric = (frame[state["numeric"]] - pd.Series(state["means"])) / pd.Series(state["stds"])
    numeric = numeric.fillna(0)
    # 类别：如房屋类型 -> one-hot；训练时未见的类别在对应列中全部为 0。
    categorical = pd.get_dummies(frame[state["categorical"]], dummy_na=True)
    categorical = categorical.reindex(columns=state["dummy_columns"], fill_value=0)
    features = pd.concat([numeric, categorical], axis=1).to_numpy(dtype=np.float32)
    if not np.isfinite(features).all():
        raise ValueError("处理后的特征出现了 NaN 或无穷大")
    return torch.from_numpy(features)


def get_net(in_features):
    # [batch, 特征数] @ [特征数, 1] + 偏置 -> [batch, 1]，输出预测房价。
    # 本章先从线性回归基线开始，暂不增加隐藏层。
    net = nn.Sequential(nn.Linear(in_features, 1))
    nn.init.normal_(net[0].weight, mean=0, std=0.01)
    nn.init.zeros_(net[0].bias)
    return net


@torch.no_grad()
def log_rmse(net, features, labels):
    """对数房价误差：越小越好；它不是分类准确率，也不是精确百分比。"""
    net.eval()
    predictions = net(features).clamp(min=1)  # 防止非正预测值无法取对数。
    return torch.sqrt(torch.mean((predictions.log() - labels.log()) ** 2)).item()


def train(net, features, labels, args, valid_features=None, valid_labels=None, tag="训练"):
    loader = DataLoader(TensorDataset(features, labels), batch_size=args.batch_size,
                        shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss = nn.MSELoss()
    history = {"train_log_rmse": [], "valid_log_rmse": []}
    for epoch in range(1, args.epochs + 1):
        net.train()
        for x, y in loader:
            # 教材训练用原始房价的 MSE；报告指标才使用 log RMSE。
            predictions = net(x)
            batch_loss = loss(predictions, y)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        train_error = log_rmse(net, features, labels)
        history["train_log_rmse"].append(train_error)
        valid_error = None
        if valid_features is not None:
            valid_error = log_rmse(net, valid_features, valid_labels)
            history["valid_log_rmse"].append(valid_error)
        if not np.isfinite(train_error) or (valid_error is not None and not np.isfinite(valid_error)):
            raise RuntimeError("训练出现非有限误差，请检查输入或降低学习率")
        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            message = f"{tag} | epoch {epoch:3d}/{args.epochs} | train log RMSE {train_error:.4f}"
            if valid_error is not None:
                message += f" | valid log RMSE {valid_error:.4f}"
            print(message, flush=True)
    return history


def k_fold(frame, labels, args):
    # 固定种子打乱后分为 k 份，每份轮流验证；所有样本都会参与一次验证。
    folds = np.array_split(np.random.default_rng(args.seed).permutation(len(frame)), args.folds)
    results = []
    for fold, valid_indices in enumerate(folds):
        train_indices = np.concatenate([indices for i, indices in enumerate(folds) if i != fold])
        train_frame = frame.iloc[train_indices]
        state = fit_preprocessor(train_frame)
        train_features = transform_features(train_frame, state)
        valid_features = transform_features(frame.iloc[valid_indices], state)
        torch.manual_seed(args.seed + fold)
        net = get_net(train_features.shape[1])  # 每一折重新初始化，不能接着上一折训练。
        history = train(net, train_features, labels[train_indices], args,
                        valid_features, labels[valid_indices], tag=f"折 {fold + 1}")
        results.append({"fold": fold + 1, "train_rows": len(train_indices),
                        "valid_rows": len(valid_indices), "features": train_features.shape[1],
                        **history})
    return results


def save_curves(folds, final_history, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = np.arange(1, len(final_history["train_log_rmse"]) + 1)
    for key, label in (("train_log_rmse", "CV train mean"), ("valid_log_rmse", "CV validation mean")):
        axes[0].plot(epochs, np.mean([fold[key] for fold in folds], axis=0), label=label)
    axes[1].plot(epochs, final_history["train_log_rmse"], label="Full-data train")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Log RMSE (lower is better)")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="D2L 4.10 房价线性回归基线")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.weight_decay < 0:
        parser.error("epochs、batch-size、lr 必须为正数，weight-decay 不能为负数")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    train_data = pd.read_csv(download_data("train"))
    test_data = pd.read_csv(download_data("test"))
    if not 2 <= args.folds <= len(train_data):
        parser.error("folds 必须在 2 和训练样本数之间")
    frame = train_data.drop(columns=["Id", "SalePrice"])  # Id 是编号，不拿来预测。
    test_frame = test_data.drop(columns=["Id"])
    labels = torch.tensor(train_data[["SalePrice"]].to_numpy(dtype=np.float32))
    if not torch.isfinite(labels).all() or not (labels > 0).all():
        raise ValueError("房价标签必须是有限正数")
    print(f"原始训练表 {train_data.shape}；测试表 {test_data.shape}；标签 {tuple(labels.shape)}", flush=True)
    folds = k_fold(frame, labels, args)
    mean_train = float(np.mean([fold["train_log_rmse"][-1] for fold in folds]))
    mean_valid = float(np.mean([fold["valid_log_rmse"][-1] for fold in folds]))
    print(f"{args.folds} 折平均：train log RMSE {mean_train:.5f}，valid log RMSE {mean_valid:.5f}", flush=True)

    # 验证结束后，使用全部有标签数据重新训练最终模型。
    state = fit_preprocessor(frame)
    train_features = transform_features(frame, state)
    test_features = transform_features(test_frame, state)
    print(f"最终训练特征 {tuple(train_features.shape)}；测试特征 {tuple(test_features.shape)}", flush=True)
    torch.manual_seed(args.seed)
    net = get_net(train_features.shape[1])
    final_history = train(net, train_features, labels, args, tag="全量训练")
    net.eval()
    with torch.no_grad():
        predictions = net(test_features).clamp(min=1).squeeze(1).numpy()
    if len(predictions) != len(test_data) or not np.isfinite(predictions).all():
        raise RuntimeError("预测数量或数值异常，停止生成提交文件")

    # 每次运行保留独立目录，避免覆盖之前的实验结果。
    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output = ROOT / "outputs" / "house_price" / timestamp
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame({"Id": test_data["Id"], "SalePrice": predictions}).to_csv(output / "submission.csv", index=False)
    torch.save({"model_state_dict": net.state_dict(), "preprocessor": state,
                "in_features": train_features.shape[1], "config": vars(args)}, output / "model.pt")
    results = {"config": vars(args), "model": "linear regression", "training_loss": "raw-price MSE",
               "metric": "log RMSE", "cv_mean_train_log_rmse": mean_train,
               "cv_mean_valid_log_rmse": mean_valid, "folds": folds, "final_history": final_history,
               "train_features_shape": list(train_features.shape), "test_features_shape": list(test_features.shape),
               "submission_rows": len(predictions), "kaggle_submitted": False}
    (output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    save_curves(folds, final_history, output / "curves.png")
    print(f"输出已保存：{output}\n已生成 submission.csv，未上传 Kaggle。", flush=True)


if __name__ == "__main__":
    main()
