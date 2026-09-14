"""作业一：使用 PyTorch 在 load_digits 上完成 MLP 控制变量实验。

默认命令：
    python mlp_digits.py --device cuda --output-dir assignment1_artifacts

程序会在本机真实运行实验，并保存曲线、CSV、运行元数据和终端日志。
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, TextIO

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
import torch.nn as nn
from matplotlib import font_manager
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


SEED = 42
EPOCHS = 30


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    name: str
    hidden: tuple[int, ...] = (32,)
    activation: str = "sigmoid"
    optimizer: str = "sgd"
    learning_rate: float = 0.1
    weight_decay: float = 0.0
    dropout: float = 0.0
    batch_norm: bool = False
    seed: int = SEED
    figure_name: str | None = None


class Tee:
    """把打印内容同时写到终端和日志文件。"""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


class MLP(nn.Module):
    """可配置的多层感知器。"""

    def __init__(
        self,
        hidden: tuple[int, ...] = (32,),
        activation: str = "sigmoid",
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        activation_types: dict[str, type[nn.Module]] = {
            "sigmoid": nn.Sigmoid,
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "gelu": nn.GELU,
        }
        if activation not in activation_types:
            raise ValueError(f"不支持的激活函数: {activation}")

        layers: list[nn.Module] = []
        previous = 64
        for width in hidden:
            layers.append(nn.Linear(previous, width))
            if batch_norm:
                # 按指导书要求：线性变换后、激活函数前加入 BatchNorm1d。
                layers.append(nn.BatchNorm1d(width))
            layers.append(activation_types[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous = width
        layers.append(nn.Linear(previous, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def configure_plot_font() -> str:
    """优先使用指导书指定的微软雅黑，避免中文显示为方框。"""
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    chosen = next((name for name in candidates if name in installed), "DejaVu Sans")
    plt.rcParams["font.sans-serif"] = [chosen]
    plt.rcParams["axes.unicode_minus"] = False
    return chosen


def set_reproducible_seed(seed: int) -> None:
    """每组实验都从相同的随机状态重新开始。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_data(device: torch.device) -> tuple[torch.Tensor, ...]:
    """加载并固定划分数据；标签必须是 CrossEntropyLoss 要求的 Long。"""
    features, labels = load_digits(return_X_y=True)
    features = features.astype(np.float32) / 16.0
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=SEED,
    )
    return (
        torch.tensor(train_x, dtype=torch.float32, device=device),
        torch.tensor(train_y, dtype=torch.long, device=device),
        torch.tensor(test_x, dtype=torch.float32, device=device),
        torch.tensor(test_y, dtype=torch.long, device=device),
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_experiment(
    config: ExperimentConfig,
    data: tuple[torch.Tensor, ...],
    device: torch.device,
) -> tuple[dict[str, object], dict[str, list[float]]]:
    set_reproducible_seed(config.seed)
    model = MLP(
        hidden=config.hidden,
        activation=config.activation,
        dropout=config.dropout,
        batch_norm=config.batch_norm,
    ).to(device)

    if config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"不支持的优化器: {config.optimizer}")

    train_x, train_y, test_x, test_y = data
    loss_fn = nn.CrossEntropyLoss()
    history: dict[str, list[float]] = {"loss": [], "accuracy": []}

    synchronize(device)
    started = time.perf_counter()
    for _ in range(EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_x)
        loss = loss_fn(logits, train_y)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            predictions = model(test_x).argmax(dim=1)
            accuracy = (predictions == test_y).float().mean().item()
        history["loss"].append(float(loss.detach().item()))
        history["accuracy"].append(float(accuracy))
    synchronize(device)
    elapsed = time.perf_counter() - started

    losses = np.asarray(history["loss"], dtype=float)
    accuracies = np.asarray(history["accuracy"], dtype=float)
    epoch_to_80 = next(
        (index + 1 for index, value in enumerate(accuracies) if value >= 0.80),
        None,
    )
    finite = bool(np.isfinite(losses).all())
    tail = losses[-10:]
    result: dict[str, object] = {
        **asdict(config),
        "hidden": "x".join(map(str, config.hidden)),
        "epochs": EPOCHS,
        "device": str(device),
        "final_accuracy_pct": float(accuracies[-1] * 100.0),
        "train_seconds": float(elapsed),
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "minimum_loss": float(np.nanmin(losses)),
        "tail_loss_std": float(np.nanstd(tail)),
        "epoch_to_80_pct": epoch_to_80,
        "finite_losses": finite,
        "status": "completed" if finite else "non_finite_loss",
    }
    return result, history


def plot_history(history: dict[str, list[float]], output_path: Path, title: str) -> None:
    epochs = np.arange(1, len(history["loss"]) + 1)
    # 横向紧凑比例适合报告模板中的图片栏，同时保留可独立阅读的坐标轴。
    figure, loss_axis = plt.subplots(figsize=(7.2, 3.1), constrained_layout=True)
    accuracy_axis = loss_axis.twinx()

    loss_line = loss_axis.plot(
        epochs,
        history["loss"],
        color="#1f4e79",
        linewidth=2.0,
        label="训练损失",
    )[0]
    accuracy_line = accuracy_axis.plot(
        epochs,
        np.asarray(history["accuracy"]) * 100.0,
        color="#c55a11",
        linewidth=2.0,
        label="测试准确率",
    )[0]
    loss_axis.set_xlabel("训练轮次")
    loss_axis.set_ylabel("交叉熵损失", color="#1f4e79")
    accuracy_axis.set_ylabel("测试准确率（%）", color="#c55a11")
    loss_axis.set_title(title)
    loss_axis.grid(True, alpha=0.25, linestyle="--")
    accuracy_axis.set_ylim(0, 100)
    loss_axis.legend(
        [loss_line, accuracy_line],
        [loss_line.get_label(), accuracy_line.get_label()],
        loc="best",
        frameon=False,
    )
    figure.savefig(output_path, dpi=200, facecolor="white")
    plt.close(figure)


def standard_experiments() -> list[ExperimentConfig]:
    return [
        ExperimentConfig("baseline", "基线 Baseline", figure_name="result_baseline.png"),
        ExperimentConfig(
            "exp1", "换激活函数", activation="relu", figure_name="result_exp1.png"
        ),
        ExperimentConfig(
            "exp2", "加深网络", hidden=(128, 128), figure_name="result_exp2.png"
        ),
        ExperimentConfig(
            "exp3",
            "换优化器",
            optimizer="adam",
            learning_rate=0.01,
            figure_name="result_exp3.png",
        ),
        ExperimentConfig(
            "exp4", "调学习率", learning_rate=0.01, figure_name="result_exp4.png"
        ),
        ExperimentConfig(
            "exp5", "加 L2 正则", weight_decay=1e-4, figure_name="result_exp5.png"
        ),
        ExperimentConfig(
            "exp6", "加 Dropout", dropout=0.2, figure_name="result_exp6.png"
        ),
        ExperimentConfig(
            "exp7", "加 BatchNorm", batch_norm=True, figure_name="result_exp7.png"
        ),
        ExperimentConfig(
            "exp8_run1",
            "最优组合第 1 次",
            hidden=(128, 128),
            activation="relu",
            optimizer="adam",
            learning_rate=0.001,
            batch_norm=True,
            figure_name="result_exp8_run1.png",
        ),
        ExperimentConfig(
            "exp8_run2",
            "最优组合第 2 次",
            hidden=(128, 128),
            activation="relu",
            optimizer="adam",
            learning_rate=0.001,
            batch_norm=True,
        ),
        ExperimentConfig(
            "exp8_run3",
            "最优组合第 3 次",
            hidden=(128, 128),
            activation="relu",
            optimizer="adam",
            learning_rate=0.001,
            batch_norm=True,
        ),
        ExperimentConfig(
            "lr_too_big",
            "失败实验",
            learning_rate=10.0,
            figure_name="result_lr_too_big.png",
        ),
    ]


def learning_rate_scan() -> list[ExperimentConfig]:
    return [
        ExperimentConfig(f"lr_scan_{lr:g}", f"学习率扫描 lr={lr:g}", learning_rate=lr)
        for lr in (0.1, 1.0, 5.0, 10.0, 20.0)
    ]


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        return
    fieldnames: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def collect_metadata(device: torch.device, font_name: str) -> dict[str, object]:
    cuda_available = torch.cuda.is_available()
    return {
        "generated_at_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": plt.matplotlib.__version__,
        "selected_device": str(device),
        "plot_font": font_name,
        "seed": SEED,
        "epochs": EPOCHS,
        "train_size": 1437,
        "test_size": 360,
        "split": "80/20 stratified, random_state=42",
    }


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("已请求 CUDA，但 torch.cuda.is_available() 为 False")
    return torch.device(requested)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("assignment1_artifacts"))
    parser.add_argument(
        "--experiment",
        default="all",
        help="all、baseline、exp1...exp7、exp8_run1...exp8_run3 或 lr_too_big",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "execution_log.txt"

    with log_path.open("w", encoding="utf-8") as log_stream:
        tee = Tee(sys.stdout, log_stream)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            device = resolve_device(args.device)
            font_name = configure_plot_font()
            data = load_data(device)
            configs = standard_experiments()
            known_ids = {config.experiment_id for config in configs}
            if args.experiment != "all":
                if args.experiment not in known_ids:
                    raise ValueError(f"未知实验: {args.experiment}; 可选值: {sorted(known_ids)}")
                configs = [c for c in configs if c.experiment_id == args.experiment]

            print(f"Python: {platform.python_version()}")
            print(f"PyTorch: {torch.__version__}; CUDA 编译版本: {torch.version.cuda}")
            print(f"运行设备: {device}")
            if device.type == "cuda":
                print(f"GPU: {torch.cuda.get_device_name(device)}")
            print(f"数据: 训练 1437 / 测试 360; 分层划分; random_state={SEED}")

            results: list[dict[str, object]] = []
            for config in configs:
                result, history = run_experiment(config, data, device)
                results.append(result)
                if config.figure_name:
                    plot_history(history, output_dir / config.figure_name, config.name)
                print(
                    f"[{config.experiment_id}] {config.name}: "
                    f"准确率={result['final_accuracy_pct']:.2f}% "
                    f"耗时={result['train_seconds']:.4f}s "
                    f"损失={result['initial_loss']:.4f}->{result['final_loss']:.4f}"
                )

            if args.experiment == "all":
                scan_rows: list[dict[str, object]] = []
                for config in learning_rate_scan():
                    result, _ = run_experiment(config, data, device)
                    scan_rows.append(result)
                    print(
                        f"[{config.experiment_id}] 准确率={result['final_accuracy_pct']:.2f}% "
                        f"损失={result['initial_loss']:.4f}->{result['final_loss']:.4f} "
                        f"尾段波动={result['tail_loss_std']:.4f}"
                    )
                write_csv(output_dir / "lr_scan.csv", scan_rows)

            write_csv(output_dir / "results.csv", results)
            metadata = collect_metadata(device, font_name)
            with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as stream:
                json.dump(metadata, stream, ensure_ascii=False, indent=2)
            print(f"结果已保存到: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
