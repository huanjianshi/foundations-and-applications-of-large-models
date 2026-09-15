"""ConvLSTM bouncing-ball experiments for coursework assignment 2.

The script generates 2,200 independent sequences (2,000 train + 200 test),
runs the baseline and six controlled comparisons, builds an empirically chosen
combined configuration, and saves machine-readable evidence plus report-ready
figures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


@dataclass(frozen=True)
class ExperimentConfig:
    group: str
    name: str
    model_type: str = "convlstm"
    layers: int = 1
    hidden: int = 32
    kernel: int = 3
    input_frames: int = 4
    loss_name: str = "mse"
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 0.001


BASELINE = ExperimentConfig(group="0", name="基线 Baseline")
STANDARD_EXPERIMENTS = [
    BASELINE,
    replace(BASELINE, group="1", name="加深层数", layers=2),
    replace(BASELINE, group="2", name="隐藏通道", hidden=64),
    replace(BASELINE, group="3", name="卷积核", kernel=5),
    replace(BASELINE, group="4", name="输入帧数", input_frames=8),
    replace(BASELINE, group="5", name="结构对照", model_type="flatten_lstm"),
    replace(BASELINE, group="6", name="损失函数", loss_name="l1"),
]

EXPERIMENT_ALIASES = {
    "baseline": "0",
    "depth": "1",
    "hidden": "2",
    "kernel": "3",
    "frames": "4",
    "flatten": "5",
    "l1": "6",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        default="all",
        choices=["all", *EXPERIMENT_ALIASES, "best"],
        help="Run all experiments, one controlled experiment, or best (requires results.csv).",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42, help="Model and shuffle seed.")
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--quick-check", action="store_true", help="Use a tiny run for code validation.")
    return parser.parse_args()


def configure_logging(output_dir: Path, append: bool = False) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("convlstm_bounce")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output_dir / "run.log", mode="a" if append else "w", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def resolve_device(choice: str) -> torch.device:
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_sequences(
    n_seq: int,
    t_steps: int = 10,
    size: int = 32,
    radius: int = 2,
    seed: int = 42,
) -> tuple[np.ndarray, list[dict[str, float | bool]]]:
    """Generate independent bouncing-ball sequences and useful sample metadata."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    sequences = np.zeros((n_seq, t_steps, 1, size, size), dtype=np.float32)
    metadata: list[dict[str, float | bool]] = []
    low, high = float(radius), float(size - radius)

    for sample in range(n_seq):
        x, y = rng.uniform(4, size - 5, 2)
        vx, vy = rng.choice([-1, 1], 2) * rng.uniform(0.8, 1.6, 2)
        initial_speed = float(math.hypot(vx, vy))
        bounce_steps: list[int] = []
        for step in range(t_steps):
            x += vx
            y += vy
            bounced = False
            if x < low:
                x = 2 * low - x
                vx = -vx
                bounced = True
            elif x > high:
                x = 2 * high - x
                vx = -vx
                bounced = True
            if y < low:
                y = 2 * low - y
                vy = -vy
                bounced = True
            elif y > high:
                y = 2 * high - y
                vy = -vy
                bounced = True
            if bounced:
                bounce_steps.append(step)
            sequences[sample, step, 0] = (
                (xx - x) ** 2 + (yy - y) ** 2 <= radius * radius
            ).astype(np.float32)
        metadata.append(
            {
                "speed": initial_speed,
                "bounced_by_frame_5": any(s <= 4 for s in bounce_steps),
                "bounced_by_frame_9": any(s <= 8 for s in bounce_steps),
            }
        )
    return sequences, metadata


def assert_disjoint(train: np.ndarray, test: np.ndarray) -> None:
    train_hashes = {
        hashlib.blake2b(sample.tobytes(), digest_size=16).digest() for sample in train
    }
    test_hashes = {
        hashlib.blake2b(sample.tobytes(), digest_size=16).digest() for sample in test
    }
    overlap = train_hashes.intersection(test_hashes)
    if overlap:
        raise AssertionError(f"train/test contain {len(overlap)} identical sequences")


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=kernel_size // 2,
        )

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, cell = state
        gates = self.conv(torch.cat((x, hidden), dim=1))
        input_gate, forget_gate, candidate, output_gate = gates.chunk(4, dim=1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        output_gate = torch.sigmoid(output_gate)
        cell_new = forget_gate * cell + input_gate * torch.tanh(candidate)
        hidden_new = output_gate * torch.tanh(cell_new)
        return hidden_new, cell_new


class ConvLSTM(nn.Module):
    def __init__(self, hidden: int = 32, kernel: int = 3, layers: int = 1):
        super().__init__()
        channel_pairs = [(1, hidden)] + [(hidden, hidden)] * (layers - 1)
        self.cells = nn.ModuleList(
            ConvLSTMCell(in_ch, hidden, kernel) for in_ch, hidden in channel_pairs
        )
        self.output = nn.Conv2d(hidden, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, _, height, width = x.shape
        states = [
            (
                x.new_zeros(batch, cell.hidden_channels, height, width),
                x.new_zeros(batch, cell.hidden_channels, height, width),
            )
            for cell in self.cells
        ]
        for step in range(x.shape[1]):
            current = x[:, step]
            for layer, cell in enumerate(self.cells):
                states[layer] = cell(current, states[layer])
                current = states[layer][0]
        return self.output(states[-1][0])


class FlattenLSTM(nn.Module):
    def __init__(self, image_size: int = 32, hidden: int = 256):
        super().__init__()
        self.image_size = image_size
        pixels = image_size * image_size
        self.lstm = nn.LSTM(pixels, hidden, batch_first=True)
        self.output = nn.Linear(hidden, pixels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps = x.shape[:2]
        flattened = x.reshape(batch, steps, -1)
        sequence, _ = self.lstm(flattened)
        prediction = self.output(sequence[:, -1])
        return prediction.reshape(batch, 1, self.image_size, self.image_size)


def build_model(config: ExperimentConfig) -> nn.Module:
    if config.model_type == "flatten_lstm":
        return FlattenLSTM()
    return ConvLSTM(hidden=config.hidden, kernel=config.kernel, layers=config.layers)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    device: torch.device,
    return_predictions: bool = False,
) -> tuple[float, float, torch.Tensor | None]:
    model.eval()
    squared_error = 0.0
    absolute_error = 0.0
    element_count = 0
    predictions: list[torch.Tensor] = []
    for start in range(0, len(inputs), batch_size):
        batch_x = inputs[start : start + batch_size].to(device, non_blocking=True)
        batch_y = targets[start : start + batch_size].to(device, non_blocking=True)
        pred = model(batch_x)
        diff = pred - batch_y
        squared_error += diff.square().sum().item()
        absolute_error += diff.abs().sum().item()
        element_count += diff.numel()
        if return_predictions:
            predictions.append(pred.detach().cpu())
    synchronize(device)
    joined = torch.cat(predictions) if predictions else None
    return squared_error / element_count, absolute_error / element_count, joined


def run_experiment(
    config: ExperimentConfig,
    seed: int,
    train_data: torch.Tensor,
    test_data: torch.Tensor,
    device: torch.device,
    logger: logging.Logger,
) -> tuple[dict[str, object], list[dict[str, object]], torch.Tensor]:
    set_seed(seed)
    model = build_model(config).to(device)
    params = parameter_count(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn: nn.Module = nn.L1Loss() if config.loss_name == "l1" else nn.MSELoss()
    train_x = train_data[:, : config.input_frames]
    train_y = train_data[:, config.input_frames]
    test_x = test_data[:, : config.input_frames]
    test_y = test_data[:, config.input_frames]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    history: list[dict[str, object]] = []

    # Warm up both forward and backward kernels before timing without updating weights.
    warm_x = train_x[:2].to(device)
    warm_y = train_y[:2].to(device)
    probe = model(warm_x)
    if probe.shape != (2, 1, 32, 32):
        raise AssertionError(f"unexpected output shape: {tuple(probe.shape)}")
    loss_fn(probe, warm_y).backward()
    optimizer.zero_grad(set_to_none=True)
    del warm_x, warm_y, probe

    synchronize(device)
    start_time = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_x), generator=generator)
        loss_sum = 0.0
        seen = 0
        for start in range(0, len(train_x), config.batch_size):
            indices = permutation[start : start + config.batch_size]
            batch_x = train_x[indices].to(device, non_blocking=True)
            batch_y = train_y[indices].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x)
            loss = loss_fn(prediction, batch_y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss in {config.name}, epoch {epoch}")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(indices)
            seen += len(indices)
        test_mse, test_mae, _ = evaluate(
            model, test_x, test_y, config.batch_size, device
        )
        epoch_row = {
            "group": config.group,
            "experiment": config.name,
            "seed": seed,
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "test_mse": test_mse,
            "test_mae": test_mae,
        }
        history.append(epoch_row)
        logger.info(
            "group=%s name=%s seed=%d epoch=%d train_loss=%.7f test_mse=%.7f test_mae=%.7f",
            config.group,
            config.name,
            seed,
            epoch,
            epoch_row["train_loss"],
            test_mse,
            test_mae,
        )
    synchronize(device)
    elapsed = time.perf_counter() - start_time
    final_mse, final_mae, predictions = evaluate(
        model, test_x, test_y, config.batch_size, device, return_predictions=True
    )
    assert predictions is not None
    result: dict[str, object] = {
        "group": config.group,
        "experiment": config.name,
        "seed": seed,
        "model_type": config.model_type,
        "layers": config.layers,
        "hidden": config.hidden,
        "kernel": config.kernel,
        "input_frames": config.input_frames,
        "loss": config.loss_name,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "parameters": params,
        "test_mse": final_mse,
        "test_mae": final_mae,
        "elapsed_seconds": elapsed,
        "device": str(device),
    }
    logger.info(
        "finished group=%s seed=%d params=%d mse=%.7f mae=%.7f time=%.2fs",
        config.group,
        seed,
        params,
        final_mse,
        final_mae,
        elapsed,
    )
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, history, predictions


def assert_controlled_configs() -> None:
    fields = ["model_type", "layers", "hidden", "kernel", "input_frames", "loss_name"]
    expected = {"1": "layers", "2": "hidden", "3": "kernel", "4": "input_frames", "5": "model_type", "6": "loss_name"}
    for config in STANDARD_EXPERIMENTS[1:]:
        changed = [field for field in fields if getattr(config, field) != getattr(BASELINE, field)]
        if changed != [expected[config.group]]:
            raise AssertionError(f"group {config.group} is not controlled: {changed}")


def plot_history(history: list[dict[str, object]], output_path: Path, title: str) -> None:
    epochs = [int(row["epoch"]) for row in history]
    train_loss = [float(row["train_loss"]) for row in history]
    test_mse = [float(row["test_mse"]) for row in history]
    test_mae = [float(row["test_mae"]) for row in history]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(epochs, train_loss, marker="o", linewidth=2, label="训练损失")
    ax.plot(epochs, test_mse, marker="s", linewidth=2, label="测试 MSE")
    ax.plot(epochs, test_mae, marker="^", linewidth=2, label="测试 MAE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("损失 / 误差")
    ax.set_title(title)
    ax.set_xticks(epochs)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def select_prediction_samples(
    metadata: list[dict[str, float | bool]], input_frames: int
) -> list[int]:
    bounce_key = "bounced_by_frame_9" if input_frames == 8 else "bounced_by_frame_5"
    speeds = np.array([float(row["speed"]) for row in metadata])
    non_bounce = [i for i, row in enumerate(metadata) if not bool(row[bounce_key])]
    bounce = [i for i, row in enumerate(metadata) if bool(row[bounce_key])]
    typical = min(non_bounce, key=lambda i: abs(speeds[i] - np.median(speeds))) if non_bounce else 0
    fast_pool = non_bounce or list(range(len(metadata)))
    fast = max(fast_pool, key=lambda i: speeds[i])
    boundary = max(bounce, key=lambda i: speeds[i]) if bounce else int(np.argmax(speeds))
    selected = [typical, fast, boundary]
    for index in range(len(metadata)):
        if len(set(selected)) == 3:
            break
        if index not in selected:
            selected[selected.index(next(x for x in selected if selected.count(x) > 1))] = index
    return selected


def plot_predictions(
    config: ExperimentConfig,
    test_data: torch.Tensor,
    test_metadata: list[dict[str, float | bool]],
    predictions: torch.Tensor,
    output_path: Path,
) -> list[dict[str, object]]:
    indices = select_prediction_samples(test_metadata, config.input_frames)
    shown_frames = min(4, config.input_frames)
    frame_start = config.input_frames - shown_frames
    columns = shown_frames + 2
    fig, axes = plt.subplots(3, columns, figsize=(2.15 * columns, 6.5))
    descriptions = ["典型样本", "高速样本", "近边界/反弹样本"]
    details: list[dict[str, object]] = []
    for row, (sample_index, description) in enumerate(zip(indices, descriptions)):
        target = test_data[sample_index, config.input_frames, 0].numpy()
        prediction = predictions[sample_index, 0].numpy()
        sample_mse = float(np.mean((prediction - target) ** 2))
        for offset in range(shown_frames):
            frame_number = frame_start + offset
            axes[row, offset].imshow(test_data[sample_index, frame_number, 0], cmap="gray", vmin=0, vmax=1)
            axes[row, offset].set_title(f"输入帧 {frame_number + 1}", fontsize=10)
        axes[row, -2].imshow(target, cmap="gray", vmin=0, vmax=1)
        axes[row, -2].set_title("真实下一帧", fontsize=10)
        axes[row, -1].imshow(prediction, cmap="gray", vmin=0, vmax=1)
        axes[row, -1].set_title(f"预测帧\nMSE={sample_mse:.5f}", fontsize=10)
        axes[row, 0].set_ylabel(description, fontsize=10)
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
        details.append(
            {
                "description": description,
                "test_index": sample_index,
                "speed": float(test_metadata[sample_index]["speed"]),
                "bounced": bool(test_metadata[sample_index]["bounced_by_frame_9" if config.input_frames == 8 else "bounced_by_frame_5"]),
                "mse": sample_mse,
            }
        )
    suffix = "（展示最后 4 帧，模型实际输入 8 帧）" if config.input_frames == 8 else ""
    fig.suptitle(f"输入帧、真实下一帧与预测帧对比{suffix}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return details


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def persist_results(output_dir: Path, results: list[dict[str, object]], histories: list[dict[str, object]]) -> None:
    result_fields = [
        "group", "experiment", "seed", "model_type", "layers", "hidden", "kernel",
        "input_frames", "loss", "epochs", "batch_size", "learning_rate", "parameters",
        "test_mse", "test_mae", "elapsed_seconds", "device",
    ]
    history_fields = ["group", "experiment", "seed", "epoch", "train_loss", "test_mse", "test_mae"]
    write_csv(output_dir / "results.csv", results, result_fields)
    write_csv(output_dir / "histories.csv", histories, history_fields)


def read_results(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def choose_best_config(results: list[dict[str, object]], logger: logging.Logger) -> tuple[ExperimentConfig, list[str]]:
    by_group = {str(row["group"]): row for row in results if str(row["group"]) in {str(i) for i in range(7)}}
    if len(by_group) != 7:
        raise RuntimeError("all seven standard experiment results are required to choose best")
    baseline_mse = float(by_group["0"]["test_mse"])
    config = replace(BASELINE, group="7", name="最优组合")
    adopted: list[str] = []
    compatible = {
        "1": ("layers", 2, "加深层数"),
        "2": ("hidden", 64, "隐藏通道"),
        "3": ("kernel", 5, "卷积核"),
        "4": ("input_frames", 8, "输入帧数"),
        "6": ("loss_name", "l1", "L1损失"),
    }
    for group, (field, value, label) in compatible.items():
        if float(by_group[group]["test_mse"]) < baseline_mse:
            config = replace(config, **{field: value})
            adopted.append(label)
    if not adopted:
        best_group = min(compatible, key=lambda group: float(by_group[group]["test_mse"]))
        field, value, label = compatible[best_group]
        config = replace(config, **{field: value})
        adopted.append(f"最低MSE单项：{label}")
    logger.info("best configuration adopted changes: %s", "、".join(adopted))
    return config, adopted


def mean_result(rows: list[dict[str, object]], config: ExperimentConfig, device: torch.device) -> dict[str, object]:
    return {
        "group": "7_mean",
        "experiment": "最优组合平均",
        "seed": "42/43/44",
        "model_type": config.model_type,
        "layers": config.layers,
        "hidden": config.hidden,
        "kernel": config.kernel,
        "input_frames": config.input_frames,
        "loss": config.loss_name,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "parameters": rows[0]["parameters"],
        "test_mse": float(np.mean([float(row["test_mse"]) for row in rows])),
        "test_mae": float(np.mean([float(row["test_mae"]) for row in rows])),
        "elapsed_seconds": float(np.mean([float(row["elapsed_seconds"]) for row in rows])),
        "device": str(device),
    }


def verify_parameter_formulas() -> None:
    baseline = ConvLSTM(hidden=32, kernel=3, layers=1)
    cell_count = parameter_count(baseline.cells[0])
    output_count = parameter_count(baseline.output)
    if (cell_count, output_count, parameter_count(baseline)) != (38144, 289, 38433):
        raise AssertionError((cell_count, output_count, parameter_count(baseline)))
    hidden64_cell = parameter_count(ConvLSTMCell(1, 64, 3))
    if hidden64_cell != 150016:
        raise AssertionError(hidden64_cell)
    flatten = parameter_count(FlattenLSTM())
    if flatten != 1575936:
        raise AssertionError(flatten)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    logger = configure_logging(output_dir, append=args.experiment == "best")
    device = resolve_device(args.device)
    assert_controlled_configs()
    verify_parameter_formulas()

    train_size = 96 if args.quick_check else args.train_size
    test_size = 24 if args.quick_check else args.test_size
    epochs = 1 if args.quick_check else args.epochs
    batch_size = min(16, train_size) if args.quick_check else args.batch_size
    logger.info("device=%s train=%d test=%d epochs=%d batch=%d", device, train_size, test_size, epochs, batch_size)

    all_sequences, all_metadata = make_sequences(train_size + test_size, seed=args.data_seed)
    train_np = all_sequences[:train_size]
    test_np = all_sequences[train_size:]
    assert train_np.shape == (train_size, 10, 1, 32, 32)
    assert test_np.shape == (test_size, 10, 1, 32, 32)
    assert_disjoint(train_np, test_np)
    train_data = torch.from_numpy(train_np)
    test_data = torch.from_numpy(test_np)
    test_metadata = all_metadata[train_size:]

    configs = [replace(config, epochs=epochs, batch_size=batch_size) for config in STANDARD_EXPERIMENTS]
    results: list[dict[str, object]] = []
    histories: list[dict[str, object]] = []
    prediction_cache: dict[tuple[str, int], torch.Tensor] = {}
    config_cache: dict[str, ExperimentConfig] = {config.group: config for config in configs}

    if args.experiment == "all":
        selected_configs = configs
    elif args.experiment in EXPERIMENT_ALIASES:
        group = EXPERIMENT_ALIASES[args.experiment]
        selected_configs = [config_cache[group]]
    else:
        selected_configs = []

    for config in selected_configs:
        result, history, predictions = run_experiment(
            config, args.seed, train_data, test_data, device, logger
        )
        results.append(result)
        histories.extend(history)
        prediction_cache[(config.group, args.seed)] = predictions
        figure_name = "result_baseline.png" if config.group == "0" else f"result_exp{config.group}.png"
        plot_history(history, output_dir / figure_name, f"{config.name}训练与测试曲线")
        persist_results(output_dir, results, histories)

    adopted_changes: list[str] = []
    best_config: ExperimentConfig | None = None
    prediction_details: list[dict[str, object]] = []
    if args.experiment == "all":
        best_config, adopted_changes = choose_best_config(results, logger)
        best_config = replace(best_config, epochs=epochs, batch_size=batch_size)
        config_cache["7"] = best_config
        best_rows: list[dict[str, object]] = []
        for seed in (42, 43, 44):
            result, history, predictions = run_experiment(
                best_config, seed, train_data, test_data, device, logger
            )
            best_rows.append(result)
            results.append(result)
            histories.extend(history)
            prediction_cache[("7", seed)] = predictions
            persist_results(output_dir, results, histories)
        results.append(mean_result(best_rows, best_config, device))
        persist_results(output_dir, results, histories)
        prediction_details = plot_predictions(
            best_config,
            test_data,
            test_metadata,
            prediction_cache[("7", 42)],
            output_dir / "result_pred.png",
        )
    elif args.experiment == "best":
        existing_path = output_dir / "results.csv"
        if not existing_path.exists():
            raise FileNotFoundError("--experiment best requires an existing results.csv from --experiment all")
        existing_results = read_results(existing_path)
        best_config, adopted_changes = choose_best_config(existing_results, logger)
        best_config = replace(best_config, epochs=epochs, batch_size=batch_size)
        result, history, predictions = run_experiment(
            best_config, args.seed, train_data, test_data, device, logger
        )
        write_csv(output_dir / "best_rerun_results.csv", [result], list(result.keys()))
        write_csv(output_dir / "best_rerun_histories.csv", history, list(history[0].keys()))
        prediction_details = plot_predictions(
            best_config, test_data, test_metadata, predictions, output_dir / "result_pred.png"
        )

    metadata = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "matplotlib": plt.matplotlib.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "train_size": train_size,
        "test_size": test_size,
        "data_seed": args.data_seed,
        "train_test_overlap": 0,
        "best_configuration": asdict(best_config) if best_config else None,
        "best_adopted_changes": adopted_changes,
        "prediction_samples": prediction_details,
        "baseline_parameter_check": {"cell": 38144, "output": 289, "total": 38433},
        "hidden64_cell_parameters": 150016,
        "flatten_lstm_parameters": 1575936,
        "quick_check": args.quick_check,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("all requested artifacts written to %s", output_dir)


if __name__ == "__main__":
    main()
