"""Innovation extensions for the ConvLSTM bouncing-ball coursework.

Two guide-aligned extensions are evaluated with the empirically selected
2-layer, 64-channel, 5x5 ConvLSTM: (1) two independently bouncing balls and
(2) robustness to false echoes confined to the image boundary.  All reported
numbers are produced by real training runs and written to machine-readable
files.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from convlstm_bounce import (
    ExperimentConfig,
    assert_disjoint,
    plot_history,
    resolve_device,
    run_experiment,
)


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


BEST = ExperimentConfig(
    group="I",
    name="创新实验",
    layers=2,
    hidden=64,
    kernel=5,
    input_frames=4,
)


def configure_innovation_logging(output_dir: Path) -> logging.Logger:
    """Use a separate log so an open baseline log never blocks the extension."""
    logger = logging.getLogger("convlstm_innovation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output_dir / "innovation_run.log", mode="w", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=4202)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--noise-probability", type=float, default=0.08)
    parser.add_argument("--border-width", type=int, default=3)
    parser.add_argument("--quick-check", action="store_true")
    return parser.parse_args()


def make_multiball_sequences(
    n_seq: int,
    *,
    balls: int = 2,
    t_steps: int = 10,
    size: int = 32,
    radius: int = 2,
    seed: int = 4202,
) -> np.ndarray:
    """Generate unions of independently moving balls with wall reflection."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    sequences = np.zeros((n_seq, t_steps, 1, size, size), dtype=np.float32)
    low, high = float(radius), float(size - radius)
    for sample in range(n_seq):
        positions = rng.uniform(4, size - 5, (balls, 2))
        velocities = rng.choice([-1, 1], (balls, 2)) * rng.uniform(0.8, 1.6, (balls, 2))
        for step in range(t_steps):
            frame = np.zeros((size, size), dtype=np.float32)
            for ball in range(balls):
                positions[ball] += velocities[ball]
                for axis in (0, 1):
                    if positions[ball, axis] < low:
                        positions[ball, axis] = 2 * low - positions[ball, axis]
                        velocities[ball, axis] *= -1
                    elif positions[ball, axis] > high:
                        positions[ball, axis] = 2 * high - positions[ball, axis]
                        velocities[ball, axis] *= -1
                x, y = positions[ball]
                frame = np.maximum(frame, ((xx - x) ** 2 + (yy - y) ** 2 <= radius**2))
            sequences[sample, step, 0] = frame
    return sequences


def add_boundary_false_echoes(
    clean: np.ndarray,
    *,
    input_frames: int,
    probability: float,
    border_width: int,
    seed: int,
) -> np.ndarray:
    """Add reproducible binary clutter only to observed border pixels.

    The prediction target and all future frames remain clean, so the task tests
    denoising-aware motion prediction rather than reproducing the corruption.
    """
    if not (0.0 <= probability <= 1.0):
        raise ValueError("probability must lie in [0, 1]")
    noisy = clean.copy()
    height, width = clean.shape[-2:]
    border = np.zeros((height, width), dtype=bool)
    border[:border_width] = True
    border[-border_width:] = True
    border[:, :border_width] = True
    border[:, -border_width:] = True
    rng = np.random.default_rng(seed)
    echoes = rng.random((len(clean), input_frames, height, width)) < probability
    echoes &= border[None, None]
    observed = noisy[:, :input_frames, 0]
    observed[echoes] = 1.0
    return noisy


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_noise_comparison(
    histories: dict[str, list[dict[str, object]]], output_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    labels = {"I2a": "无增强训练", "I2b": "边界噪声增强训练"}
    for group, history in histories.items():
        ax.plot(
            [int(row["epoch"]) for row in history],
            [float(row["test_mse"]) for row in history],
            marker="o",
            linewidth=2,
            label=labels[group],
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("含噪测试集 MSE")
    ax.set_title("创新实验 2：边界噪声鲁棒性对比")
    ax.set_xticks(range(1, len(next(iter(histories.values()))) + 1))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_examples(
    multiball: torch.Tensor,
    multiball_pred: torch.Tensor,
    noisy_test: torch.Tensor,
    clean_pred: torch.Tensor,
    augmented_pred: torch.Tensor,
    output_path: Path,
) -> dict[str, float]:
    sample = 0
    target_multi = multiball[sample, 4, 0].numpy()
    pred_multi = multiball_pred[sample, 0].numpy()
    target_noise = noisy_test[sample, 4, 0].numpy()
    pred_clean = clean_pred[sample, 0].numpy()
    pred_aug = augmented_pred[sample, 0].numpy()
    metrics = {
        "multiball_sample_mse": float(np.mean((pred_multi - target_multi) ** 2)),
        "noise_clean_train_sample_mse": float(np.mean((pred_clean - target_noise) ** 2)),
        "noise_augmented_train_sample_mse": float(np.mean((pred_aug - target_noise) ** 2)),
    }

    fig, axes = plt.subplots(2, 4, figsize=(9.0, 4.7))
    multi_images = [multiball[sample, 3, 0].numpy(), target_multi, pred_multi]
    multi_titles = ["双球输入帧 4", "双球真实帧", f"双球预测\nMSE={metrics['multiball_sample_mse']:.5f}"]
    for col, (image, title) in enumerate(zip(multi_images, multi_titles)):
        axes[0, col].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(title, fontsize=10)
    axes[0, 3].axis("off")

    noise_images = [noisy_test[sample, 3, 0].numpy(), target_noise, pred_clean, pred_aug]
    noise_titles = [
        "含边界噪声输入",
        "干净真实帧",
        f"无增强预测\nMSE={metrics['noise_clean_train_sample_mse']:.5f}",
        f"增强训练预测\nMSE={metrics['noise_augmented_train_sample_mse']:.5f}",
    ]
    for col, (image, title) in enumerate(zip(noise_images, noise_titles)):
        axes[1, col].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title(title, fontsize=10)
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    axes[0, 0].set_ylabel("创新 1", fontsize=11)
    axes[1, 0].set_ylabel("创新 2", fontsize=11)
    fig.suptitle("创新场景预测示例", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return metrics


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_innovation_logging(output_dir)
    device = resolve_device(args.device)
    train_size = 96 if args.quick_check else args.train_size
    test_size = 24 if args.quick_check else args.test_size
    epochs = 1 if args.quick_check else args.epochs
    batch_size = min(16, train_size) if args.quick_check else args.batch_size
    config = replace(BEST, epochs=epochs, batch_size=batch_size)
    logger.info(
        "innovation device=%s train=%d test=%d epochs=%d noise_p=%.3f border=%d",
        device, train_size, test_size, epochs, args.noise_probability, args.border_width,
    )

    # Use separate data seeds for innovations; each train/test split is independent.
    multi_np = make_multiball_sequences(train_size + test_size, seed=args.data_seed)
    multi_train_np, multi_test_np = multi_np[:train_size], multi_np[train_size:]
    assert_disjoint(multi_train_np, multi_test_np)
    multi_train = torch.from_numpy(multi_train_np)
    multi_test = torch.from_numpy(multi_test_np)

    clean_np = make_multiball_sequences(train_size + test_size, balls=1, seed=args.data_seed + 1)
    clean_train_np, clean_test_np = clean_np[:train_size], clean_np[train_size:]
    assert_disjoint(clean_train_np, clean_test_np)
    noisy_train_np = add_boundary_false_echoes(
        clean_train_np,
        input_frames=4,
        probability=args.noise_probability,
        border_width=args.border_width,
        seed=args.data_seed + 2,
    )
    noisy_test_np = add_boundary_false_echoes(
        clean_test_np,
        input_frames=4,
        probability=args.noise_probability,
        border_width=args.border_width,
        seed=args.data_seed + 3,
    )
    # Targets must be byte-identical to the clean data.
    if not np.array_equal(noisy_train_np[:, 4:], clean_train_np[:, 4:]):
        raise AssertionError("training targets were corrupted")
    if not np.array_equal(noisy_test_np[:, 4:], clean_test_np[:, 4:]):
        raise AssertionError("test targets were corrupted")
    clean_train = torch.from_numpy(clean_train_np)
    noisy_train = torch.from_numpy(noisy_train_np)
    noisy_test = torch.from_numpy(noisy_test_np)

    rows: list[dict[str, object]] = []
    histories: list[dict[str, object]] = []
    by_group: dict[str, list[dict[str, object]]] = {}

    multi_config = replace(config, group="I1", name="创新1：双球预测")
    result_multi, history_multi, pred_multi = run_experiment(
        multi_config, args.seed, multi_train, multi_test, device, logger
    )
    rows.append({**result_multi, "training_condition": "双球训练", "test_condition": "双球测试"})
    histories.extend(history_multi)
    plot_history(history_multi, output_dir / "result_innovation_multiball.png", "创新实验 1：双球预测")

    clean_config = replace(config, group="I2a", name="创新2a：无增强训练")
    result_clean, history_clean, pred_clean = run_experiment(
        clean_config, args.seed, clean_train, noisy_test, device, logger
    )
    rows.append({**result_clean, "training_condition": "干净单球训练", "test_condition": "边界噪声单球测试"})
    histories.extend(history_clean)
    by_group["I2a"] = history_clean

    augmented_config = replace(config, group="I2b", name="创新2b：噪声增强训练")
    result_aug, history_aug, pred_aug = run_experiment(
        augmented_config, args.seed, noisy_train, noisy_test, device, logger
    )
    rows.append({**result_aug, "training_condition": "边界噪声增强训练", "test_condition": "边界噪声单球测试"})
    histories.extend(history_aug)
    by_group["I2b"] = history_aug
    plot_noise_comparison(by_group, output_dir / "result_innovation_noise.png")

    sample_metrics = plot_examples(
        multi_test, pred_multi, noisy_test, pred_clean, pred_aug,
        output_dir / "result_innovation_examples.png",
    )
    write_rows(output_dir / "innovation_results.csv", rows)
    write_rows(output_dir / "innovation_histories.csv", histories)

    improvement = 100.0 * (
        float(result_clean["test_mse"]) - float(result_aug["test_mse"])
    ) / float(result_clean["test_mse"])
    metadata = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "device": str(device),
        "train_size": train_size,
        "test_size": test_size,
        "model": {"layers": 2, "hidden": 64, "kernel": 5, "input_frames": 4},
        "double_ball": {"balls": 2, "train_test_overlap": 0},
        "boundary_noise": {
            "type": "binary false echoes on observed border pixels only",
            "probability": args.noise_probability,
            "border_width": args.border_width,
            "targets_clean": True,
            "mse_improvement_percent_with_augmentation": improvement,
        },
        "sample_metrics": sample_metrics,
    }
    (output_dir / "innovation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("innovation artifacts written; noise robustness MSE improvement=%.2f%%", improvement)


if __name__ == "__main__":
    main()
