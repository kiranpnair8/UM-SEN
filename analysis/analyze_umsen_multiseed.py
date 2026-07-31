#!/usr/bin/env python3
"""Analyze UM-SEN multi-seed validation outputs without rerunning training."""

import argparse
import json
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


DEFAULT_INPUT_DIR = Path("results/umsen_multiseed")
DEFAULT_OUTPUT_DIR = Path("figures/umsen_multiseed")
SEEDS = (42, 43, 44)
CONFIG = "umsen"


def warn(message):
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze UM-SEN multi-seed results.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    return parser.parse_args()


def configure_plots():
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.12)
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def repo_path(path):
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[1] / path


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_results(input_dir, seeds):
    input_dir = repo_path(input_dir)
    metrics_path = input_dir / "metrics.json"
    results = {}
    if metrics_path.exists():
        payload = load_json(metrics_path)
        for seed_record in payload.get("seeds", []):
            seed = int(seed_record.get("seed"))
            config_record = seed_record.get("configs", {}).get(CONFIG)
            if seed in seeds and config_record:
                results[seed] = config_record.get("epochs", [])
    for seed in seeds:
        if seed in results:
            continue
        path = input_dir / f"seed_{seed}" / CONFIG / f"{CONFIG}.json"
        if path.exists():
            results[seed] = load_json(path).get("epochs", [])
        else:
            warn(f"Missing UM-SEN results for seed {seed}: {path}")
    missing = [seed for seed in seeds if seed not in results]
    if missing:
        raise FileNotFoundError(f"Missing UM-SEN result files for seeds: {missing}")
    return results


def save_figure(fig, output_dir, stem):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def epochs(records):
    return np.asarray([record.get("epoch", idx) for idx, record in enumerate(records)], dtype=float)


def scalar_series(records, split, key):
    values = []
    for record in records:
        value = record.get(split, {}).get(key, np.nan)
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            warn(f"Non-numeric value for {split}.{key}: {value!r}")
            values.append(np.nan)
    return np.asarray(values, dtype=float)


def block_ids(results, metric_key):
    ids = set()
    for records in results.values():
        for record in records:
            metric = record.get("train", {}).get(metric_key, {})
            if isinstance(metric, dict):
                ids.update(metric.keys())
    return sorted(ids, key=lambda item: int(item))


def block_series(records, metric_key, block_id):
    values = []
    for record in records:
        metric = record.get("train", {}).get(metric_key, {})
        value = metric.get(str(block_id), metric.get(block_id, np.nan)) if isinstance(metric, dict) else np.nan
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            warn(f"Non-numeric value for train.{metric_key}.{block_id}: {value!r}")
            values.append(np.nan)
    return np.asarray(values, dtype=float)


def subplot_grid(n_items):
    cols = min(2, max(1, n_items))
    rows = int(math.ceil(n_items / cols))
    return rows, cols


def seed_style(seed):
    if seed == 42:
        return {"linewidth": 2.8, "alpha": 1.0, "zorder": 3}
    return {"linewidth": 1.6, "alpha": 0.72, "zorder": 2}


def plot_block_metric(results, output_dir, metric_key, ylabel, title, stem):
    blocks = block_ids(results, metric_key)
    if not blocks:
        warn(f"No block metric found for {metric_key}")
        return
    rows, cols = subplot_grid(len(blocks))
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 3.6 * rows), sharex=True)
    axes = np.atleast_1d(axes).reshape(-1)
    palette = sns.color_palette("deep", n_colors=max(len(results), 3))
    seed_colors = {seed: palette[idx] for idx, seed in enumerate(sorted(results))}
    for ax, block_id in zip(axes, blocks):
        for seed in sorted(results):
            records = results[seed]
            label = f"seed {seed}" + (" (highlight)" if seed == 42 else "")
            ax.plot(
                epochs(records),
                block_series(records, metric_key, block_id),
                marker="o",
                markersize=3.8,
                color=seed_colors[seed],
                label=label,
                **seed_style(seed),
            )
        ax.set_title(f"Block {block_id}")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.35)
    for ax in axes[len(blocks):]:
        ax.axis("off")
    for ax in axes:
        if ax.has_data():
            ax.set_xlabel("Epoch")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 3), frameon=False)
    fig.suptitle(title, y=1.02)
    save_figure(fig, output_dir, stem)


def plot_scalar_metric(results, output_dir, split_key_pairs, ylabel, title, stem, percent=False):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    palette = sns.color_palette("deep", n_colors=max(len(results), 3))
    seed_colors = {seed: palette[idx] for idx, seed in enumerate(sorted(results))}
    for seed in sorted(results):
        records = results[seed]
        for split, key, label_suffix, linestyle in split_key_pairs:
            values = scalar_series(records, split, key)
            if percent:
                values = 100.0 * values
            ax.plot(
                epochs(records),
                values,
                marker="o",
                markersize=3.8,
                linestyle=linestyle,
                color=seed_colors[seed],
                label=f"seed {seed} {label_suffix}",
                **seed_style(seed),
            )
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.35)
    ax.legend(ncol=2, frameon=False)
    save_figure(fig, output_dir, stem)


def plot_validation_metrics(results, output_dir):
    fig, ax_loss = plt.subplots(figsize=(7.4, 4.6))
    ax_acc = ax_loss.twinx()
    palette = sns.color_palette("deep", n_colors=max(len(results), 3))
    seed_colors = {seed: palette[idx] for idx, seed in enumerate(sorted(results))}
    for seed in sorted(results):
        records = results[seed]
        x = epochs(records)
        style = seed_style(seed)
        ax_loss.plot(
            x,
            scalar_series(records, "validation", "loss"),
            marker="o",
            markersize=3.8,
            linestyle="-",
            color=seed_colors[seed],
            label=f"seed {seed} val loss",
            **style,
        )
        ax_acc.plot(
            x,
            100.0 * scalar_series(records, "validation", "accuracy"),
            marker="s",
            markersize=3.5,
            linestyle="--",
            color=seed_colors[seed],
            label=f"seed {seed} val acc",
            **style,
        )
    ax_loss.set_title("UM-SEN validation loss and accuracy versus epoch")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Validation loss")
    ax_acc.set_ylabel("Validation accuracy (%)")
    ax_loss.grid(True, alpha=0.35)
    loss_handles, loss_labels = ax_loss.get_legend_handles_labels()
    acc_handles, acc_labels = ax_acc.get_legend_handles_labels()
    ax_loss.legend(loss_handles + acc_handles, loss_labels + acc_labels, ncol=2, frameon=False)
    save_figure(fig, output_dir, "validation_loss_accuracy")


def first_final_change(results, metric_key, block_id=None, split="train"):
    rows = []
    for seed, records in sorted(results.items()):
        if block_id is None:
            values = scalar_series(records, split, metric_key)
        else:
            values = block_series(records, metric_key, block_id)
        if len(values) == 0:
            continue
        rows.append((seed, float(values[0]), float(values[-1]), float(values[-1] - values[0])))
    return rows


def mean_abs_seed42_gap(results, metric_key, block_id=None, split="train"):
    if 42 not in results:
        return np.nan
    if block_id is None:
        seed42 = scalar_series(results[42], split, metric_key)
    else:
        seed42 = block_series(results[42], metric_key, block_id)
    gaps = []
    for seed, records in results.items():
        if seed == 42:
            continue
        other = scalar_series(records, split, metric_key) if block_id is None else block_series(records, metric_key, block_id)
        n = min(len(seed42), len(other))
        if n:
            gaps.append(np.nanmean(np.abs(seed42[:n] - other[:n])))
    return float(np.nanmean(gaps)) if gaps else np.nan


def write_summary(results, output_dir):
    lines = [
        "UM-SEN multi-seed analysis",
        "",
        "Input configuration: UM-SEN only, seeds 42, 43, 44.",
        "Seed 42 is highlighted with a thicker, fully opaque line in every plot.",
        "",
        "Validation performance:",
    ]
    for seed, records in sorted(results.items()):
        val_acc = scalar_series(records, "validation", "accuracy")
        val_loss = scalar_series(records, "validation", "loss")
        best_idx = int(np.nanargmax(val_acc))
        loss_idx = int(np.nanargmin(val_loss))
        lines.append(
            f"seed {seed}: best_acc={100.0 * val_acc[best_idx]:.2f}% "
            f"at epoch {records[best_idx].get('epoch', best_idx)}, "
            f"final_acc={100.0 * val_acc[-1]:.2f}%, "
            f"best_loss={val_loss[loss_idx]:.4f} "
            f"at epoch {records[loss_idx].get('epoch', loss_idx)}"
        )

    lines.extend(["", "Seed 42 mean absolute gaps versus seeds 43 and 44:"])
    scalar_metrics = [
        ("gradient_norm", "train", "gradient norm"),
        ("spike_rate", "train", "spike rate"),
        ("loss", "validation", "validation loss"),
        ("accuracy", "validation", "validation accuracy"),
    ]
    for key, split, label in scalar_metrics:
        gap = mean_abs_seed42_gap(results, key, split=split)
        if key == "accuracy":
            lines.append(f"{label}: {100.0 * gap:.2f} percentage points")
        else:
            lines.append(f"{label}: {gap:.6f}")

    block_metrics = [
        ("alpha_per_block", "alpha"),
        ("centered_z_per_block", "centered z"),
        ("ema_entropy_dispersion_per_block", "EMA entropy dispersion"),
    ]
    for metric_key, label in block_metrics:
        blocks = block_ids(results, metric_key)
        if not blocks:
            continue
        gaps = [(block, mean_abs_seed42_gap(results, metric_key, block_id=block)) for block in blocks]
        largest = max(gaps, key=lambda item: item[1])
        lines.append(
            f"{label}: largest seed-42 gap is block {largest[0]} "
            f"with mean absolute gap {largest[1]:.6f}"
        )

    lines.extend(["", "First-to-final changes by seed:"])
    for metric_key, label in block_metrics:
        blocks = block_ids(results, metric_key)
        if not blocks:
            continue
        lines.append(f"{label}:")
        for block in blocks:
            changes = first_final_change(results, metric_key, block_id=block)
            change_text = ", ".join(
                f"seed {seed}: {start:.6f}->{final:.6f} ({change:+.6f})"
                for seed, start, final, change in changes
            )
            lines.append(f"  block {block}: {change_text}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    configure_plots()
    results = load_results(args.input_dir, args.seeds)

    plot_block_metric(
        results,
        output_dir,
        "alpha_per_block",
        "Applied alpha",
        "UM-SEN alpha per block versus epoch",
        "alpha_per_block",
    )
    plot_block_metric(
        results,
        output_dir,
        "centered_z_per_block",
        "Centered z",
        "UM-SEN centered z per block versus epoch",
        "centered_z_per_block",
    )
    plot_block_metric(
        results,
        output_dir,
        "ema_entropy_dispersion_per_block",
        "EMA entropy dispersion",
        "UM-SEN entropy dispersion per block versus epoch",
        "entropy_dispersion_per_block",
    )
    plot_scalar_metric(
        results,
        output_dir,
        [("train", "gradient_norm", "gradient norm", "-")],
        "Gradient norm",
        "UM-SEN gradient norm versus epoch",
        "gradient_norm",
    )
    plot_validation_metrics(results, output_dir)
    plot_scalar_metric(
        results,
        output_dir,
        [("train", "spike_rate", "spike rate", "-")],
        "Spike rate",
        "UM-SEN spike rate versus epoch",
        "spike_rate",
    )
    write_summary(results, output_dir)
    print(f"Wrote UM-SEN multi-seed figures and summary to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
