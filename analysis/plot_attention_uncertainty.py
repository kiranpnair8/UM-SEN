#!/usr/bin/env python3
"""Plot the 20-epoch CIFAR-10 attention uncertainty diagnostic results."""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


DEFAULT_METRICS = Path("results/attention_entropy_diagnostic_20ep/metrics.json")
DEFAULT_OUTPUT_DIR = Path("figures/attention_entropy_diagnostic_20ep")
TARGET_TEMPERATURE = "0.25"


def warn(message):
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot CIFAR-10 Spikformer attention uncertainty diagnostic metrics."
    )
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_metrics(path):
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    epochs = payload.get("epochs", [])
    if not epochs:
        raise ValueError(f"No epoch records found in {path}")
    return payload, epochs


def configure_plots():
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.15)
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


def save_figure(fig, output_dir, stem):
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def clean_name(value):
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("=", "")
        .replace(".", "p")
        .replace("-", "m")
    )


def epoch_numbers(epochs):
    values = []
    for idx, record in enumerate(epochs):
        values.append(record.get("epoch", idx))
    return np.asarray(values, dtype=float)


def scalar(record, path, default=np.nan):
    cursor = record
    for key in path:
        if not isinstance(cursor, dict) or key not in cursor:
            warn(f"Missing field: {'.'.join(str(p) for p in path)}")
            return default
        cursor = cursor[key]
    try:
        return float(cursor)
    except (TypeError, ValueError):
        warn(f"Non-numeric field: {'.'.join(str(p) for p in path)} = {cursor!r}")
        return default


def series(epochs, path):
    return np.asarray([scalar(record, path) for record in epochs], dtype=float)


def block_records(record):
    blocks = record.get("attention", [])
    if not isinstance(blocks, list):
        warn("Epoch attention field is missing or is not a list")
        return {}
    return {block.get("block", idx): block for idx, block in enumerate(blocks)}


def infer_blocks(epochs):
    blocks = set()
    for record in epochs:
        blocks.update(block_records(record).keys())
    return sorted(blocks)


def infer_temperatures(epochs):
    temperatures = set()
    for record in epochs:
        for block in block_records(record).values():
            for metric_name in ("temperature_entropy", "attention_sparsity", "gini_impurity"):
                metric = block.get(metric_name, {})
                if isinstance(metric, dict):
                    temperatures.update(metric.keys())
    return sorted(temperatures, key=lambda item: float(item))


def infer_heads(epochs, block_id, temperature):
    heads = set()
    for record in epochs:
        block = block_records(record).get(block_id, {})
        entries = block.get("per_head_entropy", {}).get(temperature, [])
        if isinstance(entries, list):
            for idx, entry in enumerate(entries):
                heads.add(entry.get("head", idx))
    return sorted(heads)


def block_metric_series(epochs, block_id, metric_name, temperature, stat):
    values = []
    for record in epochs:
        block = block_records(record).get(block_id)
        if block is None:
            warn(f"Missing block {block_id} for epoch {record.get('epoch')}")
            values.append(np.nan)
            continue
        metric = block.get(metric_name, {})
        temp_record = metric.get(temperature)
        if not isinstance(temp_record, dict):
            warn(f"Missing {metric_name}[{temperature}] for block {block_id}")
            values.append(np.nan)
            continue
        values.append(scalar(temp_record, [stat]))
    return np.asarray(values, dtype=float)


def per_head_matrix(epochs, block_id, temperature, stat):
    heads = infer_heads(epochs, block_id, temperature)
    matrix = np.full((len(heads), len(epochs)), np.nan, dtype=float)
    head_to_row = {head: row for row, head in enumerate(heads)}
    for col, record in enumerate(epochs):
        block = block_records(record).get(block_id, {})
        entries = block.get("per_head_entropy", {}).get(temperature, [])
        if not isinstance(entries, list):
            warn(f"Missing per-head entropy for block {block_id}, T={temperature}")
            continue
        for idx, entry in enumerate(entries):
            head = entry.get("head", idx)
            if head in head_to_row:
                matrix[head_to_row[head], col] = scalar(entry, [stat])
    return heads, matrix


def finite_pair(x_values, y_values):
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def rankdata(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def pearsonr(x_values, y_values):
    x, y = finite_pair(x_values, y_values)
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearmanr(x_values, y_values):
    x, y = finite_pair(x_values, y_values)
    if len(x) < 2:
        return np.nan
    return pearsonr(rankdata(x), rankdata(y))


def nan_first(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(finite[0]) if finite.size else np.nan


def nan_last(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(finite[-1]) if finite.size else np.nan


def nan_change(values):
    first = nan_first(values)
    last = nan_last(values)
    return last - first if np.isfinite(first) and np.isfinite(last) else np.nan


def nan_mean_by_epoch(series_by_block):
    if not series_by_block:
        return np.asarray([], dtype=float)
    return np.nanmean(np.vstack(series_by_block), axis=0)


def plot_accuracy(epochs, epoch_x, output_dir):
    train_acc = series(epochs, ["train", "accuracy"]) * 100.0
    val_acc = series(epochs, ["validation", "accuracy"]) * 100.0
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(epoch_x, train_acc, marker="o", label="Train")
    ax.plot(epoch_x, val_acc, marker="s", label="Validation")
    if np.isfinite(val_acc).any():
        best_idx = int(np.nanargmax(val_acc))
        ax.scatter(epoch_x[best_idx], val_acc[best_idx], s=80, color="crimson", zorder=5)
        ax.annotate(
            f"best val: epoch {int(epoch_x[best_idx])}, {val_acc[best_idx]:.2f}%",
            xy=(epoch_x[best_idx], val_acc[best_idx]),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=9,
            color="crimson",
        )
    ax.set_title("CIFAR-10 Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.35)
    save_figure(fig, output_dir, "accuracy_vs_epoch")
    return train_acc, val_acc


def plot_loss(epochs, epoch_x, output_dir):
    train_loss = series(epochs, ["train", "loss"])
    val_loss = series(epochs, ["validation", "loss"])
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(epoch_x, train_loss, marker="o", label="Train")
    ax.plot(epoch_x, val_loss, marker="s", label="Validation")
    ax.set_title("CIFAR-10 Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.35)
    save_figure(fig, output_dir, "loss_vs_epoch")
    return train_loss, val_loss


def plot_temperature_metric(epochs, epoch_x, blocks, temperatures, output_dir, metric_name, stat, ylabel, stem_prefix):
    for block_id in blocks:
        fig, ax = plt.subplots(figsize=(6.8, 4.1))
        plotted = False
        for temperature in temperatures:
            values = block_metric_series(epochs, block_id, metric_name, temperature, stat)
            if np.isfinite(values).any():
                ax.plot(epoch_x, values, marker="o", label=f"T={temperature}")
                plotted = True
        if not plotted:
            warn(f"No finite values for {metric_name}.{stat} in block {block_id}")
        ax.set_title(f"Block {block_id}: {ylabel}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=True)
        ax.grid(True, alpha=0.35)
        save_figure(fig, output_dir, f"{stem_prefix}_block_{block_id}")


def plot_per_head_heatmaps(epochs, epoch_x, blocks, output_dir, stat, title_suffix, stem_prefix):
    for block_id in blocks:
        heads, matrix = per_head_matrix(epochs, block_id, TARGET_TEMPERATURE, stat)
        fig_width = max(7.0, 0.38 * len(epoch_x))
        fig_height = max(3.5, 0.32 * max(len(heads), 1))
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        if len(heads) == 0:
            warn(f"No heads found for block {block_id}, T={TARGET_TEMPERATURE}")
            ax.text(0.5, 0.5, "No per-head data", ha="center", va="center", transform=ax.transAxes)
        else:
            sns.heatmap(
                matrix,
                ax=ax,
                cmap="viridis",
                cbar_kws={"label": "Normalized entropy"},
                xticklabels=[int(x) if float(x).is_integer() else x for x in epoch_x],
                yticklabels=heads,
                vmin=0.0 if stat == "mean" else None,
                vmax=1.0 if stat == "mean" else None,
            )
        ax.set_title(f"Block {block_id}: Per-Head Entropy {title_suffix} at T={TARGET_TEMPERATURE}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Attention Head")
        save_figure(fig, output_dir, f"{stem_prefix}_block_{block_id}_T{clean_name(TARGET_TEMPERATURE)}")


def plot_layerwise(epochs, epoch_x, blocks, output_dir, metric_name, ylabel, stem):
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    for block_id in blocks:
        values = block_metric_series(epochs, block_id, metric_name, TARGET_TEMPERATURE, "mean")
        if np.isfinite(values).any():
            ax.plot(epoch_x, values, marker="o", label=f"Block {block_id}")
    ax.set_title(f"Layer-Wise {ylabel} at T={TARGET_TEMPERATURE}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=True, ncol=2)
    ax.grid(True, alpha=0.35)
    save_figure(fig, output_dir, stem)


def aggregate_metric(epochs, blocks, metric_name, temperature, stat):
    block_series = [
        block_metric_series(epochs, block_id, metric_name, temperature, stat)
        for block_id in blocks
    ]
    return nan_mean_by_epoch(block_series)


def plot_scatter_relationships(epochs, blocks, val_acc_percent, output_dir):
    metrics = {
        "mean entropy at T=0.25": aggregate_metric(
            epochs, blocks, "temperature_entropy", TARGET_TEMPERATURE, "mean"
        ),
        "entropy std at T=0.25": aggregate_metric(
            epochs, blocks, "temperature_entropy", TARGET_TEMPERATURE, "std"
        ),
        "attention sparsity at T=0.25": aggregate_metric(
            epochs, blocks, "attention_sparsity", TARGET_TEMPERATURE, "mean"
        ),
    }
    correlations = {}
    for label, values in metrics.items():
        pearson = pearsonr(values, val_acc_percent)
        spearman = spearmanr(values, val_acc_percent)
        correlations[label] = {"pearson": pearson, "spearman": spearman}
        x, y = finite_pair(values, val_acc_percent)
        fig, ax = plt.subplots(figsize=(5.3, 4.3))
        ax.scatter(x, y, s=52, alpha=0.85)
        if len(x) >= 2 and np.std(x) > 0.0:
            coeff = np.polyfit(x, y, deg=1)
            xs = np.linspace(np.min(x), np.max(x), 100)
            ax.plot(xs, coeff[0] * xs + coeff[1], color="crimson", linewidth=1.5)
        ax.text(
            0.04,
            0.96,
            f"Pearson r = {pearson:.3f}\nSpearman rho = {spearman:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85},
        )
        ax.set_title(f"Validation Accuracy vs {label}")
        ax.set_xlabel(label)
        ax.set_ylabel("Validation accuracy (%)")
        ax.grid(True, alpha=0.35)
        save_figure(fig, output_dir, f"scatter_val_accuracy_vs_{clean_name(label)}")
    return metrics, correlations


def largest_head_variation(epochs, blocks):
    candidates = []
    for block_id in blocks:
        heads, matrix = per_head_matrix(epochs, block_id, TARGET_TEMPERATURE, "mean")
        for row, head in enumerate(heads):
            values = matrix[row, :]
            if np.isfinite(values).any():
                candidates.append((float(np.nanstd(values)), block_id, head))
    return sorted(candidates, reverse=True)


def largest_block_variation(epochs, blocks, metric_name, stat):
    candidates = []
    for block_id in blocks:
        values = block_metric_series(epochs, block_id, metric_name, TARGET_TEMPERATURE, stat)
        if np.isfinite(values).any():
            candidates.append((float(np.nanstd(values)), block_id))
    return sorted(candidates, reverse=True)


def meaningful_label(change, values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size < 2 or not np.isfinite(change):
        return "insufficient data"
    span = float(np.nanmax(finite) - np.nanmin(finite))
    baseline = max(abs(nan_first(finite)), 1e-12)
    rel_change = abs(change) / baseline
    if abs(change) >= 0.02 or rel_change >= 0.10 or span >= 0.03:
        return "yes"
    return "small/unclear"


def metric_line(name, values):
    first = nan_first(values)
    last = nan_last(values)
    change = nan_change(values)
    return f"{name}: first={first:.6g}, final={last:.6g}, change={change:+.6g}"


def write_summary(
    output_dir,
    epoch_x,
    train_acc,
    val_acc,
    train_loss,
    val_loss,
    aggregate_series,
    correlations,
    block_variation,
    head_variation,
):
    best_idx = int(np.nanargmax(val_acc)) if np.isfinite(val_acc).any() else None
    lines = []
    lines.append("Attention Uncertainty Diagnostic Summary")
    lines.append("=" * 42)
    lines.append("")
    if best_idx is not None:
        lines.append(
            f"Best validation epoch: {int(epoch_x[best_idx])} "
            f"with validation accuracy {val_acc[best_idx]:.3f}%"
        )
    else:
        lines.append("Best validation epoch: unavailable")
    lines.append("")
    lines.append("Major Metric Changes")
    lines.append(metric_line("Train accuracy (%)", train_acc))
    lines.append(metric_line("Validation accuracy (%)", val_acc))
    lines.append(metric_line("Train loss", train_loss))
    lines.append(metric_line("Validation loss", val_loss))
    for name, values in aggregate_series.items():
        lines.append(metric_line(name, values))
    lines.append("")
    lines.append("Correlations With Validation Accuracy")
    for name, stats in correlations.items():
        lines.append(
            f"{name}: Pearson r={stats['pearson']:.6g}, "
            f"Spearman rho={stats['spearman']:.6g}"
        )
    lines.append("")
    lines.append("Largest Entropy Variation")
    if block_variation:
        variation, block_id = block_variation[0]
        lines.append(f"Block with largest mean-entropy variation: block {block_id} (std over epochs={variation:.6g})")
    else:
        lines.append("Block with largest mean-entropy variation: unavailable")
    if head_variation:
        variation, block_id, head = head_variation[0]
        lines.append(
            f"Head with largest entropy variation: block {block_id}, head {head} "
            f"(std over epochs={variation:.6g})"
        )
    else:
        lines.append("Head with largest entropy variation: unavailable")
    lines.append("")
    lines.append("Meaningful Change Assessment")
    for name, values in aggregate_series.items():
        lines.append(f"{name}: {meaningful_label(nan_change(values), values)}")
    summary_path = output_dir / "summary.txt"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    configure_plots()
    _, epochs = load_metrics(args.metrics)
    output_dir = args.output_dir
    epoch_x = epoch_numbers(epochs)
    blocks = infer_blocks(epochs)
    temperatures = infer_temperatures(epochs)

    if len(epoch_x) != 20:
        warn(f"Expected 20 epochs for this benchmark, found {len(epoch_x)}")
    if list(epoch_x.astype(int)) != list(range(int(epoch_x[0]), int(epoch_x[0]) + len(epoch_x))):
        warn("Epoch values are not a contiguous integer sequence")
    if TARGET_TEMPERATURE not in temperatures:
        warn(f"T={TARGET_TEMPERATURE} not found; available temperatures: {temperatures}")

    train_acc, val_acc = plot_accuracy(epochs, epoch_x, output_dir)
    train_loss, val_loss = plot_loss(epochs, epoch_x, output_dir)

    plot_temperature_metric(
        epochs,
        epoch_x,
        blocks,
        temperatures,
        output_dir,
        "temperature_entropy",
        "mean",
        "Mean normalized entropy",
        "mean_temperature_entropy",
    )
    plot_temperature_metric(
        epochs,
        epoch_x,
        blocks,
        temperatures,
        output_dir,
        "temperature_entropy",
        "std",
        "Entropy standard deviation",
        "entropy_std",
    )
    plot_per_head_heatmaps(
        epochs,
        epoch_x,
        blocks,
        output_dir,
        "mean",
        "Mean",
        "per_head_entropy_mean",
    )
    plot_per_head_heatmaps(
        epochs,
        epoch_x,
        blocks,
        output_dir,
        "std",
        "Std",
        "per_head_entropy_std",
    )
    plot_temperature_metric(
        epochs,
        epoch_x,
        blocks,
        temperatures,
        output_dir,
        "attention_sparsity",
        "mean",
        "Top-k probability mass",
        "attention_sparsity",
    )
    plot_temperature_metric(
        epochs,
        epoch_x,
        blocks,
        temperatures,
        output_dir,
        "gini_impurity",
        "mean",
        "Gini impurity",
        "gini_impurity",
    )
    plot_layerwise(
        epochs,
        epoch_x,
        blocks,
        output_dir,
        "temperature_entropy",
        "mean normalized entropy",
        "layerwise_mean_entropy_T0p25",
    )
    plot_layerwise(
        epochs,
        epoch_x,
        blocks,
        output_dir,
        "attention_sparsity",
        "top-k probability mass",
        "layerwise_attention_sparsity_T0p25",
    )

    scatter_series, correlations = plot_scatter_relationships(epochs, blocks, val_acc, output_dir)
    aggregate_series = {
        "Mean entropy at T=0.25 across blocks": scatter_series["mean entropy at T=0.25"],
        "Entropy std at T=0.25 across blocks": scatter_series["entropy std at T=0.25"],
        "Attention sparsity at T=0.25 across blocks": scatter_series["attention sparsity at T=0.25"],
        "Gini impurity at T=0.25 across blocks": aggregate_metric(
            epochs, blocks, "gini_impurity", TARGET_TEMPERATURE, "mean"
        ),
    }
    write_summary(
        output_dir,
        epoch_x,
        train_acc,
        val_acc,
        train_loss,
        val_loss,
        aggregate_series,
        correlations,
        largest_block_variation(epochs, blocks, "temperature_entropy", "mean"),
        largest_head_variation(epochs, blocks),
    )
    print(f"Wrote figures and summary to {output_dir}")


if __name__ == "__main__":
    main()
