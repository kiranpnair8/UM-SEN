"""Plot diagnostic epoch means and fixed-state response matrices."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

METRICS = {"G_l": "Gradient transmission G", "directional_consistency": "Directional consistency",
           "parameter_grad_norm": "Parameter-gradient L2 norm", "firing_rate": "Firing rate"}


def save(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_response_matrix(data, path):
    path = Path(path)
    values = np.array([[np.nan if value is None else value for value in row] for row in data["matrix"]])
    sns.set_theme(context="paper", style="white")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(values, mask=~np.isfinite(values), cmap="vlag", center=0, annot=True, fmt=".3g",
                xticklabels=data["columns_perturbed_blocks"], yticklabels=data["rows_observed_blocks"], ax=ax,
                cbar_kws={"label": "Delta G / Delta alpha"})
    ax.set(xlabel="Perturbed block j (zero-based)", ylabel="Observed block i (zero-based)")
    save(fig, path)


def plot_epochs(payload, output):
    rows = payload["epochs"]
    if not rows:
        raise ValueError("No epoch records to plot")
    epochs = sorted({row["epoch"] for row in rows})
    blocks = sorted({row["block_id"] for row in rows})
    lookup = {(row["block_id"], row["epoch"]): row for row in rows}
    sns.set_theme(context="paper", style="whitegrid")
    for metric, label in METRICS.items():
        values = np.array([[lookup.get((block, epoch), {}).get(metric) for epoch in epochs] for block in blocks], dtype=float)
        fig, ax = plt.subplots(figsize=(max(5, len(epochs) * .22), max(3, len(blocks) * .6)))
        options = dict(cmap="vlag", center=0) if metric in ("G_l", "directional_consistency") else dict(cmap="viridis")
        sns.heatmap(values, mask=~np.isfinite(values), xticklabels=epochs, yticklabels=blocks,
                    ax=ax, cbar_kws={"label": label}, **options)
        ax.set(xlabel="Epoch", ylabel="Transformer block (zero-based)")
        save(fig, output / (metric + "_heatmap.png"))
        fig, ax = plt.subplots(figsize=(6, 4))
        for block, series in zip(blocks, values):
            ax.plot(epochs, series, marker="o", markersize=3, label=f"Block {block}")
        ax.set(xlabel="Epoch", ylabel=label)
        ax.legend(frameon=False)
        ax.grid(alpha=.3)
        save(fig, output / (metric + "_curves.png"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, type=Path, help="epochs.json or response_matrix.json")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    from experiments.gradient_flow.common import output_directory
    output = output_directory(args.output_dir, "figures")
    data = json.loads(args.metrics.read_text(encoding="utf-8"))
    if "matrix" in data:
        plot_response_matrix(data, output / "response_matrix.png")
    else:
        plot_epochs(data, output)
    print(output)


if __name__ == "__main__":
    main()
