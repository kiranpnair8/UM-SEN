#!/usr/bin/env python3
# Historical CogMI-era script; not the SAGE-v2 / IJCNN 2027 pipeline.
# See archive/cogmi2026/manifests/README.md from the repository root.
"""Generate five publication-quality bar plots from surrogate comparison table."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATASETS = ["CIFAR-10", "CIFAR-100", "CIFAR10-DVS"]
METHODS = ["Fixed", "Learnable", "SAGE"]
T_VALUES = [1, 2, 4, 8, 12]

RESULTS = {
    1: {
        "Fixed": [92.9, 74.2, 60.0],
        "Learnable": [93.2, 74.7, 60.8],
        "SAGE": [93.7, 75.3, 61.8],
    },
    2: {
        "Fixed": [94.0, 76.3, 68.5],
        "Learnable": [94.4, 76.8, 69.2],
        "SAGE": [94.4, 77.3, 70.3],
    },
    4: {
        "Fixed": [94.89, 77.9, 74.5],
        "Learnable": [95.2, 78.3, 75.1],
        "SAGE": [95.8, 79.0, 76.2],
    },
    8: {
        "Fixed": [95.1, 78.4, 78.3],
        "Learnable": [95.1, 78.0, 78.8],
        "SAGE": [94.8, 78.6, 79.7],
    },
    12: {
        "Fixed": [95.2, 78.6, 79.6],
        "Learnable": [94.0, 79.0, 79.9],
        "SAGE": [95.1, 79.1, 80.33],
    },
}

COLORS = {
    "Fixed": "#C7E9C0",
    "Learnable": "#74C476",
    "SAGE": "#238B45",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot surrogate comparison bars by T.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/surrogate_table_bars"),
        help="directory where PNG files will be saved",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.linewidth": 1.0,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_for_t(t_value: int, output_dir: Path, dpi: int) -> Path:
    x = np.arange(len(DATASETS))
    width = 0.24

    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    for idx, method in enumerate(METHODS):
        offset = (idx - 1) * width
        values = RESULTS[t_value][method]
        ax.bar(
            x + offset,
            values,
            width,
            label=method,
            color=COLORS[method],
            edgecolor="black",
            linewidth=0.6,
        )

    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.set_ylim(54, 99)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.45)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"surrogate_comparison_T{t_value}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    paths = [plot_for_t(t_value, args.output_dir, args.dpi) for t_value in T_VALUES]
    for path in paths:
        print(f"saved {path}")


if __name__ == "__main__":
    main()
