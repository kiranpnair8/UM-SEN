#!/usr/bin/env python3
"""Plot surrogate-comparison accuracy versus simulation time."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


DATASETS = ("cifar10", "cifar100", "imagenet200")
METHODS = ("fixed", "learnable", "adaptive", "umsen")
METHOD_LABELS = {
    "fixed": "Fixed",
    "learnable": "Learnable",
    "adaptive": "Adaptive",
    "umsen": "UM-SEN",
}
COLORS = {
    "fixed": "#2f6fbb",
    "learnable": "#7c5fb5",
    "adaptive": "#c77d2f",
    "umsen": "#16855b",
}
MARKERS = {
    "fixed": "o",
    "learnable": "s",
    "adaptive": "^",
    "umsen": "D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot surrogate comparison results.")
    parser.add_argument("--results-root", type=Path, default=Path("results/surrogate_comparison"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures/surrogate_comparison"))
    parser.add_argument("--metric", choices=("best_accuracy", "final_accuracy"), default="best_accuracy")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_point(results_root: Path, dataset: str, method: str, t_value: int) -> float | None:
    seed_dirs = sorted((results_root / dataset / method / f"T{t_value}").glob("seed*"))
    values = []
    for seed_dir in seed_dirs:
        metrics_file = seed_dir / "metrics.json"
        if not metrics_file.exists():
            continue
        try:
            metrics = json.loads(metrics_file.read_text())
        except json.JSONDecodeError:
            print(f"warning: could not parse {metrics_file}")
            continue
        if metrics.get("status") != "completed":
            reason = metrics.get("reason") or metrics.get("warning") or metrics.get("status")
            print(f"warning: skipping {metrics_file}: {reason}")
            continue
        value = metrics.get(read_point.metric_name)
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
    if not values:
        return None
    return sum(values) / len(values)


def plot_dataset(ax, results_root: Path, dataset: str, t_values: list[int]) -> None:
    for method in METHODS:
        xs = []
        ys = []
        for t_value in t_values:
            value = read_point(results_root, dataset, method, t_value)
            if value is None:
                continue
            xs.append(t_value)
            ys.append(value)
        if not xs:
            continue
        ax.plot(
            xs,
            ys,
            marker=MARKERS[method],
            linewidth=2.2,
            markersize=6,
            color=COLORS[method],
            label=METHOD_LABELS[method],
        )
    ax.set_title(dataset.upper().replace("IMAGENET200", "ImageNet-200"))
    ax.set_xlabel("Simulation time steps (T)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(t_values)
    ax.grid(True, alpha=0.3, linewidth=0.8)


def main() -> None:
    args = parse_args()
    root = repo_root()
    results_root = (root / args.results_root).resolve() if not args.results_root.is_absolute() else args.results_root
    output_dir = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    read_point.metric_name = args.metric
    t_values = sorted(
        {
            int(path.name[1:])
            for path in results_root.glob("*/*/T*")
            if path.name.startswith("T") and path.name[1:].isdigit()
        }
    )
    if not t_values:
        t_values = [1, 2, 4, 8, 12]

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "figure.dpi": 130,
            "savefig.dpi": 300,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=False)
    for ax, dataset in zip(axes, DATASETS):
        plot_dataset(ax, results_root, dataset, t_values)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False)
    fig.suptitle(f"Surrogate-gradient comparison ({args.metric.replace('_', ' ')})", y=1.04)
    fig.tight_layout()

    png = output_dir / f"accuracy_vs_T_{args.metric}.png"
    pdf = output_dir / f"accuracy_vs_T_{args.metric}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"saved {png}")
    print(f"saved {pdf}")


if __name__ == "__main__":
    main()
