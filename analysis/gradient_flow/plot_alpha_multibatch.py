"""Matplotlib-only descriptive analysis of completed SAGE multibatch probe outputs."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .plot_alpha_response import save


def heatmap(values, blocks, label, path, fraction=False):
    values = np.asarray(values, dtype=float)
    fig, ax = plt.subplots(figsize=(5, 4))
    options = dict(vmin=0, vmax=1) if fraction else {}
    image = ax.imshow(values, **options)
    ax.set(xticks=range(len(blocks)), xticklabels=blocks, yticks=range(len(blocks)),
           yticklabels=blocks, xlabel="Perturbed block j", ylabel="Measured block i")
    fig.colorbar(image, ax=ax, label=label)
    for (i, j), value in np.ndenumerate(values):
        ax.text(j, i, f"{value:.3g}", ha="center", va="center", fontsize=9)
    save(fig, path)


def boxes(groups, labels, ylabel, path):
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), 4))
    ax.boxplot([[v for v in group if v is not None] for group in groups])
    ax.set(xticks=range(1, len(labels) + 1), xticklabels=labels, ylabel=ylabel)
    ax.tick_params(axis="x", labelrotation=15)
    ax.grid(axis="y", alpha=.3)
    save(fig, path)


def plot(data, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    blocks = data["block_ids"]
    stats = data["response_stats"]
    heatmap(stats["mean"], blocks, "Mean central finite-difference response", output / "response_mean")
    if stats["std"] is not None:
        heatmap(stats["std"], blocks, "Sample SD of response", output / "response_std")
    heatmap(data["response_sign_consistency"]["fraction_positive"], blocks,
            "Positive-sign fraction", output / "response_positive_fraction", fraction=True)
    curves = data["alpha_curves"]
    for j, block in enumerate(blocks):
        fig, ax = plt.subplots(figsize=(6, 4))
        order = np.argsort(data["alpha_values"])
        x = np.asarray(data["alpha_values"])[order]
        for i, measured in enumerate(blocks):
            mean = np.asarray(curves["mean"])[j, i, order]
            ax.plot(x, mean, marker="o", label=f"Measured block {measured}")
            if curves["std"] is not None:
                sd = np.asarray(curves["std"])[j, i, order]
                ax.fill_between(x, mean-sd, mean+sd, alpha=.15)
        ax.set(xlabel=f"Alpha in perturbed block {block}", ylabel="G (batch mean +/- sample SD)")
        ax.legend(frameon=False)
        ax.grid(alpha=.3)
        save(fig, output / f"alpha_curves_block{block}")
    matrices = np.asarray(data["matrices"])
    boxes([matrices[:, i, i] for i in range(len(blocks))], [f"Block {b}" for b in blocks],
          "Diagonal central finite-difference response", output / "diagonal_distributions")
    per_batch = data["structural_metrics"]["per_batch"]
    metrics = ("mean_abs_diagonal", "mean_abs_off_diagonal", "mean_abs_nonzero_off_diagonal",
               "mean_abs_nearest_upstream", "mean_abs_distant_upstream", "mean_abs_downstream")
    boxes([[b[key] for b in per_batch] for key in metrics],
          ["Diagonal", "All off-diag.", "Nonzero off-diag.", "Nearest upstream", "Distant upstream", "Downstream"],
          "Mean absolute response per batch", output / "structural_magnitudes")
    boxes([[b[key] for b in per_batch] for key in ("ratio_all_off_diagonal", "ratio_nonzero_off_diagonal")],
          ["All off-diagonal entries", "Effectively-nonzero entries only"],
          "Diagonal / off-diagonal magnitude ratio", output / "structural_ratios")
    structure = data["structural_metrics"]
    lines = ["SAGE fixed-checkpoint multi-batch alpha responses: descriptive statistics only.",
             f"Batches: {data['batch_count']}; reference alpha: {data['reference_alpha']}",
             "Central finite difference uses the nearest symmetric measured pair, not an exact derivative.",
             f"Effective-zero tolerance: {data['zero_tolerance']}", data["tolerance_definition"],
             f"SD: {data['std_definition']}", f"CI: {data['ci_method']}",
             f"Fraction every diagonal strictly positive: {structure['fraction_every_diagonal_positive']}",
             f"Per-block fraction diagonal strictly positive: {structure['per_block_fraction_diagonal_positive']}",
             f"Fraction diagonal mean magnitude > all-off-diagonal mean: {structure['fraction_diagonal_mean_larger_than_off_diagonal']}",
             f"Fraction nearest-upstream mean magnitude > distant-upstream: {structure['fraction_nearest_larger_than_distant']}",
             f"Nearest-upstream sign fractions: {structure['nearest_upstream_sign_consistency']}"]
    for name, record in structure["aggregate"].items():
        lines.append(f"{name}: {record}")
    if curves["local_monotonicity"] is not None:
        near = curves["local_monotonicity"]
        lines.append(f"Sampled local monotonicity at alphas {near['alphas']} (overlap for flat curves):")
        for j in range(len(blocks)):
            for i in range(len(blocks)):
                fractions = {key: value[j][i] for key, value in near.items() if key != "alphas"}
                lines.append(f"perturbed={blocks[j]}, measured={blocks[i]}: {fractions}")
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    from experiments.gradient_flow.common import ROOT
    status = json.loads((args.run_dir / "status.json").read_text(encoding="utf-8"))
    if status["status"] != "complete":
        parser.error("Run is not complete; resume probes before producing aggregate figures")
    namespace = ROOT / "figures/gradient_flow"
    output = (ROOT / args.output_dir).resolve()
    if namespace.resolve() != namespace or not output.is_relative_to(namespace) or output == namespace:
        parser.error("Figures must be under figures/gradient_flow/<run-id>")
    data = json.loads((args.run_dir / "aggregate/summary.json").read_text(encoding="utf-8"))
    if data["batch_count"] != status["validated_batches"]:
        parser.error("Aggregate batch count does not match completion status")
    plot(data, output)
    print(output)


if __name__ == "__main__":
    main()
