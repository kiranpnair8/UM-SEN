"""Descriptive training-state figures from completed SAGE probe outputs only."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .plot_alpha_response import save


def heatmap(matrix, title, path, low, high):
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, vmin=low, vmax=high)
    ax.set(xticks=range(4), yticks=range(4), xlabel="Perturbed block j", ylabel="Measured block i", title=title)
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{value:.3g}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, label="Central finite-difference response")
    save(fig, path)


def plot(data, metadata, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    states = data["states"]
    epochs = [data["epochs"][s] for s in states]
    values = np.asarray([data["mean_matrices"][s] for s in states])
    low, high = values.min(), values.max()
    for s, epoch, matrix in zip(states, epochs, values):
        heatmap(matrix, f"{s}, epoch {epoch}", output / ("response_"+s), low, high)
    changes = {key: np.asarray(d["matrix"]["mean"]) for key, d in data["differences"].items()}
    low, high = min(m.min() for m in changes.values()), max(m.max() for m in changes.values())
    for key, matrix in changes.items():
        heatmap(matrix, key.replace("_", " "), output / key, low, high)
    def curves(series, labels, ylabel, name):
        fig, ax = plt.subplots(figsize=(6, 4))
        for y, label in zip(series, labels):
            ax.plot(epochs, y, marker="o", label=label)
        ax.set(xticks=epochs, xticklabels=[f"{s}\n{e}" for s, e in zip(states, epochs)],
               xlabel="Training state / completed zero-based epoch", ylabel=ylabel)
        ax.grid(alpha=.3)
        ax.legend(frameon=False)
        save(fig, output / name)
    curves(np.diagonal(values, axis1=1, axis2=2).T, [f"Block {i}" for i in range(4)],
           "Mean diagonal response", "diagonal_trajectories")
    curves(np.diagonal(values, offset=1, axis1=1, axis2=2).T,
           [f"R({i},{i+1})" for i in range(3)], "Mean signed nearest-upstream response", "nearest_upstream")
    keys = ("ratio_all_off_diagonal", "ratio_nonzero_off_diagonal")
    series = [[data["per_state"][s]["aggregate"]["structural_metrics"]["aggregate"][k].get("mean", np.nan) for s in states] for k in keys]
    curves(series, ["All off-diagonal", "Nonzero off-diagonal only"], "Mean per-batch diagonal/off-diagonal ratio", "ratios")
    curves(np.asarray([data["G_reference"][s]["mean"] for s in states]).T,
           [f"Block {i}" for i in range(4)], "Mean G at alpha=4", "G_reference")
    for block in range(4):
        fig, ax = plt.subplots(figsize=(6, 4))
        for s in states:
            aggregate = data["per_state"][s]["aggregate"]
            x = np.asarray(aggregate["alpha_values"])
            mean = np.asarray(aggregate["alpha_curves"]["mean"])[block, block]
            ax.plot(x, mean, marker="o", label=f"{s}, epoch {data['epochs'][s]}")
            if aggregate["alpha_curves"]["std"] is not None:
                sd = np.asarray(aggregate["alpha_curves"]["std"])[block, block]
                ax.fill_between(x, mean-sd, mean+sd, alpha=.15)
        ax.set(xlabel=f"Alpha in block {block}", ylabel="Own G response (mean +/- sample SD)")
        ax.grid(alpha=.3)
        ax.legend(frameon=False)
        save(fig, output / f"own_alpha_block{block}")
    fig, ax = plt.subplots(figsize=(10, 4))
    groups, labels = [], []
    for block in range(4):
        for s in states:
            groups.append(np.asarray(data["per_state"][s]["aggregate"]["matrices"])[:, block, block])
            labels.append(f"B{block}\n{s}\ne{data['epochs'][s]}")
    ax.boxplot(groups)
    ax.set(xticks=range(1, len(labels)+1), xticklabels=labels, ylabel="Diagonal response per evaluation batch")
    ax.grid(axis="y", alpha=.3)
    save(fig, output / "diagonal_distributions")
    lines = ["SAGE training-state alpha-response characterization; descriptive only.",
             f"Paired batches: {data['batch_count']}", f"Intervals: {data['bootstrap']}"]
    for s in states:
        checkpoint = metadata["manifest"]["states"][s]
        a = data["per_state"][s]["aggregate"]
        structure = a["structural_metrics"]
        lines += [f"{s}: epoch={checkpoint['epoch']}, path={checkpoint['path']}, SHA256={checkpoint['sha256']}",
                  f"Mean R:\n{np.array2string(np.asarray(data['mean_matrices'][s]), precision=6)}",
                  f"Frobenius norm of mean R: {data['frobenius_norms'][s]}",
                  f"Per-block strictly positive fractions: {structure['per_block_fraction_diagonal_positive']}",
                  f"Sign fractions: {a['response_sign_consistency']}",
                  f"Signs of mean responses (effective-zero tolerance applied): {data['mean_response_signs'][s]}",
                  f"All-downstream-effectively-zero fraction: {structure['aggregate']['every_downstream_effectively_zero'].get('mean')}",
                  f"Ratios: { {k: structure['aggregate'][k] for k in keys} }"]
    lines += [f"Diagonal mean trajectories (state x block): {data['diagonal_trajectories']}",
              f"Nonmonotonic sampled mean diagonal trajectories by block: {data['diagonal_mean_trajectory_nonmonotonic']}"]
    for name, diff in data["differences"].items():
        lines += [f"{name}: mean difference={diff['matrix']['mean']}",
                  f"Frobenius norm of mean difference: {diff['frobenius_norm_of_mean_difference']}",
                  f"Normalized mean change (unsafe baseline -> null): {diff['normalized_mean_matrix_change']}",
                  f"Paired diagonal intervals: {diff['paired_diagonal_change']}",
                  f"Paired structural intervals: {diff['paired_structural_change']}",
                  f"Paired G(alpha4) change: {diff['paired_G_reference_change']}",
                  f"Paired sign-unchanged fractions: {diff['paired_sign_unchanged_fraction']}",
                  f"Similarity: {diff['upper_triangle_similarity']}"]
    (output / "summary.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    from experiments.gradient_flow.common import ROOT
    if json.loads((args.run_dir / "status.json").read_text())["status"] != "complete":
        parser.error("All three states must complete before aggregate analysis")
    namespace = ROOT / "figures/gradient_flow"
    output = (ROOT / args.output_dir).resolve()
    if namespace.resolve() != namespace or not output.is_relative_to(namespace) or output == namespace:
        parser.error("State figures must be in figures/gradient_flow/<study>")
    plot(json.loads((args.run_dir / "aggregate/summary.json").read_text()),
         json.loads((args.run_dir / "metadata.json").read_text()), output)
    print(output)


if __name__ == "__main__":
    main()
