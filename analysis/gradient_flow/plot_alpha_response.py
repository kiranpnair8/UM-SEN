"""Descriptive plots of saved causal probes; no training or scientific conclusions."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_matrix(matrix, path, label="Central finite difference of G w.r.t. alpha"):
    values = np.array(matrix["matrix"], dtype=float)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(np.ma.masked_invalid(values), aspect="equal")
    ax.set(xticks=range(len(matrix["columns_perturbed_blocks"])),
           xticklabels=matrix["columns_perturbed_blocks"], yticks=range(len(matrix["rows_observed_blocks"])),
           yticklabels=matrix["rows_observed_blocks"], xlabel="Perturbed block j", ylabel="Measured block i")
    fig.colorbar(im, ax=ax, label=label)
    for (i, j), value in np.ndenumerate(values):
        ax.text(j, i, f"{value:.3g}", ha="center", va="center", fontsize=9)
    save(fig, path)


def plot(measurements, output):
    from experiments.gradient_flow.intervene import local_response_matrix
    blocks = [row["block_id"] for row in measurements["baseline"]["blocks"]]
    summary = ["Descriptive fixed-state alpha responses (not optimality or efficacy claims)."]
    for perturbed in measurements["block_ids"]:
        trials = sorted((t for t in measurements["trials"] if t["target_block"] == perturbed),
                        key=lambda t: t["alpha"])
        fig, ax = plt.subplots(figsize=(6, 4))
        for block in blocks:
            series = [(t["alpha"], t["blocks"][block]["G_l"]) for t in trials]
            ax.plot([a for a, _ in series], [np.nan if g is None else g for _, g in series],
                    marker="o", label=f"Measured block {block}")
            near = [(a, g) for a, g in series if abs(a - measurements["reference_alpha"]) <= 1]
            values = [g for _, g in near]
            if len(set(a for a, _ in near)) < 3 or any(g is None for g in values):
                status = "insufficient finite samples"
            else:
                differences = np.diff(values)
                status = ("nondecreasing" if np.all(differences >= 0) else
                          "nonincreasing" if np.all(differences <= 0) else "non-monotonic")
            summary.append(f"Perturbed {perturbed}, measured {block}: sampled G near reference +/-1: {status}")
        ax.set(xlabel="Perturbed surrogate alpha", ylabel="Gradient transmission G",
               title=f"Perturbed block {perturbed}")
        ax.grid(alpha=.3)
        ax.legend(frameon=False)
        save(fig, output / f"alpha_response_block{perturbed}")
    try:
        matrix = local_response_matrix(measurements)
    except ValueError as error:
        summary.append(str(error))
    else:
        values = np.array(matrix["matrix"], dtype=float)
        plot_matrix(matrix, output / "local_response_matrix")
        diagonal, off = [], []
        for i, row in enumerate(blocks):
            for j, col in enumerate(matrix["columns_perturbed_blocks"]):
                if np.isfinite(values[i, j]):
                    (diagonal if row == col else off).append((row, col, values[i, j]))
        summary.append(f"Central pair: alpha={matrix['lower_alpha']}, {matrix['upper_alpha']}; step={matrix['step_size']}")
        summary.append(f"Diagonal responses (i,j,value): {diagonal}")
        summary.append(f"Strongest absolute off-diagonal response: {max(off, key=lambda v: abs(v[2])) if off else 'unavailable'}")
        denominator = np.mean([abs(v[2]) for v in off]) if off else 0
        ratio = float(np.mean([abs(v[2]) for v in diagonal]) / denominator) if diagonal and denominator > 0 else None
        summary.append(f"Mean absolute diagonal / mean absolute off-diagonal: {ratio if ratio is not None else 'undefined (missing/zero denominator)'}")
    (output / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", required=True, type=Path)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    from experiments.gradient_flow.common import output_directory
    output = output_directory(args.output_dir, "figures")
    plot(json.loads(args.measurements.read_text(encoding="utf-8")), output)
    print(output)


if __name__ == "__main__":
    main()
