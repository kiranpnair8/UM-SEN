"""Saved-data-only online feasibility figures and predeclared summaries."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .online_summary import feasibility_gate, summarize_condition


ROOT = Path(__file__).resolve().parents[2]
METHODS = ("fixed", "response_aware", "random_matched")


def plot(run_dir, output_dir):
    run_dir, output_dir = Path(run_dir).resolve(), Path(output_dir).resolve()
    if output_dir == run_dir or output_dir.is_relative_to(run_dir) or run_dir.is_relative_to(output_dir):
        raise ValueError("Figures must not overwrite the original run tree")
    rows, identities, statuses, settings, hashes = {}, {}, {}, {}, {}
    for method in METHODS:
        directory = run_dir/method
        identities[method] = json.loads((directory/"provenance.json").read_text())["identity"]
        marker = json.loads((directory/"status.json").read_text())
        statuses[method] = marker["status"]
        if marker.get("identity") != identities[method] or statuses[method] != "complete":
            raise ValueError("Only matching complete conditions may be compared")
        rows[method] = json.loads((directory/"metrics.json").read_text())
        if marker["epochs"] != len(rows[method]) or [r["epoch"] for r in rows[method]] != list(range(len(rows[method]))):
            raise ValueError("Incomplete or reordered epoch metrics")
        hashes[method] = hashlib.sha256((directory/"metrics.json").read_bytes()).hexdigest()
        settings[method] = json.loads((directory/"config.json").read_text())["online"]
    base = {**identities["fixed"], "method": None}
    if any({**identity, "method": None} != base for identity in identities.values()):
        raise ValueError("Conditions do not share exact training/commit/batch provenance")
    if any(s != settings["fixed"] for s in settings.values()):
        raise ValueError("Controller settings differ across conditions")
    if len({len(v) for v in rows.values()}) != 1:
        raise ValueError("Condition epoch counts differ")
    summaries = {m: summarize_condition(r, settings[m]) for m, r in rows.items()}
    summary = dict(conditions=summaries, source_identity=base, metrics_sha256=hashes,
        gate=feasibility_gate(summaries, statuses, settings["fixed"], synthetic=base["synthetic"]),
        caveat="Descriptive, seed42 only. Smoke metrics are synthetic and cannot pass the real feasibility gate.")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    (output_dir/"summary.txt").write_text(summary["caveat"]+"\n"+json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .25, "savefig.dpi": 300})

    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(output_dir/(name+"."+ext), dpi=300, bbox_inches="tight")
        plt.close(fig)

    for field, label in (("loss", "Train loss"), ("top1", "Validation accuracy (%)")):
        fig, ax = plt.subplots(figsize=(6, 3.8))
        for method, data in rows.items():
            ax.plot([r["epoch"] for r in data], [r["train" if field == "loss" else "validation"][field] for r in data], label=method)
        ax.set(xlabel="Epoch (zero-based)", ylabel=label)
        ax.legend()
        save(fig, "train_loss" if field == "loss" else "validation_accuracy")
    data = rows["response_aware"]
    epochs = [r["epoch"] for r in data]
    blocks = len(data[0]["controller"]["alpha"])
    for field, label in (("alpha", "Surrogate alpha"), ("G", "FP32 training-probe G"),
                         ("G_ref", "Windowed reference G"), ("Rhat", "Local response estimate"),
                         ("applied_step", "Applied alpha step"), ("error", "G minus reference")):
        fig, ax = plt.subplots(figsize=(6, 3.8))
        for b in range(blocks):
            ax.plot(epochs, [r["controller"][field][b] if r["controller"][field] is not None else None for r in data], label=f"Block {b}")
        ax.set(xlabel="Epoch (zero-based)", ylabel=label)
        ax.legend()
        save(fig, "response_aware_"+field)
    for field, label in (("parameter_grad_norm", "Unscaled pre-clip gradient L2"),
                         ("G_l", "Training-batch mean G"), ("firing_rate", "Firing rate")):
        fig, axes = plt.subplots(1, blocks, figsize=(max(7, 3*blocks), 3.5), squeeze=False)
        for b, ax in enumerate(axes[0]):
            for method, records in rows.items():
                ax.plot(epochs, [r["gradient"][b][field] for r in records], label=method)
            ax.set(xlabel="Epoch", ylabel=label, title=f"Block {b}")
        axes[0][-1].legend()
        save(fig, field+"_comparison")
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for method, records in rows.items():
        ax.plot(epochs, [100*r["controller_seconds"]/r["epoch_seconds"] for r in records], label=method)
    ax.set(xlabel="Epoch", ylabel="Controller / epoch wall time (%)")
    ax.legend()
    save(fig, "controller_overhead")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    matrix = [[r["controller"]["response_sign"][b] if r["controller"]["response_sign"] is not None else float("nan") for r in data] for b in range(blocks)]
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", vmin=-1, vmax=1,
                      extent=[-.5, len(epochs)-.5, blocks-.5, -.5])
    ax.set(xlabel="Epoch (zero-based)", ylabel="Block", yticks=range(blocks))
    fig.colorbar(image, ax=ax, ticks=[-1, 0, 1], label="Measured response sign (tiny=0)")
    save(fig, "response_sign")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for method, records in rows.items():
        axes[0].plot(epochs, [sum(r["controller"]["bound_hit"])/blocks for r in records], label=method)
        axes[1].plot(epochs, [int(r["controller"]["oscillation_indicator"]) for r in records], label=method)
    axes[0].set(xlabel="Epoch", ylabel="Fraction of blocks at bounds")
    axes[1].set(xlabel="Epoch", ylabel="Within-window crossing indicator")
    axes[1].legend()
    save(fig, "bounds_and_crossings")
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    for method, records in rows.items():
        axes[0, 0].plot(epochs, [r["train"]["loss"] for r in records], label=method)
        axes[0, 1].plot(epochs, [r["validation"]["top1"] for r in records], label=method)
    for b in range(blocks):
        for ax, field in ((axes[1, 0], "alpha"), (axes[1, 1], "applied_step"), (axes[2, 0], "G"), (axes[2, 1], "Rhat")):
            ax.plot(epochs, [r["controller"][field][b] if r["controller"][field] is not None else None for r in data], label=f"Block {b}")
    for ax, label in zip(axes.flat, ("Train loss", "Validation top1 (%)", "Response-aware alpha", "Alpha step", "Probe G", "Rhat")):
        ax.set(xlabel="Epoch", ylabel=label)
        ax.legend(fontsize=8)
    save(fig, "compact_summary")
    # Publish aggregate summaries only after all conditions have passed identity checks.
    for name in ("summary.json", "summary.txt"):
        (run_dir/name).write_bytes((output_dir/name).read_bytes())
    print("Descriptive online gate:", summary["gate"])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="results/gradient_flow/controller_online_seed42")
    parser.add_argument("--output-dir", default="figures/gradient_flow/controller_online_seed42")
    args = parser.parse_args()
    run_dir, output = (ROOT/args.run_dir).resolve(), (ROOT/args.output_dir).resolve()
    if not run_dir.is_relative_to(ROOT/"results/gradient_flow") or not output.is_relative_to(ROOT/"figures/gradient_flow"):
        parser.error("Use dedicated gradient_flow result/figure namespaces")
    plot(run_dir, output)


if __name__ == "__main__":
    main()
