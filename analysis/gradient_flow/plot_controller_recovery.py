"""Saved-output-only descriptive plots for frozen-state recovery; matplotlib only."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load_saved_records(run_dir):
    """Validate the completed JSON collection without touching checkpoints/probes."""
    run_dir = Path(run_dir)
    if json.loads((run_dir / "status.json").read_text()).get("status") != "complete":
        raise ValueError("Recovery must finish before plotting combined results")
    metadata = json.loads((run_dir / "metadata.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    signature = hashlib.sha256(json.dumps(metadata, sort_keys=True, allow_nan=False).encode()).hexdigest()
    cases = []
    for state in ("early", "mid", "late"):
        for index in range(metadata["settings"]["num_batches"]):
            directory = run_dir / state / f"batch_{index:03d}"
            marker = json.loads((directory / "complete.json").read_text())
            path = directory / "recovery.json"
            if (marker.get("status") != "complete" or marker.get("signature") != signature or
                    marker.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest()):
                raise ValueError("Recovery figures require validated, matching raw batch outputs")
            cases.extend(json.loads(path.read_text()))
    if not cases:
        raise ValueError("No completed recovery cases")
    return metadata, summary, cases


def plot(run_dir, output_dir, summary_path=None):
    run_dir, output_dir = Path(run_dir), Path(output_dir)
    metadata, summary, cases = load_saved_records(run_dir)
    if summary_path is not None:
        summary = json.loads(Path(summary_path).read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .25, "savefig.dpi": 300})
    states = list(summary["groups"])
    disturbances = list(summary["groups"][states[0]])
    methods = list(summary["groups"][states[0]][disturbances[0]])
    def save(fig, name):
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(output_dir / f"{name}.{suffix}", bbox_inches="tight")
        plt.close(fig)
    for name in disturbances:
        fig, axes = plt.subplots(1, len(states), figsize=(4.5*len(states), 3.5), squeeze=False)
        for ax, state in zip(axes[0], states):
            subset = [c for c in cases if c["state"] == state and c["disturbance"] == name]
            for method in methods:
                traces = [c["methods"][method]["trace"] for c in subset]
                length = max(map(len, traces))
                values = np.full((len(traces), length), np.nan)
                for row, trace in enumerate(traces):
                    for i, point in enumerate(trace):
                        if point["normalized_error"] is not None:
                            values[row, i] = point["normalized_error"]
                valid = np.isfinite(values).sum(0)
                mean = np.nansum(values, axis=0)/np.maximum(valid, 1)
                sd = np.sqrt(np.nansum((values-mean)**2, axis=0)/np.maximum(valid, 1))
                mean[valid == 0] = np.nan
                sd[valid == 0] = np.nan
                ax.plot(range(length), mean, label=method.replace("_", " "))
                ax.fill_between(range(length), mean-sd, mean+sd, alpha=.15)
            ax.set(title=state, xlabel="Controller iteration", ylabel="L2 error / initial L2 error")
        axes[0][-1].legend(fontsize=8)
        save(fig, f"recovery_{name}")
        for field, label in (("alpha", "Surrogate alpha"), ("Rhat", "Local finite-difference response"), ("G", "Gradient transmission G")):
            fig, axes = plt.subplots(1, len(states), figsize=(4.5*len(states), 3.5), squeeze=False)
            for ax, state in zip(axes[0], states):
                traces = [c["methods"]["response_aware"]["trace"] for c in cases if c["state"] == state and c["disturbance"] == name]
                blocks = len(traces[0][0]["alpha"])
                for block in range(blocks):
                    color = None
                    for i, trace in enumerate(traces):
                        points = [(p["iteration"], p[field][block]) for p in trace if p[field] is not None]
                        if points:
                            line, = ax.plot(*zip(*points), label=f"Block {block}" if i == 0 else None, alpha=.55, color=color)
                            color = line.get_color()
                    if field == "G":
                        ax.axhline(np.mean([t[0]["G_ref"][block] for t in traces]), linestyle="--", linewidth=.8, color=color)
                ax.set(title=state, xlabel="Controller iteration", ylabel=label)
            axes[0][-1].legend(fontsize=8)
            save(fig, f"{field}_{name}")
    values = np.asarray([summary["reference_response_mean"][s] for s in states])
    fig, ax = plt.subplots(figsize=(6, 3))
    image = ax.imshow(values, aspect="auto")
    ax.set(xticks=range(values.shape[1]), yticks=range(len(states)), yticklabels=states,
           xlabel="Block", ylabel="Checkpoint state")
    ax.grid(False)
    for i in range(len(states)):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i,j]:+.4f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Reference local finite difference at alpha=4")
    save(fig, "response_sign_by_state")
    lines = [summary["claims"], "", "Real-data feasibility gate: " + str(summary["gate"]["passed"]),
             "Gate failures: " + repr(summary["gate"]["failures"]),
             "Pooled normalized error comparisons: " + repr(summary["pooled_comparison"]),
             "Curves: mean +/- population SD; raw per-batch trajectories remain in recovery.json.",
             "Stopped trajectories are not extrapolated; later means may use fewer batches.",
             "Random matched directions are sampled among feasible signs; update magnitudes replay the response-aware schedule."]
    for state, groups in summary["groups"].items():
        lines.append(f"\n{state}: reference response={summary['reference_response_mean'][state]}")
        for name, methods in groups.items():
            for method, record in methods.items():
                lines.append(f"{name}/{method}: {record}")
    (output_dir / "summary.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="results/gradient_flow/controller_recovery_seed42")
    parser.add_argument("--output-dir", default="figures/gradient_flow/controller_recovery_seed42")
    parser.add_argument("--summary", help="Optional probe-free regenerated summary; raw records still hash-validated")
    args = parser.parse_args()
    output = (ROOT / args.output_dir).resolve()
    namespace = ROOT / "figures/gradient_flow"
    if namespace.resolve() != namespace or not output.is_relative_to(namespace) or output == namespace:
        parser.error("Figures must use a separate figures/gradient_flow/<run-id> namespace")
    plot(ROOT / args.run_dir, output, args.summary)


if __name__ == "__main__":
    main()
