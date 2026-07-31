#!/usr/bin/env python3
"""Focused Block 1 controller comparison for UM-SEN multi-seed results."""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


DEFAULT_INPUT_DIR = Path("results/umsen_multiseed")
DEFAULT_OUTPUT_DIR = Path("figures/umsen_multiseed")
DEFAULT_FIGURE = "block1_controller_comparison.png"
DEFAULT_SUMMARY = "block1_controller_summary.txt"
SEEDS = (42, 43, 44)
CONFIG = "umsen"
BLOCK_ID = "1"


METRICS = {
    "ema": {
        "keys": ("ema_entropy_dispersion_per_block",),
        "label": "EMA entropy dispersion",
        "title": "EMA Entropy Dispersion",
    },
    "running_mean": {
        "keys": (
            "running_mean_per_block",
            "controller_running_mean_per_block",
            "running_entropy_mean_per_block",
            "ema_running_mean_per_block",
        ),
        "label": "Controller running mean",
        "title": "Running Mean",
    },
    "centered_z": {
        "keys": ("centered_z_per_block",),
        "label": "Centered z",
        "title": "Centered z",
    },
    "alpha": {
        "keys": ("alpha_per_block", "controller_alpha_per_block"),
        "label": "Alpha",
        "title": "Applied Alpha",
    },
}


def warn(message):
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze UM-SEN Block 1 controller traces.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--block-id", default=BLOCK_ID)
    return parser.parse_args()


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


def epochs(records):
    return np.asarray([record.get("epoch", idx) for idx, record in enumerate(records)], dtype=float)


def get_block_value(record, metric_keys, block_id):
    train = record.get("train", {})
    for metric_key in metric_keys:
        values = train.get(metric_key)
        if isinstance(values, dict):
            value = values.get(str(block_id), values.get(block_id))
            if value is not None:
                try:
                    return float(value), metric_key
                except (TypeError, ValueError):
                    warn(f"Non-numeric {metric_key}.{block_id}: {value!r}")
    return np.nan, None


def block_series(records, metric_keys, block_id):
    values = []
    source_key = None
    for record in records:
        value, key = get_block_value(record, metric_keys, block_id)
        values.append(value)
        source_key = source_key or key
    return np.asarray(values, dtype=float), source_key


def seed_style(seed):
    if seed == 42:
        return {"linewidth": 3.0, "alpha": 1.0, "zorder": 4}
    return {"linewidth": 1.7, "alpha": 0.72, "zorder": 3}


def sign_change_epochs(records, z_values):
    x = epochs(records)
    changes = []
    for idx in range(1, len(z_values)):
        prev = z_values[idx - 1]
        curr = z_values[idx]
        if np.isnan(prev) or np.isnan(curr):
            continue
        if np.sign(prev) != np.sign(curr):
            changes.append((float(x[idx - 1]), float(x[idx]), float(prev), float(curr)))
    return changes


def seed42_sign_mismatch_epochs(results, block_id):
    if 42 not in results:
        return []
    z42, _ = block_series(results[42], METRICS["centered_z"]["keys"], block_id)
    x42 = epochs(results[42])
    mismatches = []
    for idx, value42 in enumerate(z42):
        if np.isnan(value42):
            continue
        other_signs = []
        for seed in sorted(results):
            if seed == 42:
                continue
            other, _ = block_series(results[seed], METRICS["centered_z"]["keys"], block_id)
            if idx < len(other) and not np.isnan(other[idx]):
                other_signs.append(np.sign(other[idx]))
        if other_signs and all(np.sign(value42) != sign for sign in other_signs):
            mismatches.append(float(x42[idx]))
    return mismatches


def plot_controller(results, output_dir, block_id):
    fig, axes = plt.subplots(4, 1, figsize=(8.4, 11.0), sharex=True)
    palette = sns.color_palette("deep", n_colors=max(len(results), 3))
    seed_colors = {seed: palette[idx] for idx, seed in enumerate(sorted(results))}
    mismatch_epochs = seed42_sign_mismatch_epochs(results, block_id)

    for ax, metric_name in zip(axes, ("ema", "running_mean", "centered_z", "alpha")):
        spec = METRICS[metric_name]
        found = False
        for seed in sorted(results):
            records = results[seed]
            y, source_key = block_series(records, spec["keys"], block_id)
            if source_key is None or np.all(np.isnan(y)):
                continue
            found = True
            label = f"seed {seed}" + (" (highlight)" if seed == 42 else "")
            ax.plot(
                epochs(records),
                y,
                marker="o",
                markersize=4,
                color=seed_colors[seed],
                label=label,
                **seed_style(seed),
            )
        if metric_name == "centered_z":
            ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.65)
        for epoch in mismatch_epochs:
            ax.axvspan(epoch - 0.08, epoch + 0.08, color="crimson", alpha=0.12, linewidth=0)
        ax.set_title(spec["title"])
        ax.set_ylabel(spec["label"])
        ax.grid(True, alpha=0.35)
        if not found:
            ax.text(
                0.5,
                0.5,
                "Not present in saved multiseed results",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
                color="dimgray",
            )
    axes[-1].set_xlabel("Epoch")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 3), frameon=False)
    fig.suptitle(
        f"UM-SEN Block {block_id} Controller Comparison\n"
        "crimson bands mark epochs where seed 42 centered-z sign differs from both other seeds",
        y=0.985,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / DEFAULT_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(fig)


def describe_metric(results, block_id, metric_name):
    spec = METRICS[metric_name]
    rows = []
    for seed, records in sorted(results.items()):
        y, source_key = block_series(records, spec["keys"], block_id)
        if source_key is None or np.all(np.isnan(y)):
            continue
        rows.append(
            f"seed {seed}: first={y[0]:.6f}, final={y[-1]:.6f}, "
            f"min={np.nanmin(y):.6f}, max={np.nanmax(y):.6f}, source={source_key}"
        )
    return rows


def write_summary(results, output_dir, block_id):
    lines = [
        f"UM-SEN Block {block_id} controller comparison",
        "",
        "Input: results/umsen_multiseed/",
        "Configuration: UM-SEN only, seeds 42, 43, 44.",
        "Seed 42 is highlighted with a thicker line in the figure.",
        "",
    ]
    for metric_name in ("ema", "running_mean", "centered_z", "alpha"):
        lines.append(METRICS[metric_name]["title"] + ":")
        rows = describe_metric(results, block_id, metric_name)
        if rows:
            lines.extend(f"  {row}" for row in rows)
        else:
            lines.append("  Not present in the saved multiseed results.")
            if metric_name == "running_mean":
                lines.append(
                    "  The controller running mean was used during training but was not saved in "
                    "analysis/run_umsen_mechanism_test.py snapshots, so it cannot be reconstructed "
                    "exactly without rerunning or changing logging."
                )
        lines.append("")

    lines.append("Centered-z sign changes:")
    for seed, records in sorted(results.items()):
        z, _ = block_series(records, METRICS["centered_z"]["keys"], block_id)
        changes = sign_change_epochs(records, z)
        if not changes:
            lines.append(f"  seed {seed}: no sign change")
            continue
        text = "; ".join(
            f"epoch {start:g}->{end:g}: {prev:.6f}->{curr:.6f}"
            for start, end, prev, curr in changes
        )
        lines.append(f"  seed {seed}: {text}")

    mismatches = seed42_sign_mismatch_epochs(results, block_id)
    lines.append("")
    if mismatches:
        lines.append(
            "Seed 42 differs in centered-z sign from both seeds 43 and 44 at epochs: "
            + ", ".join(f"{epoch:g}" for epoch in mismatches)
        )
    else:
        lines.append("No epoch found where seed 42 centered-z sign differs from both seeds 43 and 44.")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / DEFAULT_SUMMARY).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    configure_plots()
    output_dir = repo_path(args.output_dir)
    results = load_results(args.input_dir, args.seeds)
    plot_controller(results, output_dir, str(args.block_id))
    write_summary(results, output_dir, str(args.block_id))
    print(f"Wrote Block 1 controller comparison to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
