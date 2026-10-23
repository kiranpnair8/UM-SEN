"""Saved-data-only method-design comparison; no training or automatic selection."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.gradient_flow import method_ablation as study
from experiments.gradient_flow.common import ROOT, write_json
from experiments.gradient_flow.controller_calibration import lock
from experiments.gradient_flow.method_policy import VARIANTS
from analysis.gradient_flow.calibration_summary import summarize, relative, calibration_gate


OUTPUT = "figures/gradient_flow/controller_method_ablation_seed42"


def method_gate(summary, comparison, criteria):
    result = calibration_gate(summary, comparison, criteria, complete=True)
    within_bounds = all(1 <= a <= 8 for a in summary["alpha_min"]+summary["alpha_max"])
    result["checks"]["alpha_within_bounds"] = within_bounds
    result["promising"] = all(result["checks"].values())
    result["stronger_positive_indicator"] = result["stronger_positive_indicator"] and within_bounds
    result["failures"] = [key for key, value in result["checks"].items() if not value]
    return result


def persistent_failure(path):
    events = Path(path)/"amp_events.jsonl"
    if not events.exists():
        return False
    return any(e.get("persistence_failure") or e.get("kind") in
        ("nonfinite_loss", "nonfinite_fp32_gradients", "unsafe_gradient_observation")
        for e in (json.loads(line) for line in events.read_text().splitlines() if line.strip()))


def ranking(summaries, comparisons, gates):
    """Lexicographic criteria, not a fitted combined score; no winner is selected."""
    eligible = []
    mechanical = {}
    for name in VARIANTS:
        checks = gates[name]["checks"]
        mechanical[name] = (all(value for key, value in checks.items() if key not in
            ("G_variance_reduced_at_least_25_percent", "final_accuracy_within_half_pp")) and
            all(1 <= a <= 8 for a in summaries[name]["alpha_min"]+summaries[name]["alpha_max"]))
        if mechanical[name] and checks["final_accuracy_within_half_pp"] and comparisons[name]["G_variance_reduction_fraction"] is not None:
            eligible.append(name)
    order = {name: i for i, name in enumerate(VARIANTS)}
    eligible.sort(key=lambda n: (-comparisons[n]["G_variance_reduction_fraction"], order[n],
        summaries[n]["controller_overhead_fraction"]))
    return dict(eligible_in_descriptive_order=eligible, mechanically_stable=mechanical,
        excluded=[n for n in VARIANTS if n not in eligible], final_sage_selected=None,
        rule="Mechanical stability first, final top1 >= Fixed-0.5pp, then decreasing G-variance reduction. "
            "Exact ties prefer simpler design, then lower overhead. No combined score or automatic selection.")


def load_results(run_dir):
    manifest = study.prior.read(run_dir/"study_manifest.json")
    if manifest["identity"]["synthetic"]:
        raise ValueError("Synthetic smoke cannot finalize a real method ablation")
    records, data = study.references(manifest["config"])
    if records != manifest["sources"] or study.choose_deadband(data["current"]) != manifest["deadband"]:
        raise ValueError("Historical references or frozen deadband changed")
    origins = dict(records)
    configs = {name: study.prior.read(ROOT/record["path"]/"config.json") for name, record in records.items()}
    for name in VARIANTS:
        path = run_dir/name
        identity, config, rows = study.prior.checked_directory(path)
        expected = dict(manifest["identity"], variant=name, deadband=manifest["deadband"]["value"])
        if (identity != expected or config != study.resolve(manifest["config"], name, identity["deadband"]) or
                config != manifest["variant_configs"][name] or
                any(r["implementation_commit"] != identity["source_commit"] for r in rows)):
            raise ValueError("Method variant provenance/config mismatch: "+name)
        if name == "deep_only" and any(c["alpha"][0] != 4. or c["alpha_before"][0] != 4. or c["applied_step"][0] != 0.
                for c in (r["controller"] for r in rows)):
            raise ValueError("DEEP_ONLY block0 was not exactly frozen")
        data[name], configs[name] = rows, config
        origins[name] = dict(path=path.relative_to(ROOT).as_posix(), identity=identity, reused=False,
            files={n: study.digest(path/n) for n in ("metrics.json", "config.json", "provenance.json", "status.json")})
    summaries = {name: summarize(rows, configs[name]["online"], persistent_failure(ROOT/origins[name]["path"]))
        for name, rows in data.items()}
    for name, s in summaries.items():
        s["bound_hit_count"] = sum(sum(r["controller"]["bound_hit"]) for r in data[name])
    criteria = manifest["config"]["method_ablation"]["criteria"]
    comparisons = {n: {key: value for key, value in relative(summaries[n], summaries["fixed"], criteria).items()
        if key != "stabilization_per_accuracy_cost"} for n in data if n != "fixed"}
    gates = {n: method_gate(summaries[n], comparisons[n], criteria) for n in VARIANTS}
    return data, dict(conditions=summaries, relative_to_fixed=comparisons, gates=gates, criteria=criteria,
        ranking=ranking(summaries, comparisons, gates), deadband=manifest["deadband"], origins=origins,
        deep_only_block0_exactly_4=True, implementation_commit=manifest["identity"]["source_commit"],
        study_config_hash=manifest["identity"]["study_config_hash"],
        caveat="Seed42 method-design ablation only. G variance is across epoch means, not minibatches. "
            "All references reused read-only. Ranking is descriptive; multi-seed follow-up requires explicit review.")


def plot(run_dir, output):
    run_dir, output = Path(run_dir).absolute(), Path(output).absolute()
    if run_dir != ROOT/study.OUTPUT or output != ROOT/OUTPUT or run_dir.resolve() != run_dir or output.resolve() != output:
        raise ValueError("Use the separate method-ablation output namespaces")
    data, result = load_results(run_dir)  # Incomplete runs cannot publish final plots/summary.
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .25})
    names = list(data)
    labels = {n: n.replace("_", " ") for n in names}
    labels["current"] = "current (reused)"
    blocks = len(result["conditions"]["fixed"]["alpha_final"])

    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(output/(name+"."+ext), dpi=300, bbox_inches="tight")
        plt.close(fig)

    for group, key, filename, ylabel in (("validation", "top1", "accuracy", "Validation top1 (%)"),
            ("train", "loss", "train_loss", "Train loss")):
        fig, ax = plt.subplots(figsize=(8, 4))
        for n, rows in data.items():
            ax.plot([r["epoch"] for r in rows], [r[group][key] for r in rows], label=labels[n])
        ax.set(xlabel="Epoch (zero-based)", ylabel=ylabel)
        ax.legend(fontsize=7)
        save(fig, filename)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, n in zip(axes, VARIANTS):
        for b in range(blocks):
            ax.plot([r["epoch"] for r in data[n]], [r["controller"]["alpha"][b] for r in data[n]], label=f"Block {b}")
        ax.set(title=labels[n], xlabel="Epoch", ylabel="Alpha")
        ax.legend(fontsize=7)
    save(fig, "alpha_trajectories")
    fig, axes = plt.subplots(1, blocks, figsize=(max(12, blocks*3.5), 4.5), squeeze=False)
    for b, ax in enumerate(axes[0]):
        ax.bar(range(len(names)), [result["conditions"][n]["train_G_variance"][b] for n in names])
        ax.set(title=f"Block {b}", xticks=range(len(names)), xticklabels=[labels[n] for n in names], ylabel="Epoch-level train G variance")
        ax.tick_params(axis="x", labelrotation=75)
    save(fig, "per_block_G_variance")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, kind in zip(axes, ("final", "best")):
        for n in names:
            s = result["conditions"][n]
            ax.scatter(s["mean_train_G_variance"], s[kind+"_validation_top1"], label=labels[n])
        ax.set(xlabel="Mean train G variance (across epoch means)", ylabel=kind.capitalize()+" top1 (%)")
        ax.legend(fontsize=7)
    save(fig, "G_variance_vs_accuracy")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for n, rows in data.items():
        epochs = [r["epoch"] for r in rows]
        axes[0].plot(epochs, [sum(abs(a-4) for a in r["controller"]["alpha"])/blocks for r in rows], label=labels[n])
        cumulative, values = 0., []
        for r in rows:
            cumulative += sum(abs(s) for s in r["controller"]["applied_step"])
            values.append(cumulative)
        axes[1].plot(epochs, values, label=labels[n])
    axes[0].set(xlabel="Epoch", ylabel="Mean absolute alpha deviation from 4")
    axes[1].set(xlabel="Epoch", ylabel="Cumulative absolute alpha movement")
    axes[1].legend(fontsize=7)
    save(fig, "alpha_drift")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, n in zip(axes, VARIANTS):
        values = [[r["controller"]["response_sign"][b] for r in data[n]] for b in range(blocks)]
        image = ax.imshow(values, aspect="auto", interpolation="nearest", vmin=-1, vmax=1)
        ax.set(title=labels[n], xlabel="Epoch", ylabel="Block", yticks=range(blocks))
        fig.colorbar(image, ax=ax, ticks=[-1, 0, 1], label="Response sign (tiny=0)")
    save(fig, "response_sign_behavior")
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.axis("off")
    table = ax.table(cellText=[[labels[n], f"{result['conditions'][n]['final_validation_top1']:.2f}",
        f"{result['conditions'][n]['best_validation_top1']:.2f}", f"{result['conditions'][n]['mean_train_G_variance']:.4f}",
        f"{result['conditions'][n]['mean_absolute_alpha_deviation_from_4']:.3f}",
        str(result["gates"][n]["promising"]) if n in VARIANTS else "reference"] for n in names],
        colLabels=["Condition", "Final top1", "Best top1", "Mean G variance", "Mean alpha drift", "Target met"], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.6)
    save(fig, "compact_summary")
    for destination in (run_dir, output):
        write_json(destination/"summary.json", result)
        (destination/"summary.txt").write_text(result["caveat"]+"\n\n"+json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result["ranking"], indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=study.OUTPUT)
    parser.add_argument("--output-dir", default=OUTPUT)
    args = parser.parse_args()
    with lock(ROOT/args.run_dir/".analysis.lock"):
        plot(ROOT/args.run_dir, ROOT/args.output_dir)


if __name__ == "__main__":
    main()
