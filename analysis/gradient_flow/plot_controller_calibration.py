"""Read-only source comparison; write figures/summary only in calibration namespaces."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.gradient_flow.calibration_spec import (OUTPUT, VARIANTS, checked_directory,
    historical_sources, read, resolve, training_invariants)
from experiments.gradient_flow.common import ROOT, write_json
from .calibration_summary import calibration_gate, relative, summarize


def load_results(run_dir):
    manifest = read(run_dir/"study_manifest.json")
    if manifest["identity"]["synthetic"]:
        raise ValueError("Synthetic smoke cannot finalize the real calibration comparison")
    config = manifest["config"]
    sources = historical_sources(config, manifest["identity"])
    if sources != manifest["sources"]:
        raise ValueError("Historical CURRENT/Fixed evidence changed since registration")
    if read(run_dir/"current/reference.json") != dict(condition="current",
            reused_from_completed_prior_experiment=True, **sources["current"]):
        raise ValueError("CURRENT read-only reference changed")
    data, summaries, origins = {}, {}, {}
    for name in ("fixed", *VARIANTS):
        path = ROOT/sources[name]["path"] if name in sources else run_dir/name
        identity, saved_config, rows = checked_directory(path, epochs=config["online"]["epochs"])
        if name not in sources:
            if (training_invariants(identity) != training_invariants(manifest["identity"]) or
                    identity.get("variant") != name or identity.get("method") != "response_aware" or
                    saved_config != resolve(config, name) or
                    any(r["implementation_commit"] != identity["source_commit"] for r in rows)):
                raise ValueError("Calibration provenance/controller comparison failed: "+name)
        events = path/"amp_events.jsonl"
        persistent = any(json.loads(line).get("persistence_failure") or
            json.loads(line).get("kind") in ("nonfinite_loss", "nonfinite_fp32_gradients", "unsafe_gradient_observation")
            for line in events.read_text().splitlines() if line.strip()) if events.exists() else False
        data[name] = rows
        summaries[name] = summarize(rows, saved_config["online"], bool(persistent))
        from experiments.gradient_flow.online_checkpoint import digest
        origins[name] = dict(path=path.relative_to(ROOT).as_posix(), reused=name in sources,
            metrics_sha256=digest(path/"metrics.json"), identity=identity)
    criteria = config["calibration"]["criteria"]
    comparisons = {name: relative(s, summaries["fixed"], criteria) for name, s in summaries.items() if name != "fixed"}
    gates = {name: calibration_gate(summaries[name], comparison, criteria, complete=True)
        for name, comparison in comparisons.items()}
    result = dict(conditions=summaries, relative_to_fixed=comparisons, criteria=criteria, gates=gates,
        origins=origins, study_config_hash=manifest["identity"]["study_config_hash"],
        implementation_commit=manifest["identity"]["source_commit"],
        caveat="Seed42 only. Variance is population variance of epoch-mean G, not minibatch variance. "
            "CURRENT and Fixed reused read-only. Descriptive tradeoff only; no automatic selection. "
            "If essentially tied, prefer the simpler/more conservative policy for subsequent multi-seed evaluation.")
    return data, result


def plot(run_dir, output_dir):
    run_dir, output_dir = Path(run_dir).absolute(), Path(output_dir).absolute()
    if (run_dir.resolve() != run_dir or output_dir.resolve() != output_dir or
            run_dir != ROOT/OUTPUT or output_dir != ROOT/"figures/gradient_flow/controller_calibration_seed42"):
        raise ValueError("Use separate declared calibration result/figure directories; original study is read-only")
    data, result = load_results(run_dir)  # Refuse incomplete variants BEFORE writing any finalization.
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .25, "savefig.dpi": 300})

    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(output_dir/(name+"."+ext), dpi=300, bbox_inches="tight")
        plt.close(fig)

    labels = dict(fixed="Fixed", current="Current (reused)", conservative="Conservative",
                  very_conservative="Very conservative", anchored="Anchored")
    for group, field, label, filename in (("validation", "top1", "Validation top1 (%)", "validation_accuracy"),
            ("train", "loss", "Train loss", "train_loss")):
        fig, ax = plt.subplots(figsize=(7, 4))
        for name, rows in data.items():
            ax.plot([r["epoch"] for r in rows], [r[group][field] for r in rows], label=labels[name])
        ax.set(xlabel="Epoch (zero-based)", ylabel=label)
        ax.legend(fontsize=8)
        save(fig, filename)
    blocks = len(result["conditions"]["fixed"]["alpha_final"])
    for field, filename, ylabel in (("alpha", "alpha_trajectories", "Alpha"),
            ("deviation", "alpha_deviation", "Mean absolute alpha deviation from 4"),
            ("movement", "cumulative_alpha_movement", "Cumulative absolute movement"),
            ("update", "controller_update_magnitude", "Absolute applied alpha step")):
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), squeeze=False)
        for ax, name in zip(axes.flat, VARIANTS):
            rows = data[name]
            cumulative = [0.]*blocks
            series = [[] for _ in range(blocks)]
            for row in rows:
                c = row["controller"]
                for b in range(blocks):
                    cumulative[b] += abs(c["applied_step"][b])
                    series[b].append(c["alpha"][b] if field == "alpha" else abs(c["alpha"][b]-4.)
                        if field == "deviation" else cumulative[b] if field == "movement" else abs(c["applied_step"][b]))
            for b in range(blocks):
                ax.plot([r["epoch"] for r in rows], series[b], label=f"Block {b}")
            ax.set(title=labels[name], xlabel="Epoch", ylabel=ylabel.replace("Mean ", ""))
            ax.legend(fontsize=8)
        save(fig, filename)
    summaries = result["conditions"]
    names = list(data)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(names)), [summaries[n]["mean_train_G_variance"] for n in names])
    ax.set(xticks=range(len(names)), xticklabels=[labels[n] for n in names], ylabel="Mean epoch-level train G variance")
    ax.tick_params(axis="x", labelrotation=15)
    save(fig, "mean_G_variance")
    fig, axes = plt.subplots(1, blocks, figsize=(max(10, 3*blocks), 4), squeeze=False)
    for b, ax in enumerate(axes[0]):
        ax.bar(range(len(names)), [summaries[n]["train_G_variance"][b] for n in names])
        ax.set(title=f"Block {b}", xticks=range(len(names)), xticklabels=[labels[n] for n in names], ylabel="Train G variance")
        ax.tick_params(axis="x", labelrotation=65)
    save(fig, "per_block_G_variance")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), squeeze=False)
    for ax, kind in zip(axes[0], ("best", "final")):
        for n in names:
            x, y = summaries[n]["mean_train_G_variance"], summaries[n][kind+"_validation_top1"]
            ax.scatter(x, y, label=labels[n])
        ax.set(xlabel="Mean train G variance (across epoch means)", ylabel=kind.capitalize()+" validation top1 (%)")
        ax.legend(fontsize=8)
    save(fig, "accuracy_vs_G_stability")
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.axis("off")
    table = ax.table(cellText=[[labels[n], f"{summaries[n]['final_validation_top1']:.2f}",
        f"{summaries[n]['best_validation_top1']:.2f}", f"{summaries[n]['mean_train_G_variance']:.4f}",
        f"{summaries[n]['controller_overhead_fraction']*100:.1f}",
        "reference" if n == "fixed" else str(result["gates"][n]["promising"])] for n in names],
        colLabels=["Condition", "Final top1", "Best top1", "Train G variance", "Overhead (%)", "Promising"], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    save(fig, "compact_calibration_summary")
    write_json(run_dir/"summary.json", result)
    write_json(output_dir/"summary.json", result)
    text = result["caveat"]+"\n\n"+json.dumps(result, indent=2, allow_nan=False)+"\n"
    (output_dir/"summary.txt").write_text(text, encoding="utf-8")
    (run_dir/"summary.txt").write_text(text, encoding="utf-8")
    print(json.dumps(dict(gates=result["gates"], observed_hours={n: s["total_epoch_seconds"]/3600 for n, s in summaries.items()}), indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=OUTPUT)
    parser.add_argument("--output-dir", default="figures/gradient_flow/controller_calibration_seed42")
    args = parser.parse_args()
    from experiments.gradient_flow.controller_calibration import lock
    run_dir = ROOT/args.run_dir
    with lock(run_dir/".analysis.lock"):
        plot(run_dir, ROOT/args.output_dir)


if __name__ == "__main__":
    main()
