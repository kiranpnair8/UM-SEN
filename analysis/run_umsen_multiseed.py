#!/usr/bin/env python3
"""Run the final 5-epoch baseline vs UM-SEN validation across multiple seeds."""

import argparse
import copy
import json
import statistics
import time
from pathlib import Path

import torch

import run_umsen_mechanism_test as mechanism_test


CONFIGS = ("baseline", "umsen")
SEEDS = (42, 43, 44)


def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-seed UM-SEN validation.")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=CONFIGS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("results/umsen_multiseed"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=6e-2)
    parser.add_argument("--time-step", type=int, default=4)
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--entropy-temperature", type=float, default=0.25)
    parser.add_argument("--ema-beta", type=float, default=0.95)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--alpha-min", type=float, default=3.0)
    parser.add_argument("--alpha-max", type=float, default=5.0)
    parser.add_argument("--debug-controller", action="store_true")
    return parser.parse_args()


def resolve_repo_path(path):
    return mechanism_test.resolve_repo_path(path)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def summarize_run(records):
    val_records = [record["validation"] for record in records]
    train_records = [record["train"] for record in records]
    best_val_accuracy = max(record["accuracy"] for record in val_records)
    final_val_accuracy = val_records[-1]["accuracy"]
    best_val_loss = min(record["loss"] for record in val_records)
    training_time_seconds = sum(record["epoch_time_seconds"] for record in train_records)
    return {
        "best_validation_accuracy": float(best_val_accuracy),
        "final_validation_accuracy": float(final_val_accuracy),
        "best_validation_loss": float(best_val_loss),
        "training_time_seconds": float(training_time_seconds),
    }


def mean_std(values):
    if not values:
        return {"mean": None, "std": None}
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"mean": float(statistics.mean(values)), "std": float(std)}


def aggregate_by_config(seed_summaries, configs):
    aggregate = {}
    metric_names = (
        "best_validation_accuracy",
        "final_validation_accuracy",
        "best_validation_loss",
        "training_time_seconds",
    )
    for config_name in configs:
        runs = [
            seed_result["configs"][config_name]["summary"]
            for seed_result in seed_summaries
            if config_name in seed_result["configs"]
        ]
        aggregate[config_name] = {
            metric: mean_std([run[metric] for run in runs])
            for metric in metric_names
        }
    return aggregate


def format_accuracy(value):
    return f"{100.0 * value:.2f}%"


def format_mean_std(metric, as_accuracy=False):
    mean = metric["mean"]
    std = metric["std"]
    if mean is None:
        return "NA"
    if as_accuracy:
        return f"{100.0 * mean:.2f}% +/- {100.0 * std:.2f}%"
    return f"{mean:.4f} +/- {std:.4f}"


def write_summary(path, payload):
    lines = [
        "UM-SEN multi-seed validation",
        "",
        f"Seeds: {', '.join(str(seed) for seed in payload['config']['seeds'])}",
        f"Epochs: {payload['config']['epochs']}",
        f"Configs: {', '.join(payload['config']['configs'])}",
        "",
        "Per-seed results:",
    ]
    for seed_result in payload["seeds"]:
        lines.append(f"seed {seed_result['seed']}:")
        for config_name in payload["config"]["configs"]:
            summary = seed_result["configs"][config_name]["summary"]
            lines.append(
                "  "
                f"{config_name}: "
                f"best_acc={format_accuracy(summary['best_validation_accuracy'])}, "
                f"final_acc={format_accuracy(summary['final_validation_accuracy'])}, "
                f"best_loss={summary['best_validation_loss']:.4f}, "
                f"train_time={summary['training_time_seconds']:.2f}s"
            )
    lines.extend(["", "Overall mean +/- std:"])
    for config_name in payload["config"]["configs"]:
        aggregate = payload["aggregate"][config_name]
        lines.append(
            f"{config_name}: "
            f"best_acc={format_mean_std(aggregate['best_validation_accuracy'], as_accuracy=True)}, "
            f"final_acc={format_mean_std(aggregate['final_validation_accuracy'], as_accuracy=True)}, "
            f"best_loss={format_mean_std(aggregate['best_validation_loss'])}, "
            f"train_time={format_mean_std(aggregate['training_time_seconds'])}s"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolved_config(args, device):
    config_args = copy.copy(args)
    if not hasattr(config_args, "seed"):
        config_args.seed = args.seeds[0] if args.seeds else None
    config = mechanism_test.resolved_config(config_args, device)
    config["configs"] = args.configs
    config["seeds"] = args.seeds
    return config


def run_seed_config(seed, config_name, args, device, output_dir):
    run_args = copy.copy(args)
    run_args.seed = seed
    run_args.configs = [config_name]
    run_output_dir = output_dir / f"seed_{seed}" / config_name
    run_args.output_dir = run_output_dir
    start = time.time()
    records = mechanism_test.run_config(config_name, run_args, device)
    wall_time_seconds = time.time() - start
    summary = summarize_run(records)
    summary["wall_time_seconds"] = float(wall_time_seconds)
    write_json(run_output_dir / f"{config_name}.json", {
        "config": resolved_config(run_args, device),
        "epochs": records,
        "summary": summary,
    })
    return {"epochs": records, "summary": summary}


def main():
    args = parse_args()
    output_dir = resolve_repo_path(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("UM-SEN multi-seed validation requires CUDA because the production model uses backend='cupy'.")

    output_dir.mkdir(parents=True, exist_ok=True)
    config = resolved_config(args, device)
    write_json(output_dir / "config.json", config)

    seed_summaries = []
    all_results = {"config": config, "seeds": [], "aggregate": {}}
    for seed in args.seeds:
        seed_result = {"seed": seed, "configs": {}}
        print(f"Starting seed {seed}", flush=True)
        for config_name in args.configs:
            print(f"Starting seed {seed}, configuration {config_name}", flush=True)
            seed_result["configs"][config_name] = run_seed_config(seed, config_name, args, device, output_dir)
        seed_summaries.append(seed_result)
        all_results["seeds"] = seed_summaries
        all_results["aggregate"] = aggregate_by_config(seed_summaries, args.configs)
        write_json(output_dir / "metrics.json", all_results)

    all_results["aggregate"] = aggregate_by_config(seed_summaries, args.configs)
    write_json(output_dir / "metrics.json", all_results)
    write_summary(output_dir / "summary.txt", all_results)
    print(f"Wrote UM-SEN multi-seed validation results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
