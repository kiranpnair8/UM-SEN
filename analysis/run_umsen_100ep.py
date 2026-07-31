#!/usr/bin/env python3
"""Run the 100-epoch CIFAR-10 baseline vs UM-SEN benchmark."""

import argparse
import copy
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional

import run_umsen_mechanism_test as mechanism_test


CONFIGS = ("baseline", "umsen")
SEEDS = (42, 43, 44)


def parse_args():
    parser = argparse.ArgumentParser(description="Run 100-epoch CIFAR-10 baseline vs UM-SEN benchmark.")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=CONFIGS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("results/umsen_100ep"))
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


def resolved_config(args, device):
    config_args = copy.copy(args)
    config_args.seed = args.seeds[0] if args.seeds else None
    config = mechanism_test.resolved_config(config_args, device)
    config["configs"] = args.configs
    config["seeds"] = args.seeds
    config["checkpoint_metric"] = "validation_accuracy"
    return config


def save_checkpoint(path, model, optimizer, epoch, config_name, seed, metrics, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "config_name": config_name,
            "seed": seed,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def summarize_records(records):
    best_validation_accuracy = max(record["validation"]["accuracy"] for record in records)
    final_validation_accuracy = records[-1]["validation"]["accuracy"]
    best_validation_loss = min(record["validation"]["loss"] for record in records)
    training_time_seconds = sum(record["train"]["epoch_time_seconds"] for record in records)
    best_epoch = max(records, key=lambda record: record["validation"]["accuracy"])["epoch"]
    return {
        "best_validation_accuracy": float(best_validation_accuracy),
        "final_validation_accuracy": float(final_validation_accuracy),
        "best_validation_loss": float(best_validation_loss),
        "best_epoch": int(best_epoch),
        "training_time_seconds": float(training_time_seconds),
    }


def mean_std(values):
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def run_one(config_name, seed, args, device, output_dir, config):
    run_args = copy.copy(args)
    run_args.seed = seed
    run_args.configs = [config_name]

    mechanism_test.set_seed(seed)
    train_loader, val_loader = mechanism_test.build_loaders(run_args)
    model = mechanism_test.build_model(run_args).to(device)
    mechanism = mechanism_test.UMSENMechanism(model, config_name, run_args)
    spike_tracker = mechanism_test.SpikeRateTracker(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=run_args.lr, weight_decay=run_args.weight_decay)

    records = []
    best_validation_accuracy = float("-inf")
    best_checkpoint = output_dir / "checkpoints" / f"{config_name}_seed{seed}_best.pth"
    run_start = time.time()
    try:
        for epoch in range(run_args.epochs):
            train_metrics = mechanism_test.train_one_epoch(
                model, train_loader, criterion, optimizer, mechanism, spike_tracker, device, epoch
            )
            val_metrics = mechanism_test.validate(model, val_loader, criterion, device)
            best_validation_accuracy = max(best_validation_accuracy, val_metrics["accuracy"])
            record = {
                "epoch": epoch,
                "train": {
                    "accuracy": train_metrics["accuracy"],
                    "loss": train_metrics["loss"],
                    "epoch_time_seconds": train_metrics["epoch_time_seconds"],
                },
                "validation": {
                    "accuracy": val_metrics["accuracy"],
                    "loss": val_metrics["loss"],
                },
                "best_validation_accuracy": float(best_validation_accuracy),
            }
            records.append(record)
            if val_metrics["accuracy"] >= best_validation_accuracy:
                save_checkpoint(
                    best_checkpoint,
                    model,
                    optimizer,
                    epoch,
                    config_name,
                    seed,
                    record,
                    config,
                )
            print(
                f"{config_name} seed {seed} epoch {epoch}: "
                f"train_acc={train_metrics['accuracy']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"best_val_acc={best_validation_accuracy:.4f} "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"time={train_metrics['epoch_time_seconds']:.2f}s",
                flush=True,
            )
            write_json(
                output_dir / f"{config_name}_seed{seed}.json",
                {
                    "config": config,
                    "config_name": config_name,
                    "seed": seed,
                    "checkpoint": str(best_checkpoint),
                    "epochs": records,
                    "summary": summarize_records(records),
                },
            )
    finally:
        spike_tracker.close()
        functional.reset_net(model)

    summary = summarize_records(records)
    summary["wall_time_seconds"] = float(time.time() - run_start)
    write_json(
        output_dir / f"{config_name}_seed{seed}.json",
        {
            "config": config,
            "config_name": config_name,
            "seed": seed,
            "checkpoint": str(best_checkpoint),
            "epochs": records,
            "summary": summary,
        },
    )
    return {
        "config_name": config_name,
        "seed": seed,
        "checkpoint": str(best_checkpoint),
        "epochs": records,
        "summary": summary,
    }


def build_summary(results):
    summary = {"runs": {}, "aggregate": {}}
    for result in results:
        key = f"{result['config_name']}_seed{result['seed']}"
        summary["runs"][key] = {
            "checkpoint": result["checkpoint"],
            **result["summary"],
        }
    for config_name in CONFIGS:
        runs = [result for result in results if result["config_name"] == config_name]
        if not runs:
            continue
        summary["aggregate"][config_name] = {
            "best_validation_accuracy": mean_std(
                [result["summary"]["best_validation_accuracy"] for result in runs]
            ),
            "final_validation_accuracy": mean_std(
                [result["summary"]["final_validation_accuracy"] for result in runs]
            ),
        }
    return summary


def main():
    args = parse_args()
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("UM-SEN 100-epoch benchmark requires CUDA because the production model uses backend='cupy'.")

    config = resolved_config(args, device)
    write_json(output_dir / "config.json", config)

    results = []
    for config_name in args.configs:
        for seed in args.seeds:
            print(f"Starting {config_name} seed {seed}", flush=True)
            result = run_one(config_name, seed, args, device, output_dir, config)
            results.append(result)
            write_json(output_dir / "summary.json", {"config": config, **build_summary(results)})

    write_json(output_dir / "summary.json", {"config": config, **build_summary(results)})
    print(f"Wrote UM-SEN 100-epoch benchmark results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
