#!/usr/bin/env python3
"""Run the 100-epoch CIFAR-10 baseline vs UM-SEN benchmark."""

import argparse
import copy
import gc
import json
import os
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


def controller_state(mechanism):
    return {
        "step": mechanism.step,
        "current_epoch": mechanism.current_epoch,
        "rng_state": mechanism.rng.getstate(),
        "controllers": [
            {
                "raw_dispersion": controller.raw_dispersion,
                "ema_dispersion": controller.ema_dispersion,
                "normalized_z": controller.normalized_z,
                "centered_z": controller.centered_z,
                "alpha": controller.alpha,
                "initialized": controller.initialized,
                "running_count": controller.running_count,
                "running_mean": controller.running_mean,
                "running_m2": controller.running_m2,
            }
            for controller in mechanism.controllers
        ],
    }


def load_controller_state(mechanism, state):
    if not state:
        return
    mechanism.step = int(state.get("step", mechanism.step))
    mechanism.current_epoch = int(state.get("current_epoch", mechanism.current_epoch))
    if state.get("rng_state") is not None:
        mechanism.rng.setstate(state["rng_state"])
    for controller, saved in zip(mechanism.controllers, state.get("controllers", [])):
        for key, value in saved.items():
            if hasattr(controller, key):
                setattr(controller, key, value)


def save_latest_checkpoint(path, model, optimizer, mechanism, epoch, config_name, seed, metrics, config, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "config_name": config_name,
            "seed": seed,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "controller_state": controller_state(mechanism),
            "metrics": metrics,
            "records": records,
            "config": config,
        },
        path,
    )


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def clear_controller_buffers(mechanism):
    for controller in mechanism.controllers:
        controller.raw_history.clear()
        controller.ema_history.clear()
        controller.z_history.clear()
        controller.centered_z_history.clear()
        controller.alpha_history.clear()


def process_rss_mb():
    statm = Path("/proc/self/statm")
    if statm.exists():
        pages = int(statm.read_text(encoding="utf-8").split()[1])
        return float(pages * os.sysconf("SC_PAGE_SIZE") / (1024 ** 2))
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return float(value / 1024.0)
    except Exception:
        return None


def memory_stats(device):
    stats = {"process_rss_mb": process_rss_mb()}
    if device.type == "cuda":
        stats.update({
            "cuda_allocated_mb": float(torch.cuda.memory_allocated(device) / (1024 ** 2)),
            "cuda_reserved_mb": float(torch.cuda.memory_reserved(device) / (1024 ** 2)),
        })
    else:
        stats.update({"cuda_allocated_mb": 0.0, "cuda_reserved_mb": 0.0})
    return stats


def sanitize_record(record):
    return {
        "epoch": int(record["epoch"]),
        "train": {
            "accuracy": float(record["train"]["accuracy"]),
            "loss": float(record["train"]["loss"]),
            "epoch_time_seconds": float(record["train"].get("epoch_time_seconds", 0.0)),
        },
        "validation": {
            "accuracy": float(record["validation"]["accuracy"]),
            "loss": float(record["validation"]["loss"]),
        },
        "best_validation_accuracy": float(record["best_validation_accuracy"]),
        "memory": record.get("memory", {}),
    }


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
    run_path = output_dir / f"{config_name}_seed{seed}.json"
    best_checkpoint = output_dir / "checkpoints" / f"{config_name}_seed{seed}_best.pth"
    latest_checkpoint = output_dir / "checkpoints" / f"{config_name}_seed{seed}_latest.pth"

    if run_path.exists():
        existing = json.loads(run_path.read_text(encoding="utf-8"))
        existing_records = existing.get("epochs", [])
        if len(existing_records) >= run_args.epochs:
            existing_summary = existing.get("summary")
            if existing_summary is None:
                existing_summary = summarize_records(existing_records[:run_args.epochs])
            print(f"Skipping complete run: {config_name} seed {seed}", flush=True)
            return {
                "config_name": config_name,
                "seed": seed,
                "checkpoint": existing.get("checkpoint", str(best_checkpoint)),
                "latest_checkpoint": str(latest_checkpoint),
                "epochs": existing_records[:run_args.epochs],
                "summary": existing_summary,
            }

    mechanism_test.set_seed(seed)
    records = []
    best_validation_accuracy = float("-inf")
    start_epoch = 0
    run_start = time.time()
    train_loader = val_loader = model = mechanism = spike_tracker = criterion = optimizer = None
    try:
        train_loader, val_loader = mechanism_test.build_loaders(run_args)
        model = mechanism_test.build_model(run_args).to(device)
        mechanism = mechanism_test.UMSENMechanism(model, config_name, run_args)
        spike_tracker = mechanism_test.SpikeRateTracker(model)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=run_args.lr, weight_decay=run_args.weight_decay)

        resume_checkpoint = latest_checkpoint if latest_checkpoint.exists() else None
        if resume_checkpoint is None and best_checkpoint.exists():
            resume_checkpoint = best_checkpoint
            print(
                f"Latest checkpoint missing for {config_name} seed {seed}; "
                "falling back to best checkpoint.",
                flush=True,
            )
        if resume_checkpoint is not None:
            checkpoint = load_checkpoint(resume_checkpoint, device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            load_controller_state(mechanism, checkpoint.get("controller_state"))
            records = [
                sanitize_record(record)
                for record in checkpoint.get("records", [])
                if int(record.get("epoch", -1)) <= int(checkpoint["epoch"])
            ]
            if not records and run_path.exists():
                existing = json.loads(run_path.read_text(encoding="utf-8"))
                records = [
                    sanitize_record(record)
                    for record in existing.get("epochs", [])
                    if int(record.get("epoch", -1)) <= int(checkpoint["epoch"])
                ]
            start_epoch = int(checkpoint["epoch"]) + 1
            if records:
                best_validation_accuracy = max(record["validation"]["accuracy"] for record in records)
            print(
                f"Resuming {config_name} seed {seed} from epoch {start_epoch}",
                flush=True,
            )

        for epoch in range(start_epoch, run_args.epochs):
            train_metrics = mechanism_test.train_one_epoch(
                model, train_loader, criterion, optimizer, mechanism, spike_tracker, device, epoch
            )
            val_metrics = mechanism_test.validate(model, val_loader, criterion, device)
            val_accuracy = float(val_metrics["accuracy"])
            is_best = val_accuracy >= best_validation_accuracy
            best_validation_accuracy = max(best_validation_accuracy, val_accuracy)
            record = {
                "epoch": int(epoch),
                "train": {
                    "accuracy": float(train_metrics["accuracy"]),
                    "loss": float(train_metrics["loss"]),
                    "epoch_time_seconds": float(train_metrics["epoch_time_seconds"]),
                },
                "validation": {
                    "accuracy": val_accuracy,
                    "loss": float(val_metrics["loss"]),
                },
                "best_validation_accuracy": float(best_validation_accuracy),
                "memory": memory_stats(device),
            }
            records.append(record)
            save_latest_checkpoint(
                latest_checkpoint,
                model,
                optimizer,
                mechanism,
                epoch,
                config_name,
                seed,
                record,
                config,
                records,
            )
            if is_best:
                save_latest_checkpoint(
                    best_checkpoint,
                    model,
                    optimizer,
                    mechanism,
                    epoch,
                    config_name,
                    seed,
                    record,
                    config,
                    records,
                )
            clear_controller_buffers(mechanism)
            print(
                f"{config_name} seed {seed} epoch {epoch}: "
                f"train_acc={train_metrics['accuracy']:.4f} "
                f"val_acc={val_accuracy:.4f} "
                f"best_val_acc={best_validation_accuracy:.4f} "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"time={train_metrics['epoch_time_seconds']:.2f}s "
                f"rss_mb={record['memory']['process_rss_mb']} "
                f"cuda_alloc_mb={record['memory']['cuda_allocated_mb']:.2f} "
                f"cuda_reserved_mb={record['memory']['cuda_reserved_mb']:.2f}",
                flush=True,
            )
            write_json(
                run_path,
                {
                    "config": config,
                    "config_name": config_name,
                    "seed": seed,
                    "checkpoint": str(best_checkpoint),
                    "latest_checkpoint": str(latest_checkpoint),
                    "epochs": records,
                    "summary": summarize_records(records),
                },
            )
    finally:
        if spike_tracker is not None:
            spike_tracker.close()
        if model is not None:
            functional.reset_net(model)
        del optimizer, criterion, spike_tracker, mechanism, model, train_loader, val_loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = summarize_records(records)
    summary["wall_time_seconds"] = float(time.time() - run_start)
    write_json(
        run_path,
        {
            "config": config,
            "config_name": config_name,
            "seed": seed,
            "checkpoint": str(best_checkpoint),
            "latest_checkpoint": str(latest_checkpoint),
            "epochs": records,
            "summary": summary,
        },
    )
    return {
        "config_name": config_name,
        "seed": seed,
        "checkpoint": str(best_checkpoint),
        "latest_checkpoint": str(latest_checkpoint),
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
