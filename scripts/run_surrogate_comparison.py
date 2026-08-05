#!/usr/bin/env python3
"""Run surrogate-gradient comparison experiments with existing trainers.

This script is intentionally an orchestration layer. It calls the checked-in
Spikformer training entry points, keeps their optimizer/scheduler/augmentation
recipes intact, and varies only the requested surrogate method and time steps
when the underlying trainer already exposes that capability.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DATASETS = ("cifar10", "cifar100", "imagenet200")
METHODS = ("fixed", "learnable", "adaptive", "umsen")
TIME_STEPS = (1, 2, 4, 8, 12)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    workdir: Path
    config: str
    fixed_entry: str
    umsen_entry: str | None
    dataset_arg: str
    num_classes: int
    supports_time_step: bool
    default_data_dir: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def dataset_specs(root: Path) -> dict[str, DatasetSpec]:
    return {
        "cifar10": DatasetSpec(
            name="cifar10",
            workdir=root / "spikformer" / "cifar10",
            config="cifar10.yml",
            fixed_entry="train.py",
            umsen_entry="train_umsen.py",
            dataset_arg="torch/cifar10",
            num_classes=10,
            supports_time_step=True,
            default_data_dir=str(root / "data"),
        ),
        "cifar100": DatasetSpec(
            name="cifar100",
            workdir=root / "spikformer" / "cifar10",
            config="cifar10.yml",
            fixed_entry="train.py",
            umsen_entry="train_umsen.py",
            dataset_arg="torch/cifar100",
            num_classes=100,
            supports_time_step=True,
            default_data_dir=str(root / "data"),
        ),
        "imagenet200": DatasetSpec(
            name="imagenet200",
            workdir=root / "spikformer" / "imagenet",
            config="imagenet.yml",
            fixed_entry="train.py",
            umsen_entry=None,
            dataset_arg="imagenet",
            num_classes=200,
            supports_time_step=False,
            default_data_dir=str(root / "data" / "imagenet200"),
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Spikformer surrogate comparison experiments.")
    parser.add_argument("--dataset", choices=DATASETS, default="cifar10")
    parser.add_argument("--method", choices=METHODS, default="fixed")
    parser.add_argument("--T", type=int, default=4, choices=TIME_STEPS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", action="store_true", help="run the full dataset x method x T grid")
    parser.add_argument("--output-root", type=Path, default=Path("results/surrogate_comparison"))
    parser.add_argument("--data-dir", type=Path, default=None, help="override data directory for a single run")
    parser.add_argument("--cifar10-data-dir", type=Path, default=None)
    parser.add_argument("--cifar100-data-dir", type=Path, default=None)
    parser.add_argument("--imagenet200-data-dir", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable, help="Python executable used to launch trainers")
    parser.add_argument("--force", action="store_true", help="rerun even if metrics.json says the run completed")
    parser.add_argument("--dry-run", action="store_true", help="print commands and write no training outputs")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[], help="extra args appended to trainer")
    return parser.parse_args()


def data_dir_for(args: argparse.Namespace, spec: DatasetSpec) -> str:
    if args.data_dir is not None and not args.all:
        return str(args.data_dir)
    override = getattr(args, f"{spec.name}_data_dir")
    if override is not None:
        return str(override)
    return spec.default_data_dir


def method_support(spec: DatasetSpec, method: str, time_step: int) -> tuple[bool, str]:
    if method == "fixed":
        if spec.supports_time_step or time_step == 4:
            return True, "fixed surrogate available through baseline trainer"
        return False, "this trainer hard-codes T=4 and has no time-step CLI"
    if method == "umsen":
        if spec.umsen_entry is None:
            return False, "no validated UM-SEN wrapper exists for this dataset trainer"
        return True, "UM-SEN available through train_umsen.py"
    if method == "learnable":
        return False, "no validated learnable surrogate implementation exists in the repository"
    if method == "adaptive":
        return False, "no non-UM-SEN adaptive surrogate implementation exists in the repository"
    return False, f"unknown method: {method}"


def run_dir(root: Path, dataset: str, method: str, time_step: int, seed: int) -> Path:
    return root / dataset / method / f"T{time_step}" / f"seed{seed}"


def latest_checkpoint(path: Path) -> Path | None:
    preferred = path / "last.pth.tar"
    if preferred.exists():
        return preferred
    candidates = list(path.glob("*.pth.tar")) + list(path.glob("*.pth"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def best_checkpoint(path: Path) -> Path | None:
    for name in ("model_best.pth.tar", "best.pth.tar", "model_best.pth"):
        candidate = path / name
        if candidate.exists():
            return candidate
    return None


def summary_path(path: Path) -> Path:
    return path / "summary.csv"


def metrics_path(path: Path) -> Path:
    return path / "metrics.json"


def is_complete(path: Path, epochs: int) -> bool:
    metrics_file = metrics_path(path)
    if not metrics_file.exists():
        return False
    try:
        metrics = json.loads(metrics_file.read_text())
    except json.JSONDecodeError:
        return False
    return metrics.get("status") == "completed" and int(metrics.get("epochs", -1)) >= epochs


def build_command(
    args: argparse.Namespace,
    spec: DatasetSpec,
    method: str,
    time_step: int,
    seed: int,
    out_dir: Path,
) -> list[str]:
    entry = spec.umsen_entry if method == "umsen" else spec.fixed_entry
    if entry is None:
        raise ValueError(f"No entry point for {spec.name}/{method}")

    cmd = [
        args.python,
        entry,
        "-c",
        spec.config,
        "--epochs",
        str(args.epochs),
        "--seed",
        str(seed),
        "-data-dir",
        data_dir_for(args, spec),
        "--dataset",
        spec.dataset_arg,
        "--num-classes",
        str(spec.num_classes),
        "--output",
        str(out_dir.parent),
        "--experiment",
        out_dir.name,
    ]
    if spec.supports_time_step:
        cmd.extend(["-T", str(time_step)])
    if method == "umsen":
        cmd.append("--umsen")
    ckpt = latest_checkpoint(out_dir)
    if ckpt is not None:
        cmd.extend(["--resume", str(ckpt)])
    cmd.extend(args.extra_args)
    return cmd


def parse_summary(path: Path) -> dict:
    summary = summary_path(path)
    if not summary.exists():
        return {
            "best_accuracy": None,
            "final_accuracy": None,
            "best_epoch": None,
            "rows": [],
            "warning": f"missing {summary}",
        }

    with summary.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {
            "best_accuracy": None,
            "final_accuracy": None,
            "best_epoch": None,
            "rows": [],
            "warning": f"empty {summary}",
        }

    acc_key = next((k for k in ("eval_top1", "top1", "accuracy", "acc1") if k in rows[0]), None)
    if acc_key is None:
        return {
            "best_accuracy": None,
            "final_accuracy": None,
            "best_epoch": None,
            "rows": rows,
            "warning": f"no accuracy column in {summary}",
        }

    best_acc = -math.inf
    best_epoch = None
    for row in rows:
        try:
            acc = float(row[acc_key])
        except (TypeError, ValueError):
            continue
        if acc > best_acc:
            best_acc = acc
            try:
                best_epoch = int(float(row.get("epoch", len(rows) - 1)))
            except (TypeError, ValueError):
                best_epoch = None

    final_acc = float(rows[-1][acc_key])
    loss_key = next((k for k in ("eval_loss", "loss") if k in rows[-1]), None)
    final_loss = float(rows[-1][loss_key]) if loss_key is not None else None
    return {
        "best_accuracy": best_acc,
        "final_accuracy": final_acc,
        "best_epoch": best_epoch,
        "final_loss": final_loss,
        "accuracy_column": acc_key,
        "rows": rows,
    }


def write_metrics(path: Path, payload: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    metrics_path(path).write_text(json.dumps(payload, indent=2) + "\n")


def write_unsupported(path: Path, dataset: str, method: str, time_step: int, seed: int, reason: str) -> None:
    write_metrics(
        path,
        {
            "status": "unsupported",
            "dataset": dataset,
            "method": method,
            "T": time_step,
            "seed": seed,
            "reason": reason,
        },
    )


def run_one(args: argparse.Namespace, spec: DatasetSpec, method: str, time_step: int, seed: int) -> dict:
    ok, reason = method_support(spec, method, time_step)
    out_dir = run_dir(args.output_root, spec.name, method, time_step, seed)
    if not ok:
        if args.all:
            if not args.dry_run:
                write_unsupported(out_dir, spec.name, method, time_step, seed, reason)
            print(f"SKIP {spec.name}/{method}/T{time_step}/seed{seed}: {reason}")
            return {"status": "unsupported", "reason": reason}
        raise SystemExit(f"Unsupported experiment {spec.name}/{method}/T{time_step}: {reason}")

    if is_complete(out_dir, args.epochs) and not args.force:
        print(f"SKIP completed {out_dir}")
        return json.loads(metrics_path(out_dir).read_text())

    cmd = build_command(args, spec, method, time_step, seed, out_dir)
    print("RUN", " ".join(cmd))
    if args.dry_run:
        return {"status": "dry_run", "command": cmd}

    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    log_file = out_dir / "train.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{spec.workdir}{os.pathsep}{env.get('PYTHONPATH', '')}"
    with log_file.open("ab") as log:
        log.write(("\n\n===== surrogate comparison run =====\n" + " ".join(cmd) + "\n").encode())
        result = subprocess.run(cmd, cwd=spec.workdir, stdout=log, stderr=subprocess.STDOUT, env=env)

    elapsed = time.time() - start
    parsed = parse_summary(out_dir)
    status = "completed" if result.returncode == 0 else "failed"
    payload = {
        "status": status,
        "returncode": result.returncode,
        "dataset": spec.name,
        "method": method,
        "T": time_step,
        "seed": seed,
        "epochs": args.epochs,
        "output_dir": str(out_dir),
        "training_log": str(log_file),
        "checkpoint": str(latest_checkpoint(out_dir)) if latest_checkpoint(out_dir) else None,
        "best_checkpoint": str(best_checkpoint(out_dir)) if best_checkpoint(out_dir) else None,
        "elapsed_seconds": elapsed,
        "best_accuracy": parsed.get("best_accuracy"),
        "final_accuracy": parsed.get("final_accuracy"),
        "best_epoch": parsed.get("best_epoch"),
        "final_loss": parsed.get("final_loss"),
        "accuracy_column": parsed.get("accuracy_column"),
        "warning": parsed.get("warning"),
        "command": cmd,
    }
    write_metrics(out_dir, payload)
    if result.returncode != 0:
        raise SystemExit(f"Run failed with exit code {result.returncode}; see {log_file}")
    return payload


def grid(args: argparse.Namespace) -> Iterable[tuple[str, str, int, int]]:
    if args.all:
        for dataset in DATASETS:
            for method in METHODS:
                for time_step in TIME_STEPS:
                    yield dataset, method, time_step, args.seed
    else:
        yield args.dataset, args.method, args.T, args.seed


def main() -> None:
    args = parse_args()
    root = repo_root()
    args.output_root = (root / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root
    specs = dataset_specs(root)

    results = []
    for dataset, method, time_step, seed in grid(args):
        results.append(run_one(args, specs[dataset], method, time_step, seed))

    if args.all and not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / "manifest.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
