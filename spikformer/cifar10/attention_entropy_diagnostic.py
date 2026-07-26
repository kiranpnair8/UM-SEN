import argparse
import json
import math
import random
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional
from torch.utils.data import DataLoader

from model import Spikformer


ATTENTION_TEMPERATURES = (0.25, 0.5, 1.0, 2.0)
EPS = 1e-12


def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-10 Spikformer attention entropy diagnostic")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="./output/attention_entropy_diagnostic")
    parser.add_argument("--entropy-log-interval", type=int, default=1)
    parser.add_argument("--attention-top-k", type=int, default=5)
    parser.add_argument("--data-dir", type=str, default="./data")
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
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(args):
    from torchvision import datasets, transforms

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_set = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=train_transform)
    val_set = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=val_transform)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def build_model(args):
    model = Spikformer(
        img_size_h=32,
        img_size_w=32,
        patch_size=args.patch_size,
        in_channels=3,
        num_classes=10,
        embed_dims=args.dim,
        num_heads=args.num_heads,
        mlp_ratios=args.mlp_ratio,
        depths=args.layer,
        sr_ratios=1,
        T=args.time_step,
    )
    install_attention_metric_collectors(model, args.attention_top_k)
    model.set_attention_entropy_collection(True)
    return model


def accuracy(output, target):
    pred = output.argmax(dim=1)
    return (pred == target).float().mean().item()


def should_collect(batch_idx, interval):
    return interval > 0 and batch_idx % interval == 0


def train_one_epoch(model, loader, criterion, optimizer, device, entropy_log_interval):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        model.set_attention_entropy_collection(should_collect(batch_idx, entropy_log_interval))

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        functional.reset_net(model)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_seen += batch_size
    return {
        "loss": total_loss / total_seen,
        "accuracy": total_correct / total_seen,
    }


@torch.no_grad()
def validate(model, loader, criterion, device, entropy_log_interval):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        model.set_attention_entropy_collection(should_collect(batch_idx, entropy_log_interval))

        logits = model(images)
        loss = criterion(logits, targets)
        functional.reset_net(model)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_seen += batch_size
    return {
        "loss": total_loss / total_seen,
        "accuracy": total_correct / total_seen,
    }


def tensor_to_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def summarize_tensor(tensor):
    tensor = tensor.detach()
    return {
        "mean": tensor_to_float(tensor.mean()),
        "std": tensor_to_float(tensor.std(unbiased=False)),
        "min": tensor_to_float(tensor.min()),
        "max": tensor_to_float(tensor.max()),
    }


def summarize_stat_records(records):
    if not records:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "num_records": 0,
        }
    means = torch.tensor([tensor_to_float(record["mean"]) for record in records], dtype=torch.float64)
    stds = torch.tensor([tensor_to_float(record["std"]) for record in records], dtype=torch.float64)
    return {
        "mean": float(means.mean().item()),
        "std": float(stds.mean().item()),
        "min": min(tensor_to_float(record["min"]) for record in records),
        "max": max(tensor_to_float(record["max"]) for record in records),
        "num_records": len(records),
    }


def normalized_entropy(probs):
    num_tokens = probs.shape[-1]
    entropy = -(probs * (probs + EPS).log()).sum(dim=-1)
    if num_tokens > 1:
        entropy = entropy / math.log(num_tokens)
    return entropy


def per_head_summaries(values):
    return [
        {
            "head": head,
            **summarize_tensor(values[:, :, head, :]),
        }
        for head in range(values.shape[2])
    ]


def attention_metric_record(attn, top_k):
    raw_attn = attn.detach()
    num_tokens = raw_attn.shape[-1]
    k = min(top_k, num_tokens)
    record = {
        "raw_attention_scores": summarize_tensor(raw_attn),
        "temperature_entropy": {},
        "per_head_entropy": {},
        "attention_sparsity": {},
        "gini_impurity": {},
    }
    for temperature in ATTENTION_TEMPERATURES:
        temperature_key = str(temperature)
        probs = torch.softmax(raw_attn / temperature, dim=-1)
        entropy = normalized_entropy(probs)
        topk_mass = probs.topk(k=k, dim=-1).values.sum(dim=-1)
        gini = 1.0 - probs.square().sum(dim=-1)
        record["temperature_entropy"][temperature_key] = summarize_tensor(entropy)
        record["per_head_entropy"][temperature_key] = per_head_summaries(entropy)
        record["attention_sparsity"][temperature_key] = {
            "top_k": k,
            **summarize_tensor(topk_mass),
        }
        record["gini_impurity"][temperature_key] = summarize_tensor(gini)
    return record


def install_attention_metric_collectors(model, top_k):
    for block in getattr(model, "block"):
        attn_module = block.attn
        attn_module.attention_metric_stats = []

        def record_metrics(self, attn):
            if not self.collect_attention_entropy:
                return
            with torch.no_grad():
                self.attention_metric_stats.append(attention_metric_record(attn, top_k))

        attn_module._record_attention_entropy = types.MethodType(record_metrics, attn_module)


def clear_attention_metric_stats(model):
    for block in getattr(model, "block"):
        block.attn.attention_metric_stats.clear()


def aggregate_summary_records(records):
    if not records:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "num_records": 0,
        }
    means = torch.tensor([record["mean"] for record in records], dtype=torch.float64)
    stds = torch.tensor([record["std"] for record in records], dtype=torch.float64)
    return {
        "mean": float(means.mean().item()),
        "std": float(stds.mean().item()),
        "min": min(record["min"] for record in records),
        "max": max(record["max"] for record in records),
        "num_records": len(records),
    }


def aggregate_per_head_records(records, temperature):
    if not records:
        return []
    num_heads = len(records[0]["per_head_entropy"][temperature])
    return [
        {
            "head": head,
            **aggregate_summary_records([
                record["per_head_entropy"][temperature][head]
                for record in records
            ]),
        }
        for head in range(num_heads)
    ]


def aggregate_temperature_records(records, metric_name, temperature):
    if metric_name == "per_head_entropy":
        return aggregate_per_head_records(records, temperature)
    aggregated = aggregate_summary_records([
        record[metric_name][temperature]
        for record in records
    ])
    if metric_name == "attention_sparsity" and records:
        aggregated["top_k"] = records[0][metric_name][temperature]["top_k"]
    return aggregated


def aggregate_attention_metric_records(records):
    return {
        "raw_attention_scores": aggregate_summary_records([
            record["raw_attention_scores"]
            for record in records
        ]),
        "temperature_entropy": {
            str(temperature): aggregate_temperature_records(records, "temperature_entropy", str(temperature))
            for temperature in ATTENTION_TEMPERATURES
        },
        "per_head_entropy": {
            str(temperature): aggregate_temperature_records(records, "per_head_entropy", str(temperature))
            for temperature in ATTENTION_TEMPERATURES
        },
        "attention_sparsity": {
            str(temperature): aggregate_temperature_records(records, "attention_sparsity", str(temperature))
            for temperature in ATTENTION_TEMPERATURES
        },
        "gini_impurity": {
            str(temperature): aggregate_temperature_records(records, "gini_impurity", str(temperature))
            for temperature in ATTENTION_TEMPERATURES
        },
    }


def collect_attention_epoch_stats(model):
    return [
        {
            "block": i,
            **aggregate_attention_metric_records(block.attn.attention_metric_stats),
        }
        for i, block in enumerate(getattr(model, "block"))
    ]


def resolved_config(args, device):
    return {
        "epochs": args.epochs,
        "seed": args.seed,
        "output_dir": args.output_dir,
        "entropy_log_interval": args.entropy_log_interval,
        "attention_top_k": args.attention_top_k,
        "attention_temperatures": list(ATTENTION_TEMPERATURES),
        "data_dir": args.data_dir,
        "batch_size": args.batch_size,
        "val_batch_size": args.val_batch_size,
        "workers": args.workers,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "time_step": args.time_step,
        "layer": args.layer,
        "dim": args.dim,
        "num_heads": args.num_heads,
        "patch_size": args.patch_size,
        "mlp_ratio": args.mlp_ratio,
        "surrogate_alpha": 4,
        "backend": "cupy",
        "device": str(device),
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This diagnostic preserves backend='cupy' and requires CUDA/CuPy training.")

    output_dir = Path(args.output_dir)
    config = resolved_config(args, device)
    write_json(output_dir / "config.json", config)

    train_loader, val_loader = build_loaders(args)
    model = build_model(args).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    epoch_metrics = []
    for epoch in range(args.epochs):
        model.clear_attention_entropy_stats()
        clear_attention_metric_stats(model)
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, args.entropy_log_interval
        )
        val_metrics = validate(model, val_loader, criterion, device, args.entropy_log_interval)
        attention_stats = collect_attention_epoch_stats(model)
        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
            "attention": attention_stats,
        }
        epoch_metrics.append(epoch_record)
        write_json(output_dir / "metrics.json", {"config": config, "epochs": epoch_metrics})
        write_json(output_dir / f"epoch_{epoch:03d}.json", epoch_record)
        model.clear_attention_entropy_stats()
        clear_attention_metric_stats(model)


if __name__ == "__main__":
    main()
