import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional
from torch.utils.data import DataLoader

from model import Spikformer


def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-10 Spikformer attention entropy diagnostic")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="./output/attention_entropy_diagnostic")
    parser.add_argument("--entropy-log-interval", type=int, default=1)
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


def collect_attention_epoch_stats(model):
    entropy_blocks = model.get_attention_entropy_stats()
    score_blocks = model.get_attention_score_stats()
    score_by_block = {block["block"]: block["stats"] for block in score_blocks}
    return [
        {
            "block": block["block"],
            "normalized_entropy": summarize_stat_records(block["stats"]),
            "raw_attention_scores": summarize_stat_records(score_by_block.get(block["block"], [])),
        }
        for block in entropy_blocks
    ]


def resolved_config(args, device):
    return {
        "epochs": args.epochs,
        "seed": args.seed,
        "output_dir": args.output_dir,
        "entropy_log_interval": args.entropy_log_interval,
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


if __name__ == "__main__":
    main()
