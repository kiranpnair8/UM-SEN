#!/usr/bin/env python3
"""Run a short CIFAR-10 UM-SEN mechanism test without editing model code."""

import argparse
import json
import math
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
CIFAR10_DIR = REPO_ROOT / "spikformer" / "cifar10"

if str(CIFAR10_DIR) not in sys.path:
    sys.path.insert(0, str(CIFAR10_DIR))

from model import Spikformer  # noqa: E402


CONFIGS = ("baseline", "umsen", "shuffled")


def parse_args():
    parser = argparse.ArgumentParser(description="Run a 5-epoch UM-SEN mechanism test.")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=CONFIGS)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results/umsen_mechanism_test"))
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
    parser.add_argument("--alpha-min", type=float, default=2.0)
    parser.add_argument("--alpha-max", type=float, default=6.0)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def resolve_repo_path(path):
    return path if path.is_absolute() else REPO_ROOT / path


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def build_loaders(args):
    from torchvision import datasets, transforms

    data_dir = resolve_repo_path(args.data_dir)
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
    train_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
    val_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=val_transform)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=generator,
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
    return Spikformer(
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


def lif_nodes(module):
    return [m for m in module.modules() if isinstance(m, MultiStepLIFNode)]


def set_surrogate_alpha(module, alpha):
    for node in lif_nodes(module):
        surrogate = getattr(node, "surrogate_function", None)
        if surrogate is not None and hasattr(surrogate, "alpha"):
            current = getattr(surrogate, "alpha")
            if torch.is_tensor(current):
                current.data.fill_(float(alpha))
            else:
                setattr(surrogate, "alpha", float(alpha))


class SpikeRateTracker:
    def __init__(self, model):
        self.total = 0.0
        self.count = 0
        self.handles = [
            node.register_forward_hook(self._hook)
            for node in lif_nodes(model)
        ]

    def _hook(self, module, inputs, output):
        if torch.is_tensor(output):
            self.total += float(output.detach().float().mean().cpu().item())
            self.count += 1

    def reset(self):
        self.total = 0.0
        self.count = 0

    def value(self):
        return self.total / self.count if self.count else 0.0

    def close(self):
        for handle in self.handles:
            handle.remove()


class BlockController:
    def __init__(self, block_id, beta, alpha_min, alpha_max, temperature):
        self.block_id = block_id
        self.beta = beta
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.temperature = temperature
        self.raw_dispersion = 0.0
        self.ema_dispersion = 0.5
        self.alpha = 4.0
        self.initialized = False

    def observe(self, attn):
        with torch.no_grad():
            probs = torch.softmax(attn.detach() / self.temperature, dim=-1)
            entropy = -(probs * (probs + 1e-12).log()).sum(dim=-1)
            num_tokens = attn.shape[-1]
            if num_tokens > 1:
                entropy = entropy / math.log(num_tokens)
            per_head_entropy = entropy.mean(dim=(0, 1, 3))
            raw = per_head_entropy.std(unbiased=False)
            raw_value = float(raw.detach().cpu().item())
            if self.initialized:
                self.ema_dispersion = self.beta * self.ema_dispersion + (1.0 - self.beta) * raw_value
            else:
                self.ema_dispersion = raw_value
                self.initialized = True
            self.raw_dispersion = raw_value
            normalized = min(max(self.ema_dispersion, 0.0), 1.0)
            self.alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * normalized


class UMSENMechanism:
    def __init__(self, model, mode, args):
        self.model = model
        self.mode = mode
        self.rng = random.Random(args.seed)
        self.controllers = [
            BlockController(
                block_id=i,
                beta=args.ema_beta,
                alpha_min=args.alpha_min,
                alpha_max=args.alpha_max,
                temperature=args.entropy_temperature,
            )
            for i, _ in enumerate(getattr(model, "block"))
        ]
        self.last_applied_alphas = [4.0 for _ in self.controllers]
        self.handles_installed = False
        if mode in ("umsen", "shuffled"):
            self.install_attention_hooks()
        self.apply_step_alphas()

    def install_attention_hooks(self):
        for idx, block in enumerate(getattr(self.model, "block")):
            attn_module = block.attn
            controller = self.controllers[idx]

            def record_attention(self_attn, attn, block_controller=controller):
                if self_attn.training:
                    block_controller.observe(attn)

            attn_module._record_attention_entropy = types.MethodType(record_attention, attn_module)
        self.handles_installed = True

    def current_controller_alphas(self):
        if self.mode == "baseline":
            return [4.0 for _ in self.controllers]
        return [controller.alpha for controller in self.controllers]

    def apply_step_alphas(self):
        alphas = self.current_controller_alphas()
        if self.mode == "shuffled" and len(alphas) > 1:
            alphas = list(alphas)
            self.rng.shuffle(alphas)
        for block, alpha in zip(getattr(self.model, "block"), alphas):
            set_surrogate_alpha(block, alpha)
        if self.mode == "baseline":
            set_surrogate_alpha(self.model.patch_embed, 4.0)
        self.last_applied_alphas = [float(alpha) for alpha in alphas]
        return self.last_applied_alphas

    def snapshot(self):
        return {
            str(controller.block_id): {
                "applied_alpha": self.last_applied_alphas[controller.block_id],
                "controller_alpha": controller.alpha,
                "raw_entropy_dispersion": controller.raw_dispersion,
                "ema_entropy_dispersion": controller.ema_dispersion,
            }
            for controller in self.controllers
        }


def mean_block_stats(records, key):
    if not records:
        return {}
    block_ids = sorted(records[0].keys(), key=int)
    summary = {}
    for block_id in block_ids:
        values = [record[block_id][key] for record in records]
        summary[block_id] = float(np.mean(values))
    return summary


def gradient_norm(model):
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            norm = parameter.grad.detach().data.norm(2)
            total += float(norm.item()) ** 2
    return math.sqrt(total)


def train_one_epoch(model, loader, criterion, optimizer, mechanism, spike_tracker, device):
    model.train()
    spike_tracker.reset()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    grad_norms = []
    block_records = []
    start_time = time.time()
    for images, targets in loader:
        mechanism.apply_step_alphas()
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        grad_norms.append(gradient_norm(model))
        optimizer.step()
        block_records.append(mechanism.snapshot())
        functional.reset_net(model)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_seen += batch_size
    return {
        "loss": total_loss / total_seen,
        "accuracy": total_correct / total_seen,
        "gradient_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
        "spike_rate": spike_tracker.value(),
        "epoch_time_seconds": time.time() - start_time,
        "alpha_per_block": mean_block_stats(block_records, "applied_alpha"),
        "controller_alpha_per_block": mean_block_stats(block_records, "controller_alpha"),
        "raw_entropy_dispersion_per_block": mean_block_stats(block_records, "raw_entropy_dispersion"),
        "ema_entropy_dispersion_per_block": mean_block_stats(block_records, "ema_entropy_dispersion"),
    }


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
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


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def resolved_config(args, device):
    return {
        "configs": args.configs,
        "epochs": args.epochs,
        "seed": args.seed,
        "output_dir": str(resolve_repo_path(args.output_dir)),
        "data_dir": str(resolve_repo_path(args.data_dir)),
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
        "entropy_temperature": args.entropy_temperature,
        "ema_beta": args.ema_beta,
        "alpha_min": args.alpha_min,
        "alpha_max": args.alpha_max,
        "fixed_baseline_alpha": 4.0,
        "device": str(device),
    }


def run_config(config_name, args, device):
    set_seed(args.seed)
    train_loader, val_loader = build_loaders(args)
    model = build_model(args).to(device)
    mechanism = UMSENMechanism(model, config_name, args)
    spike_tracker = SpikeRateTracker(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    records = []
    try:
        for epoch in range(args.epochs):
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, mechanism, spike_tracker, device
            )
            val_metrics = validate(model, val_loader, criterion, device)
            record = {
                "epoch": epoch,
                "train": train_metrics,
                "validation": val_metrics,
            }
            records.append(record)
            print(
                f"{config_name} epoch {epoch}: "
                f"train_acc={train_metrics['accuracy']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"grad_norm={train_metrics['gradient_norm']:.4f} "
                f"spike_rate={train_metrics['spike_rate']:.4f}",
                flush=True,
            )
    finally:
        spike_tracker.close()
    return records


def main():
    args = parse_args()
    output_dir = resolve_repo_path(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("UM-SEN mechanism test requires CUDA because the production model uses backend='cupy'.")

    set_seed(args.seed)
    config = resolved_config(args, device)
    write_json(output_dir / "config.json", config)

    all_results = {
        "config": config,
        "runs": {},
    }
    for config_name in args.configs:
        print(f"Starting configuration: {config_name}", flush=True)
        records = run_config(config_name, args, device)
        all_results["runs"][config_name] = records
        write_json(output_dir / f"{config_name}.json", {"config": config, "epochs": records})
        write_json(output_dir / "metrics.json", all_results)
    print(f"Wrote UM-SEN mechanism test results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
