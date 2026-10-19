"""Save the first deterministic real CIFAR-10 evaluation minibatch (no augmentation)."""

import argparse
from pathlib import Path

import torch
from torchvision.datasets import CIFAR10

from .common import ROOT, load_config, provenance, seed_all


def make_batch(data_dir, config, batch_size=32):
    # Same transform factory and eval arguments as the official CIFAR loader;
    # CPU normalization replaces the CUDA prefetcher's equivalent normalization.
    from transforms_factory import create_transform
    transform = create_transform(input_size=(3, config["img_size"], config["img_size"]),
        is_training=False, use_prefetcher=False, interpolation=config.get("interpolation", "bicubic"),
        mean=config["mean"], std=config["std"], crop_pct=config.get("crop_pct", 1.))
    dataset = CIFAR10(data_dir, train=False, download=False, transform=transform)
    if not 1 <= batch_size <= len(dataset):
        raise ValueError("Batch size must be within the test dataset size")
    images, targets = next(iter(torch.utils.data.DataLoader(dataset, batch_size=batch_size,
        shuffle=False, num_workers=0, drop_last=False)))
    return dict(images=images.float(), targets=targets.long(), synthetic=False,
                dataset="cifar10", split="test", indices=list(range(batch_size)),
                preprocessing=dict(mean=config["mean"], std=config["std"],
                    interpolation=config.get("interpolation", "bicubic"), crop_pct=config.get("crop_pct", 1.)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gradient_flow/cifar10_diagnostics.yml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/gradient_flow/probe_batches/cifar10_eval_bs32_seed42.pt")
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    seed_all(args.seed)
    batch = make_batch(args.data_dir, config, args.batch_size)
    batch.update(seed=args.seed, source=provenance())
    path = Path(args.output)
    if not path.is_absolute():
        path = ROOT / path
    if not path.resolve().is_relative_to((ROOT / "results/gradient_flow").resolve()):
        parser.error("Probe data must be saved under results/gradient_flow/")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save(batch, stream)
    print("Saved:", path)
    for key in ("images", "targets"):
        print(key, "shape:", tuple(batch[key].shape), "dtype:", batch[key].dtype)
    print("Image min/max:", float(batch["images"].min()), float(batch["images"].max()))
    print("First target IDs:", batch["targets"][:10].tolist())


if __name__ == "__main__":
    main()
