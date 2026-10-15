"""Controlled alpha probes and finite-difference response matrices; never step an optimizer."""

import argparse
import copy
from contextlib import nullcontext
import hashlib
import math
import random

import numpy as np
import torch
from spikingjelly.clock_driven import functional

from .alpha_scope import BlockAlphas
from .common import ROOT, build_model, load_config, load_weights, make_scaler, output_directory, provenance, seed_all, write_json
from .gradient_monitor import GradientMonitor
from .step_adapter import backward_step


def rng_snapshot():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def intervention(model, images, targets, block_ids, alpha_values, reference_alpha=4., beta=.95,
                 amp=False, mode="train"):
    """Restore caller weights, BN, gradients, mode, neuron memory and RNG even on error."""
    scope = BlockAlphas(model)
    caller_alpha = scope.snapshot()
    trial_alpha = caller_alpha
    images, targets = images.detach(), targets.detach()
    original_weights = copy.deepcopy(model.state_dict())
    original_grads = [p.grad.detach().clone() if p.grad is not None else None for p in model.parameters()]
    original_modes = [(module, module.training) for module in model.modules()]
    memories = [(module, dict(module._memories)) for module in model.modules() if hasattr(module, "_memories")]
    rng = rng_snapshot()
    monitor = GradientMonitor(model, beta=beta)
    # A zero-LR optimizer is only an AMP unscale target; .step() is NEVER called.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.)

    def trial(block_id=None, alpha=None, reference=None):
        model.load_state_dict(original_weights)
        scope.restore(trial_alpha)
        if block_id is not None:
            scope.set(block_id, alpha)
        model.train(mode == "train")
        functional.reset_net(model)
        model.zero_grad(set_to_none=True)
        restore_rng(rng)
        monitor.references = [{} for _ in model.block]
        if reference is not None:
            monitor.load_state_dict(reference)
        scaler = make_scaler(images.device, amp)
        monitor.begin(0, 0, scaler.get_scale() if scaler is not None else 1.)
        with torch.cuda.amp.autocast() if amp else nullcontext():
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, targets)
        records = backward_step(loss, optimizer, monitor, scaler, update=False)
        return dict(target_block=block_id, alpha=alpha, loss=float(loss.detach().cpu()),
                    logits=logits.detach().cpu().tolist(), blocks=records)

    try:
        # Reference starts from exactly the requested checkpoint/block alpha state.
        # Use reference_alpha for all blocks, with SPS untouched, in this experiment.
        scope.set_all([reference_alpha] * len(scope.nodes))
        trial_alpha = scope.snapshot()
        baseline = trial()
        initial_reference = monitor.state_dict()
        trials = []
        for block_id in block_ids:
            for alpha in alpha_values:
                trials.append(trial(block_id, alpha, initial_reference))
        return dict(baseline=baseline, trials=trials, reference_alpha=reference_alpha,
                    alpha_values=list(alpha_values), block_ids=list(block_ids), amp=amp, mode=mode,
                    direction_reference="same baseline parameter-gradient reference reset before each trial",
                    optimizer_updates=0)
    finally:
        monitor.close()
        model.load_state_dict(original_weights)
        scope.restore(caller_alpha)
        for parameter, grad in zip(model.parameters(), original_grads):
            parameter.grad = grad
        for module, memory in memories:
            module._memories.clear()
            module._memories.update(memory)
        for module, training in original_modes:
            module.training = training
        restore_rng(rng)


def response_matrix(measurements):
    """Two-point secant, or central difference for symmetric endpoints; not a derivative at all states."""
    lower, upper = min(measurements["alpha_values"]), max(measurements["alpha_values"])
    if lower == upper:
        raise ValueError("Response matrix requires two distinct alpha values")
    matrix = []
    for observed in range(len(measurements["baseline"]["blocks"])):
        row = []
        for perturbed in measurements["block_ids"]:
            lo = next(t for t in measurements["trials"] if t["target_block"] == perturbed and t["alpha"] == lower)
            hi = next(t for t in measurements["trials"] if t["target_block"] == perturbed and t["alpha"] == upper)
            a, b = lo["blocks"][observed]["G_l"], hi["blocks"][observed]["G_l"]
            row.append((b - a) / (upper - lower) if a is not None and b is not None else None)
        matrix.append(row)
    return dict(rows_observed_blocks=list(range(len(matrix))), columns_perturbed_blocks=measurements["block_ids"],
                lower_alpha=lower, upper_alpha=upper, reference_alpha=measurements["reference_alpha"],
                estimator="endpoint finite difference (secant); local central difference only for a small symmetric sweep",
                matrix=matrix)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gradient_flow/cifar10_diagnostics.yml")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint")
    group.add_argument("--checkpoints", nargs="+", help="label=path pairs, e.g. early=... mid=... late=...")
    parser.add_argument("--batch", required=True, help="Trusted torch file containing images and hard targets")
    parser.add_argument("--block", type=int, default=0, help="Zero-based transformer block")
    parser.add_argument("--alphas", type=float, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--reference-alpha", type=float, default=4.)
    parser.add_argument("--matrix", action="store_true", help="Perturb each block independently")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", choices=("torch", "cupy"), default="cupy")
    parser.add_argument("--amp", action="store_true", help="Default FP32 probes avoid loss-scale confounds")
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    args = parser.parse_args()
    if args.checkpoints and any("=" not in item for item in args.checkpoints):
        parser.error("--checkpoints requires label=path entries")
    labels = {"checkpoint": args.checkpoint} if args.checkpoint else dict(item.split("=", 1) for item in args.checkpoints)
    if args.checkpoints and len(labels) != len(args.checkpoints):
        parser.error("Checkpoint labels must be unique")
    if any(not math.isfinite(alpha) or alpha <= 0 for alpha in args.alphas):
        parser.error("Alphas must be finite and positive")
    if args.matrix and len(set(args.alphas)) < 2:
        parser.error("Response matrix requires at least two distinct alpha values")
    if any(not label or "/" in label or "\\" in label or label in (".", "..") for label in labels):
        parser.error("Checkpoint labels must be single safe path components")
    device = torch.device(args.device)
    if device.type == "cpu" and args.backend != "torch":
        parser.error("CPU probes require --backend torch")
    config = load_config(ROOT / args.config)
    seed_all(config.get("seed", 42))
    model = build_model(config, args.backend).to(device)
    if not args.matrix and not 0 <= args.block < len(model.block):
        parser.error("Invalid zero-based transformer block")
    try:
        batch = torch.load(args.batch, map_location="cpu", weights_only=False)
    except TypeError:
        batch = torch.load(args.batch, map_location="cpu")
    images, targets = batch["images"].to(device), batch["targets"].to(device)
    if images.ndim != 4 or tuple(images.shape[1:]) != (3, config["img_size"], config["img_size"]):
        parser.error("Fixed minibatch shape does not match backbone configuration")
    if targets.ndim != 1 or targets.shape[0] != images.shape[0]:
        parser.error("Probes require one hard class label per sample")
    destination = output_directory(args.output)
    for label, checkpoint in labels.items():
        metadata = load_weights(model, checkpoint)
        seed_all(config.get("seed", 42))
        rows = intervention(model, images, targets, list(range(len(model.block))) if args.matrix else [args.block],
                            args.alphas, args.reference_alpha, config.get("diagnostic_beta", .95), args.amp, args.mode)
        directory = destination / label
        directory.mkdir()
        with open(args.batch, "rb") as stream:
            batch_sha256 = hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(stream.read()).hexdigest()
        write_json(directory / "measurements.json", dict(**rows, checkpoint=metadata, batch_path=args.batch,
                   batch_sha256=batch_sha256, loss="cross_entropy, fixed hard targets; no augmentation/Mixup during probes",
                   config=config, device=str(device), backend=args.backend, batch_shape=list(images.shape),
                   synthetic_batch=bool(batch.get("synthetic", False)), source=provenance()))
        if args.matrix:
            matrix = response_matrix(rows)
            write_json(directory / "response_matrix.json", matrix)
            from analysis.gradient_flow.plot_diagnostics import plot_response_matrix
            plot_response_matrix(matrix, directory / "response_matrix.png")
    print(destination)


if __name__ == "__main__":
    main()
