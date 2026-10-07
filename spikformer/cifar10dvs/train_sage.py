#!/usr/bin/env python3
"""Isolated single-run DVS SAGE orchestration over the original DVS loops."""

import argparse
import copy
import itertools
import json
import os
from pathlib import Path
import random
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.append(str(HERE))
sys.path.append(str(HERE.parent / "cifar10"))
from training_args import get_args_parser


def parse_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(ROOT / "configs/sage_v2/sage_cifar10dvs_t10.yml"))
    preliminary, _ = pre.parse_known_args(argv)
    import yaml
    with open(preliminary.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    parser = get_args_parser()
    parser.add_argument("--config", default=preliminary.config)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-batches", type=int, default=4)
    parser.add_argument("--run-id", default="full")
    allowed = {action.dest for action in parser._actions}
    unknown = set(config) - allowed
    if unknown:
        parser.error(f"Unknown configuration fields: {sorted(unknown)}")
    parser.set_defaults(**config)
    args = parser.parse_args(argv)
    if args.dataset != "cifar10dvs" or args.model != "spikformer" or args.num_classes != 10:
        parser.error("This entry point supports only the existing CIFAR10-DVS Spikformer")
    if args.resume or args.test_only or args.start_epoch != 0 or args.world_size != 1 or args.sync_bn:
        parser.error("This entry point is fresh, single-GPU SAGE training only")
    if args.T_train is not None or args.T < 1 or args.batch_size < 2 or args.batch_size % 2:
        parser.error("Use full positive T and an even batch size >= 2 for the existing batch Mixup")
    if args.epochs <= args.cooldown_epochs or args.smoke_batches < 1:
        parser.error("Total epochs must exceed cooldown; smoke-batches must be positive")
    if args.mixup <= 0 or args.cutmix != 0 or args.cutmix_minmax is not None:
        parser.error("Preserve the DVS soft-target Mixup recipe; frame CutMix is unsupported")
    if args.sched != "cosine" or args.lr_cycle_limit != 1 or args.lr_cycle_mul != 1:
        parser.error("This entry point preserves the single-cycle cosine recipe")
    # Fixed controller settings, deliberately not experimental CLI knobs here.
    args.umsen_entropy_temperature = 0.25
    args.umsen_ema_beta = 0.95
    args.umsen_warmup_steps = 100
    args.umsen_alpha_min = 3.0
    args.umsen_alpha_max = 5.0
    return args


def output_path(args, root=ROOT):
    namespace = Path(root).resolve() / "results/sage_v2/cifar10dvs"
    if namespace.resolve() != namespace:
        raise ValueError("Outputs must remain in an unsymlinked SAGE-v2 namespace")
    base = Path(args.output_dir)
    if not base.is_absolute():
        base = Path(root) / base
    if not args.run_id or Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        raise ValueError("run-id must be one nonempty path component")
    mode = "smoke" if args.smoke_test else "full"
    destination = (base / "sage" / f"T{args.T}" / f"seed{args.seed}" / f"{mode}_{args.run_id}").resolve()
    if not destination.is_relative_to(namespace):
        raise ValueError("Outputs must remain under results/sage_v2/cifar10dvs")
    if destination.exists():
        raise ValueError(f"Refusing to overwrite an existing run: {destination}")
    return destination


def validate_frame_shape(shape, steps):
    if len(shape) != 5 or tuple(shape[1:]) != (steps, 2, 128, 128) or shape[0] < 1:
        raise ValueError(f"Expected [B,{steps},2,128,128], received {tuple(shape)}")


class LimitedBatches:
    """Smoke-only prefix; no replacement sampler or dataset."""
    def __init__(self, loader, count):
        self.loader, self.count = loader, min(len(loader), count)

    def __len__(self):
        return self.count

    def __iter__(self):
        return itertools.islice(self.loader, self.count)


def prepare_surrogates(model):
    from spikingjelly.clock_driven.neuron import MultiStepLIFNode
    from spikingjelly.clock_driven import surrogate
    from sage_controller import set_surrogate_alpha
    if len(model.block) != 2:
        raise ValueError("Expected the original two-block DVS backbone")
    # SpikingJelly constructor defaults may share a surrogate instance.
    # Copy its exact behavior before assigning independent per-block alphas.
    for node in model.modules():
        if isinstance(node, MultiStepLIFNode):
            if not isinstance(node.surrogate_function, surrogate.Sigmoid):
                raise ValueError("Expected the historical default Sigmoid surrogate")
            node.surrogate_function = copy.deepcopy(node.surrogate_function)
    set_surrogate_alpha(model.patch_embed, 4.0)
    for block in model.block:
        set_surrogate_alpha(block, 4.0)


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
    os.replace(temporary, path)


def main(argv=None):
    args = parse_args(argv)
    destination = output_path(args)
    import numpy as np
    import torch
    import train as legacy
    import model as dvs_model
    from sage_controller import UMSENController
    from experiment_state import source_provenance

    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("Actual DVS training requires a CUDA GPU and the production CuPy backend")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    train_set, test_set, _, _ = legacy.load_data(args.data_path, False, args.T)
    validate_frame_shape((1, *train_set[0][0].shape), args.T)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                             num_workers=args.workers, drop_last=True, pin_memory=True,
                                             persistent_workers=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                            num_workers=args.workers, drop_last=False, pin_memory=True,
                                            persistent_workers=False)
    if len(train_loader) == 0 or len(test_loader) == 0:
        raise ValueError("Empty training/test loader")
    if args.smoke_test:
        train_loader = LimitedBatches(train_loader, args.smoke_batches)
        test_loader = LimitedBatches(test_loader, args.smoke_batches)
    # Same factory/architecture, avoiding timm-only metadata unsupported by its constructor.
    model = dvs_model.spikformer(pretrained=False, drop_rate=0., drop_path_rate=0.1).to(device)
    prepare_surrogates(model)
    model.register_forward_pre_hook(lambda module, inputs: validate_frame_shape(inputs[0].shape, args.T))
    controller = UMSENController(model, args)
    optimizer = legacy.create_optimizer(args, model)
    schedule_args = copy.copy(args)
    schedule_args.epochs = args.epochs - args.cooldown_epochs
    scheduler, num_epochs = legacy.create_scheduler(schedule_args, optimizer)
    if num_epochs != args.epochs:
        raise ValueError(f"Installed timm resolved {num_epochs} epochs, expected {args.epochs}")
    scaler = legacy.amp.GradScaler() if args.amp else None
    train_loss_fn = legacy.SoftTargetCrossEntropy().to(device)
    test_loss_fn = torch.nn.CrossEntropyLoss()
    aug = legacy.transforms.Compose([legacy.transforms.RandomHorizontalFlip(p=0.5)])
    trivial_aug = legacy.autoaugment.SNNAugmentWide()
    mixup = legacy.Mixup(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
                        cutmix_minmax=args.cutmix_minmax, prob=args.mixup_prob,
                        switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                        label_smoothing=args.smoothing, num_classes=args.num_classes)
    destination.mkdir(parents=True, exist_ok=False)
    resolved = dict(vars(args), method="sage", total_epochs=num_epochs,
                    scheduler_epochs=schedule_args.epochs, smoke_test=args.smoke_test,
                    executed_epoch_limit=1 if args.smoke_test else num_epochs,
                    train_samples=len(train_set), test_samples=len(test_set),
                    output_dir=str(destination), source=[source_provenance(path) for path in
                        (HERE / "train_sage.py", HERE / "train.py", HERE / "model.py",
                         HERE / "training_args.py", HERE / "autoaugment.py", HERE / "utils.py",
                         HERE.parent / "cifar10/sage_controller.py", Path(args.config))],
                    architecture={"blocks": 2, "dimension": 256, "heads": 16,
                                  "input": [args.T, 2, 128, 128], "backend": "cupy"})
    write_json(destination / "config.json", resolved)
    history, best, best_epoch, best_acc5 = [], -float("inf"), None, None
    print(f"SAGE DVS outputs: {destination}", flush=True)
    for epoch in range(1 if args.smoke_test else num_epochs):
        controller.start_epoch(epoch)
        if epoch >= 75:
            mixup.mixup_enabled = False
        lr = float(optimizer.param_groups[0]["lr"])
        started = time.perf_counter()
        loss, accuracy, _ = legacy.train_one_epoch(model, train_loss_fn, optimizer, train_loader,
                                                   device, epoch, args.print_freq, scaler, args.T_train,
                                                   aug, trivial_aug, mixup)
        train_seconds = time.perf_counter() - started
        scheduler.step(epoch + 1)
        test_loss, test_accuracy, test_acc5 = legacy.evaluate(model, test_loss_fn, test_loader, device)
        improved = test_accuracy > best
        if improved:
            best, best_epoch, best_acc5 = float(test_accuracy), epoch, float(test_acc5)
        controller.epoch_record(epoch)
        history.append({"epoch": epoch, "train_loss": float(loss), "train_accuracy": float(accuracy),
                        "test_loss": float(test_loss), "test_accuracy": float(test_accuracy),
                        "best_test_accuracy": best, "lr": lr, "training_time_seconds": float(train_seconds),
                        "sage": copy.deepcopy(controller.history[-1])})
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "lr_scheduler": scheduler.state_dict(), "scaler": scaler.state_dict() if scaler else None,
                      "epoch": epoch, "args": vars(args), "max_test_acc1": best,
                      "test_acc5_at_max_test_acc1": best_acc5, "best_epoch": best_epoch,
                      "sage_controller": controller.state_dict()}
        for filename in (["last.pth", "model_best.pth"] if improved else ["last.pth"]):
            temporary = destination / (filename + ".tmp")
            torch.save(checkpoint, temporary)
            os.replace(temporary, destination / filename)
        write_json(destination / "metrics.json", {"method": "sage", "seed": args.seed, "T": args.T,
                   "smoke_test": args.smoke_test, "accuracy_unit": "percent", "epochs": history,
                   "best_test_accuracy": best, "best_epoch": best_epoch})
        controller.save_history(destination)
        print(json.dumps(history[-1], allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
