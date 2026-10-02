"""Checkpoint-adaptation infrastructure; no model or controller mathematics."""

import copy
from collections.abc import Mapping
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import subprocess
import warnings


LOGGER = logging.getLogger("train")
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path, seen=()):
    import yaml
    path = Path(path).resolve()
    if path in seen:
        raise ValueError("Cyclic base_config inheritance")
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    base = config.pop("base_config", None)
    if base:
        inherited, sources = load_config(path.parent / base, (*seen, path))
        inherited.update(config)
        return inherited, [*sources, str(path)]
    return config, [str(path)]


def validate_experiment_args(args):
    if getattr(args, "init_use_ema", False) and not args.init_checkpoint:
        raise ValueError("--init-use-ema requires --init-checkpoint")
    if args.init_checkpoint and (args.resume or args.initial_checkpoint):
        raise ValueError("--init-checkpoint cannot be combined with --resume or --initial-checkpoint")
    if args.init_checkpoint and args.start_epoch not in (None, 0):
        raise ValueError("Checkpoint initialization must start at adaptation epoch 0")
    if args.initial_checkpoint:
        raise ValueError("The historical trainer did not consume --initial-checkpoint; use --init-checkpoint")
    if not math.isfinite(args.fixed_alpha) or args.fixed_alpha <= 0:
        raise ValueError("--fixed-alpha must be finite and positive")
    if not math.isfinite(args.sps_alpha) or args.sps_alpha <= 0:
        raise ValueError("--sps-alpha must be finite and positive")
    if getattr(args, "umsen", False) and args.fixed_alpha != 4.0:
        raise ValueError("The unchanged SAGE controller requires a baseline alpha of 4.0")
    args.sage_v2 = args.sage_v2 or bool(args.init_checkpoint)
    if args.sage_v2:
        if not args.output or not args.experiment:
            raise ValueError("SAGE-v2 requires explicit --output and --experiment")
        destination = safe_output_path(args.output, args.experiment)
        if destination.exists() and any(destination.iterdir()) and not args.resume:
            raise ValueError("Refusing to overwrite an existing SAGE-v2 run; use a new run ID or resume")
        if args.resume and Path(args.resume).resolve().parent != destination:
            raise ValueError("SAGE-v2 resume checkpoint must belong to the requested run directory")
        if getattr(args, "distributed", False):
            raise ValueError("SAGE-v2 continuation currently supports single-process training")
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise ValueError("SAGE-v2 RNG continuation currently supports one process")
        if args.use_multi_epochs_loader and args.deterministic:
            raise ValueError("Paired deterministic mode requires ordinary epoch DataLoaders")


def safe_output_path(output, experiment, root=REPO_ROOT):
    namespace = (Path(root) / "results" / "sage_v2").resolve()
    base = Path(output).expanduser().resolve()
    path = (base / experiment).resolve()
    if not path.is_relative_to(namespace) or not base.is_relative_to(namespace) or path == namespace:
        raise ValueError("Adaptation output must stay within results/sage_v2 (including resolved symlinks)")
    return path


def load_trusted_checkpoint(path):
    import torch
    # Historical timm files contain argparse.Namespace, not just safe tensor objects.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def initialize_model(model, checkpoint, use_ema=False):
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must be a model-state mapping or a checkpoint dictionary")
    keys = ("state_dict_ema", "model_ema") if use_ema else ("state_dict", "model_state_dict", "model")
    selected = next((key for key in keys if checkpoint.get(key) is not None), None)
    if selected is None:
        reserved = {"epoch", "optimizer", "optimizer_state_dict", "metric", "metrics", "args",
                    "state_dict", "model_state_dict", "model", "state_dict_ema", "model_ema",
                    "version", "sage_training_state"}
        if use_ema or reserved.intersection(checkpoint):
            raise ValueError("No requested model weights found; expected " + " > ".join(keys))
        weights = checkpoint
        selected = "raw_state_dict"
    else:
        weights = checkpoint[selected]
    if not isinstance(weights, Mapping) or not weights or not all(isinstance(key, str) for key in weights):
        raise ValueError("Selected weight key is not a nonempty state dictionary: " + selected)
    cleaned = {key.removeprefix("module."): value for key, value in weights.items()}
    if len(cleaned) != len(weights):
        raise ValueError("Ambiguous module-prefix collision in checkpoint weights")
    weights = cleaned
    model.load_state_dict(weights, strict=True)
    LOGGER.info("Initialization model-weight key: %s (EMA explicitly requested: %s)", selected, use_ema)
    metadata = {key: checkpoint[key] for key in ("epoch", "metric") if key in checkpoint}
    return {"source_metadata": metadata, "start_epoch": 0, "best_metric": None,
            "controller_restored": False, "model_weight_key": selected, "use_ema": use_ema}


def apply_fixed_alpha(model, alpha, sps_alpha=4.0):
    from spikingjelly.clock_driven.neuron import MultiStepLIFNode
    sections = {"transformer": (model.block, alpha), "sps": ([model.patch_embed], sps_alpha)}
    counts = {}
    for scope, (modules, value) in sections.items():
        nodes = [node for module in modules for node in module.modules() if isinstance(node, MultiStepLIFNode)]
        if not nodes:
            raise ValueError("No LIF surrogates found in " + scope)
        for node in nodes:
            if not hasattr(node.surrogate_function, "alpha"):
                raise ValueError("LIF surrogate has no configurable alpha")
            node.surrogate_function.alpha = float(value)
        assert all(node.surrogate_function.alpha == value for node in nodes)
        counts[scope] = len(nodes)
        LOGGER.info("Fixed surrogate scope=%s; effective alpha=%s; LIF nodes=%d", scope, value, len(nodes))
    return counts


def configure_determinism(seed, rank=0):
    import numpy as np
    import torch
    effective_seed = seed + rank
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(effective_seed)
    np.random.seed(effective_seed % (2 ** 32))
    torch.manual_seed(effective_seed)
    torch.cuda.manual_seed_all(effective_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    settings = {"seed": effective_seed, "cudnn_benchmark": False,
                "cudnn_deterministic": True, "deterministic_algorithms": "warn_only",
                "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                "exact_determinism_guaranteed": False,
                "worker_seeding": "torch DataLoader seed -> Python/NumPy/Torch",
                "limitation": "Custom SpikingJelly/CuPy kernels are not audited by torch determinism checks"}
    LOGGER.warning("Paired reproducibility settings: %s", settings)
    return settings


def capture_rng(loaders=()):
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loaders": [loader.generator.get_state() for loader in loaders]}


def restore_rng(state, loaders=()):
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if len(state["loaders"]) != len(loaders):
        raise ValueError("Checkpoint DataLoader generator count does not match")
    for loader, generator_state in zip(loaders, state["loaders"]):
        loader.generator.set_state(generator_state)


def controller_for(model):
    return getattr(getattr(model, "module", model), "_sage_controller", None)


def restore_extra_state(checkpoint, scheduler, controller=None):
    state = checkpoint.get("sage_training_state")
    if state is None:
        warnings.warn("Legacy checkpoint lacks scheduler/best/controller/RNG continuation state; these cannot be recovered exactly")
        return {"best_metric": None, "best_epoch": None}
    if state.get("controller") is not None and controller is None:
        raise ValueError("SAGE resume requires train_umsen.py --umsen; use --init-checkpoint for a new Fixed branch")
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if controller is not None:
        if state.get("controller") is None:
            warnings.warn("Checkpoint lacks controller state; using a fresh controller")
        else:
            controller.load_state_dict(state["controller"])
    return state


def state_saver_class(base_class):
    class StateCheckpointSaver(base_class):
        def __init__(self, *args, scheduler=None, loaders=(), provenance=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.scheduler = scheduler
            self.loaders = loaders
            self.provenance = provenance

        def _save(self, save_path, epoch, metric=None):
            # Keep timm's complete/versioned format and enrich its temporary file
            # before timm links it as last, best, periodic, or recovery checkpoint.
            super()._save(save_path, epoch, metric)
            checkpoint = load_trusted_checkpoint(save_path)
            best_metric, best_epoch = self.best_metric, self.best_epoch
            if metric is not None and (best_metric is None or self.cmp(metric, best_metric)):
                best_metric, best_epoch = metric, epoch
            controller = controller_for(self.model)
            checkpoint["sage_training_state"] = {
                "version": 1, "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "controller": controller.state_dict() if controller else None,
                "best_metric": best_metric, "best_epoch": best_epoch,
                "rng": capture_rng(self.loaders), "provenance": self.provenance,
                "continuation_scope": "epoch-boundary; recovery files restart their epoch, not their minibatch"}
            import torch
            torch.save(checkpoint, save_path)
    return StateCheckpointSaver


def source_provenance(path):
    path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def write_provenance(args, output_dir, source=None, inherited=None):
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    record = {"schema_version": 1, "protocol": "experimental checkpoint adaptation",
              "git_commit": commit, "git_dirty": dirty, "resolved_args": vars(args),
              "method": "SAGE" if getattr(args, "umsen", False) else "Fixed-SG",
              "architecture": {"model": args.model, "blocks": args.layer, "dim": args.dim,
                               "heads": args.num_heads, "T": args.time_step},
              "output_path": str(Path(output_dir).resolve()),
              "initialization": source, "parent_run_provenance": inherited,
              "resume_checkpoint": str(Path(args.resume).resolve()) if args.resume else None}
    path = Path(output_dir) / "provenance.json"
    with path.open("w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    return record


def snapshot_fields(instance, fields):
    return {field: copy.deepcopy(getattr(instance, field)) for field in fields}


def restore_fields(instance, state, fields):
    missing = set(fields) - state.keys()
    if missing:
        raise ValueError("Incomplete controller state: " + ", ".join(sorted(missing)))
    for field in fields:
        setattr(instance, field, copy.deepcopy(state[field]))
