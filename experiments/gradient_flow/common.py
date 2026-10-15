"""Runtime helpers; the shared Spikformer source stays in its original location."""

import importlib.util
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CIFAR = ROOT / "spikformer/cifar10"


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_config(path):
    if str(CIFAR) not in sys.path:
        sys.path.insert(0, str(CIFAR))
    from experiment_state import load_config as inherited_config
    config = inherited_config(path)[0]
    # argparse normally casts these inherited YAML scientific-notation strings.
    for name in ("lr", "weight_decay", "min_lr", "warmup_lr"):
        if name in config:
            config[name] = float(config[name])
    return config


def build_model(config, backend="cupy"):
    if backend not in ("torch", "cupy"):
        raise ValueError("Unsupported backend")
    if str(CIFAR) not in sys.path:
        sys.path.insert(0, str(CIFAR))
    spec = importlib.util.spec_from_file_location("gradient_flow_cifar_backbone", CIFAR / "model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    node = module.MultiStepLIFNode
    def lif(*args, **kwargs):
        kwargs["backend"] = backend
        return node(*args, **kwargs)
    # Only this newly loaded module's constructor alias is changed. Production
    # model.py and other imported model modules retain their CuPy defaults.
    module.MultiStepLIFNode = lif
    try:
        model = module.Spikformer(img_size_h=config["img_size"], img_size_w=config["img_size"],
            patch_size=config["patch_size"], in_channels=3, num_classes=config["num_classes"],
            embed_dims=config["dim"], num_heads=config["num_heads"], mlp_ratios=config["mlp_ratio"],
            depths=config["layer"], sr_ratios=1, T=config["time_step"])
    finally:
        module.MultiStepLIFNode = node
    return model


def load_weights(model, path):
    if str(CIFAR) not in sys.path:
        sys.path.insert(0, str(CIFAR))
    from experiment_state import initialize_model, load_trusted_checkpoint, source_provenance
    checkpoint = load_trusted_checkpoint(path)
    metadata = initialize_model(model, checkpoint, use_ema=False)
    metadata.update(source_provenance(path))
    print("Checkpoint weight key:", metadata["model_weight_key"], flush=True)
    return metadata


def output_directory(path, kind="results"):
    namespace = ROOT / kind / "gradient_flow"
    if namespace.resolve() != namespace:
        raise ValueError("Symlinked runtime namespace is not allowed")
    path = Path(path)
    destination = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not destination.is_relative_to(namespace) or destination == namespace:
        raise ValueError(f"Outputs must be inside {kind}/gradient_flow/<run-id>")
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def write_json(path, payload):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def provenance():
    packages = {}
    for name in ("torchvision", "timm", "spikingjelly", "numpy", "PyYAML"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    try:
        git = ["git", "-c", "safe.directory=" + ROOT.as_posix(), "-C", str(ROOT)]
        return dict(commit=subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip(),
                    dirty=bool(subprocess.check_output(git + ["status", "--porcelain"], text=True).strip()),
                    torch=torch.__version__, cuda=torch.version.cuda, packages=packages)
    except (OSError, subprocess.CalledProcessError):
        return dict(commit=None, torch=torch.__version__, packages=packages)


def make_scaler(device, enabled):
    if not enabled:
        return None
    if device.type != "cuda":
        raise ValueError("Training/probe AMP currently requires CUDA; CPU smoke uses FP32")
    return torch.cuda.amp.GradScaler()


class LimitedLoader:
    def __init__(self, loader, limit):
        self.loader, self.limit = loader, min(len(loader), limit) if limit else len(loader)

    def __iter__(self):
        from itertools import islice
        return islice(self.loader, self.limit)

    def __len__(self):
        return self.limit

    def __getattr__(self, name):
        return getattr(self.loader, name)

    @property
    def mixup_enabled(self):
        return self.loader.mixup_enabled

    @mixup_enabled.setter
    def mixup_enabled(self, value):
        self.loader.mixup_enabled = value
