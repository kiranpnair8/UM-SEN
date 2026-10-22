"""Closed calibration declaration and read-only historical evidence checks."""

import copy
import json
from pathlib import Path

from .common import ROOT, load_config
from .multi_batch_intervene import digest_json
from .online_checkpoint import digest


CONFIG = "configs/gradient_flow/controller_calibration.yml"
BASE = "configs/gradient_flow/controller_online.yml"
ORIGINAL = "results/gradient_flow/controller_online_seed42"
OUTPUT = "results/gradient_flow/controller_calibration_seed42"
VARIANTS = {
    "current": dict(eta=.25, max_step=.25, reference_mode="frozen_window", reference_ema=.99),
    "conservative": dict(eta=.10, max_step=.10, reference_mode="frozen_window", reference_ema=.99),
    "very_conservative": dict(eta=.05, max_step=.05, reference_mode="frozen_window", reference_ema=.99),
    "anchored": dict(eta=.10, max_step=.10, reference_mode="ema_window", reference_ema=.90),
}
CRITERIA = dict(epochs=60, max_overhead_fraction=.20, min_variance_reduction=.25,
    max_final_accuracy_drop_pp=.50, strong_min_variance_reduction=.40,
    strong_max_accuracy_drop_pp=.25, tradeoff_accuracy_floor_pp=.10)
APPROVED_HISTORICAL = {
    "d8721251988691af90a7e01d5e4ff12cdd4571a8",
    "9a9c0a77773b8e962fe90f2558224033d1140c43",
    "dda6b31d05a1ab7b835ed1fc0061f8e4e54aa92e",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def declaration(data_dir=None):
    config = load_config(ROOT/CONFIG)
    declared = config["calibration"]
    if (declared["variants"] != VARIANTS or declared["criteria"] != CRITERIA or
            declared["original_study"] != ORIGINAL or config["output"] != OUTPUT):
        raise ValueError("Calibration must contain exactly the four predeclared variants/criteria")
    baseline = load_config(ROOT/BASE)
    inherited = {k: v for k, v in config.items() if k not in ("calibration", "output")}
    if inherited != {k: v for k, v in baseline.items() if k != "output"}:
        raise ValueError("Calibration changed inherited training/probe/safety settings")
    if data_dir is not None:
        config["data_dir"] = data_dir
    return config


def resolve(config, variant):
    if variant not in VARIANTS or config["calibration"]["variants"] != VARIANTS:
        raise ValueError("Not a predeclared calibration variant")
    result = copy.deepcopy(config)
    result["online"]["controller"].update(VARIANTS[variant])
    return result


def original_config(config):
    result = copy.deepcopy(config)
    result.pop("calibration")
    result["output"] = ORIGINAL
    result["online"]["controller"].update(VARIANTS["current"])
    return result


def training_signature(config, backend="cupy", synthetic=False, smoke_epochs=None):
    return digest_json(dict(config=original_config(config), backend=backend,
        synthetic=synthetic, smoke_epochs=smoke_epochs if synthetic else None))


def training_invariants(identity):
    return {k: identity[k] for k in ("source_commit", "training_signature", "batch_fingerprint",
        "runtime", "synthetic", "study_config_hash", "seed", "dataset", "amp_safety", "backend", "smoke_epochs")}


def checked_directory(path, *, complete=True, epochs=60):
    path = Path(path)
    if path.absolute() != path.resolve():
        raise ValueError("Symlinked evidence paths are not supported")
    identity = read(path/"provenance.json")["identity"]
    status, rows = read(path/"status.json"), read(path/"metrics.json")
    if (status.get("identity") != identity or status.get("epochs") != len(rows) or
            [r["epoch"] for r in rows] != list(range(len(rows))) or
            (complete and (status.get("status") != "complete" or len(rows) != epochs))):
        raise ValueError("Incomplete or inconsistent condition: "+str(path))
    if any(not isinstance(r.get("implementation_commit"), str) for r in rows):
        raise ValueError("Missing actual per-epoch implementation provenance")
    return identity, read(path/"config.json"), rows


def historical_sources(config, proposed=None):
    """JSON only: never deserialize, copy, update or open historical checkpoints."""
    source = ROOT/ORIGINAL
    expected = original_config(config)
    evidence, common = {}, None
    for label, condition in (("fixed", "fixed"), ("current", "response_aware")):
        path = source/condition
        identity, saved_config, rows = checked_directory(path)
        if (saved_config != expected or identity.get("method") != condition or identity.get("synthetic") or
                identity.get("source_commit") not in APPROVED_HISTORICAL or
                identity.get("training_signature") != training_signature(config) or
                any(r["implementation_commit"] not in APPROVED_HISTORICAL for r in rows)):
            raise ValueError("Historical result is not an exact compatible completed source: "+condition)
        neutral = {**identity, "method": None}
        if common is not None and common != neutral:
            raise ValueError("Historical Fixed and CURRENT provenance differ")
        common = neutral
        if proposed is not None and any(identity[k] != proposed[k] for k in
                ("training_signature", "batch_fingerprint", "runtime", "synthetic")):
            raise ValueError("Calibration training/runtime/probe differs from historical source")
        if label == "fixed" and any(r["controller"]["alpha"] != [4.]*config["layer"] for r in rows):
            raise ValueError("Historical Fixed alpha differs from 4")
        evidence[label] = dict(path=path.relative_to(ROOT).as_posix(), identity=identity,
            files={name: digest(path/name) for name in
                ("metrics.json", "config.json", "provenance.json", "status.json")},
            implementation_commits=sorted({r["implementation_commit"] for r in rows}),
            reused=True, epochs=len(rows), total_epoch_seconds=sum(r["epoch_seconds"] for r in rows))
    return evidence


def validate_resume_config(config, identity, destination):
    """Extra authorization required before issuing a calibration pickle capability."""
    variant = Path(destination).name
    declared = declaration(config["data_dir"])
    if (variant not in VARIANTS or variant == "current" or config != resolve(declared, variant) or
            identity.get("variant") != variant or identity.get("method") != "response_aware" or
            identity.get("study_config_hash") != digest_json(declared) or
            identity.get("training_signature") != training_signature(config,
                identity.get("backend", "cupy"), identity.get("synthetic", False), identity.get("smoke_epochs"))):
        raise ValueError("Invalid predeclared calibration resume identity/config")
    return True
