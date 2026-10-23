"""Exactly three predeclared method designs; historical studies are read-only."""

import argparse
import copy
import math
import os
from pathlib import Path
import signal

import torch
import yaml

from . import calibration_spec as prior
from . import online_train
from .amp_safety import DEFAULT_SAFETY
from .common import ROOT, load_config, provenance, write_json
from .controller_calibration import fixed_batch, lock
from .method_policy import MethodRuntime, VARIANTS
from .multi_batch_intervene import batch_fingerprint, digest_json
from .online_checkpoint import digest


CONFIG = "configs/gradient_flow/controller_method_ablation.yml"
OUTPUT = "results/gradient_flow/controller_method_ablation_seed42"
RULE = dict(source="current", statistic="positive_nonrefresh_absolute_error_quantile",
    quantile=.25, interpolation="linear", minimum_observations=16)


def declaration(data_dir=None):
    config = load_config(ROOT/CONFIG)
    base = load_config(ROOT/prior.BASE)
    if ({k: v for k, v in config.items() if k not in ("method_ablation", "output")} !=
            {k: v for k, v in base.items() if k != "output"} or config["output"] != OUTPUT or
            config["method_ablation"] != dict(calibration_source=prior.OUTPUT,
                variants=VARIANTS, deadband_rule=RULE, criteria=prior.CRITERIA)):
        raise ValueError("Method ablation must retain exactly the predeclared designs, rule and training protocol")
    if data_dir is not None:
        config["data_dir"] = data_dir
    return config


def resolve(config, variant, deadband):
    if variant not in VARIANTS or not math.isfinite(deadband) or deadband <= 0:
        raise ValueError("Invalid method variant or preregistered deadband")
    result = copy.deepcopy(config)
    p = VARIANTS[variant]
    result["online"]["controller"].update(eta=p["eta"], max_step=p["max_step"])
    result["online"]["method_design"] = dict(variant=variant, parameters=p.copy(), deadband=deadband)
    return result


def choose_deadband(rows):
    """A single global small-error scale; reads ONLY past training-probe errors."""
    values = []
    for row in rows:
        c = row["controller"]
        if not c["reference_refreshed"]:
            for error in c["error"]:
                value = abs(float(error))
                if not math.isfinite(value):
                    raise ValueError("Nonfinite historical training-probe error")
                if value > 0:
                    values.append(value)
    values.sort()
    if len(values) < RULE["minimum_observations"]:
        raise ValueError("Too few existing training errors to predeclare a deadband")
    position = RULE["quantile"]*(len(values)-1)
    low = math.floor(position)
    value = values[low]+(position-low)*(values[math.ceil(position)]-values[low])
    return dict(value=value, rule=RULE, observations=len(values), min=min(values), max=max(values),
        rationale="Lower quartile of positive non-refresh absolute CURRENT probe errors: suppress small errors without treating boundary zeros as noise. "
            "Empirical scale heuristic, not an estimated confidence interval; frozen once before all three runs. No validation metrics used.")


def references(config):
    """Read complete Fixed/CURRENT and the two completed global-gain controls."""
    calibration = prior.declaration(config["data_dir"])
    historical = prior.historical_sources(calibration)
    root = ROOT/prior.OUTPUT
    manifest = prior.read(root/"study_manifest.json")
    if (manifest["config"] != calibration or manifest["sources"] != historical or
            manifest["identity"]["synthetic"] or manifest["identity"]["source_commit"] not in
            {"15d510befc67125cd29d8689086499ad1fba2a6d", "7fb359804367c3b44bc9e60d35421c6bf4e1041d"}):
        raise ValueError("Completed calibration references do not match reviewed provenance")
    records, data = {}, {}
    for name in ("fixed", "current", "conservative", "very_conservative"):
        path = ROOT/historical[name]["path"] if name in historical else root/name
        identity, saved_config, rows = prior.checked_directory(path, require_epoch_provenance=name not in historical)
        if name not in historical and (saved_config != prior.resolve(calibration, name) or
                prior.training_invariants(identity) != prior.training_invariants(manifest["identity"]) or
                identity.get("variant") != name or identity.get("method") != "response_aware" or
                any(r["implementation_commit"] != identity["source_commit"] for r in rows)):
            raise ValueError("Completed global-gain reference mismatch: "+name)
        data[name] = rows
        records[name] = dict(path=path.relative_to(ROOT).as_posix(), identity=identity,
            files={n: digest(path/n) for n in ("metrics.json", "config.json", "status.json", "provenance.json")},
            legacy_epoch_provenance=historical.get(name, {}).get("legacy_epoch_provenance"), reused=True)
    return records, data


def run_identity(config, args, batch, source):
    normalized = prior.declaration(config["data_dir"])
    return dict(source_commit=source["commit"], method="response_aware", seed=config["seed"],
        dataset=config["dataset"], training_signature=prior.training_signature(normalized, args.backend, args.smoke, 2),
        study_config_hash=digest_json(config), batch_fingerprint=batch_fingerprint(batch),
        runtime={k: source.get(k) for k in ("torch", "cuda", "packages")},
        amp_safety={**DEFAULT_SAFETY, **config["online"].get("amp_safety", {})},
        synthetic=args.smoke, backend=args.backend, smoke_epochs=2 if args.smoke else None)


def register(config, args, root, batch, source):
    identity = run_identity(config, args, batch, source)
    sources, rows = ({}, {}) if args.smoke else references(config)
    if not args.smoke:
        old = sources["fixed"]["identity"]
        if any(old[k] != identity[k] for k in ("training_signature", "batch_fingerprint", "runtime", "synthetic")):
            raise ValueError("Method ablation training/probe/runtime differs from Fixed")
    deadband = dict(value=.01, rule="synthetic smoke only; NOT an empirical real deadband") if args.smoke else choose_deadband(rows["current"])
    payload = dict(schema_version=1, identity=identity, config=config, sources=sources, deadband=deadband,
        implementation_source=source, variant_configs={v: resolve(config, v, deadband["value"]) for v in VARIANTS},
        interpretation="Predeclared seed42 method design, no general superiority claim. Deadband fixed BEFORE GPU runs.")
    root.mkdir(parents=True, exist_ok=True)
    with lock(root/".preparation.lock"):
        path = root/"study_manifest.json"
        if path.exists():
            old = prior.read(path)
            if any(old[k] != payload[k] for k in ("identity", "config", "sources", "deadband", "variant_configs")):
                raise ValueError("Method preregistration differs; refusing rewrite")
        else:
            write_json(path, payload)
    print("Predeclared deadband:", deadband, flush=True)
    return payload


def checked_manifest(root, config, args, batch, source):
    manifest = prior.read(root/"study_manifest.json")
    if manifest["config"] != config or manifest["identity"] != run_identity(config, args, batch, source):
        raise ValueError("Method preparation/config/commit/runtime/probe mismatch")
    if not args.smoke:
        sources, rows = references(config)
        if manifest["sources"] != sources or manifest["deadband"] != choose_deadband(rows["current"]):
            raise ValueError("Historical evidence or frozen deadband changed")
    if manifest["variant_configs"] != {v: resolve(config, v, manifest["deadband"]["value"]) for v in VARIANTS}:
        raise ValueError("Variant declaration changed")
    return manifest


def validate_resume(destination, identity, config):
    root = Path(destination).parent
    variant = Path(destination).name
    namespace = ROOT/"results/gradient_flow"
    if (root.parent != namespace or not root.name.startswith("controller_method_ablation_") or
            (not identity.get("synthetic") and root != ROOT/OUTPUT) or variant not in VARIANTS):
        raise ValueError("Not an exact method-ablation resume namespace")
    manifest = prior.read(root/"study_manifest.json")
    expected = dict(manifest["identity"], variant=variant, deadband=manifest["deadband"]["value"])
    if (identity != expected or config != manifest["variant_configs"][variant] or
            config != resolve(manifest["config"], variant, identity["deadband"])):
        raise ValueError("Method resume identity/config/deadband mismatch")


def execute(args, config, directory, batch, identity):
    """Only this condition is written during training; shared manifest is read-only."""
    checkpoint = None
    if directory.resolve() != directory.absolute():
        raise ValueError("Symlinked condition refused")
    if directory.exists():
        if not args.resume or prior.read(directory/"provenance.json")["identity"] != identity or prior.read(directory/"config.json") != config:
            raise ValueError("Existing method run requires matching explicit resume")
        status = prior.read(directory/"status.json")
        if status.get("status") == "failed":
            raise ValueError("Failed method run requires manual review; no blind replay")
        from .online_checkpoint import authorize
        args._checkpoint_authorization = authorize(directory, identity, config, resume=True)
        checkpoint = online_train.load_checkpoint(directory/"last.pth.tar", args._checkpoint_authorization)
        horizon = 2 if args.smoke else config["online"]["epochs"]
        rows = checkpoint["gradient_flow_online"]["rows"]
        if len(rows) == horizon:
            prior.checked_directory(directory, epochs=horizon)
            if prior.read(directory/"metrics.json") != rows or not (directory/"model_best.pth.tar").is_file():
                raise ValueError("Completed publication differs from saved state")
            print("Already complete, skipping:", directory)
            return
        if len(rows) >= horizon:
            raise ValueError("Checkpoint exceeds horizon")
    else:
        directory.mkdir(exist_ok=False)
        write_json(directory/"config.json", config)
        write_json(directory/"provenance.json", dict(identity=identity, source=provenance(),
            slurm_job_id=os.environ.get("SLURM_JOB_ID"), design=config["online"].get("method_design")))
        if getattr(args, "_run_manifest", None) is not None:
            write_json(directory/"run_manifest.json", args._run_manifest)
        (directory/"resolved_training.yml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    path = directory/"resolved_training.yml"
    if yaml.safe_load(path.read_text()) != config:
        raise ValueError("Resolved official recipe differs")
    args.config = path.relative_to(ROOT).as_posix()
    previous = signal.signal(signal.SIGUSR1, online_train.request_pause) if hasattr(signal, "SIGUSR1") else None
    try:
        write_json(directory/"status.json", dict(status="running", identity=identity))
        runner = online_train.run_smoke if args.smoke else online_train.run_official
        runner(args, config, directory, batch, identity, checkpoint, None, runtime_class=MethodRuntime)
    except online_train.OnlinePause as error:
        write_json(directory/"status.json", dict(status="paused", identity=identity, reason=str(error)))
        raise SystemExit(75)
    except BaseException as error:
        write_json(directory/"status.json", dict(status="failed", identity=identity, error=repr(error)))
        raise
    finally:
        if previous is not None:
            signal.signal(signal.SIGUSR1, previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data-dir")
    parser.add_argument("--output", help="Synthetic smoke namespace only")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", choices=("torch", "cupy"), default="cupy")
    parser.add_argument("--expected-commit")
    args = parser.parse_args()
    args.smoke_epochs, args.condition = 2, "response_aware"
    if not args.prepare_only and args.variant is None:
        parser.error("Choose one of the three predeclared variants")
    config, source = declaration(args.data_dir), provenance()
    if args.expected_commit and args.expected_commit != source.get("commit"):
        parser.error("Unexpected source commit")
    if not args.smoke and (source.get("dirty") or not source.get("commit") or args.device != "cuda" or args.backend != "cupy"):
        parser.error("Real method ablation requires clean committed CUDA/CuPy protocol")
    root = (ROOT/(args.output or ("results/gradient_flow/controller_method_ablation_smoke_seed42" if args.smoke else OUTPUT))).absolute()
    if (root.resolve() != root or root.parent != ROOT/"results/gradient_flow" or
            not root.name.startswith("controller_method_ablation_") or (not args.smoke and root != ROOT/OUTPUT)):
        parser.error("Use the separate method-ablation namespace")
    batch = fixed_batch(args, config)
    if args.prepare_only or args.smoke:
        manifest = register(config, args, root, batch, source)
        if args.prepare_only:
            return
    else:
        manifest = checked_manifest(root, config, args, batch, source)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; refusing to create a failed training condition")
    identity = dict(manifest["identity"], variant=args.variant, deadband=manifest["deadband"]["value"])
    with lock(root/("."+args.variant+".lock")):
        execute(args, manifest["variant_configs"][args.variant], root/args.variant, batch, identity)


if __name__ == "__main__":
    main()
