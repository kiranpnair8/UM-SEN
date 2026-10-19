"""Audit real baseline artifacts, then prepare one same-trajectory stage manifest."""

import argparse
from argparse import Namespace
import csv
import gc
import json
from pathlib import Path
import subprocess
import sys
import uuid

import torch
import yaml

from .common import CIFAR, ROOT, provenance, write_json
from .multi_batch_intervene import digest_file, digest_json

sys.path.insert(0, str(CIFAR))
from experiment_state import load_trusted_checkpoint

STAGES = ("early", "mid", "late")
OPERATIONAL = {"output", "experiment", "resume", "initial_checkpoint", "init_checkpoint", "init_use_ema",
               "start_epoch", "rank", "world_size", "distributed", "device", "local_rank", "config_sources",
               "determinism_settings", "sage_v2", "milestone_checkpoints", "trajectory_id", "no_resume_opt"}
OPERATIONAL.update(("log_interval", "checkpoint_hist", "recovery_interval"))


def args_dict(value):
    return dict(value) if isinstance(value, dict) else vars(value) if hasattr(value, "__dict__") else {}


def checkpoint_epoch(checkpoint):
    epoch = checkpoint.get("epoch")
    if not isinstance(epoch, int):
        return None
    if checkpoint.get("version") == 2:
        return epoch
    if checkpoint.get("version") == 1:
        return epoch - 1  # legacy timm stored the next epoch
    return None  # Do not guess semantics for unversioned historical files.


def recipe_signature(args):
    # Compare the full recipe, not just optimizer/backbone headline values.
    return json.loads(json.dumps({key: value for key, value in args.items() if key not in OPERATIONAL}, sort_keys=True))


def scheduled_epochs(recipe):
    from timm.scheduler import create_scheduler
    required = ("epochs", "sched", "warmup_epochs", "min_lr", "warmup_lr", "cooldown_epochs")
    if any(key not in recipe for key in required):
        raise ValueError("Authoritative args lack scheduler settings; manual provenance review required")
    args = Namespace(**recipe)
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=float(recipe["lr"]))
    return int(create_scheduler(args, optimizer)[1])


def audit(directory):
    directory = Path(directory).resolve()
    report = dict(path=str(directory), available=directory.is_dir(), checkpoints=[], artifacts={})
    if not report["available"]:
        return report
    for name in ("args.yaml", "summary.csv", "provenance.json"):
        path = directory / name
        if path.is_file():
            report["artifacts"][name] = dict(path=str(path), sha256=digest_file(path))
    if (directory / "provenance.json").is_file():
        report["provenance"] = json.loads((directory / "provenance.json").read_text())
    if (directory / "args.yaml").is_file():
        report["args"] = yaml.safe_load((directory / "args.yaml").read_text())
    if (directory / "summary.csv").is_file():
        with (directory / "summary.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        report["summary"] = dict(rows=len(rows), first=rows[0] if rows else None, final=rows[-1] if rows else None,
                                 best=max(rows, key=lambda r: float(r.get("eval_top1", "-inf"))) if rows else None)
    paths = sorted(p for p in directory.rglob("*") if p.is_file() and
                   (p.name.endswith(".pth.tar") or p.suffix in (".pth", ".pt")))
    for path in paths:
        try:
            checkpoint = load_trusted_checkpoint(path)
            saved_args = args_dict(checkpoint.get("args"))
            extra = checkpoint.get("sage_training_state") or {}
            report["checkpoints"].append(dict(path=str(path), sha256=digest_file(path), epoch=checkpoint_epoch(checkpoint), raw_epoch=checkpoint.get("epoch"),
                metric=checkpoint.get("metric"), version=checkpoint.get("version"), args=saved_args,
                trajectory_id=saved_args.get("trajectory_id"), keys=sorted(checkpoint),
                scheduler_present=extra.get("scheduler") is not None, rng_present=extra.get("rng") is not None,
                controller_present=extra.get("controller") is not None,
                optimizer_present="optimizer" in checkpoint,
                scaler_keys=[k for k in ("amp_scaler", "loss_scaler") if k in checkpoint],
                best_metric=extra.get("best_metric"), best_epoch=extra.get("best_epoch"),
                model_keys=[k for k in ("state_dict", "model_state_dict", "model") if k in checkpoint]))
            del checkpoint
        except Exception as error:
            report["checkpoints"].append(dict(path=str(path), error=f"{type(error).__name__}: {error}"))
        gc.collect()
    return report


def validate_baseline(recipe):
    expected = dict(seed=42, time_step=4, layer=4, dim=384, num_heads=12, patch_size=4,
                    mlp_ratio=4, img_size=32, num_classes=10)
    if any(recipe.get(k) != v for k, v in expected.items()) or "cifar10" not in recipe.get("dataset", "").lower():
        raise ValueError("Historical args do not describe the requested seed42 CIFAR-10 backbone")
    if any(recipe.get(k, False) for k in ("umsen", "model_ema", "pretrained", "init_checkpoint", "initial_checkpoint")) or any(
            recipe.get(k, 4.) != 4. for k in ("fixed_alpha", "sps_alpha")):
        raise ValueError("Need the historical Fixed-SG non-EMA alpha4/SPS4 baseline")


def choose_stages(report, recipe, epochs, trajectory_id=None):
    result = {}
    for label, epoch in zip(STAGES, epochs):
        candidates = [c for c in report["checkpoints"] if "error" not in c and c.get("epoch") == epoch and
                      c.get("model_keys") and not c.get("controller_present") and
                      recipe_signature(c["args"]) == recipe_signature(recipe) and
                      (trajectory_id is None or c.get("trajectory_id") == trajectory_id) and
                      not Path(c["path"]).name.startswith(("recovery", "tmp"))]
        if candidates:
            candidates.sort(key=lambda c: (not Path(c["path"]).name.startswith("checkpoint_epoch"),
                                           Path(c["path"]).name == "model_best.pth.tar", c["path"]))
            result[label] = {k: candidates[0][k] for k in ("path", "sha256", "epoch", "metric")}
    if len(result) == 3 and len({c["sha256"] for c in result.values()}) != 3:
        raise ValueError("Different state labels must not use duplicate checkpoint content")
    if len(result) == 3:
        selected_ids = {c.get("trajectory_id") for c in report["checkpoints"]
                        if c.get("sha256") in {r["sha256"] for r in result.values()}}
        if len(selected_ids) != 1:
            raise ValueError("Selected checkpoints have conflicting embedded trajectory IDs")
    return result


def compare_historical(historical, regenerated):
    result = dict(status="comparison unavailable until both summaries/checkpoints exist",
                  exact_reproduction_claimed=False,
                  limitation="Regeneration is one NEW deterministic trajectory, never mixed with historical states; custom CUDA kernels are not guaranteed bitwise deterministic")
    if "summary" in historical and "summary" in regenerated:
        result.update(status="descriptive comparison", historical=historical["summary"], regenerated=regenerated["summary"])
        for which in ("final", "best"):
            a, b = historical["summary"].get(which), regenerated["summary"].get(which)
            if a and b and "eval_top1" in a and "eval_top1" in b:
                result[which + "_accuracy_difference"] = float(b["eval_top1"]) - float(a["eval_top1"])
    return result


def compare_final_checkpoints(historical, regenerated, recipe, final_epoch, data_dir):
    result = dict(status="target-final historical checkpoint unavailable", bitwise_equal=None)
    old = next((c for c in historical["checkpoints"] if c.get("epoch") == final_epoch and c.get("model_keys")), None)
    new = next((c for c in regenerated["checkpoints"] if c.get("epoch") == final_epoch and c.get("model_keys")), None)
    if old is None or new is None:
        return result
    from .common import build_model, load_weights
    a, b = load_trusted_checkpoint(old["path"]), load_trusted_checkpoint(new["path"])
    def weights(c):
        return {k.removeprefix("module."): v for k, v in next(c[key] for key in ("state_dict", "model_state_dict", "model") if key in c).items()}
    a, b = weights(a), weights(b)
    if a.keys() != b.keys():
        return dict(status="incompatible state-dict keys", missing=sorted(a.keys()-b.keys()), unexpected=sorted(b.keys()-a.keys()))
    result.update(status="measured checkpoint comparison", bitwise_equal=all(torch.equal(a[k], b[k]) for k in a),
                  maximum_absolute_weight_or_buffer_difference=max(float((a[k].double()-b[k].double()).abs().max()) for k in a),
                  historical=old["path"], regenerated=new["path"], epoch=final_epoch)
    del a, b
    if torch.cuda.is_available():
        from spikingjelly.clock_driven import functional
        from .make_probe_batch import make_batch
        batch = make_batch(data_dir, recipe, 32)
        model = build_model(recipe, "cupy").cuda().eval()
        predictions = []
        try:
            for checkpoint in (old, new):
                load_weights(model, checkpoint["path"])
                functional.reset_net(model)
                with torch.no_grad():
                    predictions.append(model(batch["images"].cuda()).cpu())
            result["fixed_test_batch"] = dict(indices=batch["indices"],
                maximum_logit_difference=float((predictions[0]-predictions[1]).abs().max()),
                prediction_agreement=float((predictions[0].argmax(1) == predictions[1].argmax(1)).float().mean()),
                loss=[float(torch.nn.functional.cross_entropy(p, batch["targets"])) for p in predictions])
        finally:
            functional.reset_net(model)
            del model
            torch.cuda.empty_cache()
    else:
        result["prediction_comparison"] = "not measured: CUDA unavailable"
    return result


def prepare(historical_dir, study_dir, data_dir, regenerate=False):
    study_dir = Path(study_dir)
    study_dir.mkdir(parents=True, exist_ok=True)
    report = audit(historical_dir)
    path = study_dir / ("checkpoint_inventory_latest.json" if (study_dir / "checkpoint_inventory.json").exists() else "checkpoint_inventory.json")
    write_json(path, report)
    if not report.get("args"):
        raise ValueError("Historical args.yaml unavailable: cannot verify recipe or authorize automatic regeneration")
    recipe = report["args"]
    validate_baseline(recipe)
    total = scheduled_epochs(recipe)
    epochs = [20, 150, total - 1]
    print(f"Audit: {len(report['checkpoints'])} checkpoint files; resolved training epochs={total}; target states={epochs}", flush=True)
    if not epochs[0] < epochs[1] < epochs[2]:
        raise ValueError("Resolved schedule does not permit early20/mid150/final stage comparison")
    source = provenance()
    if source.get("dirty") is not False:
        raise ValueError("Prepare/research requires a clean Git checkout")
    plan_path = study_dir / "trajectory_plan.json"
    identity = dict(source=source, args_sha256=report["artifacts"]["args.yaml"]["sha256"],
                    historical_path=report["path"], data_dir=str(Path(data_dir).resolve()), epochs=epochs)
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan["identity"] != identity:
            raise ValueError("Existing state study uses different code/args/data; choose a NEW study directory")
    else:
        found = choose_stages(report, recipe, epochs)
        ambiguous = [epoch for label, epoch in zip(STAGES, epochs) if label not in found and any(
            c.get("epoch") == epoch and c.get("model_keys") and
            not Path(c["path"]).name.startswith(("recovery", "tmp")) for c in report["checkpoints"])]
        if ambiguous:
            raise ValueError(f"Target epochs {ambiguous} exist but their recipe/provenance conflicts; manual review required before regeneration")
        if len(found) != 3 and not regenerate:
            raise ValueError("Missing historical milestones. Explicit --regenerate-missing required; no training launched")
        plan = dict(identity=identity, origin="historical" if len(found) == 3 else "regenerated",
                    trajectory_id=digest_json(identity) if len(found) == 3 else "fixed_seed42_"+uuid.uuid4().hex,
                    recipe=recipe,
                    historical_available_stages=found,
                    evidence=("same original run directory plus matching embedded recipe/seed; no cryptographic proof of historical data ordering"
                              if len(found) == 3 else "one NEW from-scratch run with unique persisted trajectory ID on all milestones; no historical states mixed"))
        write_json(plan_path, plan)
    print(f"Selected trajectory origin: {plan['origin']}; historical available states: {list(plan['historical_available_stages'])}", flush=True)
    if plan["origin"] == "historical":
        selected = choose_stages(report, recipe, epochs)
        if len(selected) != 3:
            raise ValueError("Previously selected historical trajectory is incomplete; refuse silent trajectory switch")
        if selected != plan["historical_available_stages"]:
            raise ValueError("Previously selected historical checkpoint identities changed; manual review required")
        selected_recipe = recipe
    else:
        if not regenerate:
            raise ValueError("Missing historical milestones. Explicit --regenerate-missing required; no training launched")
        selected_recipe = {k: v for k, v in recipe.items() if k not in OPERATIONAL}
        selected_recipe.update(data_dir=str(Path(data_dir).resolve()), output=str(study_dir / "trajectory"),
            experiment="fixed_seed42", deterministic=True, sage_v2=False, fixed_alpha=4., sps_alpha=4.,
            trajectory_id=plan["trajectory_id"], milestone_checkpoints=epochs)
        recipe_path = study_dir / "regeneration.yml"
        recipe_path.write_text(yaml.safe_dump(selected_recipe), encoding="utf-8")
        directory = study_dir / "trajectory/fixed_seed42"
        current = audit(directory)
        selected = choose_stages(current, selected_recipe, epochs, plan["trajectory_id"])
        if len(selected) != 3:
            last = directory / "last.pth.tar"
            resume = []
            if last.exists():
                checkpoint = load_trusted_checkpoint(last)
                if args_dict(checkpoint.get("args")).get("trajectory_id") != plan["trajectory_id"]:
                    raise ValueError("Regeneration resume belongs to another trajectory")
                past_missing = [e for label, e in zip(STAGES, epochs) if e <= checkpoint["epoch"] and label not in selected]
                if past_missing:
                    raise ValueError(f"Past milestones missing {past_missing}; cannot reconstruct from later weights. Use a new study directory")
                resume = ["--resume", str(last)]
                del checkpoint
            elif directory.exists() and any(directory.iterdir()):
                raise ValueError("Nonempty regeneration directory without last checkpoint; review manually")
            command = [sys.executable, "-u", "-m", "experiments.gradient_flow.train_milestones", "-c", str(recipe_path), *resume]
            print("Regenerating one NEW trajectory:", " ".join(command), flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            current = audit(directory)
            selected = choose_stages(current, selected_recipe, epochs, plan["trajectory_id"])
        if len(selected) != 3:
            raise RuntimeError("Trajectory incomplete; milestones retained. Resubmit to resume; no state probes launched")
        validation = compare_historical(report, current)
        validation["recipe_differences"] = {key: [recipe.get(key), selected_recipe.get(key)]
            for key in sorted(set(recipe) | set(selected_recipe)) if recipe.get(key) != selected_recipe.get(key)}
        validation["final_checkpoint_comparison"] = compare_final_checkpoints(report, current, selected_recipe, epochs[-1], data_dir)
        write_json(study_dir / "trajectory_validation.json", validation)
    manifest = dict(schema_version=1, trajectory_id=plan["trajectory_id"], origin=plan["origin"],
                    trajectory_evidence=plan["evidence"], states=selected, recipe=selected_recipe,
                    trajectory_directory=report["path"] if plan["origin"] == "historical" else str(directory.resolve()),
                    training_seed=42, resolved_training_epochs=total, source=source)
    write_json(study_dir / "states.json", manifest)
    for label in STAGES:
        c = selected[label]
        print(f"{label.upper()} epoch={c['epoch']} path={c['path']} SHA256={c['sha256']}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-dir", default="results/umsen_official_recipe/baseline_seed42")
    parser.add_argument("--study-dir", default="results/gradient_flow/alpha_state_dependence_seed42")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--regenerate-missing", action="store_true")
    args = parser.parse_args()
    destination = (ROOT / args.study_dir).resolve()
    namespace = ROOT / "results/gradient_flow"
    if namespace.resolve() != namespace or not destination.is_relative_to(namespace) or destination == namespace:
        parser.error("Study output must be a new results/gradient_flow/<study> namespace")
    prepare(args.historical_dir, destination, args.data_dir, args.regenerate_missing)


if __name__ == "__main__":
    main()
