"""Isolated CIFAR100 adaptation; never updates active CIFAR10 source or outputs."""

import argparse
import hashlib
from pathlib import Path
import signal
import subprocess

import torch
import yaml

from . import method_ablation as study
from .common import ROOT, load_config, provenance, write_json
from .controller_calibration import lock
from .intervene import restore_rng, rng_snapshot
from .method_policy import MethodRuntime
from .multi_batch_intervene import batch_fingerprint, digest_json
from .online_checkpoint import OnlineCheckpoint, digest


CONFIG = "configs/gradient_flow/cifar100_deep_only_followup.yml"
OUTPUT = "results/gradient_flow/cifar100_deep_only_followup"
APPROVAL = OUTPUT+"/approval.json"
PARENT_COMMIT = "7cd92a8d7402957083c855823dca906d5ac681b6"
CONDITIONS = ("fixed", "final_sage")
SEEDS = (42, 43, 44)
_AUTHORITY = object()


def declaration(data_dir=None):
    config = load_config(ROOT/CONFIG)
    baseline = load_config(ROOT/study.prior.BASE)
    expected = dict(native_cifar100_recipe_found=False,
        normalization="inherited_cifar10_explicit_review_required",
        recipe="inherited_released_cifar10_60_epoch_prefix", seeds=list(SEEDS),
        conditions=list(CONDITIONS), final_sage="deep_only")
    if (config.get("cifar100_audit") != expected or config["dataset"] != "torch/cifar100" or
            config["num_classes"] != 100 or config["output"] != OUTPUT or
            {k: v for k, v in config.items() if k not in ("dataset", "num_classes", "output", "cifar100_audit")} !=
            {k: v for k, v in baseline.items() if k not in ("dataset", "num_classes", "output")}):
        raise ValueError("CIFAR100 changed an unreviewed inherited recipe setting")
    if data_dir is not None:
        config["data_dir"] = data_dir
    return config


def parent_evidence(source_root):
    """Read completed method-ablation JSON/selection, never active CIFAR10 outputs."""
    source_root = Path(source_root).resolve()
    root = source_root/study.OUTPUT
    files = dict(selection=root/"final_selection.json", ablation_manifest=root/"study_manifest.json",
        ablation_summary=root/"summary.json", followup_config=source_root/"configs/gradient_flow/controller_followup.yml")
    selection = study.prior.read(files["selection"])
    if (selection.get("approved") is not True or selection.get("final_sage") != "deep_only" or
            not selection.get("reviewed_by") or selection.get("implementation_commit") != PARENT_COMMIT):
        raise ValueError("Need the original reviewed deep_only selection; do not rewrite it for CIFAR100")
    hashes = {name: digest(path) for name, path in files.items()}
    if any(hashes[key] != selection[key+"_sha256"] for key in ("ablation_manifest", "ablation_summary", "followup_config")):
        raise ValueError("Original selection evidence changed")
    git = ["git", "-c", "safe.directory="+ROOT.as_posix(), "-C", str(ROOT)]
    committed = subprocess.check_output(git+["show", PARENT_COMMIT+":configs/gradient_flow/controller_followup.yml"])
    if hashlib.sha256(committed).hexdigest() != hashes["followup_config"]:
        raise ValueError("Parent config differs from its launch-commit configuration")
    manifest = study.prior.read(files["ablation_manifest"])
    summary = study.prior.read(files["ablation_summary"])
    if ("deep_only" not in summary["ranking"]["eligible_in_descriptive_order"] or
            manifest["variant_configs"]["deep_only"]["online"]["method_design"]["parameters"] !=
            study.VARIANTS["deep_only"] or selection["deadband"] != manifest["deadband"]["value"]):
        raise ValueError("Parent method/threshold is not the reviewed deep_only design")
    return dict(source_root=str(source_root), selection_commit=PARENT_COMMIT,
        files_sha256=hashes, deadband=selection["deadband"], selected_method="deep_only",
        provisional=True, stabilization_target_met=summary["gates"]["deep_only"]["promising"])


def runtime_signature(source):
    return {key: source.get(key) for key in ("torch", "cuda", "packages")}


def prepare(args):
    if not args.source_root or not args.reviewed_by or not args.acknowledge_inherited_normalization:
        raise ValueError("Need --source-root, --reviewed-by and --acknowledge-inherited-normalization")
    source, config = provenance(), declaration(args.data_dir)
    if source.get("dirty") or not source.get("commit"):
        raise ValueError("Prepare from a clean committed CIFAR100 checkout")
    parent = parent_evidence(args.source_root)
    if Path(parent["source_root"]) == ROOT or source["commit"] == PARENT_COMMIT:
        raise ValueError("Use a separate CIFAR100 worktree; leave active CIFAR10 checkout unchanged")
    approval = dict(schema_version=1, implementation_commit=source["commit"], parent=parent,
        reviewed_by=args.reviewed_by, acknowledged_inherited_normalization=True,
        config_sha256=digest(ROOT/CONFIG), resolved_base=config, recipe_hash=digest_json(config),
        runtime=runtime_signature(source), seeds=list(SEEDS), conditions=list(CONDITIONS),
        note="Explicit inherited CIFAR10 adaptation, NOT official CIFAR100 reproduction; preparation launches no training.")
    root = ROOT/OUTPUT
    if root.resolve() != root:
        raise ValueError("Symlinked CIFAR100 output namespace refused")
    root.mkdir(parents=True, exist_ok=True)
    with lock(root/".approval.lock"):
        path = ROOT/APPROVAL
        if path.exists() and study.prior.read(path) != approval:
            raise ValueError("CIFAR100 approval differs; refusing overwrite")
        if not path.exists():
            write_json(path, approval)
    print(approval)


def read_approval():
    if not (ROOT/APPROVAL).is_file():
        raise ValueError("Prepare/review CIFAR100 explicitly before launching")
    record, source = study.prior.read(ROOT/APPROVAL), provenance()
    if (record["implementation_commit"] != source.get("commit") or source.get("dirty") or
            record["config_sha256"] != digest(ROOT/CONFIG) or
            record["resolved_base"] != declaration(record["resolved_base"]["data_dir"]) or
            record["recipe_hash"] != digest_json(record["resolved_base"]) or
            record["runtime"] != runtime_signature(source) or
            not record["acknowledged_inherited_normalization"] or
            parent_evidence(record["parent"]["source_root"]) != record["parent"]):
        raise ValueError("CIFAR100 approval/recipe/runtime/source mismatch")
    return record


def resolved_config(approval, seed):
    if seed not in SEEDS:
        raise ValueError("Unreviewed seed")
    config = study.resolve(approval["resolved_base"], "deep_only", approval["parent"]["deadband"])
    config["seed"], config["output"] = seed, f"{OUTPUT}/seed{seed}"
    return config


def identity_for(config, approval, condition, fingerprint):
    if condition not in CONDITIONS:
        raise ValueError("Unreviewed condition")
    return dict(source_commit=approval["implementation_commit"], method=condition,
        seed=config["seed"], dataset="torch/cifar100", synthetic=False,
        parent_selection_commit=approval["parent"]["selection_commit"],
        parent_selection_sha256=approval["parent"]["files_sha256"]["selection"],
        approval_sha256=digest(ROOT/APPROVAL), training_signature=digest_json(config),
        batch_fingerprint=fingerprint, runtime=approval["runtime"])


class Cifar100Checkpoint(OnlineCheckpoint):
    """Separately sealed authorization; shared/general resume guards stay unchanged."""
    def check(self, path=None):
        p = self.path if path is None else Path(path).absolute()
        expected = ROOT/OUTPUT/f"seed{self.identity['seed']}"/self.identity["method"]/"last.pth.tar"
        if (self._authority is not _AUTHORITY or self.identity["method"] not in CONDITIONS or
                self.identity["seed"] not in SEEDS or p != expected or p != self.path or
                p.resolve() != p or not p.is_file() or digest(p) != self.sha256):
            raise ValueError("Unvalidated/changed CIFAR100 checkpoint")


def authorize(directory, identity, config):
    approval = read_approval()
    expected_config = resolved_config(approval, identity["seed"])
    expected_identity = identity_for(expected_config, approval, identity["method"], identity["batch_fingerprint"])
    directory = Path(directory).absolute()
    if (directory != ROOT/OUTPUT/f"seed{identity['seed']}"/identity["method"] or
            directory.resolve() != directory or identity != expected_identity or config != expected_config):
        raise ValueError("Not the approved CIFAR100 checkpoint identity/config/path")
    if (study.prior.read(directory/"provenance.json")["identity"] != identity or
            study.prior.read(directory/"config.json") != config or
            study.prior.read(directory/"run_manifest.json") != dict(identity=identity, config=config)):
        raise ValueError("CIFAR100 checkpoint provenance/config mismatch")
    evidence = study.prior.read(directory/"checkpoint_integrity.json")
    rows = study.prior.read(directory/"metrics.json")
    epoch = evidence["epoch"]
    if (evidence["identity"] != identity or type(epoch) is not int or epoch < 0 or
            len(rows) != epoch+1 or [r["epoch"] for r in rows] != list(range(epoch+1))):
        raise ValueError("CIFAR100 checkpoint epoch/integrity mismatch")
    capability = Cifar100Checkpoint(directory/"last.pth.tar", evidence["sha256"], identity, epoch, _AUTHORITY)
    capability.check()
    return capability


def train(args):
    approval = read_approval()
    if not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; no condition outputs created")
    config = resolved_config(approval, args.seed)
    if args.data_dir and args.data_dir != config["data_dir"]:
        raise ValueError("Data directory differs from CIFAR100 approval")
    rng = rng_snapshot()
    try:
        from .make_probe_batch import iter_batches
        batch = next(iter_batches(config["data_dir"], config, config["online"]["probe_batch_size"], split="train"))
    finally:
        restore_rng(rng)
    if (batch["dataset"] != "cifar100" or batch["targets"].dtype != torch.long or
            int(batch["targets"].min()) < 0 or int(batch["targets"].max()) >= 100):
        raise ValueError("Wrong CIFAR100 probe dataset/targets")
    requested = args.condition
    identity = identity_for(config, approval, requested, batch_fingerprint(batch))
    root = ROOT/config["output"]
    if root.resolve() != root:
        raise ValueError("Symlinked CIFAR100 run root refused")
    root.mkdir(parents=True, exist_ok=True)
    directory = root/requested
    with lock(root/("."+requested+".lock")):
        peer = root/("final_sage" if requested == "fixed" else "fixed")/"provenance.json"
        if peer.is_file() and {**study.prior.read(peer)["identity"], "method": requested} != identity:
            raise ValueError("Paired CIFAR100 condition uses different data/probe/training provenance")
        checkpoint = None
        if directory.resolve() != directory:
            raise ValueError("Symlinked CIFAR100 condition refused")
        if directory.exists():
            if not args.resume:
                raise ValueError("Existing CIFAR100 run requires --resume; no restart")
            if study.prior.read(directory/"status.json").get("status") == "failed":
                raise ValueError("Failed CIFAR100 run requires manual review")
            args._checkpoint_authorization = authorize(directory, identity, config)
            checkpoint = args._checkpoint_authorization.load()
            horizon = config["online"]["epochs"]
            if checkpoint["epoch"]+1 > horizon:
                raise ValueError("CIFAR100 checkpoint exceeds declared horizon")
            if checkpoint["epoch"]+1 == horizon:
                _, _, rows = study.prior.checked_directory(directory, epochs=horizon)
                if rows != checkpoint["gradient_flow_online"]["rows"] or not (directory/"model_best.pth.tar").is_file():
                    raise ValueError("Completed CIFAR100 publication differs from checkpoint")
                print("Already complete, skipping:", directory)
                return
        else:
            directory.mkdir(exist_ok=False)
            write_json(directory/"config.json", config)
            write_json(directory/"provenance.json", dict(identity=identity, source=provenance(), parent=approval["parent"]))
            write_json(directory/"run_manifest.json", dict(identity=identity, config=config))
            (directory/"resolved_training.yml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        if yaml.safe_load((directory/"resolved_training.yml").read_text()) != config:
            raise ValueError("CIFAR100 resolved training config changed")
        args.config = (directory/"resolved_training.yml").relative_to(ROOT).as_posix()
        args.condition = "fixed" if requested == "fixed" else "response_aware"
        previous = signal.signal(signal.SIGUSR1, study.online_train.request_pause) if hasattr(signal, "SIGUSR1") else None
        try:
            write_json(directory/"status.json", dict(status="running", identity=identity))
            study.online_train.run_official(args, config, directory, batch, identity, checkpoint, None, runtime_class=MethodRuntime)
        except study.online_train.OnlinePause as error:
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
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--show-config", action="store_true")
    parser.add_argument("--source-root", help="Original CIFAR10 clone, READ ONLY")
    parser.add_argument("--reviewed-by")
    parser.add_argument("--acknowledge-inherited-normalization", action="store_true")
    parser.add_argument("--data-dir")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=42)
    parser.add_argument("--condition", choices=CONDITIONS, default="fixed")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.show_config:
        print(yaml.safe_dump(declaration(args.data_dir), sort_keys=False))
    elif args.prepare:
        prepare(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
