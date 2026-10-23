"""PREPARED follow-ups; explicit post-ablation approval is mandatory before launch."""

import argparse
import statistics

import torch

from . import method_ablation as study
from .common import ROOT, load_config, provenance, write_json
from .controller_calibration import lock
from .intervene import restore_rng, rng_snapshot
from .method_policy import VARIANTS
from .multi_batch_intervene import batch_fingerprint, digest_json


CONFIG = "configs/gradient_flow/controller_followup.yml"
SELECTION = study.OUTPUT+"/final_selection.json"


def read_selection():
    if not (ROOT/SELECTION).is_file():
        raise ValueError("Follow-ups are prepared only: review and explicitly approve FINAL_SAGE before launch")
    record = study.prior.read(ROOT/SELECTION)
    if (record.get("approved") is not True or record.get("final_sage") not in VARIANTS or
            not record.get("reviewed_by") or record["implementation_commit"] != provenance().get("commit") or
            record["ablation_manifest_sha256"] != study.digest(ROOT/study.OUTPUT/"study_manifest.json") or
            record["ablation_summary_sha256"] != study.digest(ROOT/study.OUTPUT/"summary.json") or
            record["followup_config_sha256"] != study.digest(ROOT/CONFIG)):
        raise ValueError("Missing, changed or unreviewed FINAL_SAGE selection; do not launch follow-ups")
    return record


def approve(args):
    if not args.final_sage or not args.reviewed_by:
        raise ValueError("Approval requires explicit --final-sage and --reviewed-by after scientific review")
    from analysis.gradient_flow.plot_controller_method_ablation import load_results
    _, summary = load_results(ROOT/study.OUTPUT)
    published = study.prior.read(ROOT/study.OUTPUT/"summary.json")
    if published != summary or args.final_sage not in summary["ranking"]["eligible_in_descriptive_order"]:
        raise ValueError("Finalize/review the ablation first; selected design must be mechanically stable and accuracy-eligible")
    source = provenance()
    if source.get("dirty"):
        raise ValueError("Approval requires a clean committed checkout")
    payload = dict(approved=True, final_sage=args.final_sage, reviewed_by=args.reviewed_by,
        implementation_commit=source["commit"], ablation_manifest_sha256=study.digest(ROOT/study.OUTPUT/"study_manifest.json"),
        ablation_summary_sha256=study.digest(ROOT/study.OUTPUT/"summary.json"),
        followup_config_sha256=study.digest(ROOT/CONFIG),
        approved_datasets=["cifar10", "cifar100"] if args.acknowledge_cifar100_recipe else ["cifar10"],
        deadband=summary["deadband"]["value"],
        note="Explicit user review, not automatic winner selection. CIFAR100 inherits the CIFAR10 recipe/normalization and is not a published CIFAR100 reproduction.")
    with lock(ROOT/study.OUTPUT/".selection.lock"):
        path = ROOT/SELECTION
        if path.exists() and study.prior.read(path) != payload:
            raise ValueError("Existing selection is immutable; manual provenance review required")
        if not path.exists():
            write_json(path, payload)
    print(payload)


def resolved_config(dataset, seed, selection, data_dir=None):
    config = load_config(ROOT/CONFIG)
    future = config["followup"]
    if (future["seeds"] != [42, 43, 44] or future["final_sage"] is not None or
            dataset not in future["datasets"] or seed not in future["seeds"]):
        raise ValueError("Unreviewed follow-up grid")
    config.update(future["datasets"][dataset])
    config["seed"] = seed
    config["online"]["epochs"] = future["epochs"]
    if data_dir is not None:
        config["data_dir"] = data_dir
    config["output"] = f"results/gradient_flow/controller_followup_{dataset}/seed{seed}"
    return study.resolve(config, selection["final_sage"], selection["deadband"])


def validate_resume(destination, identity, config):
    from pathlib import Path
    destination = Path(destination)
    dataset, seed = identity.get("followup_dataset"), identity.get("seed")
    expected = ROOT/f"results/gradient_flow/controller_followup_{dataset}/seed{seed}"/identity.get("followup_condition", "invalid")
    record = read_selection()
    if (dataset not in record["approved_datasets"] or destination != expected or destination.name not in ("fixed", "final_sage") or
            identity.get("selection_sha256") != study.digest(ROOT/SELECTION) or
            config != resolved_config(dataset, seed, record, config["data_dir"]) or
            study.prior.read(destination/"run_manifest.json") != dict(identity=identity, config=config)):
        raise ValueError("Follow-up resume provenance/approval/path mismatch")


def summarize(dataset):
    selection = read_selection()
    if dataset not in selection["approved_datasets"]:
        raise ValueError("Dataset has not been approved")
    runs = {m: [] for m in ("fixed", "final_sage")}
    for seed in (42, 43, 44):
        paired = []
        for method in runs:
            path = ROOT/f"results/gradient_flow/controller_followup_{dataset}/seed{seed}/{method}"
            identity, config, rows = study.prior.checked_directory(path)
            validate_resume(path, identity, config)
            if any(r["implementation_commit"] != selection["implementation_commit"] for r in rows):
                raise ValueError("Follow-up implementation mismatch")
            paired.append({k: v for k, v in identity.items() if k not in ("method", "followup_condition")})
            runs[method].append(dict(seed=seed, best_top1=max(r["validation"]["top1"] for r in rows),
                final_top1=rows[-1]["validation"]["top1"], training_seconds=sum(r["epoch_seconds"] for r in rows)))
        if paired[0] != paired[1]:
            raise ValueError("Fixed/FINAL_SAGE pairing mismatch")
    result = dict(selection=selection, runs=runs, aggregate={method: {key: dict(
        mean=statistics.mean(r[key] for r in rows), std=statistics.stdev(r[key] for r in rows))
        for key in ("best_top1", "final_top1", "training_seconds")} for method, rows in runs.items()},
        std_definition="sample standard deviation across three seeds")
    write_json(ROOT/f"results/gradient_flow/controller_followup_{dataset}/summary.json", result)
    print(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar10")
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    parser.add_argument("--condition", choices=("fixed", "final_sage"), default="fixed")
    parser.add_argument("--final-sage", choices=VARIANTS)
    parser.add_argument("--approve-selection", action="store_true")
    parser.add_argument("--reviewed-by")
    parser.add_argument("--acknowledge-cifar100-recipe", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--data-dir")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.approve_selection:
        approve(args)
        return
    selection = read_selection()  # Fail BEFORE constructing loaders or writing outputs.
    if args.final_sage and args.final_sage != selection["final_sage"]:
        parser.error("FINAL_SAGE differs from the reviewed selection")
    if args.dataset not in selection["approved_datasets"]:
        parser.error("Dataset recipe requires explicit review/approval")
    if args.summarize:
        summarize(args.dataset)
        return
    config = resolved_config(args.dataset, args.seed, selection, args.data_dir)
    if args.dry_run:
        print(dict(config=config, condition=args.condition, selected=selection["final_sage"], training_launched=False))
        return
    source = provenance()
    if source.get("dirty") or not torch.cuda.is_available():
        parser.error("Follow-up requires clean source and an accessible CUDA GPU")
    args.smoke, args.smoke_epochs, args.device, args.backend = False, 2, "cuda", "cupy"
    requested = args.condition
    args.condition = "fixed" if requested == "fixed" else "response_aware"
    rng = rng_snapshot()
    try:
        from .make_probe_batch import iter_batches
        from torchvision.datasets import CIFAR10, CIFAR100
        batch = next(iter_batches(config["data_dir"], config, config["online"]["probe_batch_size"], split="train",
            dataset_class=CIFAR10 if args.dataset == "cifar10" else CIFAR100))
    finally:
        restore_rng(rng)
    identity = dict(source_commit=source["commit"], method=requested, seed=args.seed,
        followup_dataset=args.dataset, followup_condition=requested, selection_sha256=study.digest(ROOT/SELECTION),
        training_signature=digest_json(config), batch_fingerprint=batch_fingerprint(batch),
        runtime={k: source.get(k) for k in ("torch", "cuda", "packages")}, synthetic=False)
    root = ROOT/config["output"]
    if root.resolve() != root.absolute() or root.parent.parent != ROOT/"results/gradient_flow":
        parser.error("Invalid follow-up namespace")
    root.mkdir(parents=True, exist_ok=True)
    directory = root/requested
    with lock(root/("."+requested+".lock")):
        payload = dict(identity=identity, config=config)
        if directory.exists() and study.prior.read(directory/"run_manifest.json") != payload:
            raise ValueError("Follow-up manifest differs")
        args._run_manifest = payload
        study.execute(args, config, directory, batch, identity)


if __name__ == "__main__":
    main()
