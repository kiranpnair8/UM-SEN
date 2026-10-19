"""Reuse SAGE multi-batch probes for verified early/mid/late checkpoints."""

import argparse
import csv
import gc
import json
from pathlib import Path
import socket

import torch

from .checkpoint_stages import STAGES, args_dict, checkpoint_epoch, load_trusted_checkpoint, recipe_signature, validate_baseline
from .common import ROOT, build_model, load_weights, provenance, seed_all, write_json
from .make_probe_batch import iter_batches
from .multi_batch_intervene import batch_fingerprint, digest_file, run_batches
from .state_response import compare_states


def validate_manifest(manifest):
    if (set(manifest["states"]) != set(STAGES) or manifest["training_seed"] != 42 or
            manifest["origin"] not in ("historical", "regenerated")):
        raise ValueError("Need same-trajectory early/mid/late seed42 manifest")
    if (len({e["sha256"] for e in manifest["states"].values()}) != 3 or
            len({str(Path(e["path"]).resolve()) for e in manifest["states"].values()}) != 3):
        raise ValueError("Duplicate checkpoint supplied for multiple state labels")
    expected = [20, 150, manifest["resolved_training_epochs"]-1]
    if not expected[0] < expected[1] < expected[2]:
        raise ValueError("Stage epochs must be ordered early < mid < late")
    hashes, paths, trajectory_ids = [], [], []
    directory = Path(manifest["trajectory_directory"]).resolve()
    for label, epoch in zip(STAGES, expected):
        entry = manifest["states"][label]
        path = Path(entry["path"]).resolve()
        if entry["epoch"] != epoch or not path.is_relative_to(directory) or digest_file(path) != entry["sha256"]:
            raise ValueError("Checkpoint label/epoch/path/hash does not match this trajectory")
        checkpoint = load_trusted_checkpoint(path)
        args = args_dict(checkpoint.get("args"))
        if checkpoint_epoch(checkpoint) != epoch or recipe_signature(args) != recipe_signature(manifest["recipe"]):
            raise ValueError("Checkpoint epoch or embedded recipe mismatch")
        if manifest["origin"] == "regenerated" and args.get("trajectory_id") != manifest["trajectory_id"]:
            raise ValueError("Regenerated states must have exactly the same trajectory ID")
        if (checkpoint.get("sage_training_state") or {}).get("controller") is not None:
            raise ValueError("Primary state study requires Fixed-SG checkpoints")
        hashes.append(entry["sha256"])
        paths.append(str(path))
        trajectory_ids.append(args.get("trajectory_id"))
        del checkpoint
    if len(set(hashes)) != 3 or len(set(paths)) != 3:
        raise ValueError("Duplicate checkpoint supplied for multiple state labels")
    if len(set(trajectory_ids)) != 1:
        raise ValueError("Conflicting checkpoint trajectory IDs")


def run_states(manifest, batches, destination, settings, resume=False, model_factory=None):
    validate_manifest(manifest)
    batches = list(batches)
    identity = dict(manifest=manifest, settings=settings, batches=[dict(batch_index=b["batch_index"],
                    indices=b["indices"], fingerprint=batch_fingerprint(b)) for b in batches])
    identity = json.loads(json.dumps(identity))
    destination = Path(destination)
    if (destination / "metadata.json").exists():
        if not resume or json.loads((destination / "metadata.json").read_text()) != identity:
            raise ValueError("Existing state study requires --resume and identical states/batches/settings/code")
    else:
        if destination.exists() and any(destination.iterdir()):
            raise ValueError("Refusing nonempty state output without metadata")
        destination.mkdir(parents=True, exist_ok=True)
        write_json(destination / "metadata.json", identity)
    records = {}
    write_json(destination / "status.json", dict(status="running", completed_states=[]))
    try:
        for label in STAGES:
            entry = manifest["states"][label]
            print(f"{label.upper()} epoch={entry['epoch']} path={entry['path']} SHA256={entry['sha256']}", flush=True)
        for label in STAGES:
            model = None
            try:
                seed_all(settings["seed"])
                model = (model_factory or (lambda: build_model(manifest["recipe"], settings["backend"])))().to(settings["device"])
                checkpoint = load_weights(model, manifest["states"][label]["path"])
                if checkpoint["sha256"] != manifest["states"][label]["sha256"]:
                    raise ValueError("Checkpoint changed during study")
                metadata = dict(settings, checkpoint=checkpoint, config=manifest["recipe"],
                    protocol="SAGE same-trajectory state-response v1", state=label, trajectory_id=manifest["trajectory_id"],
                    checkpoint_epoch=manifest["states"][label]["epoch"], block_count=4,
                    shared_batch_identities=identity["batches"], mode="eval", amp=False)
                aggregate = run_batches(model, batches, destination / label, metadata, resume)
                fingerprints, reference = [], []
                for index in range(settings["num_batches"]):
                    path = destination / label / "batches" / f"batch_{index:03d}"
                    marker = json.loads((path / "complete.json").read_text())
                    fingerprints.append(marker["batch_fingerprint"])
                    data = json.loads((path / "measurements.json").read_text())
                    reference.append([b["G_l"] for b in data["baseline"]["blocks"]])
                if fingerprints != [b["fingerprint"] for b in identity["batches"]]:
                    raise ValueError("Probe batch identities differ between training states")
                records[label] = dict(aggregate=aggregate, batch_fingerprints=fingerprints, G_reference=reference)
                write_json(destination / "status.json", dict(status="running", completed_states=list(records)))
            finally:
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        epochs = {s: manifest["states"][s]["epoch"] for s in STAGES}
        result = compare_states(records, epochs, settings["bootstrap_samples"], settings["seed"], settings["zero_tolerance"])
        directory = destination / "aggregate"
        directory.mkdir(exist_ok=True)
        write_json(directory / "summary.json", result)
        for key in ("mean", "std", "ci95"):
            write_json(directory / ("state_response_" + key + ".json"),
                       {s: records[s]["aggregate"]["response_stats"][key] for s in STAGES})
        write_json(directory / "state_differences.json", result["differences"])
        write_json(directory / "state_structural_metrics.json", {s: records[s]["aggregate"]["structural_metrics"] for s in STAGES})
        with (directory / "state_raw_measurements.jsonl").open("w") as stream, \
                (directory / "state_raw_measurements.csv").open("w", newline="") as csvfile:
            writer = None
            for label in STAGES:
                for line in (destination / label / "aggregate/raw_measurements.jsonl").read_text().splitlines():
                    row = dict(json.loads(line), state=label, checkpoint_epoch=epochs[label])
                    if writer is None:
                        writer = csv.DictWriter(csvfile, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
        write_json(destination / "status.json", dict(status="complete", completed_states=list(STAGES)))
        return result
    except Exception as error:
        write_json(destination / "status.json", dict(status="failed", completed_states=list(records),
                   error_type=type(error).__name__, error=str(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="results/gradient_flow/alpha_state_dependence_seed42/probes")
    parser.add_argument("--num-batches", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", choices=("cupy", "torch"), default="cupy")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    source = provenance()
    if source.get("dirty") is not False:
        parser.error("State probes require a clean Git checkout")
    manifest = json.loads(args.manifest.read_text())
    validate_baseline(manifest["recipe"])
    namespace = ROOT / "results/gradient_flow"
    destination = (ROOT / args.output).resolve()
    if namespace.resolve() != namespace or not destination.is_relative_to(namespace) or destination == namespace:
        parser.error("State outputs must be inside results/gradient_flow/<new-study>")
    if args.backend == "cupy" and torch.device(args.device).type != "cuda":
        parser.error("CuPy requires CUDA; no backend fallback")
    seed_all(42)
    source["hostname"] = socket.gethostname()
    try:
        from importlib.metadata import version
        source["cupy"] = version("cupy-cuda12x")
    except Exception:
        try:
            import cupy
            source["cupy"] = cupy.__version__
        except ImportError:
            source["cupy"] = None
    settings = dict(source=source, num_batches=args.num_batches, batch_size=args.batch_size,
        seed=42, alpha_values=[2, 3, 4, 5, 6], reference_alpha=4., zero_tolerance=1e-8,
        bootstrap_samples=10000, backend=args.backend, device=args.device, dataset_path=str(Path(args.data_dir).resolve()),
        cuda_version=torch.version.cuda, gpu_name=torch.cuda.get_device_name(args.device) if torch.device(args.device).type == "cuda" else None)
    run_states(manifest, iter_batches(args.data_dir, manifest["recipe"], args.batch_size, args.num_batches), destination, settings, args.resume)


if __name__ == "__main__":
    main()
