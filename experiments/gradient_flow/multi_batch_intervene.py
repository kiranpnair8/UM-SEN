"""SAGE: deterministic multi-batch, fixed-checkpoint alpha probes; no training."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import torch

from .common import ROOT, build_model, load_config, load_weights, provenance, seed_all, write_json
from .intervene import intervention, save_flat_records
from .make_probe_batch import iter_batches
from .multi_batch_stats import aggregate_measurements, validate_measurements

FILES = ("batch.pt", "measurements.json", "local_response_matrix.json",
         "raw_measurements.jsonl", "raw_measurements.csv")


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def batch_fingerprint(batch):
    digest = hashlib.sha256()
    digest.update(json.dumps(batch["indices"]).encode())
    for name in ("images", "targets"):
        tensor = batch[name].detach().cpu().contiguous()
        digest.update(str((name, tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def completed_batch(directory, signature, fingerprint, metadata):
    """Only a matching last-written completion marker plus validated files can skip work."""
    try:
        marker = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
        if marker.get("status") != "complete":
            return None
        if not all(isinstance(marker.get(key), str) for key in ("run_signature", "batch_fingerprint")):
            raise ValueError("completion marker missing protocol/input identity")
        if any(digest_file(directory / name) != marker["sha256"][name] for name in FILES):
            raise ValueError("artifact checksum mismatch")
        data = json.loads((directory / "measurements.json").read_text(encoding="utf-8"))
        matrix = validate_measurements(data, metadata["block_count"], metadata["alpha_values"], metadata["reference_alpha"])
        if matrix != json.loads((directory / "local_response_matrix.json").read_text(encoding="utf-8")):
            raise ValueError("saved local matrix differs from raw data")
    except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration) as error:
        print(f"Recompute {directory.name}: incomplete/corrupt artifacts ({error})", flush=True)
        return None
    if marker["run_signature"] != signature or marker["batch_fingerprint"] != fingerprint:
        raise ValueError("Validated batch belongs to different inputs/protocol; use a NEW output directory")
    return data


def rebuild_aggregate(destination, payloads, metadata):
    aggregate = aggregate_measurements(payloads, metadata["zero_tolerance"], metadata["bootstrap_samples"], metadata["seed"])
    directory = destination / "aggregate"
    directory.mkdir(exist_ok=True)
    write_json(directory / "summary.json", aggregate)
    for key in ("mean", "std", "median", "minimum", "maximum", "ci95"):
        write_json(directory / f"response_{key}.json", dict(block_ids=aggregate["block_ids"],
                   values=aggregate["response_stats"][key], ci_method=aggregate["ci_method"]))
    for key in ("response_sign_consistency", "structural_metrics", "alpha_curves"):
        write_json(directory / (key + ".json"), aggregate[key])
    with (directory / "raw_measurements.jsonl").open("w", encoding="utf-8") as jsonl, \
            (directory / "raw_measurements.csv").open("w", newline="", encoding="utf-8") as csvfile:
        writer = None
        for index in range(len(payloads)):
            source = destination / "batches" / f"batch_{index:03d}" / "raw_measurements.jsonl"
            for line in source.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if writer is None:
                    writer = csv.DictWriter(csvfile, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                jsonl.write(json.dumps(row, allow_nan=False) + "\n")
    return aggregate


def run_batches(model, batches, destination, metadata, resume=False, probe=None):
    """Shared orchestration for real input traversal and tiny synthetic correctness tests."""
    probe = intervention if probe is None else probe
    destination = Path(destination)
    metadata = json.loads(json.dumps(metadata))
    signature = digest_json(metadata)
    if (destination / "metadata.json").exists():
        previous = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
        if not resume or previous != metadata:
            raise ValueError("Existing run requires --resume and identical metadata/code/checkpoint/settings")
    else:
        if destination.exists() and any(destination.iterdir()):
            raise ValueError("Refusing nonempty output without run metadata")
        destination.mkdir(parents=True, exist_ok=True)
        write_json(destination / "metadata.json", metadata)
    count, started, fresh, durations, payloads = 0, time.perf_counter(), 0, [], []
    directory = None
    write_json(destination / "status.json", dict(status="running", run_signature=signature))
    try:
        for index, batch in enumerate(batches):
            if index >= metadata["num_batches"]:
                raise ValueError("Too many batches supplied")
            expected = list(range(index * metadata["batch_size"], (index + 1) * metadata["batch_size"]))
            if batch["batch_index"] != index or batch["indices"] != expected:
                raise ValueError("Batch indices must be sequential, deterministic and non-overlapping")
            images, targets = batch["images"], batch["targets"]
            size = metadata["config"]["img_size"]
            if (images.shape != (metadata["batch_size"], 3, size, size) or images.dtype != torch.float32 or
                    not bool(torch.isfinite(images).all()) or targets.shape != (metadata["batch_size"],) or
                    targets.dtype != torch.long or not bool(((targets >= 0) & (targets < metadata["config"]["num_classes"])).all())):
                raise ValueError("Invalid finite FP32 image / hard-label batch")
            fingerprint = batch_fingerprint(batch)
            directory = destination / "batches" / f"batch_{index:03d}"
            cached = completed_batch(directory, signature, fingerprint, metadata) if resume and directory.exists() else None
            if cached is not None:
                payloads.append(cached)
                print(f"batch {index + 1}/{metadata['num_batches']}: validated, skipped; elapsed {time.perf_counter()-started:.1f}s", flush=True)
            else:
                directory.mkdir(parents=True, exist_ok=True)
                write_json(directory / "complete.json", dict(status="running", run_signature=signature))
                print(f"batch {index + 1}/{metadata['num_batches']}: starting; elapsed {time.perf_counter()-started:.1f}s", flush=True)
                saved_batch = dict(batch, seed=metadata["seed"], source=metadata.get("source"))
                temporary = directory / "batch.pt.tmp"
                torch.save(saved_batch, temporary)
                temporary.replace(directory / "batch.pt")
                device = next(model.parameters()).device
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                tick = time.perf_counter()
                data = probe(model, images.to(device), targets.to(device), list(range(metadata["block_count"])),
                             metadata["alpha_values"], metadata["reference_alpha"], mode="eval", amp=False)
                matrix = validate_measurements(data, metadata["block_count"], metadata["alpha_values"], metadata["reference_alpha"])
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                duration = time.perf_counter() - tick
                durations.append(duration)
                fresh += 1
                data.update(batch_index=index, indices=batch["indices"], batch_fingerprint=fingerprint,
                            elapsed_seconds=duration, synthetic_batch=bool(batch.get("synthetic", False)))
                write_json(directory / "measurements.json", data)
                write_json(directory / "local_response_matrix.json", matrix)
                save_flat_records(directory, data, dict(checkpoint=metadata["checkpoint"]["path"],
                    batch_file=str(directory / "batch.pt"), batch_index=index,
                    backend=metadata["backend"], device=metadata["device"], mode="eval"))
                write_json(directory / "complete.json", dict(status="complete", run_signature=signature,
                    batch_fingerprint=fingerprint, elapsed_seconds=duration,
                    sha256={name: digest_file(directory / name) for name in FILES}))
                payloads.append(data)
                eta = sum(durations) / fresh * (metadata["num_batches"] - index - 1)
                print(f"batch {index + 1}/{metadata['num_batches']}: {duration:.1f}s; elapsed {time.perf_counter()-started:.1f}s; estimated remaining {eta:.1f}s", flush=True)
            count += 1
            write_json(destination / "status.json", dict(status="running", validated_batches=count,
                       expected_batches=metadata["num_batches"], run_signature=signature))
        if count != metadata["num_batches"]:
            raise ValueError("Too few batches supplied")
        aggregate = rebuild_aggregate(destination, payloads, metadata)
        write_json(destination / "status.json", dict(status="complete", validated_batches=count,
                   expected_batches=count, run_signature=signature, invocation_seconds=time.perf_counter()-started))
        return aggregate
    except Exception as error:
        failure = dict(status="failed", validated_batches=count, error_type=type(error).__name__,
                       error=str(error), run_signature=signature)
        write_json(destination / "status.json", failure)
        if directory is not None and (directory / "complete.json").exists():
            marker = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
            if marker.get("status") != "complete":
                write_json(directory / "failure.json", failure)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gradient_flow/cifar10_diagnostics.yml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--num-batches", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alphas", type=float, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--reference-alpha", type=float, default=4.)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", choices=("cupy", "torch"), default="cupy")
    parser.add_argument("--zero-tolerance", type=float, default=1e-8)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--output", default="results/gradient_flow/alpha_multibatch_seed42")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    import math
    if (args.num_batches < 1 or args.batch_size < 1 or args.bootstrap_samples < 100 or
            not math.isfinite(args.zero_tolerance) or args.zero_tolerance < 0 or
            len(set(args.alphas)) != len(args.alphas) or args.reference_alpha not in args.alphas or
            any(not math.isfinite(a) or a <= 0 for a in [args.reference_alpha, *args.alphas])):
        parser.error("Need positive batches/alphas, unique sweep including reference, valid tolerance and >=100 bootstrap draws")
    # Fail before touching output if no symmetric local finite difference exists.
    if not any(a < args.reference_alpha < b and math.isclose(args.reference_alpha-a, b-args.reference_alpha)
               for a in args.alphas for b in args.alphas):
        parser.error("Need a symmetric alpha pair around reference")
    device = torch.device(args.device)
    if device.type not in ("cuda", "cpu") or (args.backend == "cupy" and device.type != "cuda"):
        parser.error("CuPy requires CUDA; CPU requires explicit --backend torch")
    namespace = ROOT / "results/gradient_flow"
    destination = (ROOT / args.output).resolve()
    if namespace.resolve() != namespace or not destination.is_relative_to(namespace) or destination == namespace:
        parser.error("Output must be inside results/gradient_flow/<new-run>")
    source = provenance()
    if source.get("dirty") is not False:
        parser.error("Research probes require a clean Git checkout so resume cannot mix uncommitted implementations")
    config = load_config(ROOT / args.config)
    config["seed"] = args.seed
    seed_all(args.seed)
    model = build_model(config, args.backend).to(device)
    checkpoint = load_weights(model, args.checkpoint)
    if len(model.block) != 4:
        parser.error("This experiment requires four transformer blocks")
    metadata = dict(protocol="SAGE fixed-checkpoint multibatch alpha response v1", checkpoint=checkpoint,
        config=config, dataset_path=str(Path(args.data_dir).resolve()), dataset="cifar10", split="test",
        num_batches=args.num_batches, batch_size=args.batch_size, seed=args.seed, alpha_values=args.alphas,
        reference_alpha=args.reference_alpha, sps_alpha=4., block_count=len(model.block),
        mode="eval", amp=False, backend=args.backend, device=str(device), source=source,
        cuda_version=torch.version.cuda, gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        zero_tolerance=args.zero_tolerance, bootstrap_samples=args.bootstrap_samples)
    run_batches(model, iter_batches(args.data_dir, config, args.batch_size, args.num_batches),
                destination, metadata, args.resume)
    print("Completed:", destination, flush=True)


if __name__ == "__main__":
    main()
