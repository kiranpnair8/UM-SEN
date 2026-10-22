"""Predeclared four-condition calibration; CURRENT is a read-only reference."""

import argparse
from contextlib import contextmanager
import gc
import os
from pathlib import Path
import signal

import torch
import yaml

from . import online_train
from .calibration_spec import (ORIGINAL, OUTPUT, VARIANTS, checked_directory,
    declaration, historical_sources, read, resolve, training_signature)
from .common import ROOT, provenance, write_json
from .intervene import restore_rng, rng_snapshot
from .multi_batch_intervene import batch_fingerprint, digest_json
from .amp_safety import DEFAULT_SAFETY


@contextmanager
def lock(path):
    """Nonblocking locks protect CLI and SLURM writers equally, on each variant."""
    path = Path(path).absolute()
    if path.resolve() != path:
        raise ValueError("Symlinked calibration lock refused")
    with Path(path).open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            stream.seek(0, 2)
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise ValueError("Another calibration writer holds "+str(path)) from error
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise ValueError("Another calibration writer holds "+str(path)) from error
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def output_root(args):
    path = (ROOT/(args.output or ("results/gradient_flow/controller_calibration_smoke_seed42" if args.smoke else OUTPUT))).absolute()
    namespace = ROOT/"results/gradient_flow"
    if (path.resolve() != path or namespace.resolve() != namespace or path.parent != namespace or
            not path.name.startswith("controller_calibration_") or
            (not args.smoke and path != ROOT/OUTPUT) or path == ROOT/ORIGINAL):
        raise ValueError("Use the separate calibration namespace, never the original online study")
    return path


def fixed_batch(args, config):
    rng = rng_snapshot()
    try:
        if args.smoke:
            generator = torch.Generator().manual_seed(420)
            return dict(images=torch.randn(2, 3, 8, 8, generator=generator),
                targets=torch.tensor([1, 2]), indices=[0, 1], synthetic=True)
        from .make_probe_batch import iter_batches
        return next(iter_batches(config["data_dir"], config, config["online"]["probe_batch_size"], split="train"))
    finally:
        restore_rng(rng)


def register(args, config, root, batch):
    source = provenance()
    if args.expected_commit and source.get("commit") != args.expected_commit:
        raise ValueError("Unexpected implementation commit")
    if not args.smoke and (source.get("dirty") or not source.get("commit")):
        raise ValueError("Real calibration requires a clean committed checkout")
    if not args.smoke and (args.device != "cuda" or args.backend != "cupy"):
        raise ValueError("Real calibration requires the unchanged CUDA/CuPy protocol")
    identity = dict(source_commit=source.get("commit"), method="response_aware",
        training_signature=training_signature(config, args.backend, args.smoke, args.smoke_epochs),
        batch_fingerprint=batch_fingerprint(batch),
        runtime={k: source.get(k) for k in ("torch", "cuda", "packages")}, synthetic=args.smoke,
        study_config_hash=digest_json(config), seed=42, dataset=config["dataset"],
        amp_safety={**DEFAULT_SAFETY, **config["online"].get("amp_safety", {})},
        backend=args.backend, smoke_epochs=args.smoke_epochs if args.smoke else None)
    # All evidence is checked BEFORE creating calibration directories; sources
    # are opened read-only and no checkpoint in the original study is loaded.
    sources = {} if args.smoke else historical_sources(config, identity)
    manifest = dict(schema_version=1, identity=identity, source=source, config=config,
        variants=VARIANTS, sources=sources,
        caveat="Seed42 calibration only; no automatic policy selection or publication claim.")
    root.mkdir(parents=True, exist_ok=True)
    with lock(root/".registration.lock"):
        path = root/"study_manifest.json"
        if path.exists():
            # Dirty status itself is not a scientific identity field in smoke.
            previous = read(path)
            if any(previous[k] != manifest[k] for k in ("identity", "config", "variants", "sources")):
                raise ValueError("Calibration preregistration/provenance mismatch; refusing mutation")
        else:
            write_json(path, manifest)
        if sources:
            current = root/"current"
            if current.resolve() != current.absolute():
                raise ValueError("Symlinked CURRENT reference directory refused")
            current.mkdir(exist_ok=True)
            reference = dict(condition="current", reused_from_completed_prior_experiment=True, **sources["current"])
            ref = current/"reference.json"
            if ref.exists() and read(ref) != reference:
                raise ValueError("Existing CURRENT reference differs")
            if not ref.exists():
                write_json(ref, reference)
    return dict(identity, variant=args.variant), source


def prepare_variant(args, config, root, identity, source):
    directory = root/args.variant
    if directory.resolve() != directory.absolute():
        raise ValueError("Symlinked calibration variant directory refused")
    checkpoint = None
    if directory.exists():
        if not args.resume:
            raise ValueError("Variant exists; use --resume (no restart or overwrite)")
        if read(directory/"provenance.json")["identity"] != identity or read(directory/"config.json") != config:
            raise ValueError("Variant implementation/config/provenance mismatch")
        status = read(directory/"status.json")
        if status.get("status") == "failed":
            raise ValueError("Failed calibration requires manual review, not automatic replay")
        from .online_checkpoint import authorize
        args._checkpoint_authorization = authorize(directory, identity, config, resume=True)
        checkpoint = online_train.load_checkpoint(directory/"last.pth.tar", args._checkpoint_authorization)
        horizon = args.smoke_epochs if args.smoke else config["online"]["epochs"]
        rows = checkpoint["gradient_flow_online"]["rows"]
        if len(rows) == horizon:
            checked_directory(directory, epochs=horizon)
            if read(directory/"metrics.json") != rows or not (directory/"model_best.pth.tar").is_file():
                raise ValueError("Completed calibration publication differs from checkpoint")
            print("Already complete, skipping:", directory)
            return directory, checkpoint, True
        if len(rows) >= horizon:
            raise ValueError("Checkpoint exceeds declared horizon")
    else:
        directory.mkdir(exist_ok=False)
        write_json(directory/"config.json", config)
        write_json(directory/"provenance.json", dict(identity=identity, source=source,
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            variant_controller=config["online"]["controller"],
            protocol="Predeclared calibration; same inherited online training, probes, safety and schedule"))
        (directory/"resolved_training.yml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    # Pass the FULLY RESOLVED variant YAML to official argparse as well.
    resolved = directory/"resolved_training.yml"
    if yaml.safe_load(resolved.read_text()) != config:
        raise ValueError("Resolved official training YAML differs from variant config")
    args.config = resolved.relative_to(ROOT).as_posix()
    return directory, checkpoint, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, default="current")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output", default=None, help="Only synthetic smoke may choose another calibration namespace")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--register-only", action="store_true", help="Validate sources and preregister; no model/training")
    parser.add_argument("--expected-commit")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", choices=("torch", "cupy"), default="cupy")
    parser.add_argument("--smoke", action="store_true", help="Synthetic 2 epochs x 2 batches; not CIFAR evidence")
    args = parser.parse_args()
    args.smoke_epochs, args.condition = 2, "response_aware"
    if args.smoke and args.variant == "current":
        parser.error("CURRENT is reference-only; smoke B/C/D individually")
    config = declaration(args.data_dir)
    root = output_root(args)
    batch = fixed_batch(args, config)
    identity, source = register(args, config, root, batch)
    if args.register_only or args.variant == "current":
        manifest = read(root/"study_manifest.json")
        prior = manifest["sources"].get("current")
        print("Registered exactly four variants; CURRENT reused read-only.")
        if prior:
            print("Observed CURRENT runtime (hours, checkpoint I/O excluded):", prior["total_epoch_seconds"]/3600)
        return
    previous = signal.signal(signal.SIGUSR1, online_train.request_pause) if hasattr(signal, "SIGUSR1") else None
    destination = None
    try:
        with lock(root/("."+args.variant+".lock")):
            resolved = resolve(config, args.variant)
            destination, checkpoint, complete = prepare_variant(args, resolved, root, identity, source)
            if complete:
                return
            write_json(destination/"status.json", dict(status="running", identity=identity))
            try:
                runner = online_train.run_smoke if args.smoke else online_train.run_official
                runner(args, resolved, destination, batch, identity, checkpoint, None)
            except online_train.OnlinePause as error:
                write_json(destination/"status.json", dict(status="paused", reason=str(error), identity=identity))
                raise SystemExit(75)
            except BaseException as error:
                write_json(destination/"status.json", dict(status="failed", error=repr(error), identity=identity))
                raise
    finally:
        if previous is not None:
            signal.signal(signal.SIGUSR1, previous)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
