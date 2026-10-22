"""Short online feasibility wrapper; the official CIFAR training recipe is reused."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import sys

import torch
from spikingjelly.clock_driven import functional

from .common import ROOT, build_model, load_config, provenance, seed_all, write_json
from .intervene import restore_rng, rng_snapshot
from .multi_batch_intervene import batch_fingerprint, digest_json
from .online_policy import METHODS
from .online_runtime import OnlineRuntime, TrainingMetrics


STOP_REQUESTED = False


class OnlinePause(RuntimeError):
    """A walltime signal requests a safe stop BEFORE the next epoch starts."""


def request_pause(*unused):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def load_checkpoint(path, authorization=None):
    if authorization is not None:
        from .online_checkpoint import OnlineCheckpoint
        if not isinstance(authorization, OnlineCheckpoint):
            raise ValueError("Invalid online checkpoint authorization")
        authorization.check(path)
        return authorization.load()
    # Model-only/general calls remain safe. Never retry a failed safe load unsafely.
    return torch.load(path, map_location="cpu", weights_only=True)


def prepare(args, config):
    source = provenance()
    if args.expected_commit and source.get("commit") != args.expected_commit:
        raise ValueError("Unexpected source commit")
    if not args.smoke and (source.get("dirty") or not source.get("commit")):
        raise ValueError("Real online experiments require a clean committed checkout")
    if not args.smoke and (args.device != "cuda" or args.backend != "cupy" or config["seed"] != 42):
        raise ValueError("Reviewed real protocol is seed42, CUDA, explicit CuPy backend")
    root = (ROOT/args.output).resolve()
    namespace = ROOT/"results/gradient_flow"
    if namespace.resolve() != namespace or not root.is_relative_to(namespace) or root == namespace:
        raise ValueError("Use a new results/gradient_flow/<study> namespace")
    destination = root/args.condition
    signature = digest_json(dict(config=config, backend=args.backend, synthetic=args.smoke,
                                 smoke_epochs=args.smoke_epochs if args.smoke else None))
    rng = rng_snapshot()
    try:
        if args.smoke:
            generator = torch.Generator().manual_seed(420)
            batch = dict(images=torch.randn(2, 3, 8, 8, generator=generator),
                         targets=torch.tensor([1, 2]), indices=[0, 1], synthetic=True)
        else:
            from .make_probe_batch import iter_batches
            # Training split ONLY; held fixed across epochs and conditions. The
            # normalized eval transform removes augmentation noise from control.
            batch = next(iter_batches(config["data_dir"], config, config["online"]["probe_batch_size"], split="train"))
    finally:
        restore_rng(rng)
    identity = dict(source_commit=source.get("commit"), training_signature=signature,
                    batch_fingerprint=batch_fingerprint(batch), method=args.condition, synthetic=args.smoke,
                    runtime={k: source.get(k) for k in ("torch", "cuda", "packages")})
    from .amp_resume import reconcile
    identity, transition = reconcile(root, identity, config, source,
        getattr(args, "allow_amp_semantics_resume", False), load_checkpoint)
    if args.condition == "random_matched":
        parent_status = root/"response_aware/status.json"
        if not parent_status.is_file() or json.loads(parent_status.read_text()).get("status") != "complete":
            raise ValueError("Random control requires a completed response-aware magnitude schedule before creating outputs")
    checkpoint = None
    if destination.exists():
        if not args.resume:
            raise ValueError("Output already exists; use --resume or a new --output")
        record = json.loads((destination/"provenance.json").read_text())
        if record["identity"] != identity:
            raise ValueError("Existing output provenance/config/commit mismatch")
        status_path = destination/"status.json"
        previous_status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if previous_status.get("status") == "failed" and any(message in previous_status.get("error", "") for message in
                ("Persistent AMP overflow", "Persistent invalid diagnostics", "Nonfinite training loss", "Nonfinite FP32")):
            raise ValueError("Previous persistent/model failure requires manual review; refusing blind replay from an older checkpoint")
        from .online_checkpoint import authorize
        args._checkpoint_authorization = authorize(destination, identity, config, resume=args.resume,
            allow_amp=getattr(args, "allow_amp_semantics_resume", False))
        checkpoint = load_checkpoint(destination/"last.pth.tar", args._checkpoint_authorization)
        saved = checkpoint.get("gradient_flow_online", {})
        if saved.get("identity") != identity:
            raise ValueError("Checkpoint provenance mismatch")
        expected = args.smoke_epochs if args.smoke else config["online"]["epochs"]
        if len(saved.get("rows", [])) == expected:
            # Do not silently accept a half-published/corrupted completed run.
            if not all((destination/name).is_file() for name in
                       ("metrics.json", "controller_metrics.json", "model_best.pth.tar", "status.json")):
                raise ValueError("Completed checkpoint has missing published outputs; manual review required")
            marker = json.loads((destination/"status.json").read_text())
            if (marker.get("status") != "complete" or marker.get("identity") != identity or
                    marker.get("epochs") != expected or
                    json.loads((destination/"metrics.json").read_text()) != saved["rows"] or
                    json.loads((destination/"controller_metrics.json").read_text()) != [r["controller"] for r in saved["rows"]]):
                raise ValueError("Completed checkpoint and published output disagree; manual review required")
            print("Already complete, skipping:", destination)
            return destination, batch, identity, checkpoint, None, True
    else:
        destination.mkdir(parents=True, exist_ok=False)
        write_json(destination/"config.json", config)
        write_json(destination/"provenance.json", dict(identity=identity, source=source,
            protocol="single-seed online feasibility; 60-epoch prefix of official released recipe",
            probe="FP32/eval, first sequential training-split batch, hard targets, no augmentation",
            reference="local nonzero reference held across configured windows; never a G=0 objective",
            condition=args.condition, command=sys.argv))
    if transition is not None:
        from .amp_resume import preserve_lineage
        preserve_lineage(root, transition, source["commit"])
        write_json(destination/"continuation_provenance.json", dict(original_identity=identity,
            implementation_source=source, compatibility=transition,
            source_checkpoint_epoch=None if checkpoint is None else checkpoint["epoch"],
            preservation="original config/provenance/checkpoint prefix not relabeled or rewritten"))
    matched = None
    if args.condition == "random_matched":
        parent = root/"response_aware"
        parent_identity = json.loads((parent/"provenance.json").read_text())["identity"]
        if {**parent_identity, "method": args.condition} != identity:
            raise ValueError("Random control source must have identical training/batch/commit provenance")
        if json.loads((parent/"status.json").read_text())["status"] != "complete":
            raise ValueError("Random control requires a completed response-aware magnitude schedule")
        matched = json.loads((parent/"metrics.json").read_text())
        write_json(destination/"matched_source.json", dict(path=str(parent), identity=parent_identity,
            metrics_sha256=hashlib.sha256((parent/"metrics.json").read_bytes()).hexdigest(),
            matching="same epoch/per-block applied absolute steps; independent feasible random signs"))
    return destination, batch, identity, checkpoint, matched, False


def run_official(args, config, destination, batch, identity, checkpoint, matched):
    import train as official
    original = {key: getattr(official, key) for key in
                ("_parse_args", "create_scheduler", "train_one_epoch", "validate", "CheckpointSaver", "resume_checkpoint")}
    runtime = None
    old_argv = sys.argv
    stop = config["online"]["epochs"]

    def parse():
        parsed, text = original["_parse_args"]()
        if (parsed.model_ema or parsed.apex_amp or parsed.sage_v2 or parsed.init_checkpoint or
                parsed.initial_checkpoint or parsed.pretrained or getattr(parsed, "umsen", False) or
                parsed.fixed_alpha != 4 or parsed.sps_alpha != 4 or not parsed.deterministic or
                int(os.environ.get("WORLD_SIZE", "1")) != 1 or parsed.recovery_interval or
                parsed.no_resume_opt or parsed.start_epoch is not None or parsed.use_multi_epochs_loader):
            raise ValueError("Online protocol requires deterministic fresh/native-AMP official training; no SAGE/EMA/alternate resume")
        write_json(destination/"resolved_args.json", vars(parsed))
        return parsed, text

    def scheduler(*a, **kw):
        result, scheduled_epochs = original["create_scheduler"](*a, **kw)
        if scheduled_epochs < stop:
            raise ValueError("Online prefix exceeds inherited official schedule")
        return result, stop

    class OnlineSaver(original["CheckpointSaver"]):
        def _save(self, path, epoch, metric=None):
            super()._save(path, epoch, metric)
            # Internal just-created timm save file, not a caller-supplied resume path.
            from experiment_state import load_trusted_checkpoint
            if Path(path).absolute().parent != destination.absolute():
                raise ValueError("Unexpected internally generated checkpoint path")
            saved = load_trusted_checkpoint(path)
            saved["gradient_flow_online"] = runtime.state_dict()
            torch.save(saved, path)

        def save_checkpoint(self, epoch, metric=None):
            result = super().save_checkpoint(epoch, metric)
            runtime.persist(complete=epoch+1 == stop)
            return result

    def train_epoch(epoch, model, loader, optimizer, loss_fn, parsed, **kwargs):
        nonlocal runtime
        if STOP_REQUESTED:
            raise OnlinePause("Walltime signal: last completed epoch is resumable")
        if runtime is None:
            native = kwargs.get("loss_scaler")
            if native is not None and not hasattr(native, "_scaler"):
                raise ValueError("Online measurements support native AMP only")
            runtime = OnlineRuntime(model, optimizer, kwargs.get("lr_scheduler"),
                getattr(native, "_scaler", None), batch, args.condition, config["online"], destination, identity, matched)
            if checkpoint is not None:
                runtime.load_state_dict(checkpoint["gradient_flow_online"])
                if checkpoint["epoch"]+1 != len(runtime.rows) or epoch != len(runtime.rows):
                    raise ValueError("Checkpoint epoch/controller history mismatch")
        runtime.start_epoch(epoch)
        metric_loss = TrainingMetrics(loss_fn)
        kwargs["loss_scaler"] = runtime.adapter
        result = original["train_one_epoch"](epoch, model, loader, optimizer, metric_loss, parsed, **kwargs)
        runtime.training = metric_loss.finish(result["loss"])
        return {key: runtime.training[key] for key in ("loss", "top1", "top5")}

    def validate(model, loader, loss_fn, parsed, **kwargs):
        result = original["validate"](model, loader, loss_fn, parsed, **kwargs)
        runtime.finish_epoch(runtime.training, result)
        return result

    trusted_resume = None
    if checkpoint is not None:
        from .online_checkpoint import scoped_timm_resume
        trusted_resume = scoped_timm_resume(original["resume_checkpoint"],
            getattr(args, "_checkpoint_authorization", None))
    official._parse_args, official.create_scheduler = parse, scheduler
    if trusted_resume is not None:
        official.resume_checkpoint = trusted_resume
    official.train_one_epoch, official.validate, official.CheckpointSaver = train_epoch, validate, OnlineSaver
    sys.argv = [old_argv[0], "--config", str(ROOT/args.config), "-data-dir", config["data_dir"],
                "--output", str(destination.parent), "--experiment", destination.name]
    if checkpoint is not None:
        sys.argv += ["--resume", str(destination/"last.pth.tar")]
    try:
        official.main()
        if runtime is None or len(runtime.rows) != stop:
            raise RuntimeError("Training stopped before requested online horizon")
    finally:
        if runtime is not None:
            runtime.close()
        for key, value in original.items():
            setattr(official, key, value)
        sys.argv = old_argv


def run_smoke(args, config, destination, batch, identity, checkpoint, matched):
    """Explicit synthetic 2x2-batch plumbing check; never a CIFAR accuracy claim."""
    from experiment_state import capture_rng, restore_rng as restore_training_rng
    seed_all(42)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type != "cuda" and args.backend != "torch":
        raise ValueError("CPU smoke requires explicit --backend torch")
    model = build_model(dict(img_size=8, patch_size=4, dim=16, num_heads=4,
        mlp_ratio=2, layer=4, time_step=2, num_classes=10), args.backend).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.06)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.smoke_epochs)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    runtime = OnlineRuntime(model, optimizer, scheduler, scaler, batch, args.condition,
                            config["online"], destination, identity, matched)
    best = -1.
    if checkpoint is not None:
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler is not None:
            scaler.load_state_dict(checkpoint["amp_scaler"])
        runtime.load_state_dict(checkpoint["gradient_flow_online"])
        best = checkpoint["best_metric"]
        restore_training_rng(checkpoint["rng"])
    images, targets = batch["images"].to(device), batch["targets"].to(device)
    try:
        for epoch in range(len(runtime.rows), args.smoke_epochs):
            if STOP_REQUESTED:
                raise OnlinePause("Smoke paused at epoch boundary")
            runtime.start_epoch(epoch)
            model.train()
            loss_fn = TrainingMetrics(torch.nn.CrossEntropyLoss())
            loss_sum = 0.
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=scaler is not None):
                    loss = loss_fn(model(images), targets)
                runtime.adapter(loss, optimizer)
                loss_sum += float(loss.detach())
                functional.reset_net(model)
            model.eval()
            with torch.no_grad():
                logits = model(images)
                val = dict(loss=float(torch.nn.functional.cross_entropy(logits, targets)),
                    top1=float((logits.argmax(1) == targets).float().mean())*100,
                    top5=float((logits.topk(5, 1).indices == targets[:, None]).any(1).float().mean())*100)
            functional.reset_net(model)
            runtime.finish_epoch(loss_fn.finish(loss_sum/2), val)
            scheduler.step()
            is_best = val["top1"] > best
            best = max(best, val["top1"])
            saved = dict(version=2, state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(), epoch=epoch, best_metric=best,
                amp_scaler=None if scaler is None else scaler.state_dict(), rng=capture_rng(),
                gradient_flow_online=runtime.state_dict())
            tmp = destination/"last.pth.tar.tmp"
            torch.save(saved, tmp)
            tmp.replace(destination/"last.pth.tar")
            if is_best:
                import shutil
                shutil.copyfile(destination/"last.pth.tar", destination/"model_best.pth.tar")
            runtime.persist(complete=epoch+1 == args.smoke_epochs)
            print("SYNTHETIC SMOKE", args.condition, epoch, "alpha", runtime.policy.alpha,
                  "G", runtime.control["G"], "Rhat", runtime.control["Rhat"], flush=True)
    finally:
        runtime.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gradient_flow/controller_online.yml")
    parser.add_argument("--condition", choices=METHODS, required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-amp-semantics-resume", action="store_true",
                        help="Explicit audited continuation of the d872125 all-finite Fixed prefix; no recipe changes")
    parser.add_argument("--validate-resume-only", action="store_true",
                        help="Read-only full-state restoration check; no forwards, updates or study writes")
    parser.add_argument("--expected-commit")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", choices=("torch", "cupy"), default="cupy")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=2)
    args = parser.parse_args()
    if args.smoke and args.smoke_epochs not in (2, 3):
        parser.error("Smoke horizon must be 2 or 3 epochs")
    config = load_config(ROOT/args.config)
    from .online_policy import OnlinePolicy
    OnlinePolicy(config["layer"], args.condition, config["online"])
    if not 1 <= config["online"]["epochs"] <= config["epochs"]:
        parser.error("Online epochs must be a positive prefix of the inherited recipe")
    if args.data_dir:
        config["data_dir"] = args.data_dir
    args.output = args.output or ("results/gradient_flow/controller_online_smoke_seed42" if args.smoke else config["output"])
    if args.validate_resume_only:
        if not args.resume or args.smoke:
            parser.error("Read-only validation requires --resume on the real study")
        from .validate_online_resume import validate
        validate(args, config)
        return
    destination = None
    previous_signal = signal.signal(signal.SIGUSR1, request_pause) if hasattr(signal, "SIGUSR1") else None
    try:
        destination, batch, identity, checkpoint, matched, complete = prepare(args, config)
        if complete:
            return
        write_json(destination/"status.json", dict(status="running", identity=identity))
        (run_smoke if args.smoke else run_official)(args, config, destination, batch, identity, checkpoint, matched)
    except OnlinePause as error:
        if destination is not None:
            write_json(destination/"status.json", dict(status="paused", reason=str(error)))
        raise SystemExit(75)
    except BaseException as error:
        if destination is not None:
            write_json(destination/"status.json", dict(status="failed", error=repr(error)))
        raise
    finally:
        if previous_signal is not None:
            signal.signal(signal.SIGUSR1, previous_signal)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
