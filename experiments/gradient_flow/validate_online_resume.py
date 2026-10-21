"""Read-only full-state restoration audit. No forward, backward, step or study writes."""

import json
import inspect
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import torch

from .common import ROOT, build_model, provenance
from .intervene import same_state
from .online_checkpoint import authorize, scoped_timm_resume
from .amp_resume import reconcile


def cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu(v) for v in value)
    return value


def restore_and_check(model, optimizer, scheduler, scaler, runtime, checkpoint, capability, resume):
    from experiment_state import capture_rng, restore_rng, restore_extra_state
    next_epoch = scoped_timm_resume(resume, capability)(model, str(capability.path), optimizer=optimizer,
        loss_scaler=scaler, log_info=True)
    extra = restore_extra_state(checkpoint, scheduler)
    runtime.load_state_dict(checkpoint["gradient_flow_online"])
    loaders = [SimpleNamespace(generator=torch.Generator()) for _ in extra["rng"]["loaders"]]
    restore_rng(extra["rng"], loaders)
    checks = dict(model=same_state(cpu(model.state_dict()), checkpoint["state_dict"]),
        optimizer=same_state(cpu(optimizer.state_dict()), checkpoint["optimizer"]),
        scheduler=same_state(cpu(scheduler.state_dict()), extra["scheduler"]),
        scaler=same_state(scaler.state_dict(), checkpoint[scaler.state_dict_key]),
        rng=same_state(cpu(capture_rng(loaders)), cpu(extra["rng"])),
        online_runtime=runtime.policy.state_dict() == checkpoint["gradient_flow_online"]["policy"],
        online_step=runtime.adapter.step == checkpoint["gradient_flow_online"]["step"],
        online_rows=runtime.rows == checkpoint["gradient_flow_online"]["rows"],
        controller_timing=runtime.controller_seconds == checkpoint["gradient_flow_online"]["controller_seconds"],
        diagnostic_state=same_state(runtime.monitor.state_dict(), checkpoint["gradient_flow_online"]["monitor"]))
    if not all(checks.values()) or next_epoch != checkpoint["epoch"]+1:
        raise AssertionError("Full-state resume audit failed: "+str(checks))
    return dict(checkpoint_epoch=checkpoint["epoch"], online_rows=len(runtime.rows), next_epoch=next_epoch,
                checks=checks, optimizer_updates=0, forwards=0, study_files_written=0)


def validate(args, config):
    source = provenance()
    if source.get("dirty") or args.expected_commit and source["commit"] != args.expected_commit:
        raise ValueError("Read-only real checkpoint validation requires the expected clean source")
    root = (ROOT/args.output).resolve()
    destination = root/args.condition
    original = json.loads((destination/"provenance.json").read_text())["identity"]
    if original["source_commit"] == "d8721251988691af90a7e01d5e4ff12cdd4571a8" and not (root/"amp_compatibility.json").is_file():
        raise ValueError("Historical AMP marker is absent. Stop for provenance review; do not recreate/delete protected artifacts")
    proposed = {**original, "source_commit": source["commit"],
                "runtime": {k: source.get(k) for k in ("torch", "cuda", "packages")}}
    # Only safe JSON and protected hashes are read in this preflight. No marker
    # is edited/created; the operational prepare path publishes lineage later.
    identity, transition = reconcile(root, proposed, config, source, args.allow_amp_semantics_resume,
                            lambda unused: (_ for _ in ()).throw(ValueError("Existing compatibility marker required")))
    capability = authorize(destination, identity, config, resume=True, allow_amp=args.allow_amp_semantics_resume)
    from .make_probe_batch import iter_batches
    from .multi_batch_intervene import batch_fingerprint, digest_json
    from .online_runtime import OnlineRuntime
    from experiment_state import capture_rng, restore_rng
    batch = next(iter_batches(config["data_dir"], config, config["online"]["probe_batch_size"], split="train"))
    signature = digest_json(dict(config=config, backend=args.backend, synthetic=False, smoke_epochs=None))
    if signature != identity["training_signature"] or batch_fingerprint(batch) != identity["batch_fingerprint"]:
        raise ValueError("Read-only probe fingerprint/config do not match the original protocol")
    checkpoint = capability.load()
    import train as official
    old_argv = sys.argv
    sys.argv = [old_argv[0], "--config", str(ROOT/args.config), "-data-dir", config["data_dir"]]
    try:
        parsed, _ = official._parse_args()
    finally:
        sys.argv = old_argv
    caller_rng = capture_rng()
    runtime = None
    try:
        model = build_model(config, args.backend).to(args.device)
        optimizer = official.create_optimizer_v2(model, **official.optimizer_kwargs(cfg=parsed))
        scheduler, _ = official.create_scheduler(parsed, optimizer)  # Same construction-before-restore order.
        scaler = official.NativeScaler()
        if args.device == "cpu":
            scaler._scaler = torch.amp.GradScaler("cpu")
        with tempfile.TemporaryDirectory(prefix="online-resume-validation-") as temporary:
            runtime = OnlineRuntime(model, optimizer, scheduler, scaler._scaler, batch, args.condition,
                                    config["online"], Path(temporary), identity)
            report = restore_and_check(model, optimizer, scheduler, scaler, runtime, checkpoint, capability,
                                       official.resume_checkpoint)
            import timm
            report.update(timm_version=timm.__version__, timm_resume_signature=str(inspect.signature(official.resume_checkpoint)),
                amp_compatibility_exists=(root/"amp_compatibility.json").is_file(),
                patch_lineage=None if transition is None else transition.get("approved_patch_lineage"),
                checkpoint_sha256=capability.sha256)
            print(json.dumps(report, indent=2))
    finally:
        if runtime is not None:
            runtime.close()
        restore_rng(caller_rng)
    return report
