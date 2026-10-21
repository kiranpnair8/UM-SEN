"""Explicit, narrowly allowlisted continuation of the pre-fix all-finite prefix."""

import hashlib
import json
import math
from pathlib import Path
import subprocess
import shutil
import os
from contextlib import contextmanager

from .amp_safety import DEFAULT_SAFETY
from .common import ROOT, write_json


LEGACY_COMMIT = "d8721251988691af90a7e01d5e4ff12cdd4571a8"
AMP_FIX_COMMIT = "9a9c0a77773b8e962fe90f2558224033d1140c43"
RESUME_FILES = {
    "experiments/gradient_flow/amp_resume.py", "experiments/gradient_flow/online_checkpoint.py",
    "experiments/gradient_flow/validate_online_resume.py", "experiments/gradient_flow/online_train.py",
    "experiments/gradient_flow/online_runtime.py", "experiments/gradient_flow/CONTROLLER_ONLINE.md",
    "jobs/gradient_flow/run_controller_online_cifar10.sh", "tests/gradient_flow/test_online_checkpoint.py",
    "tests/gradient_flow/test_online_controller.py",
}
ALLOWED_FILES = {
    "experiments/gradient_flow/amp_safety.py", "experiments/gradient_flow/amp_resume.py",
    "experiments/gradient_flow/online_runtime.py", "experiments/gradient_flow/online_train.py",
    "experiments/gradient_flow/CONTROLLER_ONLINE.md", "analysis/gradient_flow/online_summary.py",
    "jobs/gradient_flow/run_controller_online_cifar10.sh", "tests/gradient_flow/test_online_amp.py",
}


def verify_patch(current):
    git = ["git", "-c", "safe.directory="+ROOT.as_posix(), "-C", str(ROOT)]
    subprocess.run(git+["merge-base", "--is-ancestor", LEGACY_COMMIT, current], check=True)
    files = set(subprocess.check_output(git+["diff", "--name-only", LEGACY_COMMIT, current], text=True).splitlines())
    if not files or not files <= ALLOWED_FILES:
        raise ValueError("AMP continuation refuses unrelated changes: "+str(sorted(files-ALLOWED_FILES)))
    return sorted(files)


def verify_resume_patch(current):
    git = ["git", "-c", "safe.directory="+ROOT.as_posix(), "-C", str(ROOT)]
    parents = subprocess.check_output(git+["show", "-s", "--format=%P", current], text=True).strip()
    title = subprocess.check_output(git+["show", "-s", "--format=%s", current], text=True).strip()
    files = set(subprocess.check_output(git+["diff", "--name-only", AMP_FIX_COMMIT, current], text=True).splitlines())
    if (parents != AMP_FIX_COMMIT or title != "fix: allow validated full-state online checkpoint resume" or
            not files or not files <= RESUME_FILES or "experiments/gradient_flow/online_checkpoint.py" not in files):
        raise ValueError("Not the reviewed single resume-only patch; arbitrary later commits are refused")
    verify_patch(AMP_FIX_COMMIT)
    return sorted(files)


def patch_lineage(record, current):
    if current == record["implementation_commit"]:
        verify_patch(current)
        return [LEGACY_COMMIT, current]
    if record["implementation_commit"] != AMP_FIX_COMMIT:
        raise ValueError("Unexpected existing compatibility implementation commit")
    verify_resume_patch(current)
    return [LEGACY_COMMIT, AMP_FIX_COMMIT, current]


@contextmanager
def lineage_lock(root):
    with (Path(root)/".amp_resume_lineage.lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            if stream.seek(0, 2) == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def preserve_lineage(root, record, current):
    """New immutable sidecar; never rewrite amp_compatibility.json/protected files."""
    lineage = patch_lineage(record, current)
    if len(lineage) < 3:
        return
    payload = dict(patch_lineage=lineage, original_marker_sha256=digest(Path(root)/"amp_compatibility.json"),
                   original_fixed_identity=record["original_fixed_identity"], reviewed_resume_files=verify_resume_patch(current))
    destination = Path(root)/("amp_resume_lineage_"+current+".json")
    with lineage_lock(root):
        if destination.exists():
            if json.loads(destination.read_text()) != payload:
                raise ValueError("Existing immutable lineage sidecar disagrees")
        else:
            write_json(destination, payload)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compatible_identity(a, b):
    return {**a, "source_commit": LEGACY_COMMIT} == {**b, "source_commit": LEGACY_COMMIT}


def validate_legacy_prefix(checkpoint, identity, config):
    saved = checkpoint.get("gradient_flow_online", {})
    rows = saved.get("rows", [])
    extra = checkpoint.get("sage_training_state", {})
    if (saved.get("identity") != identity or identity["method"] != "fixed" or identity["synthetic"] or
            not rows or checkpoint.get("epoch") != len(rows)-1 or
            [r["epoch"] for r in rows] != list(range(len(rows))) or len(rows) >= config["online"]["epochs"] or
            "optimizer" not in checkpoint or not checkpoint.get("amp_scaler") or
            extra.get("scheduler") is None or extra.get("rng") is None or
            saved.get("policy", {}).get("settings") != config["online"] or
            saved.get("policy", {}).get("alpha") != [4.]*config["layer"]):
        raise ValueError("Need a complete native-AMP Fixed epoch-boundary prefix, not weights-only/legacy initialization")
    for row in rows:
        if (row["controller"]["alpha"] != [4.]*config["layer"] or
                any(v != 0 for v in row["controller"]["applied_step"]) or
                len(row["gradient"]) != config["layer"] or
                any(g["observations"] <= 0 or any(g[k] is None or not math.isfinite(g[k]) for k in
                    ("grad_input_norm", "grad_output_norm", "parameter_grad_norm", "G_l")) for g in row["gradient"])):
            raise ValueError("Legacy prefix did not contain exclusively valid Fixed observations")
    return rows


def reconcile(root, proposed, config, source, allowed, load_checkpoint):
    """Retain the original study identity; record the actual new code separately.

    No old checkpoint, config, metrics, status or provenance is rewritten here.
    Fresh paired conditions inherit the same original study identity only after
    explicit authorization and patch/config/runtime/batch verification.
    """
    root = Path(root)
    marker = root/"amp_compatibility.json"
    fixed_provenance = root/"fixed/provenance.json"
    legacy = json.loads(fixed_provenance.read_text())["identity"] if fixed_provenance.exists() else None
    if marker.exists():
        record = json.loads(marker.read_text())
        if not allowed or record["legacy_commit"] != LEGACY_COMMIT:
            raise ValueError("Explicit --allow-amp-semantics-resume and the recorded patch commit are required")
        lineage = patch_lineage(record, source["commit"])
        expected = {**record["original_fixed_identity"], "method": proposed["method"]}
        if not compatible_identity(expected, proposed):
            raise ValueError("AMP continuation config/runtime/batch/method mismatch")
        return expected, dict(record, approved_patch_lineage=lineage)
    if legacy is None or legacy["source_commit"] == proposed["source_commit"]:
        return proposed, None
    if not allowed or proposed["method"] != "fixed" or legacy["source_commit"] != LEGACY_COMMIT:
        raise ValueError("Cross-commit resume needs the explicit, reviewed AMP-semantics compatibility path")
    if not compatible_identity(legacy, proposed):
        raise ValueError("AMP continuation must preserve config, method, runtime and probe batch")
    files = verify_patch(source["commit"])
    path = root/"fixed/last.pth.tar"
    checkpoint = load_checkpoint(path)
    rows = validate_legacy_prefix(checkpoint, legacy, config)
    if json.loads((root/"fixed/metrics.json").read_text()) != rows:
        raise ValueError("Published prefix metrics disagree with the epoch checkpoint")
    backup = root/"fixed"/f"amp_resume_origin_epoch{checkpoint['epoch']:03d}.pth.tar"
    original_hash = digest(path)
    if not backup.exists():
        shutil.copyfile(path, backup)
    if digest(backup) != original_hash:
        raise ValueError("Protected pre-fix checkpoint copy hash mismatch")
    protected = {}
    for name in ("config.json", "provenance.json", "args.yaml", "resolved_args.json", "metrics.json", "metrics.csv", "status.json"):
        original = root/"fixed"/name
        if original.is_file():
            target = original.with_name("amp_resume_origin_"+name)
            if not target.exists():
                shutil.copyfile(original, target)
            if digest(target) != digest(original):
                raise ValueError("Protected prefix artifact hash mismatch: "+name)
            protected[name] = dict(path=str(target), sha256=digest(target))
    record = dict(version=1, legacy_commit=LEGACY_COMMIT, implementation_commit=source["commit"],
        original_fixed_identity=legacy, reviewed_changed_files=files, source_checkpoint=str(path),
        source_checkpoint_sha256=original_hash, protected_checkpoint=str(backup), prefix_epochs=[r["epoch"] for r in rows],
        prefix_metrics_sha256=digest(root/"fixed/metrics.json"), safety_defaults=DEFAULT_SAFETY,
        protected_prefix_artifacts=protected,
        reason="Native AMP overflow recovery restored; all-finite historical prefix retained exactly; no recipe/controller changes",
        prefix_rewritten=False)
    write_json(marker, record)
    return legacy, record
