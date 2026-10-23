"""Hash-validated local online checkpoints; never a safe-to-unsafe fallback."""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import torch

from .common import ROOT

_ISSUER = object()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class OnlineCheckpoint:
    path: Path
    sha256: str
    identity: dict
    epoch: int
    _authority: object = field(default=None, repr=False, compare=False)

    def check(self, path=None):
        p = self.path if path is None else Path(path).absolute()
        if self._authority is not _ISSUER or p != self.path or p.resolve() != p or not p.is_file() or digest(p) != self.sha256:
            raise ValueError("Online checkpoint path/hash changed after authorization")

    def load(self):
        self.check()
        # This checkpoint is trusted because it was generated locally by this
        # experiment and its provenance/hash/identity were validated before full
        # deserialization. Only authorize() issues this capability to the CLI.
        with self.path.open("rb") as stream:
            h = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024*1024), b""):
                h.update(chunk)
            if h.hexdigest() != self.sha256:
                raise ValueError("Checkpoint changed before full deserialization")
            stream.seek(0)
            saved = torch.load(stream, map_location="cpu", weights_only=False)
        state = saved.get("gradient_flow_online", {})
        rows = state.get("rows", [])
        if (state.get("identity") != self.identity or saved.get("epoch") != self.epoch or
                saved.get("version", 0) <= 1 or len(rows) != self.epoch+1 or
                [r["epoch"] for r in rows] != list(range(self.epoch+1))):
            raise ValueError("Deserialized full state does not match validated online metadata")
        return saved


def authorize(destination, identity, config, *, resume, allow_amp=False):
    """Validate JSON/local hash evidence BEFORE any full pickle deserialization."""
    destination = Path(destination).absolute()
    root = destination.parent
    namespace = ROOT/"results/gradient_flow"
    calibration = root.name.startswith("controller_calibration_")
    method_design = root.name.startswith("controller_method_ablation_")
    followup = root.parent.name in ("controller_followup_cifar10", "controller_followup_cifar100")
    conditions = (("deep_only", "regularized", "deadband_regularized") if method_design else
        ("fixed", "final_sage") if followup else
        ("conservative", "very_conservative", "anchored") if calibration else ("fixed", "response_aware", "random_matched"))
    if (not resume or destination.resolve() != destination or namespace.resolve() != namespace or
            not destination.is_relative_to(namespace) or destination.name not in conditions):
        raise ValueError("Not an exact local online condition resume path")
    if method_design:
        from .method_ablation import validate_resume
        validate_resume(destination, identity, config)
    elif followup:
        from .controller_followup import validate_resume
        validate_resume(destination, identity, config)
    elif calibration:
        from .calibration_spec import OUTPUT, validate_resume_config
        if root.parent != namespace or (not identity.get("synthetic") and root != ROOT/OUTPUT):
            raise ValueError("Not the exact calibration resume namespace")
        validate_resume_config(config, identity, destination)
    elif not identity.get("synthetic") and root != namespace/"controller_online_seed42":
        raise ValueError("Real full-state resume is restricted to controller_online_seed42/<condition>")
    record = json.loads((destination/"provenance.json").read_text())
    if record["identity"] != identity or json.loads((destination/"config.json").read_text()) != config:
        raise ValueError("Online checkpoint provenance/config/identity not validated")
    path = destination/"last.pth.tar"
    integrity = destination/"checkpoint_integrity.json"
    if integrity.is_file():
        evidence = json.loads(integrity.read_text())
        if evidence["identity"] != identity:
            raise ValueError("Checkpoint integrity identity mismatch")
        epoch, expected = evidence["epoch"], evidence["sha256"]
    else:
        # Historical epoch6 is enrolled by the already reviewed AMP marker and
        # its immutable protected copy, NOT by trusting a freshly computed hash.
        if not allow_amp or destination.name != "fixed":
            raise ValueError("No locally generated checkpoint integrity evidence; no unsafe fallback")
        evidence = json.loads((root/"amp_compatibility.json").read_text())
        if evidence["original_fixed_identity"] != identity or evidence["prefix_epochs"] != list(range(7)):
            raise ValueError("Historical continuation requires exactly epochs0-6")
        expected = evidence["source_checkpoint_sha256"]
        backup = Path(evidence["protected_checkpoint"]).absolute()
        if (backup.parent != destination or not backup.name.startswith("amp_resume_origin_") or
                backup.resolve() != backup or digest(backup) != expected):
            raise ValueError("Protected epoch6 checkpoint hash/path mismatch")
        for item in evidence.get("protected_prefix_artifacts", {}).values():
            p = Path(item["path"]).absolute()
            if p.parent != destination or p.resolve() != p or not p.name.startswith("amp_resume_origin_") or digest(p) != item["sha256"]:
                raise ValueError("Protected prefix artifact changed")
        if digest(destination/"metrics.json") != evidence["prefix_metrics_sha256"]:
            raise ValueError("Published epoch0-6 metrics changed")
        epoch = 6
    rows = json.loads((destination/"metrics.json").read_text())
    if type(epoch) is not int or epoch < 0 or len(rows) != epoch+1 or [r["epoch"] for r in rows] != list(range(epoch+1)):
        raise ValueError("Epoch metadata is not a contiguous committed prefix")
    capability = OnlineCheckpoint(path, expected, identity, epoch, _ISSUER)
    capability.check()
    return capability


def scoped_timm_resume(original, capability):
    import inspect
    if "weights_only" not in inspect.signature(original).parameters:
        raise RuntimeError("Installed timm lacks explicit weights_only support; no unsafe fallback")
    def resume(model, checkpoint_path, *args, **kwargs):
        if not isinstance(capability, OnlineCheckpoint):
            raise ValueError("Full-state online resume has not been authorized")
        capability.check(checkpoint_path)
        if "weights_only" in kwargs:
            raise ValueError("Caller cannot override the validated resume policy")
        result = original(model, str(capability.path), *args, weights_only=False, **kwargs)
        capability.check()
        if result != capability.epoch+1:
            raise ValueError("timm resume epoch does not match validated next epoch")
        return result
    return resume
