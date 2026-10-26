"""NRP infrastructure wrapper. Never edits the immutable scientific checkout."""

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace


PIN = "01ad13da81eed92b64247c5dc3fcd44608f31a1f"
NAMESPACE = "kc-ai-research-lab"
RUNS = {(c, s) for s in (42, 43, 44) for c in ("fixed", "deep_only")}
EXPECTED = {"torch": "2.5.1+cu121", "torchvision": "0.20.1+cu121",
    "timm": "1.0.28", "spikingjelly": "0.0.0.0.14", "numpy": "2.2.6",
    "PyYAML": "6.0.3", "cupy-cuda12x": "13.3.0"}
PARENT_FILES = {
    "final_selection.json": "results/gradient_flow/controller_method_ablation_seed42/final_selection.json",
    "study_manifest.json": "results/gradient_flow/controller_method_ablation_seed42/study_manifest.json",
    "summary.json": "results/gradient_flow/controller_method_ablation_seed42/summary.json",
    "controller_followup.yml": "configs/gradient_flow/controller_followup.yml",
}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def immutable(path, payload):
    if Path(path).exists():
        if read(path) != payload:
            raise ValueError("Existing persistent marker differs; use a reviewed NEW study ID: "+str(path))
    else:
        write(path, payload)


@contextmanager
def locked(path, *, blocking=False):
    import fcntl
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def command(args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def git(repo, *args):
    return command(["git", "-c", "safe.directory="+str(repo), "-C", str(repo), *args])


def verify_repo(repo):
    head = git(repo, "rev-parse", "HEAD")
    if head != PIN or git(repo, "status", "--porcelain"):
        raise ValueError("Scientific code must be clean at EXACT commit "+PIN)
    return head


def paths():
    root = Path(os.environ["NRP_ROOT"])
    base = Path("/storage/umsen-nrp")
    if (not root.is_absolute() or root.parent != base or root.resolve() != root or
            not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,62}", root.name) or
            os.environ.get("NRP_SOURCE_COMMIT") != PIN or
            os.environ.get("POD_NAMESPACE") != NAMESPACE or os.environ.get("NRP_PVC") != "umsen-storage-rwx" or
            not os.environ.get("POD_NODE")):
        raise ValueError("Unexpected namespace/source/RWX PVC/path or missing actual node provenance")
    if not re.fullmatch(r".+@sha256:[a-f0-9]{64}", os.environ.get("NRP_IMAGE", "")):
        raise ValueError("Build/push the image and rerender ALL manifests with image@sha256:DIGEST")
    if shutil.disk_usage("/storage").free < 10*1024**3:
        raise ValueError("Less than 10GiB free on PVC; do not automatically delete research artifacts")
    root.mkdir(parents=True, exist_ok=True)
    return root


def checkout(root, run_id):
    if not re.fullmatch(r"[a-z0-9_]+", run_id):
        raise ValueError("Invalid run directory name")
    base, repo = root/"code/UM-SEN", root/"runs"/run_id/"repo"
    with locked(root/"locks/git.lock", blocking=True):
        if not base.exists():
            base.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                "https://github.com/kiranpnair8/UM-SEN.git", str(base)], check=True)
            subprocess.run(["git", "-C", str(base), "checkout", "--detach", PIN], check=True)
        verify_repo(base)
        if not repo.exists():
            repo.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "-c", "safe.directory="+str(base), "-C", str(base),
                            "worktree", "add", "--detach", str(repo), PIN], check=True)
        verify_repo(repo)
    return repo


def source_module(repo):
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    module = importlib.import_module("experiments.gradient_flow.cifar100_followup")
    if Path(module.__file__).resolve() != repo/"experiments/gradient_flow/cifar100_followup.py":
        raise ValueError("Wrong scientific source imported")
    return module


def snapshot(repo, *, gpu):
    import torch
    versions = {name: importlib.metadata.version(name) for name in EXPECTED}
    if versions != EXPECTED or sys.version_info[:2] != (3,10) or torch.version.cuda != "12.1":
        raise ValueError("Software stack mismatch; no automatic version/backend fallback: "+str(versions))
    hardware = {}
    if gpu:
        print(command(["nvidia-smi"]), flush=True)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Exactly ONE accessible CUDA GPU is required")
        import cupy
        assert int(cupy.arange(4).sum()) == 6
        cupy.cuda.runtime.deviceSynchronize()
        p = torch.cuda.get_device_properties(0)
        hardware = dict(gpu_model=p.name, gpu_memory_bytes=p.total_memory,
            compute_capability=[p.major,p.minor],
            driver=command(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"]))
    disk = shutil.disk_usage("/storage")
    print(command(["df","-h","/storage"]), flush=True)
    result = dict(source_commit=verify_repo(repo), repo=str(repo), python=platform.python_version(),
        versions=versions, cuda_compile_version=torch.version.cuda, cuda_available=torch.cuda.is_available(),
        platform=platform.platform(), cpu=platform.processor(), logical_cpus=os.cpu_count(),
        requested_resources=json.loads(os.environ.get("NRP_RESOURCES","{}")),
        cgroup_limits={name: path.read_text().strip() for name, path in
            (("memory",Path("/sys/fs/cgroup/memory.max")),("cpu",Path("/sys/fs/cgroup/cpu.max"))) if path.is_file()},
        hardware=hardware, gpu_uuid=command(["nvidia-smi","--query-gpu=uuid","--format=csv,noheader"]) if gpu else None,
        node=os.environ["POD_NODE"], pod=os.environ["POD_NAME"], pod_uid=os.environ["POD_UID"],
        namespace=NAMESPACE, pvc="umsen-storage-rwx", mount_path="/storage",
        disk=dict(total_bytes=disk.total, free_bytes=disk.free),
        image_reference=os.environ["NRP_IMAGE"], support_script_sha256=digest(__file__))
    print(json.dumps(result, indent=2), flush=True)
    if Path("/opt/nrp/pip-freeze.txt").is_file():
        print("IMAGE PIP FREEZE\n"+Path("/opt/nrp/pip-freeze.txt").read_text(), flush=True)
    return result


def environment_key(report):
    # Software/storage readiness is shared across nodes. Actual hardware remains
    # in each attempt report, but is not a cross-job scheduling constraint.
    return {k: report[k] for k in ("source_commit","python","versions","cuda_compile_version",
        "image_reference","support_script_sha256","namespace","pvc","mount_path")}


def require_environment(root, report):
    marker = read(root/"checks/environment-ready.json")
    if marker["key"] != environment_key(report):
        raise ValueError("Environment check differs; software/source/storage must match across NRP nodes")


def stage_parent(root, entry):
    destination = root/"evidence/parent-selection"
    for key, relative in PARENT_FILES.items():
        incoming = Path("/parent-input")/key
        if not incoming.is_file():
            raise ValueError("Mount the FOUR real, immutable parent-selection files via ConfigMap; do not fabricate them")
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if digest(target) != digest(incoming):
                raise ValueError("Persistent parent evidence changed")
        else:
            target.write_bytes(incoming.read_bytes())
    return entry.parent_evidence(destination)


def data_hashes(path):
    return {name: digest(Path(path)/"cifar-100-python"/name) for name in ("train","test","meta")}


def require_data(root, entry):
    marker = read(root/"checks/data-ready.json")
    if (marker["source_commit"] != PIN or marker["files_sha256"] != data_hashes(root/"data/cifar100") or
            marker["parent"] != entry.parent_evidence(root/"evidence/parent-selection")):
        raise ValueError("Shared CIFAR100 data or parent evidence changed")
    from torchvision.datasets import CIFAR100
    for train in (True,False):
        dataset = CIFAR100(root/"data/cifar100", train=train, download=False)
        if len(dataset) != (50000 if train else 10000) or len(dataset.classes) != 100:
            raise ValueError("Wrong real CIFAR100 dataset")
    return marker


def approval(entry, root):
    reviewer = os.environ.get("NRP_REVIEWED_BY", "")
    if not reviewer or os.environ.get("NRP_ACK_RECIPE") != "true":
        raise ValueError("Create the explicit review ConfigMap; inherited normalization must be acknowledged")
    entry.prepare(SimpleNamespace(source_root=str(root/"evidence/parent-selection"),
        reviewed_by=reviewer, acknowledge_inherited_normalization=True, data_dir=str(root/"data/cifar100")))
    return entry.read_approval()


def record_attempt(root, run_id, report, config=None, identity=None):
    result = dict(report=report, condition=os.environ.get("NRP_CONDITION"), seed=int(os.environ["NRP_SEED"]),
        dataset="torch/cifar100", source_commit=PIN, resolved_config=config, native_identity=identity,
        comparison_policy="Do NOT compare raw runtime/overhead to Lawrence without accounting for recorded hardware/software differences.")
    write(root/"runs"/run_id/"attempts"/(os.environ["POD_UID"]+".json"), result)
    return result


def smoke(entry, repo, root, report):
    import torch
    from experiments.gradient_flow.common import LimitedLoader, write_json
    from experiments.gradient_flow.method_policy import MethodRuntime
    from experiments.gradient_flow.intervene import rng_snapshot, restore_rng
    from experiments.gradient_flow.make_probe_batch import iter_batches
    approved = approval(entry,root)
    config = entry.resolved_config(approved,42)
    config["online"]["epochs"] = 2  # SMOKE ONLY; final 60-epoch config is never changed.
    destination = repo/"results/gradient_flow/nrp_cifar100_smoke"
    if destination.exists():
        raise ValueError("Existing smoke outputs are preserved; review them or choose a new study ID")
    destination.mkdir(parents=True)
    config_path = destination/"smoke.yml"
    import yaml
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    write_json(destination/"config.json", config)
    rng = rng_snapshot()
    try:
        batch = next(iter_batches(config["data_dir"],config,config["online"]["probe_batch_size"],split="train"))
    finally:
        restore_rng(rng)
    import train as official
    original_loader = official.create_loader
    official.create_loader = lambda *a, **kw: LimitedLoader(original_loader(*a, **kw),8 if kw.get("is_training") else 2)
    args = SimpleNamespace(config=str(config_path.relative_to(repo)),condition="response_aware")
    identity = dict(source_commit=PIN, synthetic=False, purpose="SMOKE ONLY: two epochs x eight real CIFAR100 training batches")
    try:
        entry.study.online_train.run_official(args,config,destination,batch,identity,None,None,runtime_class=MethodRuntime)
    finally:
        official.create_loader = original_loader
    rows = read(destination/"metrics.json")
    if (len(rows) != 2 or any(r["amp"]["optimizer_updates"] < 1 or len(r["gradient"]) != 4 or
            r["controller"]["alpha"][0] != 4. for r in rows)):
        raise ValueError("Smoke did not establish valid AMP updates/all-block diagnostics/block0 alpha4")
    marker = dict(key=environment_key(report), data_marker_sha256=digest(root/"checks/data-ready.json"),
        config_sha256=digest(config_path), metrics_sha256=digest(destination/"metrics.json"),
        output_directory=str(destination), source_commit=PIN, purpose=identity["purpose"], passed=True)
    immutable(root/"checks/smoke-ready.json",marker)
    record_attempt(root,"smoke",report,config)
    print("SMOKE PASSED; this is NOT a benchmark result",flush=True)


def train(entry,repo,root,report,run_id):
    condition, seed = os.environ["NRP_CONDITION"], int(os.environ["NRP_SEED"])
    if (condition,seed) not in RUNS or run_id != f"{condition}_seed{seed}":
        raise ValueError("Only the six declared Fixed/Deep-only seed42/43/44 runs are allowed")
    marker = read(root/"checks/smoke-ready.json")
    if (not marker["passed"] or marker["key"] != environment_key(report) or
            marker["data_marker_sha256"] != digest(root/"checks/data-ready.json")):
        raise ValueError("Matching environment/data/real-CIFAR100 smoke must pass before long jobs")
    approved = approval(entry,root)
    config = entry.resolved_config(approved,seed)
    native = "fixed" if condition == "fixed" else "final_sage"
    from experiments.gradient_flow.make_probe_batch import iter_batches
    from experiments.gradient_flow.intervene import rng_snapshot, restore_rng
    from experiments.gradient_flow.multi_batch_intervene import batch_fingerprint
    rng = rng_snapshot()
    try:
        batch = next(iter_batches(config["data_dir"],config,config["online"]["probe_batch_size"],split="train"))
    finally:
        restore_rng(rng)
    identity = entry.identity_for(config,approved,native,batch_fingerprint(batch))
    record_attempt(root,run_id,report,config,identity)
    entry.train(SimpleNamespace(seed=seed,condition=native,resume=True,data_dir=config["data_dir"]))
    native_path = repo/entry.OUTPUT/f"seed{seed}"/native
    if read(native_path/"status.json")["status"] != "complete":
        raise ValueError("Native training did not complete")
    write(root/"runs"/run_id/"result-index.json",dict(source_commit=PIN, native_condition=native,
        condition=condition, seed=seed, output_directory=str(native_path),
        provenance_sha256=digest(native_path/"provenance.json"), gpu_model=report["hardware"]["gpu_model"],
        node=report["node"], image_reference=report["image_reference"]))


def inspect_results(root):
    marker = root/"checks/smoke-ready.json"
    if marker.is_file():
        ready = read(marker)
        rows = read(Path(ready["output_directory"])/"metrics.json")
        print(json.dumps(dict(smoke=ready, epochs=[dict(epoch=r["epoch"], train=r["train"],
            validation=r["validation"], alpha=r["controller"]["alpha"], amp=r["amp"],
            gradient=r["gradient"]) for r in rows]), indent=2), flush=True)
    for condition,seed in sorted(RUNS):
        path = root/"runs"/f"{condition}_seed{seed}"
        for name in ("nrp-status.json","result-index.json"):
            if (path/name).is_file():
                print(str(path/name), json.dumps(read(path/name)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("env-check","data","smoke","inspect","train"))
    args = parser.parse_args()
    root, run_id = paths(), os.environ["NRP_RUN_ID"]
    status = root/"runs"/run_id/"nrp-status.json"
    with locked(root/"locks"/(run_id+".lock")):
        try:
            repo = checkout(root,run_id)
            entry = source_module(repo)
            report = snapshot(repo,gpu=args.mode in ("env-check","smoke","train"))
            write(status,dict(status="running",source_commit=PIN,pod_uid=os.environ["POD_UID"]))
            if args.mode == "env-check":
                immutable(root/"checks/environment-ready.json",dict(key=environment_key(report),passed=True))
                record_attempt(root,run_id,report)
            elif args.mode == "data":
                with locked(root/"locks/data.lock"):
                    parent = stage_parent(root,entry)
                    from torchvision.datasets import CIFAR100
                    for training in (True,False):
                        CIFAR100(root/"data/cifar100",train=training,download=True)
                    marker = dict(source_commit=PIN,parent=parent,files_sha256=data_hashes(root/"data/cifar100"))
                    immutable(root/"checks/data-ready.json",marker)
                    record_attempt(root,run_id,report)
            elif args.mode == "inspect":
                inspect_results(root)
                record_attempt(root,run_id,report)
            else:
                require_environment(root,report)
                require_data(root,entry)
                if args.mode == "smoke":
                    smoke(entry,repo,root,report)
                else:
                    train(entry,repo,root,report,run_id)
            write(status,dict(status="complete",source_commit=PIN,pod_uid=os.environ["POD_UID"]))
        except BaseException as error:
            paused = isinstance(error,SystemExit) and error.code == 75
            write(status,dict(status="paused" if paused else "failed",error=repr(error),
                              source_commit=PIN,pod_uid=os.environ["POD_UID"]))
            raise


if __name__ == "__main__":
    main()
