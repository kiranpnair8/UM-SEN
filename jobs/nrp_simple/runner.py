"""Standalone CIFAR100 jobs. No Kubernetes client, file mutex or shared checkout."""

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from types import SimpleNamespace


PIN = "01ad13da81eed92b64247c5dc3fcd44608f31a1f"
ROOT = Path("/storage/umsen-nrp/cifar100-simple")
DATA = ROOT/"data/cifar100"
SOURCE = Path("/opt/scientific")
RUNS = {(c,s) for s in (42,43,44) for c in ("fixed","deep")}
EXPECTED = {"torch":"2.5.1+cu121","torchvision":"0.20.1+cu121","timm":"1.0.28",
    "spikingjelly":"0.0.0.0.14","numpy":"2.2.6"}
PARENT = {"final_selection.json":"results/gradient_flow/controller_method_ablation_seed42/final_selection.json",
    "study_manifest.json":"results/gradient_flow/controller_method_ablation_seed42/study_manifest.json",
    "summary.json":"results/gradient_flow/controller_method_ablation_seed42/summary.json",
    "controller_followup.yml":"configs/gradient_flow/controller_followup.yml"}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def git(repo,*args):
    return subprocess.check_output(["git","--no-optional-locks","-c","safe.directory="+str(repo),"-C",str(repo),*args],text=True).strip()


def verify_source(repo):
    if git(repo,"rev-parse","HEAD") != PIN or git(repo,"status","--porcelain"):
        raise ValueError("Scientific checkout must be clean at exactly "+PIN)
    return PIN


def private_source(destination):
    verify_source(SOURCE)
    repo = destination/"repo"
    if not repo.exists():
        shutil.copytree(SOURCE,repo)
    verify_source(repo)  # Partial copies fail safely; no reset or overwrite.
    return repo


def data_hashes():
    return {n:digest(DATA/"cifar-100-python"/n) for n in ("train","test","meta")}


def validate_data():
    from torchvision.datasets import CIFAR100
    for training,size in ((True,50000),(False,10000)):
        dataset = CIFAR100(str(DATA),train=training,download=False)
        if len(dataset) != size or len(dataset.classes) != 100:
            raise ValueError("Wrong CIFAR100 splits/classes")
    return dict(source_commit=PIN,dataset="cifar100",train_size=50000,test_size=10000,
        files_sha256=data_hashes())


def prepare_data():
    marker = ROOT/"data/CIFAR100_READY"
    if marker.exists():
        if read(marker) != validate_data():
            raise ValueError("Ready marker/data differs; preserve and review, do not overwrite")
        print("Existing CIFAR100_READY and data integrity verified; reused",flush=True)
        return
    DATA.mkdir(parents=True,exist_ok=True)
    from torchvision.datasets import CIFAR100
    for training in (True,False):
        CIFAR100(str(DATA),train=training,download=True)
    write(marker,validate_data())
    print("CIFAR100_READY: "+str(marker),flush=True)


def require_data():
    if read(ROOT/"data/CIFAR100_READY") != validate_data():
        raise ValueError("Shared CIFAR100 data/marker mismatch")


def scientific_entry(repo):
    os.chdir(repo)
    sys.path.insert(0,str(repo))
    entry = importlib.import_module("experiments.gradient_flow.cifar100_followup")
    if Path(entry.__file__).resolve() != repo/"experiments/gradient_flow/cifar100_followup.py":
        raise ValueError("Wrong scientific module")
    @contextmanager
    def independent_scope(path):
        # Only suppress infrastructure mutexes in this private standalone run.
        # Exactly one ordinary Job must target each deterministic directory.
        path = Path(path).absolute()
        if path.resolve() != path or not path.is_relative_to(repo):
            raise ValueError("Native output scope escaped the independent checkout")
        yield
    entry.lock = independent_scope
    return entry


def configuration(entry,destination,seed):
    evidence = destination/"parent-evidence"
    for name,relative in PARENT.items():
        incoming,target = Path("/parent-input")/name,evidence/relative
        if not incoming.is_file():
            raise ValueError("Missing original reviewed selection evidence: "+name)
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if target.read_bytes() != incoming.read_bytes():
                raise ValueError("Existing original selection evidence differs")
        else:
            target.write_bytes(incoming.read_bytes())
    reviewer = os.environ.get("NRP_REVIEWED_BY","")
    if not reviewer or os.environ.get("NRP_ACK_RECIPE") != "true":
        raise ValueError("Retain the existing scientific normalization review/approval ConfigMap")
    entry.prepare(SimpleNamespace(source_root=str(evidence),reviewed_by=reviewer,
        acknowledge_inherited_normalization=True,data_dir=str(DATA)))
    approval = entry.read_approval()
    config = entry.resolved_config(approval,seed)
    check_recipe(config)
    return approval,config


def check_recipe(c):
    expected = dict(dataset="torch/cifar100",num_classes=100,time_step=4,layer=4,dim=384,
        num_heads=12,patch_size=4,mlp_ratio=4,fixed_alpha=4.,sps_alpha=4.)
    if any(c[k] != v for k,v in expected.items()):
        raise ValueError("Frozen CIFAR100 backbone/surrogate settings changed")
    online,control = c["online"],c["online"]["controller"]
    if (online["epochs"] != 60 or online["update_every_epochs"] != 1 or online["reference_window_epochs"] != 5 or
        any(control[k] != v for k,v in dict(eta=.1,max_step=.1,alpha_min=1.,alpha_max=8.,reference_mode="frozen_window").items()) or
        online["method_design"]["variant"] != "deep_only" or
        online["method_design"]["parameters"] != dict(eta=.1,max_step=.1,gamma=0.,freeze_block0=True,use_deadband=False)):
        raise ValueError("Frozen Deep-only controller settings changed")


def runtime(gpu):
    import torch
    versions = {p:importlib.metadata.version(p) for p in EXPECTED}
    if versions != EXPECTED or sys.version_info[:2] != (3,10) or torch.version.cuda != "12.1":
        raise ValueError("Required software versions differ: "+str(versions))
    report = dict(source_commit=PIN,python=platform.python_version(),versions=versions,
        dataset="cifar100",seed=int(os.environ["NRP_SEED"]),condition=os.environ.get("NRP_CONDITION",""),
        cuda=torch.version.cuda,cuda_available=torch.cuda.is_available(),node=os.environ.get("POD_NODE"),
        pod=os.environ.get("POD_NAME"),pod_uid=os.environ.get("POD_UID"),image=os.environ["NRP_IMAGE"],
        support_sha256=digest(__file__),pvc="umsen-storage-rwx",data=str(DATA))
    if gpu:
        subprocess.run(["nvidia-smi"],check=True)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Exactly one accessible CUDA GPU is required")
        print("torch.__version__:",torch.__version__,"torch.version.cuda:",torch.version.cuda,flush=True)
        print("torch.cuda.is_available():",torch.cuda.is_available(),flush=True)
        print("torch.cuda.get_device_name(0):",torch.cuda.get_device_name(0),flush=True)
        print("torch.cuda.get_device_properties(0):",torch.cuda.get_device_properties(0),flush=True)
        import cupy
        assert int(cupy.arange(4).sum()) == 6
        cupy.cuda.runtime.deviceSynchronize()
        p = torch.cuda.get_device_properties(0)
        report.update(gpu_model=p.name,gpu_memory_bytes=p.total_memory,
            driver=subprocess.check_output(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],text=True).strip())
    print(json.dumps(report,indent=2),flush=True)
    return report


def smoke(config,destination):
    import torch
    from spikingjelly.clock_driven import functional
    from experiments.gradient_flow.common import build_model,seed_all
    from experiments.gradient_flow.alpha_scope import BlockAlphas
    from experiments.gradient_flow.make_probe_batch import iter_batches
    from experiments.gradient_flow.controller import ControllerConfig
    from experiments.gradient_flow.online_probe import OnlineProbe
    from experiments.gradient_flow.method_policy import MethodPolicy
    seed_all(42)
    batch = next(iter_batches(str(DATA),config,config["online"]["probe_batch_size"],split="train"))
    model = build_model(config,backend="cupy").cuda().eval()
    optimizer = torch.optim.AdamW(model.parameters(),lr=config["lr"],weight_decay=config["weight_decay"])
    scope = BlockAlphas(model)
    original = scope.snapshot()
    def backward():
        functional.reset_net(model)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["images"].cuda())
        loss = torch.nn.functional.cross_entropy(logits,batch["targets"].cuda())
        loss.backward()
        if not torch.isfinite(loss) or not any(p.grad is not None for p in model.parameters()) or any(
                p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("Smoke backward/loss is nonfinite or absent")
        return logits.detach(),float(loss.detach())
    try:
        before,loss = backward()
        scope.set(1,5.)
        assert scope.get() == [4.,5.,4.,4.]
        assert all(a==4. for n,a in scope.snapshot().items() if n.startswith("patch_embed."))
        after,changed_loss = backward()
        if not torch.equal(before,after) or loss != changed_loss:
            raise ValueError("Surrogate alpha changed forward behavior")
        scope.restore(original)
        functional.reset_net(model)
        optimizer.zero_grad(set_to_none=True)
        probe = OnlineProbe(model,batch,ControllerConfig(**config["online"]["controller"]),optimizer)
        observation = probe([4.]*4,estimate=True)
        policy = MethodPolicy(4,"response_aware",config["online"],42)
        control = policy.advance(0,observation)
        assert control["alpha"][0] == 4. and len(observation["Rhat"]) == 4
        report = dict(passed=True,synthetic=False,loss=loss,alpha_modification=[4.,5.,4.,4.],
            forward_unchanged=True,optimizer_updates=0,controller=control,probe=observation,
            note="Real CIFAR100 CuPy FP32 plumbing smoke; no training/accuracy claim")
        write(destination/"smoke.json",report)  # Also verifies persistent write access.
        print("SMOKE PASSED "+json.dumps(dict(loss=loss,G=observation["G"],Rhat=observation["Rhat"],
            alpha=control["alpha"],forward_unchanged=True,optimizer_updates=0)),flush=True)
        return report
    finally:
        scope.restore(original)
        functional.reset_net(model)
        optimizer.zero_grad(set_to_none=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("data","smoke","train"))
    args = parser.parse_args()
    seed,condition = int(os.environ["NRP_SEED"]),os.environ.get("NRP_CONDITION","")
    run_id = os.environ["NRP_RUN_ID"]
    expected = f"{condition}_seed{seed}" if args.mode=="train" else args.mode
    if (os.environ.get("NRP_ROOT") != str(ROOT) or os.environ.get("NRP_SOURCE_COMMIT") != PIN or
            run_id != expected or (args.mode=="train" and (condition,seed) not in RUNS)):
        raise ValueError("Unexpected standalone run identity/path/source")
    destination = ROOT/"runs"/run_id
    if destination.resolve() != destination:
        raise ValueError("Symlinked run destination refused")
    destination.mkdir(parents=True,exist_ok=True)
    status = destination/"status.json"
    try:
        verify_source(SOURCE)
        report = runtime(args.mode!="data")
        write(destination/"attempts"/(os.environ["POD_UID"]+".json"),report)
        write(status,dict(status="running",source_commit=PIN,seed=seed,condition=condition))
        if args.mode=="data":
            prepare_data()
        else:
            require_data()
            repo = private_source(destination)
            entry = scientific_entry(repo)
            approval,config = configuration(entry,destination,seed)
            config_path = destination/"resolved_config.json"
            if config_path.exists() and read(config_path) != config:
                raise ValueError("Existing resolved configuration differs; refusing overwrite")
            if not config_path.exists():
                write(config_path,config)
            if args.mode=="smoke":
                smoke(config,destination)
            else:
                native = "fixed" if condition=="fixed" else "final_sage"
                entry.train(SimpleNamespace(seed=seed,condition=native,resume=True,data_dir=str(DATA)))
                outputs = repo/entry.OUTPUT/f"seed{seed}"/native
                if read(outputs/"status.json")["status"] != "complete":
                    raise ValueError("Native training did not complete")
                result = dict(source_commit=PIN,seed=seed,condition=condition,dataset="cifar100",
                    output_directory=str(outputs),summary=read(outputs/"summary.json"),runtime=report,
                    provenance=read(outputs/"provenance.json")["identity"])
                write(destination/"result.json",result)
                print("SIMPLE_RESULT_JSON="+json.dumps(result,allow_nan=False),flush=True)
        write(status,dict(status="complete",source_commit=PIN,seed=seed,condition=condition))
    except BaseException as error:
        paused = isinstance(error,SystemExit) and error.code==75
        write(status,dict(status="paused" if paused else "failed",source_commit=PIN,error=repr(error)))
        raise


if __name__ == "__main__":
    main()
