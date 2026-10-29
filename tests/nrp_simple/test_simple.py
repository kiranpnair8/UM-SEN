"""No cluster writes, downloads, Docker builds, or GPU runs in these tests."""

import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from jobs.nrp_simple import manifests as manifests
from jobs.nrp_simple import runner


REPO = Path(__file__).resolve().parents[2]
IMAGE = "example.org/simple@sha256:"+"a"*64


def save_manifests(path,documents):
    for name,doc in documents.items():
        (path/name).write_text(yaml.safe_dump(doc,sort_keys=False))
    (path/"cifar100-final-jobs.yaml").write_text(yaml.safe_dump_all(
        [d for d in documents.values() if d["metadata"]["labels"]["nrp-stage"]=="train"],sort_keys=False))


def test_exact_grid_plain_jobs_readonly_data_and_independent_paths(tmp_path):
    documents = manifests.render(IMAGE)
    assert len(documents)==8
    paths,pairs=set(),set()
    for d in documents.values():
        assert d["kind"]=="Job" and d["metadata"]["namespace"]=="kc-ai-research-lab"
        assert d["metadata"]["annotations"]["umsen/source-commit"]==runner.PIN
        assert d["spec"]["backoffLimit"]==0 and d["spec"]["parallelism"]==1 and d["spec"]["completions"]==1
        p=d["spec"]["template"]["spec"]
        assert p["serviceAccountName"]=="default" and not p["automountServiceAccountToken"]
        assert p["imagePullSecrets"]==[{"name":"ghcr-cred"}]
        assert p["nodeSelector"]=={"kubernetes.io/arch":"amd64"}
        assert not any(k in p for k in ("affinity","nodeName"))
        assert p["volumes"][0]["persistentVolumeClaim"]["claimName"]=="umsen-storage-rwx"
        assert not any("projected" in v for v in p["volumes"])
        c=p["containers"][0]
        env={e["name"]:e["value"] for e in c["env"] if "value" in e}
        assert env["NRP_SOURCE_COMMIT"]==runner.PIN and env["NRP_ROOT"]=="/storage/umsen-nrp/cifar100-simple"
        gpu=c["resources"]["requests"].get("nvidia.com/gpu")
        if d["metadata"]["labels"]["nrp-stage"]=="data":
            assert gpu is None
        else:
            assert gpu==c["resources"]["limits"]["nvidia.com/gpu"]=="1"
            data=next(v for v in c["volumeMounts"] if v["mountPath"]=="/storage/umsen-nrp/cifar100-simple/data")
            assert data["readOnly"] is True
        if env["NRP_CONDITION"]:
            pairs.add((env["NRP_CONDITION"],int(env["NRP_SEED"])))
            paths.add(str(runner.ROOT/"runs"/env["NRP_RUN_ID"]))
    assert pairs==runner.RUNS and len(paths)==6
    save_manifests(tmp_path,documents)
    assert manifests.validate(tmp_path)==documents


def test_validator_rejects_extra_coordination_resources(tmp_path):
    save_manifests(tmp_path,manifests.render(IMAGE))
    (tmp_path/"extra.yaml").write_text("kind: Role\n")
    with pytest.raises(ValueError,match="Unexpected YAML"):
        manifests.validate(tmp_path)


@pytest.mark.parametrize("field",["namespace","source","pvc","secret","account","output"])
def test_validator_rejects_unsafe_mutations(tmp_path,field):
    docs=manifests.render(IMAGE)
    d=docs["cifar100-fixed-seed42.yaml"]
    p=d["spec"]["template"]["spec"]
    if field=="namespace": d["metadata"]["namespace"]="wrong"
    elif field=="source": d["metadata"]["annotations"]["umsen/source-commit"]="wrong"
    elif field=="pvc": p["volumes"][0]["persistentVolumeClaim"]["claimName"]="wrong"
    elif field=="secret": p["imagePullSecrets"]=[]
    elif field=="account": p["serviceAccountName"]="custom"
    else: next(e for e in p["containers"][0]["env"] if e["name"]=="NRP_RUN_ID")["value"]="shared"
    save_manifests(tmp_path,docs)
    with pytest.raises(ValueError,match="Unexpected/unsafe"):
        manifests.validate(tmp_path)


def test_data_staged_once_valid_marker_reused_and_corruption_refused(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,"ROOT",tmp_path)
    monkeypatch.setattr(runner,"DATA",tmp_path/"data/cifar100")
    calls=[]
    class CIFAR100:
        classes=list(range(100))
        def __init__(self,path,train,download):
            self.train=train
            calls.append(download)
            if download:
                root=Path(path)/"cifar-100-python"
                root.mkdir(parents=True,exist_ok=True)
                for n in ("train","test","meta"):
                    (root/n).write_text(n)
        def __len__(self):
            return 50000 if self.train else 10000
    monkeypatch.setitem(sys.modules,"torchvision.datasets",SimpleNamespace(CIFAR100=CIFAR100))
    runner.prepare_data()
    original=(tmp_path/"data/CIFAR100_READY").read_bytes()
    assert calls.count(True)==2
    runner.prepare_data()
    runner.require_data()
    assert calls.count(True)==2
    (runner.DATA/"cifar-100-python/train").write_text("changed")
    with pytest.raises(ValueError,match="differs"):
        runner.prepare_data()
    assert (tmp_path/"data/CIFAR100_READY").read_bytes()==original
    assert calls.count(True)==2


def test_source_pin_and_private_copy_never_overwrites(tmp_path,monkeypatch):
    source=tmp_path/"image-source"
    source.mkdir()
    (source/"science.py").write_text("original")
    monkeypatch.setattr(runner,"SOURCE",source)
    monkeypatch.setattr(runner,"git",lambda p,*a:runner.PIN if a[0]=="rev-parse" else "")
    destination=tmp_path/"run"
    repo=runner.private_source(destination)
    assert (repo/"science.py").read_text()=="original"
    (repo/"science.py").write_text("local alteration")
    monkeypatch.setattr(runner,"git",lambda p,*a:runner.PIN if a[0]=="rev-parse" else " M science.py" if p==repo else "")
    with pytest.raises(ValueError,match="clean"):
        runner.private_source(destination)
    assert (repo/"science.py").read_text()=="local alteration"
    monkeypatch.setattr(runner,"git",lambda *a:"wrong SHA")
    with pytest.raises(ValueError,match="exactly"):
        runner.verify_source(repo)


def test_frozen_config_verified_from_existing_source():
    from experiments.gradient_flow import cifar100_followup as entry
    config=entry.study.resolve(entry.declaration(),"deep_only",1.)  # Test-only unused Deep-only deadband.
    runner.check_recipe(config)
    assert config["epochs"]==300 and config["online"]["epochs"]==60
    assert config["opt"]=="adamw" and config["batch_size"]==128 and config["lr"]==.0005
    assert config["warmup_epochs"]==20 and config["mixup"]==.5 and config["reprob"]==.25
    bad=copy.deepcopy(config)
    bad["online"]["controller"]["eta"]=.2
    with pytest.raises(ValueError,match="controller"):
        runner.check_recipe(bad)


def test_private_native_scope_needs_no_filesystem_mutex(tmp_path,monkeypatch):
    repo=tmp_path/"repo"
    repo.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys,"path",list(sys.path))
    fake=SimpleNamespace(__file__=str(repo/"experiments/gradient_flow/cifar100_followup.py"))
    monkeypatch.setattr(runner.importlib,"import_module",lambda name:fake)
    entry=runner.scientific_entry(repo)
    with entry.lock(repo/"results/.approval.lock"):
        pass
    assert not (repo/"results").exists()
    with pytest.raises(ValueError,match="escaped"):
        with entry.lock(tmp_path/"elsewhere.lock"):
            pass


def test_simple_runner_has_no_coordination_and_image_has_no_results():
    folder=REPO/"jobs/nrp_simple"
    code=(folder/"runner.py").read_text()
    assert all(token not in code for token in ("fcntl","flock","KubernetesLease","KUBERNETES_SERVICE_HOST","lease_lock"))
    assert "optimizer.step(" not in code
    docker=(folder/"Dockerfile").read_text()
    assert runner.PIN in docker and "COPY results" not in docker and "COPY data" not in docker
    assert 'ENTRYPOINT ["/opt/simple/entrypoint.sh"]' in docker
    prefix=["git","-c","safe.directory="+str(REPO),"-C",str(REPO)]
    assert subprocess.check_output(prefix+["diff",runner.PIN,"--","spikformer","experiments/gradient_flow","configs/gradient_flow","jobs/gradient_flow"])==b""
