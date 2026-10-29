"""NRP support unit checks: no cluster mutation, image pull or GPU execution."""

import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from jobs.nrp import generate_manifests as generator
from jobs.nrp import nrp_runner as runner
from jobs.nrp.validate_manifests import server_dry_run, validate


ROOT = Path(__file__).resolve().parents[2]
IMAGE = "example.org/umsen@sha256:"+"1"*64


def test_exact_six_jobs_and_rwx_independent_scheduling():
    documents = generator.render(IMAGE)
    finals = [v for v in documents.values() if v["metadata"]["labels"]["nrp-stage"] == "train"]
    assert len(finals) == 6 and len(documents) == 10
    pairs, paths = set(), set()
    for doc in documents.values():
        assert doc["metadata"]["namespace"] == "kc-ai-research-lab"
        assert doc["metadata"]["annotations"]["umsen/source-commit"] == runner.PIN == generator.SOURCE_COMMIT
        spec = doc["spec"]["template"]["spec"]
        assert spec["nodeSelector"] == {"kubernetes.io/arch":"amd64"}
        assert not any(key in spec for key in ("nodeName","affinity","topologySpreadConstraints"))
        assert spec["restartPolicy"] == "Never" and doc["spec"]["backoffLimit"] == 0
        assert spec["volumes"][0]["persistentVolumeClaim"]["claimName"] == "umsen-storage-rwx"
        assert doc["metadata"]["annotations"]["umsen/pvc-access-mode"] == "ReadWriteMany"
        c = spec["containers"][0]
        assert not any(e["name"] == "NRP_NODE" for e in c["env"])
        assert next(e["value"] for e in c["env"] if e["name"] == "NRP_PVC") == generator.PVC
        assert c["volumeMounts"][0]["mountPath"] == "/storage"
        assert c["resources"]["requests"] == c["resources"]["limits"]
        mode = doc["metadata"]["labels"]["nrp-stage"]
        assert c["resources"]["requests"].get("nvidia.com/gpu") == (None if mode in ("data","inspect") else "1")
        if mode == "train":
            env = {v["name"]:v["value"] for v in c["env"] if "value" in v}
            pairs.add((env["NRP_CONDITION"],int(env["NRP_SEED"])))
            paths.add(env["NRP_RUN_ID"])
            assert doc["spec"]["activeDeadlineSeconds"] == 172800
    assert pairs == runner.RUNS == set(generator.RUNS)
    assert len(paths) == 6


def test_checked_in_manifests_deterministic_and_digest_guard():
    actual = validate(ROOT/"jobs/nrp")
    assert actual == generator.render()
    with pytest.raises(ValueError,match="BUILT"):
        validate(ROOT/"jobs/nrp", production=True)


def test_generator_custom_digest_and_secret_roundtrip(tmp_path):
    documents = generator.render(IMAGE,"new-study-v1","registry-read")
    for name, doc in documents.items():
        (tmp_path/name).write_text(yaml.safe_dump(doc,sort_keys=False))
    finals = [d for d in documents.values() if d["metadata"]["labels"]["nrp-stage"] == "train"]
    (tmp_path/"cifar100-final-jobs.yaml").write_text(yaml.safe_dump_all(finals,sort_keys=False))
    (tmp_path/"cifar100-coordination.yaml").write_text(yaml.safe_dump_all(generator.coordination("new-study-v1"),sort_keys=False))
    assert validate(tmp_path,production=True) == documents
    assert all(v["spec"]["template"]["spec"]["imagePullSecrets"] == [{"name":"registry-read"}] for v in documents.values())
    changed = documents["cifar100-fixed-seed42.yaml"]
    changed["spec"]["template"]["spec"]["nodeSelector"]["kubernetes.io/hostname"] = "other-node"
    (tmp_path/"cifar100-fixed-seed42.yaml").write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError,match="mismatched"):
        validate(tmp_path)


@pytest.mark.parametrize("study",["", "../outside", "two studies"])
def test_study_namespace_validation(study):
    with pytest.raises(ValueError):
        generator.render(IMAGE,study)


def test_repo_pin_and_cleanliness(monkeypatch):
    monkeypatch.setattr(runner,"git",lambda p,*a: runner.PIN if a[0]=="rev-parse" else "")
    assert runner.verify_repo(Path("repo")) == runner.PIN
    monkeypatch.setattr(runner,"git",lambda p,*a: "wrong")
    with pytest.raises(ValueError,match="EXACT"):
        runner.verify_repo(Path("repo"))
    monkeypatch.setattr(runner,"git",lambda p,*a: runner.PIN if a[0]=="rev-parse" else " M model.py")
    with pytest.raises(ValueError,match="clean"):
        runner.verify_repo(Path("repo"))


def test_marker_never_silently_overwrites_old_study(tmp_path):
    path = tmp_path/"marker.json"
    runner.immutable(path,{"source":runner.PIN})
    before = path.read_bytes()
    runner.immutable(path,{"source":runner.PIN})
    assert path.read_bytes() == before
    with pytest.raises(ValueError,match="NEW study"):
        runner.immutable(path,{"source":"new"})
    assert path.read_bytes() == before


def test_environment_readiness_allows_different_nodes_and_gpus():
    report = dict(source_commit=runner.PIN,python="3.10.12",versions=runner.EXPECTED,
        cuda_compile_version="12.1", image_reference=IMAGE, support_script_sha256="support",
        namespace=runner.NAMESPACE,pvc=generator.PVC,mount_path="/storage",
        node="first-node",hardware=dict(gpu_model="Tesla V100",gpu_memory_bytes=32*1024**3,driver="driver"),
        gpu_uuid="device0")
    other = dict(report,gpu_uuid="device1")
    assert runner.environment_key(report) == runner.environment_key(other)
    other = copy.deepcopy(report)
    other["hardware"]["gpu_model"] = "different"
    other["hardware"]["driver"] = "different driver"
    other["node"] = "second-node"
    assert runner.environment_key(report) == runner.environment_key(other)
    # Hardware provenance was not stripped from the original attempt reports.
    assert report["hardware"]["gpu_model"] == "Tesla V100"
    assert other["hardware"]["gpu_model"] == "different"
    for key in ("source_commit","image_reference","pvc","support_script_sha256"):
        assert runner.environment_key(report) != runner.environment_key(dict(report,**{key:"changed"}))


def test_git_bootstrap_lock_serializes_without_weakening_run_locks(tmp_path,monkeypatch):
    from contextlib import contextmanager
    calls = []
    class FakeLease:
        def __init__(self, study, scope):
            self.scope = scope
        @contextmanager
        def held(self, **kwargs):
            calls.append((self.scope,kwargs["blocking"]))
            yield self
    monkeypatch.setenv("NRP_ROOT",str(tmp_path))
    monkeypatch.setattr(runner,"KubernetesLease",FakeLease)
    with runner.locked(tmp_path/"locks/git.lock",blocking=True):
        pass
    with runner.locked(tmp_path/"locks/data.lock"):
        pass
    assert calls == [("git",True),("data",False)]
    assert not (tmp_path/"locks").exists()


def test_training_and_smoke_paths_are_explicitly_different():
    source = (ROOT/"jobs/nrp/nrp_runner.py").read_text()
    assert 'config["online"]["epochs"] = 2' in source
    assert 'nrp_cifar100_smoke' in source
    assert 'LimitedLoader(original_loader(*a, **kw),8 if kw.get("is_training") else 2)' in source
    assert 'entry.train(SimpleNamespace(seed=seed,condition=native,resume=True' in source
    assert 'download=True' in source and 'download=False' in source
    assert "Do NOT compare raw runtime/overhead to Lawrence" in source
    assert "/home/rizk_lab" not in source and "/envs/snn" not in source


def test_docker_core_stack_and_no_scientific_source_copied():
    docker = (ROOT/"jobs/nrp/Dockerfile").read_text()
    req = (ROOT/"jobs/nrp/requirements.txt").read_text()
    assert "python3.10" in docker and "12.1.1-devel-ubuntu22.04@sha256:" in docker
    assert "torch==2.5.1+cu121 torchvision==0.20.1+cu121" in docker
    for pkg in ("timm==1.0.28","spikingjelly==0.0.0.0.14","numpy==2.2.6"):
        assert pkg in req
    assert "COPY spikformer" not in docker
    assert runner.PIN == "01ad13da81eed92b64247c5dc3fcd44608f31a1f"


def test_no_changes_to_pinned_scientific_tree():
    git = ["git","-c","safe.directory="+ROOT.as_posix(),"-C",str(ROOT)]
    protected = ["spikformer","experiments/gradient_flow","configs/gradient_flow","jobs/gradient_flow"]
    assert subprocess.check_output(git+["diff",runner.PIN,"--",*protected]) == b""


def test_real_data_guard_rejects_modified_files_before_training(tmp_path):
    data = tmp_path/"data/cifar100/cifar-100-python"
    data.mkdir(parents=True)
    for name in ("train","test","meta"):
        (data/name).write_bytes(name.encode())
    parent = {"selected_method":"deep_only"}
    entry = SimpleNamespace(parent_evidence=lambda path: parent)
    runner.write(tmp_path/"checks/data-ready.json",dict(source_commit=runner.PIN,
        parent=parent,files_sha256=runner.data_hashes(tmp_path/"data/cifar100")))
    (data/"train").write_bytes(b"modified")
    with pytest.raises(ValueError,match="changed"):
        runner.require_data(tmp_path,entry)


def test_training_refuses_missing_smoke_without_calling_native_pipeline(tmp_path,monkeypatch):
    monkeypatch.setenv("NRP_CONDITION","fixed")
    monkeypatch.setenv("NRP_SEED","42")
    entry = SimpleNamespace(train=lambda *a: pytest.fail("Native training must not start"))
    with pytest.raises(FileNotFoundError):
        runner.train(entry,tmp_path/"repo",tmp_path,{},"fixed_seed42")
    assert not (tmp_path/"repo").exists()


def test_environment_check_detects_different_image_before_training(tmp_path):
    report = dict(source_commit=runner.PIN,python="3.10.12",versions=runner.EXPECTED,
        cuda_compile_version="12.1",image_reference=IMAGE,support_script_sha256="support",
        namespace=runner.NAMESPACE,pvc=generator.PVC,mount_path="/storage",
        node="first-node",hardware={"gpu_model":"V100"})
    runner.write(tmp_path/"checks/environment-ready.json",dict(key=runner.environment_key(report),passed=True))
    runner.require_environment(tmp_path,report)
    runner.require_environment(tmp_path,dict(report,node="second-node",hardware={"gpu_model":"A100"}))
    with pytest.raises(ValueError,match="Environment check differs"):
        runner.require_environment(tmp_path,dict(report,image_reference="different"))


def test_server_validation_uses_bound_rwx_and_dry_runs_every_manifest(tmp_path,monkeypatch):
    pvc = dict(status={"phase":"Bound"},spec={"accessModes":["ReadWriteMany"],"storageClassName":"rook-cephfs"})
    commands = []
    monkeypatch.setattr(subprocess,"check_output",lambda args,**kw: json.dumps(pvc))
    monkeypatch.setattr(subprocess,"run",lambda args,**kw: commands.append(args))
    documents = generator.render(IMAGE)
    server_dry_run(tmp_path,documents)
    assert len(commands) == len(documents)+2 == 12
    assert all("--context=nautilus" in c and "--namespace=kc-ai-research-lab" in c and
               "--dry-run=server" in c for c in commands)
    assert Path(commands[-1][-1]).name == "cifar100-final-jobs.yaml"


@pytest.mark.parametrize("change",["access","class","phase"])
def test_server_validation_fails_before_apply_on_wrong_storage(monkeypatch,change):
    pvc = dict(status={"phase":"Bound"},spec={"accessModes":["ReadWriteMany"],"storageClassName":"rook-cephfs"})
    if change == "access":
        pvc["spec"]["accessModes"] = ["ReadWriteOnce"]
    elif change == "class":
        pvc["spec"]["storageClassName"] = "different"
    else:
        pvc["status"]["phase"] = "Pending"
    monkeypatch.setattr(subprocess,"check_output",lambda args,**kw: json.dumps(pvc))
    monkeypatch.setattr(subprocess,"run",lambda *a,**kw: pytest.fail("No apply before valid RWX storage"))
    with pytest.raises(ValueError,match="PVC"):
        server_dry_run(ROOT/"jobs/nrp",generator.render(IMAGE))
