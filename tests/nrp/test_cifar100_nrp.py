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
from jobs.nrp.validate_manifests import validate


ROOT = Path(__file__).resolve().parents[2]
IMAGE = "example.org/umsen@sha256:"+"1"*64


def test_exact_six_jobs_and_rwo_colocation():
    documents = generator.render(IMAGE, "example-node")
    finals = [v for v in documents.values() if v["metadata"]["labels"]["nrp-stage"] == "train"]
    assert len(finals) == 6 and len(documents) == 10
    pairs, paths = set(), set()
    for doc in documents.values():
        assert doc["metadata"]["namespace"] == "kc-ai-research-lab"
        assert doc["metadata"]["annotations"]["umsen/source-commit"] == runner.PIN == generator.SOURCE_COMMIT
        spec = doc["spec"]["template"]["spec"]
        assert spec["nodeSelector"]["kubernetes.io/hostname"] == "example-node"
        assert spec["restartPolicy"] == "Never" and doc["spec"]["backoffLimit"] == 0
        assert spec["volumes"][0]["persistentVolumeClaim"]["claimName"] == "umsen-storage"
        c = spec["containers"][0]
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


def test_generator_custom_digest_node_and_secret_roundtrip(tmp_path):
    documents = generator.render(IMAGE,"chosen-node","new-study-v1","registry-read")
    for name, doc in documents.items():
        (tmp_path/name).write_text(yaml.safe_dump(doc,sort_keys=False))
    finals = [d for d in documents.values() if d["metadata"]["labels"]["nrp-stage"] == "train"]
    (tmp_path/"cifar100-final-jobs.yaml").write_text(yaml.safe_dump_all(finals,sort_keys=False))
    assert validate(tmp_path,production=True) == documents
    assert all(v["spec"]["template"]["spec"]["imagePullSecrets"] == [{"name":"registry-read"}] for v in documents.values())
    changed = documents["cifar100-fixed-seed42.yaml"]
    changed["spec"]["template"]["spec"]["nodeSelector"]["kubernetes.io/hostname"] = "other-node"
    (tmp_path/"cifar100-fixed-seed42.yaml").write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError,match="mismatched"):
        validate(tmp_path)


@pytest.mark.parametrize("node",["", "../outside", "two nodes"])
def test_rwo_requires_concrete_single_node(node):
    with pytest.raises(ValueError):
        generator.render(IMAGE,node)


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


def test_environment_key_records_model_not_gpu_uuid():
    report = dict(source_commit=runner.PIN,python="3.10.12",versions=runner.EXPECTED,
        cuda_compile_version="12.1", image_reference=IMAGE, support_script_sha256="support",
        node="same-node",hardware=dict(gpu_model="Tesla V100",gpu_memory_bytes=32*1024**3,driver="driver"),
        gpu_uuid="device0")
    other = dict(report,gpu_uuid="device1")
    assert runner.environment_key(report) == runner.environment_key(other)
    other = copy.deepcopy(report)
    other["hardware"]["gpu_model"] = "different"
    assert runner.environment_key(report) != runner.environment_key(other)


def test_git_bootstrap_lock_serializes_without_weakening_run_locks(tmp_path,monkeypatch):
    calls = []
    fake = SimpleNamespace(LOCK_EX=2,LOCK_NB=4,LOCK_UN=8,flock=lambda fd,flags: calls.append(flags))
    monkeypatch.setitem(sys.modules,"fcntl",fake)
    with runner.locked(tmp_path/"git.lock",blocking=True):
        pass
    with runner.locked(tmp_path/"run.lock"):
        pass
    assert calls == [2,8,6,8]


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
        node="same-node",hardware={"gpu_model":"V100"})
    runner.write(tmp_path/"checks/environment-ready.json",dict(key=runner.environment_key(report),passed=True))
    runner.require_environment(tmp_path,report)
    with pytest.raises(ValueError,match="Environment check differs"):
        runner.require_environment(tmp_path,dict(report,image_reference="different"))
