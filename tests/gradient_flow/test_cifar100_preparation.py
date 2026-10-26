"""CPU/config-only preparation checks; no download, GPU or HPC job access."""

import argparse
import copy
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from experiments.gradient_flow import cifar100_followup as entry
from experiments.gradient_flow.common import ROOT, build_model, load_config, write_json
from experiments.gradient_flow.controller_calibration import lock


def test_explicit_adaptation_changes_only_dataset_classes_and_namespace():
    c = entry.declaration()
    b = load_config(ROOT/entry.study.prior.BASE)
    changed = {k for k in b if b[k] != c[k]}
    assert changed == {"dataset", "num_classes", "output"}
    assert c["mean"] == [.4914, .4822, .4465] and c["std"] == [.2470, .2435, .2616]
    assert c["dataset"] == "torch/cifar100" and c["num_classes"] == 100
    assert c["online"]["epochs"] == 60 and c["epochs"] == 300 and c["warmup_epochs"] == 20
    assert c["cifar100_audit"]["native_cifar100_recipe_found"] is False


def test_head_optimizer_and_scheduler_are_resolved_on_cpu(monkeypatch):
    torch.set_num_threads(2)
    # load_config installs the existing CIFAR module path; parser is CPU-only.
    c = entry.declaration()
    import train as official
    monkeypatch.setattr("sys.argv", ["audit", "--config", str(ROOT/entry.CONFIG)])
    args, _ = official._parse_args()
    model = build_model(c, "torch")
    assert tuple(model.head.weight.shape) == (100, 384) and tuple(model.head.bias.shape) == (100,)
    assert len(model.block) == 4 and model.T == 4
    from timm.optim import create_optimizer
    from timm.scheduler import create_scheduler
    optimizer = create_optimizer(args, model)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["betas"] == (.9, .999) and optimizer.defaults["eps"] == 1e-8
    assert set(g["weight_decay"] for g in optimizer.param_groups) == {0., .06}
    scheduler, horizon = create_scheduler(args, optimizer)
    assert horizon == 310 and scheduler.t_initial == 300 and scheduler.warmup_t == 20
    assert args.clip_grad is None and not args.no_prefetcher
    assert not args.pin_mem and args.hflip == .5 and args.vflip == 0


def test_dataset_factory_uses_cifar100_fine_label_class_without_download(monkeypatch):
    import timm.data.dataset_factory as factory
    calls = []
    class Dataset:
        def __init__(self, root, train, download, **kwargs):
            calls.append((root, train, download))
            self.classes = list(range(100))
    monkeypatch.setitem(factory._TORCH_BASIC_DS, "cifar100", Dataset)
    train = factory.create_dataset("torch/cifar100", "unit-no-data", split="train", is_training=True)
    val = factory.create_dataset("torch/cifar100", "unit-no-data", split="validation", is_training=False)
    assert calls == [("unit-no-data", True, False), ("unit-no-data", False, False)]
    assert len(train.classes) == len(val.classes) == 100


def test_transforms_sampling_mixup_targets_are_cifar100_compatible():
    c = entry.declaration()
    from transforms_factory import create_transform
    from loader import create_loader
    from timm.data import FastCollateMixup
    import numpy as np
    from PIL import Image
    transform = create_transform((3,32,32), is_training=False, use_prefetcher=False,
        mean=c["mean"], std=c["std"], crop_pct=1., interpolation="bicubic")
    image = Image.fromarray(np.zeros((32,32,3), dtype=np.uint8))
    result = transform(image)
    assert result.shape == (3,32,32)
    assert result[:,0,0].tolist() == pytest.approx([-m/s for m,s in zip(c["mean"],c["std"])])
    class Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return 50000
        def __getitem__(self, i):
            return image, i % 100
    train = create_loader(Dataset(), input_size=(3,32,32), batch_size=128, is_training=True,
        use_prefetcher=False, num_workers=0, persistent_workers=False, mean=c["mean"], std=c["std"])
    val = create_loader(Dataset(), input_size=(3,32,32), batch_size=128, is_training=False,
        use_prefetcher=False, num_workers=0, persistent_workers=False, mean=c["mean"], std=c["std"])
    assert isinstance(train.sampler, torch.utils.data.RandomSampler) and train.drop_last
    assert isinstance(val.sampler, torch.utils.data.SequentialSampler) and not val.drop_last
    assert len(train) == 390
    mix = FastCollateMixup(mixup_alpha=.5, cutmix_alpha=0., prob=1.,
        mode="batch", label_smoothing=.1, num_classes=100)
    _, target = mix([(np.zeros((3,32,32),dtype=np.uint8),99), (np.zeros((3,32,32),dtype=np.uint8),42)])
    assert target.shape == (2,100) and torch.allclose(target.sum(1), torch.ones(2))


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    config = entry.declaration("existing-cifar100-cache")
    parent_bytes = (ROOT/"configs/gradient_flow/controller_followup.yml").read_text().encode()
    new, old = tmp_path/"cifar100", tmp_path/"active-cifar10"
    source = dict(commit="unit-new-cifar100-commit", dirty=False, torch="unit", cuda=None, packages={})
    parent_root = old/entry.study.OUTPUT
    parent_root.mkdir(parents=True)
    (old/"configs/gradient_flow").mkdir(parents=True)
    (old/"configs/gradient_flow/controller_followup.yml").write_bytes(parent_bytes)
    manifest = dict(deadband=dict(value=.02), variant_configs=dict(deep_only=dict(online=dict(
        method_design=dict(parameters=entry.study.VARIANTS["deep_only"])))))
    summary = dict(ranking=dict(eligible_in_descriptive_order=["deep_only"]),
        gates=dict(deep_only=dict(promising=False)))
    write_json(parent_root/"study_manifest.json", manifest)
    write_json(parent_root/"summary.json", summary)
    selection = dict(approved=True, final_sage="deep_only", reviewed_by="unit",
        implementation_commit=entry.PARENT_COMMIT, deadband=.02,
        ablation_manifest_sha256=entry.digest(parent_root/"study_manifest.json"),
        ablation_summary_sha256=entry.digest(parent_root/"summary.json"),
        followup_config_sha256=entry.digest(old/"configs/gradient_flow/controller_followup.yml"))
    write_json(parent_root/"final_selection.json", selection)
    active = old/"results/gradient_flow/controller_followup_cifar10/seed42/fixed"
    active.mkdir(parents=True)
    write_json(active/"status.json", dict(status="running", source_commit=entry.PARENT_COMMIT))
    (new/Path(entry.CONFIG).parent).mkdir(parents=True)
    (new/entry.CONFIG).write_bytes((ROOT/entry.CONFIG).read_bytes())
    monkeypatch.setattr(entry, "ROOT", new)
    monkeypatch.setattr(entry, "provenance", lambda: copy.deepcopy(source))
    def declared(data_dir=None):
        value = copy.deepcopy(config)
        if data_dir is not None:
            value["data_dir"] = data_dir
        return value
    monkeypatch.setattr(entry, "declaration", declared)
    monkeypatch.setattr(entry.subprocess, "check_output", lambda *a, **k: parent_bytes)
    before = {str(p):p.read_bytes() for p in old.rglob("*") if p.is_file()}
    args = SimpleNamespace(source_root=str(old), reviewed_by="unit",
        acknowledge_inherited_normalization=True, data_dir=config["data_dir"])
    entry.prepare(args)
    return new, old, before, args


def test_separate_approval_preserves_active_cifar10_and_commit_attribution(prepared):
    new, old, before, args = prepared
    approval = entry.read_approval()
    assert approval["implementation_commit"] == "unit-new-cifar100-commit"
    assert approval["parent"]["selection_commit"] == entry.PARENT_COMMIT
    assert approval["parent"]["stabilization_target_met"] is False
    assert before == {str(p):p.read_bytes() for p in old.rglob("*") if p.is_file()}
    raw = (new/entry.APPROVAL).read_bytes()
    entry.prepare(args)
    assert raw == (new/entry.APPROVAL).read_bytes()


@pytest.mark.parametrize("seed", entry.SEEDS)
def test_fixed_sage_exact_recipe_pairing_and_independent_namespaces(prepared, seed):
    _, _, _, _ = prepared
    approval = entry.read_approval()
    config = entry.resolved_config(approval, seed)
    a = entry.identity_for(config, approval, "fixed", "same-probe")
    b = entry.identity_for(config, approval, "final_sage", "same-probe")
    assert {**a,"method":None} == {**b,"method":None}
    assert config["seed"] == seed and config["num_classes"] == 100
    assert config["online"]["method_design"]["parameters"]["freeze_block0"]
    assert config["online"]["method_design"]["parameters"]["gamma"] == 0


def test_source_mutation_and_missing_acknowledgment_fail_closed(prepared):
    _, old, _, args = prepared
    with pytest.raises(ValueError, match="acknowledge"):
        entry.prepare(SimpleNamespace(**{**vars(args),"acknowledge_inherited_normalization":False}))
    path = old/entry.study.OUTPUT/"summary.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="evidence changed"):
        entry.read_approval()


def test_scoped_checkpoint_load_hash_and_issuer_guards(prepared):
    new, _, _, _ = prepared
    approval = entry.read_approval()
    config = entry.resolved_config(approval,42)
    identity = entry.identity_for(config,approval,"fixed","unit-probe")
    directory = new/entry.OUTPUT/"seed42/fixed"
    directory.mkdir(parents=True)
    rows = [dict(epoch=i) for i in range(2)]
    saved = dict(version=2, epoch=1, args=argparse.Namespace(unit_only=True),
        gradient_flow_online=dict(identity=identity, rows=rows))
    path = directory/"last.pth.tar"
    torch.save(saved,path)
    for name,payload in (("config.json",config),("provenance.json",dict(identity=identity)),
            ("run_manifest.json",dict(identity=identity,config=config)),("metrics.json",rows),
            ("checkpoint_integrity.json",dict(identity=identity,epoch=1,sha256=entry.digest(path)))):
        write_json(directory/name,payload)
    capability = entry.authorize(directory,identity,config)
    assert capability.load()["args"].unit_only is True
    unissued = entry.Cifar100Checkpoint(path,capability.sha256,identity,1)
    with pytest.raises(ValueError,match="Unvalidated"):
        unissued.load()
    with path.open("ab") as stream:
        stream.write(b"unit-tamper")
    with pytest.raises(ValueError,match="Unvalidated"):
        capability.load()


def test_array_parallel_structure_and_unique_locks(tmp_path):
    text = (ROOT/"jobs/gradient_flow/run_cifar100_deep_only_followup.sh").read_text()
    active = (ROOT/"jobs/gradient_flow/run_controller_followup.sh").read_text()
    assert "#SBATCH --array=0-5" in text and "#SBATCH --array=0-5" in active
    assert "SEED=$((42+TASK/2))" in text and "CONDITION=final_sage" in text
    assert "#SBATCH --time=2-00:00:00" in text and "#SBATCH --nodelist" not in text
    assert "--prepare" not in text and "git pull" not in text
    with lock(tmp_path/".fixed.lock"):
        with pytest.raises(ValueError,match="Another"):
            with lock(tmp_path/".fixed.lock"):
                pytest.fail("Duplicate")
        with lock(tmp_path/".final_sage.lock"):
            pass


def test_existing_cifar10_sources_configs_and_launcher_are_unchanged():
    protected = ["spikformer", "configs/gradient_flow/controller_followup.yml",
        "configs/gradient_flow/controller_online.yml", "experiments/gradient_flow/controller_followup.py",
        "experiments/gradient_flow/online_train.py", "experiments/gradient_flow/online_runtime.py",
        "experiments/gradient_flow/method_policy.py", "experiments/gradient_flow/online_checkpoint.py",
        "jobs/gradient_flow/run_controller_followup.sh"]
    git = ["git","-c","safe.directory="+ROOT.as_posix(),"-C",str(ROOT)]
    assert subprocess.check_output(git+["diff",entry.PARENT_COMMIT,"--",*protected]) == b""
