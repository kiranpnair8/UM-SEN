"""CPU-only infrastructure tests using small stateful protocol doubles."""

import argparse
import importlib.util
from pathlib import Path
import pickle
import sys
import types

import pytest


CIFAR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CIFAR))
import experiment_state as infrastructure


class StateObject:
    def __init__(self, state=None):
        self.state = state or {}

    def state_dict(self):
        return self.state.copy()

    def load_state_dict(self, state, **kwargs):
        self.state = state.copy()


@pytest.fixture
def controllers(monkeypatch):
    # Import the real controller classes without importing the GPU trainer.
    trainer = types.ModuleType("train")
    trainer.parser = argparse.ArgumentParser()
    for name in ("_parse_args", "create_model", "train_one_epoch"):
        setattr(trainer, name, lambda *args, **kwargs: None)
    torch = types.ModuleType("torch")
    torch.is_tensor = lambda value: False
    neuron = types.ModuleType("spikingjelly.clock_driven.neuron")

    class LIF:
        def __init__(self):
            self.surrogate_function = types.SimpleNamespace(alpha=4.0)

    neuron.MultiStepLIFNode = LIF
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "train", trainer)
    monkeypatch.setitem(sys.modules, "spikingjelly", types.ModuleType("spikingjelly"))
    monkeypatch.setitem(sys.modules, "spikingjelly.clock_driven", types.ModuleType("spikingjelly.clock_driven"))
    monkeypatch.setitem(sys.modules, "spikingjelly.clock_driven.neuron", neuron)
    spec = importlib.util.spec_from_file_location("controller_under_test", CIFAR / "train_umsen.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, LIF


def controller_model(LIF):
    blocks = []
    for _ in range(4):
        node = LIF()
        blocks.append(types.SimpleNamespace(attn=types.SimpleNamespace(training=True), modules=lambda node=node: [node]))
    return types.SimpleNamespace(block=blocks, training=True, forward=lambda *args: "logits")


def controller_args():
    return types.SimpleNamespace(umsen_ema_beta=0.95, umsen_entropy_temperature=0.25,
                                 umsen_alpha_min=3.0, umsen_alpha_max=5.0, umsen_warmup_steps=100)


def test_initialization_is_weights_only(controllers):
    module, LIF = controllers
    controller = module.UMSENController(controller_model(LIF), controller_args())
    model = StateObject()
    optimizer, scheduler, scaler = StateObject(), StateObject(), StateObject()
    checkpoint = {"state_dict": {"module.weight": 123}, "optimizer": {"momentum": 99},
                  "epoch": 262, "metric": 94.81, "sage_training_state": {"step": 999}}
    result = infrastructure.initialize_model(model, checkpoint)
    assert model.state == {"weight": 123}
    assert optimizer.state == scheduler.state == scaler.state == {}
    assert result["start_epoch"] == 0 and result["best_metric"] is None
    assert result["source_metadata"] == {"epoch": 262, "metric": 94.81}
    assert controller.step == 0 and controller.controllers[0].running_count == 0
    assert not result["controller_restored"]


def test_controller_roundtrip_and_alpha_logging(controllers):
    module, LIF = controllers
    controller = module.UMSENController(controller_model(LIF), controller_args())
    controller.current_epoch, controller.step = 2, 333
    block = controller.controllers[0]
    block.running_count, block.running_mean, block.running_m2 = 31, 0.21, 0.03
    block.initialized, block.ema_dispersion, block.raw_dispersion = True, 0.3, 0.4
    block.normalized_z, block.centered_z = 0.7, 0.8
    block.set_centered_alpha(0.8)
    block.raw_history.append(0.4)
    controller.history.append({"epoch": 1})
    # New observation is a candidate, not the alpha already applied.
    assert block.current()["applied_alpha"] == 4.0
    assert block.current()["candidate_alpha"] != 4.0
    restored = module.UMSENController(controller_model(LIF), controller_args())
    restored.load_state_dict(pickle.loads(pickle.dumps(controller.state_dict())))
    assert restored.state_dict() == controller.state_dict()
    restored.apply_step_alphas()
    assert restored.controllers[0].applied_alpha == block.alpha
    assert not restored.warmup_active()
    restored.current_epoch = 0
    assert restored.current_alphas() == [4.0] * 4


def test_resume_extra_state_and_legacy_warning(controllers):
    module, LIF = controllers
    original = module.UMSENController(controller_model(LIF), controller_args())
    original.step = 1000
    resumed = module.UMSENController(controller_model(LIF), controller_args())
    scheduler = StateObject()
    optimizer = types.SimpleNamespace(param_groups=[{"lr": 1e-5}, {"lr": 2e-5}])
    checkpoint = {"sage_training_state": {"scheduler": {"last_epoch": 7},
                  "controller": original.state_dict(), "best_metric": 95.0, "best_epoch": 6},
                  "optimizer": {"param_groups": [{"lr": 1e-5}, {"lr": 2e-5}]}}
    result = infrastructure.restore_extra_state(checkpoint, scheduler, resumed)
    assert scheduler.state == {"last_epoch": 7}
    assert optimizer.param_groups == [{"lr": 1e-5}, {"lr": 2e-5}]
    assert resumed.step == 1000 and result["best_metric"] == 95.0
    with pytest.warns(UserWarning, match="Legacy checkpoint"):
        infrastructure.restore_extra_state({}, scheduler, resumed)


def test_fixed_alpha_targets_blocks_not_sps(controllers):
    _, LIF = controllers
    blocks = [[LIF() for _ in range(7)] for _ in range(4)]
    sps = [LIF() for _ in range(5)]
    scope = lambda nodes: types.SimpleNamespace(modules=lambda: nodes)
    model = types.SimpleNamespace(block=[scope(nodes) for nodes in blocks], patch_embed=scope(sps))
    assert infrastructure.apply_fixed_alpha(model, 4.0) == {"transformer": 28, "sps": 5}
    assert all(node.surrogate_function.alpha == 4.0 for nodes in blocks + [sps] for node in nodes)
    for alpha in (3.0, 5.0):
        infrastructure.apply_fixed_alpha(model, alpha)
        assert all(node.surrogate_function.alpha == alpha for nodes in blocks for node in nodes)
        assert all(node.surrogate_function.alpha == 4.0 for node in sps)
    infrastructure.apply_fixed_alpha(model, 4.0, sps_alpha=3.5)
    assert all(node.surrogate_function.alpha == 3.5 for node in sps)


@pytest.mark.parametrize("checkpoint,key,value", [
    ({"state_dict": {"weight": 1}, "model_state_dict": {"weight": 2}, "model": {"weight": 3}, "state_dict_ema": {"weight": 9}}, "state_dict", 1),
    ({"model_state_dict": {"weight": 2}, "model": {"weight": 3}}, "model_state_dict", 2),
    ({"model": {"weight": 3}, "model_ema": {"weight": 9}}, "model", 3),
    ({"weight": 4}, "raw_state_dict", 4),
])
def test_weight_key_precedence(checkpoint, key, value, caplog):
    caplog.set_level("INFO", logger="train")
    model = StateObject()
    result = infrastructure.initialize_model(model, checkpoint)
    assert model.state == {"weight": value} and result["model_weight_key"] == key
    assert key in caplog.text


def test_ema_only_by_explicit_request():
    checkpoint = {"state_dict": {"weight": 1}, "state_dict_ema": {"weight": 2}, "model_ema": {"weight": 3}}
    model = StateObject()
    result = infrastructure.initialize_model(model, checkpoint, use_ema=True)
    assert model.state == {"weight": 2} and result["model_weight_key"] == "state_dict_ema"
    result = infrastructure.initialize_model(model, {"model_ema": {"weight": 3}}, use_ema=True)
    assert result["model_weight_key"] == "model_ema"
    with pytest.raises(ValueError, match="No requested model weights"):
        infrastructure.initialize_model(model, {"state_dict_ema": {"weight": 2}})
    with pytest.raises(ValueError, match="No requested model weights"):
        infrastructure.initialize_model(model, {"state_dict": {"weight": 1}}, use_ema=True)


def test_invalid_primary_key_does_not_silently_fall_back():
    with pytest.raises(ValueError, match="Selected weight key"):
        infrastructure.initialize_model(StateObject(), {"state_dict": [], "model": {"weight": 3}})


def test_resume_order_preserves_optimizer_group_lrs():
    optimizer = types.SimpleNamespace(param_groups=[{"lr": 0.1}, {"lr": 0.2}])

    class WarmupScheduler(StateObject):
        def __init__(self, optimizer):
            super().__init__()
            for group in optimizer.param_groups:
                group["lr"] = 1e-6

    scheduler = WarmupScheduler(optimizer)
    # timm's optimizer.load_state_dict occurs after construction.
    optimizer.param_groups = [{"lr": 0.003}, {"lr": 0.006}]
    checkpoint = {"sage_training_state": {"scheduler": {"base_values": [0.1, 0.2]}}}
    infrastructure.restore_extra_state(checkpoint, scheduler)
    assert optimizer.param_groups == [{"lr": 0.003}, {"lr": 0.006}]
    assert scheduler.state["base_values"] == [0.1, 0.2]

    import ast
    tree = ast.parse((CIFAR / "train.py").read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
    resume_line = next(node.lineno for node in calls if isinstance(node.func, ast.Name) and node.func.id == "resume_checkpoint")
    assert any(node.lineno < resume_line for node in calls if isinstance(node.func, ast.Name) and node.func.id == "create_scheduler")


@pytest.mark.parametrize("output,experiment", [
    ("results/umsen_official_recipe", "baseline_seed42"),
    ("results/sage_v2", "../umsen_official_recipe/baseline_seed42"),
    ("results/sage_v2", "/tmp/historical"),
])
def test_reject_historical_output(tmp_path, output, experiment):
    with pytest.raises(ValueError, match="results/sage_v2"):
        infrastructure.safe_output_path(tmp_path / output, experiment, tmp_path)


def test_safe_namespace(tmp_path):
    path = infrastructure.safe_output_path(tmp_path / "results/sage_v2", "cifar10/fixed/run", tmp_path)
    assert path == tmp_path / "results/sage_v2/cifar10/fixed/run"


def test_ambiguous_initialization_rejected():
    args = types.SimpleNamespace(init_checkpoint="checkpoint", resume="resume", initial_checkpoint="")
    with pytest.raises(ValueError, match="cannot be combined"):
        infrastructure.validate_experiment_args(args)


def test_initialization_epoch_override_rejected():
    args = types.SimpleNamespace(init_checkpoint="checkpoint", resume="", initial_checkpoint="", start_epoch=10)
    with pytest.raises(ValueError, match="epoch 0"):
        infrastructure.validate_experiment_args(args)


def test_source_hash_read_only(tmp_path):
    import hashlib
    path = tmp_path / "source.pth.tar"
    contents = b"read-only source evidence"
    path.write_bytes(contents)
    before = path.stat().st_mtime_ns
    metadata = infrastructure.source_provenance(path)
    assert metadata["sha256"] == hashlib.sha256(contents).hexdigest()
    assert path.read_bytes() == contents and path.stat().st_mtime_ns == before


def test_sage_state_cannot_silently_resume_as_fixed():
    with pytest.raises(ValueError, match="SAGE resume requires"):
        infrastructure.restore_extra_state({"sage_training_state": {"controller": {"step": 3}}}, None)


def test_inherited_config_preserves_recipe():
    pytest.importorskip("yaml")
    original, _ = infrastructure.load_config(CIFAR / "cifar10.yml")
    pilot, sources = infrastructure.load_config(infrastructure.REPO_ROOT / "configs/sage_v2/fixed_cifar10_adaptation.yml")
    for key in ("opt", "weight_decay", "batch_size", "aa", "mixup", "cutmix", "reprob", "smoothing", "amp", "dim", "num_heads", "layer"):
        assert pilot[key] == original[key]
    assert pilot["epochs"] == 30 and pilot["warmup_epochs"] == 0 and len(sources) == 2


def test_saver_enriches_checkpoint_before_publication(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint"
    fake_torch = types.ModuleType("torch")
    fake_torch.save = lambda state, filename: Path(filename).write_bytes(pickle.dumps(state))
    fake_torch.load = lambda filename, **kwargs: pickle.loads(Path(filename).read_bytes())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(infrastructure, "capture_rng", lambda loaders: {"paired": 42})

    class Saver:
        def __init__(self):
            self.model = StateObject()
            self.best_metric, self.best_epoch = 90.0, 3
            self.cmp = lambda a, b: a > b

        def _save(self, filename, epoch, metric=None):
            fake_torch.save({"state_dict": {"weight": 1}, "optimizer": {"step": 5},
                             "epoch": epoch, "metric": metric, "version": 2}, filename)

    saver = infrastructure.state_saver_class(Saver)(scheduler=StateObject({"epoch": 7}))
    saver._save(path, 7, 95.0)
    checkpoint = fake_torch.load(path)
    assert checkpoint["optimizer"] == {"step": 5} and checkpoint["version"] == 2
    assert checkpoint["sage_training_state"]["best_metric"] == 95.0
    assert checkpoint["sage_training_state"]["scheduler"] == {"epoch": 7}
