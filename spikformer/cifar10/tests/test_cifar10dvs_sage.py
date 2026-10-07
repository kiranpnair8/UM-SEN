"""CPU protocol checks and optional real torch-backend DVS shape/regression checks."""

import ast
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest
from test_experiment_state import controllers, controller_args

ROOT = Path(__file__).resolve().parents[3]
DVS = ROOT / "spikformer/cifar10dvs"
spec = importlib.util.spec_from_file_location("dvs_runner_under_test", DVS / "train_sage.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_default_recipe_and_smoke_namespace(tmp_path):
    args = runner.parse_args(["--smoke-test", "--run-id", "test"])
    assert (args.T, args.seed, args.epochs, args.batch_size, args.lr) == (10, 42, 106, 16, .001)
    assert args.epochs - args.cooldown_epochs == 96
    assert (args.umsen_entropy_temperature, args.umsen_ema_beta, args.umsen_warmup_steps) == (.25, .95, 100)
    destination = runner.output_path(args, tmp_path)
    assert destination == tmp_path / "results/sage_v2/cifar10dvs/sage/T10/seed42/smoke_test"
    destination.mkdir(parents=True)
    runner.write_json(destination / "metrics.json", {"smoke_test": True, "epochs": []})
    assert json.loads((destination / "metrics.json").read_text())["smoke_test"]
    with pytest.raises(ValueError, match="overwrite"):
        runner.output_path(args, tmp_path)


@pytest.mark.parametrize("output", ["results/umsen_official_recipe", "results/sage_v2/cifar10", "../historical"])
def test_reject_historical_destinations(tmp_path, output):
    args = runner.parse_args(["--output-dir", output])
    with pytest.raises(ValueError, match="Outputs must remain"):
        runner.output_path(args, tmp_path)
    assert not list(tmp_path.iterdir())


def test_t10_shape_contract():
    runner.validate_frame_shape((16, 10, 2, 128, 128), 10)
    for shape in [(16, 16, 2, 128, 128), (16, 10, 3, 128, 128), (16, 10, 2, 32, 32)]:
        with pytest.raises(ValueError):
            runner.validate_frame_shape(shape, 10)


def test_smoke_prefix_and_run_id_safety(tmp_path):
    batches = runner.LimitedBatches(list(range(20)), 4)
    assert len(batches) == 4 and list(batches) == [0, 1, 2, 3]
    args = runner.parse_args(["--run-id", "../historical"])
    with pytest.raises(ValueError, match="run-id"):
        runner.output_path(args, tmp_path)


def test_two_blocks_scope_centering_warmup_and_delayed_alpha(controllers, monkeypatch):
    module, LIF = controllers
    shared = types.SimpleNamespace(alpha=4.0)
    node_groups = [[LIF() for _ in range(8)] for _ in range(2)]
    sps = [LIF() for _ in range(5)]
    all_nodes = sum(node_groups, []) + sps
    for node in all_nodes:
        node.surrogate_function = shared
    scope = lambda nodes: types.SimpleNamespace(modules=lambda: nodes)
    blocks = [types.SimpleNamespace(attn=types.SimpleNamespace(training=True), modules=lambda nodes=nodes: nodes)
              for nodes in node_groups]
    model = types.SimpleNamespace(block=blocks, patch_embed=scope(sps), modules=lambda: all_nodes,
                                  training=True, forward=lambda *args: "unchanged logits")
    monkeypatch.setattr(sys.modules["spikingjelly.clock_driven"], "surrogate",
                        types.SimpleNamespace(Sigmoid=types.SimpleNamespace), raising=False)
    runner.prepare_surrogates(model)
    assert len({id(node.surrogate_function) for node in all_nodes}) == 21
    controller = module.UMSENController(model, controller_args())
    assert len(controller.controllers) == 2
    controller.controllers[0].normalized_z = 3.
    controller.controllers[1].normalized_z = 1.
    controller.update_centered_alphas()
    assert [c.centered_z for c in controller.controllers] == [1., -1.]
    assert controller.controllers[0].alpha > 4 > controller.controllers[1].alpha
    controller.step = 150
    controller.apply_step_alphas()
    assert all(node.surrogate_function.alpha == 4 for node in all_nodes)
    controller.start_epoch(1)
    controller.apply_step_alphas()
    applied = [c.applied_alpha for c in controller.controllers]
    for nodes, alpha in zip(node_groups, applied):
        assert all(node.surrogate_function.alpha == alpha for node in nodes)
    controller.controllers[0].normalized_z = -2.
    controller.controllers[1].normalized_z = 2.
    controller.update_centered_alphas()
    assert [c.applied_alpha for c in controller.controllers] == applied
    assert all(node.surrogate_function.alpha == 4 for node in sps)
    controller.epoch_record(1)
    record = controller.history[-1]
    assert set(record["blocks"]) == {"0", "1"}
    assert not record["warmup_active"]
    assert {"normalized_z", "centered_z", "candidate_alpha", "applied_alpha"} <= record["blocks"]["0"].keys()
    json.dumps(record, allow_nan=False)


def test_dvs_hook_leaves_attention_expression_unchanged():
    tree = ast.parse((DVS / "model.py").read_text(encoding="utf-8"))
    ssa = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SSA")
    forward = next(node for node in ssa.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    expressions = [ast.unparse(node) for node in forward.body]
    index = expressions.index("attn = q @ k.transpose(-2, -1)")
    assert expressions[index + 1] == "self._record_attention_entropy(attn)"
    assert expressions[index + 2] == "x = attn @ v * self.scale"


def test_real_t10_forward_backward_baseline_equivalence(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    neuron = pytest.importorskip("spikingjelly.clock_driven.neuron")
    functional = pytest.importorskip("spikingjelly.clock_driven.functional")
    real_lif = neuron.MultiStepLIFNode
    spec = importlib.util.spec_from_file_location("dvs_model_torch_test", DVS / "model.py")
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    def torch_lif(*args, **kwargs):
        kwargs["backend"] = "torch"
        return real_lif(*args, **kwargs)
    monkeypatch.setattr(model_module, "MultiStepLIFNode", torch_lif)
    from sage_controller import UMSENController
    torch.manual_seed(42)
    original = model_module.spikformer().train()
    modified = model_module.spikformer().train()
    modified.load_state_dict(original.state_dict())
    runner.prepare_surrogates(modified)
    controller = UMSENController(modified, controller_args())
    assert all(node.backend == "torch" for model in (original, modified)
               for node in model.modules() if isinstance(node, real_lif))
    batch = torch.rand(1, 10, 2, 128, 128)
    logits = [model(batch) for model in (original, modified)]
    assert logits[0].shape == (1, 10)
    torch.testing.assert_close(*logits, rtol=0, atol=0)
    assert all(block.running_count == 1 for block in controller.controllers)
    assert controller.warmup_active()
    losses = [torch.nn.functional.cross_entropy(value, torch.tensor([3])) for value in logits]
    torch.testing.assert_close(*losses, rtol=0, atol=0)
    for loss in losses:
        loss.backward()
    for (_, a), (_, b) in zip(original.named_parameters(), modified.named_parameters()):
        if a.grad is None or b.grad is None:
            assert a.grad is b.grad is None
        else:
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    for model in (original, modified):
        functional.reset_net(model)
