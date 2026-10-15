import copy
import json
import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
functional = pytest.importorskip("spikingjelly.clock_driven.functional")
from experiments.gradient_flow.alpha_scope import BlockAlphas
from experiments.gradient_flow.common import build_model
from experiments.gradient_flow.gradient_monitor import GradientMonitor
from experiments.gradient_flow.intervene import intervention, response_matrix, rng_snapshot
from experiments.gradient_flow.metrics_io import MetricWriter
from experiments.gradient_flow.step_adapter import backward_step


@pytest.fixture
def model():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    return build_model(dict(img_size=8, patch_size=4, dim=16, num_heads=4, mlp_ratio=2,
                            layer=2, time_step=2, num_classes=10), "torch").train()


def batch():
    generator = torch.Generator().manual_seed(123)
    return torch.randn(2, 3, 8, 8, generator=generator), torch.tensor([1, 3])


def observe(model, monitor, step=0, scaler=None, update=False, **kwargs):
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    optimizer.zero_grad()
    monitor.begin(0, step, scaler.get_scale() if scaler else 1.)
    images, targets = batch()
    logits = model(images)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    records = backward_step(loss, optimizer, monitor, scaler, update=update, **kwargs)
    grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    functional.reset_net(model)
    return logits.detach(), loss.detach(), grads, records


@pytest.mark.parametrize("enabled", [False, True])
def test_observation_preserves_logits_loss_gradients_and_updates(model, enabled):
    other = copy.deepcopy(model)
    monitor = GradientMonitor(model, enabled=enabled)
    baseline = GradientMonitor(other, enabled=False)
    try:
        for step in range(2):
            a = observe(model, monitor, step, update=True)
            b = observe(other, baseline, step, update=True)
            torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
            torch.testing.assert_close(a[1], b[1], rtol=0, atol=0)
            assert a[2].keys() == b[2].keys()
            for name in a[2]:
                torch.testing.assert_close(a[2][name], b[2][name], rtol=0, atol=0)
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(tensor, other.state_dict()[name], rtol=0, atol=0)
            if not enabled:
                assert a[3] == [] and monitor.handles == []
    finally:
        monitor.close()
        baseline.close()


def test_alpha_scope_independence_restore_and_backward(model):
    scope = BlockAlphas(model)
    saved = scope.snapshot()
    node = next(iter(scope.nodes[0].values()))
    inputs = torch.tensor([-.1, 0., .2], requires_grad=True)
    initial = node.surrogate_function(inputs)
    initial.sum().backward()
    gradient = inputs.grad.clone()
    scope.set(0, 6.)
    assert scope.get() == [6., 4.]
    assert all(node.surrogate_function.alpha == 4 for node in model.patch_embed.modules()
               if hasattr(node, "surrogate_function"))
    inputs.grad = None
    changed = node.surrogate_function(inputs)
    changed.sum().backward()
    torch.testing.assert_close(initial, changed, rtol=0, atol=0)
    expected = 6 * torch.sigmoid(6 * inputs.detach()) * (1 - torch.sigmoid(6 * inputs.detach()))
    torch.testing.assert_close(inputs.grad, expected)
    assert not torch.equal(gradient, inputs.grad)
    scope.restore(saved)
    assert scope.get() == [4., 4.]
    with pytest.raises(ValueError):
        scope.set_all([3.])


def test_norms_G_and_element_weighted_firing(model):
    monitor = GradientMonitor(model)
    boundaries, spikes, handles = [], [], []
    for block in model.block:
        def boundary(module, inputs, output):
            boundaries.append((inputs[0], output))
        handles.append(block.register_forward_hook(boundary))
    for node in model.block[0].modules():
        if hasattr(node, "surrogate_function"):
            def spike(module, inputs, output):
                spikes.append((float(output.detach().sum()), output.numel()))
            handles.append(node.register_forward_hook(spike))
    monitor.begin(3, 11)
    images, targets = batch()
    loss = torch.nn.functional.cross_entropy(model(images), targets)
    expected = [torch.autograd.grad(loss, tensors, retain_graph=True) for tensors in boundaries]
    loss.backward()
    records = monitor.finish()
    for row, (gi, go) in zip(records, expected):
        assert row["grad_input_norm"] == pytest.approx(float(gi.norm()), rel=1e-6)
        assert row["grad_output_norm"] == pytest.approx(float(go.norm()), rel=1e-6)
        assert row["G_l"] == pytest.approx(math.log((float(gi.norm()) + 1e-12) / (float(go.norm()) + 1e-12)), abs=1e-6)
        assert row["directional_consistency"] is None
        assert row["epoch"] == 3 and row["step"] == 11
    assert records[0]["firing_rate"] == pytest.approx(sum(a for a, n in spikes) / sum(n for a, n in spikes))
    for handle in handles:
        handle.remove()
    monitor.close()
    functional.reset_net(model)


def test_previous_EMA_direction_and_detached_logging(model, tmp_path):
    monitor = GradientMonitor(model, beta=.8)
    first = observe(model, monitor)[3]
    assert all(row["directional_consistency"] is None for row in first)
    state = monitor.state_dict()
    second = observe(model, monitor, 1)[3]
    assert all(row["directional_consistency"] == pytest.approx(1, abs=1e-6) for row in second)
    assert all(not value.requires_grad for block in monitor.references for value in block.values())
    assert not monitor.tensor_handles and not monitor.batch
    writer = MetricWriter(tmp_path)
    writer.write(first + second)
    writer.close()
    payload = json.loads((tmp_path / "epochs.json").read_text())
    assert len(payload["epochs"]) == 2
    json.dumps(second, allow_nan=False)
    monitor.load_state_dict(state)
    for block, saved in zip(monitor.references, state["references"]):
        for name in block:
            torch.testing.assert_close(block[name], saved[name])
    monitor.close()


def test_direction_uses_previous_not_updated_reference(model):
    monitor = GradientMonitor(model, beta=.6)
    observe(model, monitor)
    old = monitor.state_dict()["references"]
    model.zero_grad()
    monitor.begin(0, 1)
    images, targets = batch()
    loss = torch.nn.functional.cross_entropy(model(images.flip(0) * .7), targets)
    loss.backward()
    expected, gradients = [], []
    for block, reference in zip(model.block, old):
        grads = {name: p.grad.detach().clone() for name, p in block.named_parameters() if p.grad is not None}
        gradients.append(grads)
        current = torch.cat([g.flatten() for g in grads.values()])
        previous = torch.cat([reference[name].flatten() for name in grads])
        expected.append(float(torch.nn.functional.cosine_similarity(current, previous, dim=0)))
    records = monitor.finish()
    for row, value, ref, grads, previous in zip(records, expected, monitor.references, gradients, old):
        assert row["directional_consistency"] == pytest.approx(value, abs=1e-6)
        for name in grads:
            torch.testing.assert_close(ref[name], .6 * previous[name] + .4 * grads[name])
    monitor.close()
    functional.reset_net(model)


def test_neuron_reset_reproduces_forward(model):
    images, _ = batch()
    first = model(images).detach()
    nodes = [node for node in model.modules() if hasattr(node, "surrogate_function")]
    assert any(torch.is_tensor(node.v) for node in nodes)
    functional.reset_net(model)
    assert all(node.v == 0. for node in nodes)
    second = model(images).detach()
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    functional.reset_net(model)


def test_nonfinite_scaled_step_is_logged_and_skipped(model):
    if not hasattr(torch, "amp") or not hasattr(torch.amp, "GradScaler"):
        pytest.skip("CPU GradScaler unavailable")
    parameters = [p.detach().clone() for p in model.parameters()]
    monitor = GradientMonitor(model)
    handle = model.head.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
    try:
        records = observe(model, monitor, scaler=torch.amp.GradScaler("cpu", init_scale=128.), update=True)[3]
        assert all(not row["valid"] and row["G_l"] is None for row in records)
        assert all(not reference for reference in monitor.references)
        for parameter, original in zip(model.parameters(), parameters):
            torch.testing.assert_close(parameter, original, rtol=0, atol=0)
    finally:
        handle.remove()
        monitor.close()


def test_scaled_gradients_match_FP32_and_unscale_once(model, monkeypatch):
    if not hasattr(torch, "amp") or not hasattr(torch.amp, "GradScaler"):
        pytest.skip("CPU scaling test needs modern torch.amp; CUDA test is separate")
    other = copy.deepcopy(model)
    scaler = torch.amp.GradScaler("cpu", init_scale=128.)
    calls = []
    original = scaler.unscale_
    def counted(optimizer):
        calls.append(1)
        return original(optimizer)
    monkeypatch.setattr(scaler, "unscale_", counted)
    a_monitor, b_monitor = GradientMonitor(model), GradientMonitor(other)
    a = observe(model, a_monitor, scaler=scaler, update=True)
    b = observe(other, b_monitor, update=True)
    assert len(calls) == 1
    for row_a, row_b in zip(a[3], b[3]):
        for name in ("grad_input_norm", "grad_output_norm", "G_l", "parameter_grad_norm"):
            assert row_a[name] == pytest.approx(row_b[name], rel=1e-6, abs=1e-8)
    for pa, pb in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(pa, pb, rtol=0, atol=0)
    a_monitor.close()
    b_monitor.close()


def test_preclip_measurement_and_invalid_EMA(model):
    monitor = GradientMonitor(model)
    result = observe(model, monitor, clip_grad=1e-5, parameters=model.parameters())
    assert all(row["parameter_grad_norm"] > 1e-5 for row in result[3])
    assert math.sqrt(sum(float(p.grad.norm()) ** 2 for p in model.parameters() if p.grad is not None)) <= 1.01e-5
    before = monitor.state_dict()
    monitor.begin(0, 1)
    images, targets = batch()
    torch.nn.functional.cross_entropy(model(images), targets).backward()
    rows = monitor.finish(valid=False, reason="test invalid step")
    assert all(not row["valid"] and row["G_l"] is None for row in rows)
    for block, old in zip(monitor.references, before["references"]):
        for name in block:
            torch.testing.assert_close(block[name], old[name])
    monitor.close()
    functional.reset_net(model)


def test_intervention_state_rng_reset_determinism_and_response(model, monkeypatch):
    def forbidden_step(*args, **kwargs):
        raise AssertionError("Interventions must never call optimizer.step")
    monkeypatch.setattr(torch.optim.SGD, "step", forbidden_step)
    scope = BlockAlphas(model)
    scope.set(1, 3.)
    first_node = next(iter(scope.nodes[0].values()))
    first_node.v = torch.tensor(.17)
    model.block[0].attn.q_bn.eval()
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    weights = copy.deepcopy(model.state_dict())
    modes = [module.training for module in model.modules()]
    alphas = scope.snapshot()
    rng = rng_snapshot()
    images, targets = batch()
    rows = intervention(model, images, targets, [0, 1], [2., 4., 4., 6.])
    assert rows["optimizer_updates"] == 0
    same = [trial for trial in rows["trials"] if trial["target_block"] == 0 and trial["alpha"] == 4]
    assert same[0] == same[1]
    assert same[0]["logits"] == rows["baseline"]["logits"]
    assert all(trial["logits"] == rows["baseline"]["logits"] for trial in rows["trials"])
    assert rows["trials"][0]["blocks"][0]["grad_input_norm"] != rows["trials"][3]["blocks"][0]["grad_input_norm"]
    matrix = response_matrix(rows)
    assert len(matrix["matrix"]) == 2 and len(matrix["matrix"][0]) == 2
    assert matrix["matrix"][1][0] == pytest.approx(0, abs=1e-6)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
    assert scope.snapshot() == alphas
    assert float(first_node.v) == pytest.approx(.17)
    assert modes == [module.training for module in model.modules()]
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in model.parameters())
    after = rng_snapshot()
    assert torch.equal(after["torch"], rng["torch"])
    assert after["python"] == rng["python"]
    assert after["numpy"][0] == rng["numpy"][0]
    assert after["numpy"][1].tolist() == rng["numpy"][1].tolist()
    assert after["numpy"][2:] == rng["numpy"][2:]
    with pytest.raises(ValueError):
        intervention(model, images, targets, [100], [4.])
    assert scope.snapshot() == alphas
    assert modes == [module.training for module in model.modules()]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP/CuPy unavailable locally")
@pytest.mark.parametrize("backend", ["torch", "cupy"])
def test_cuda_AMP_smoke(model, backend):
    if backend == "cupy":
        pytest.importorskip("cupy")
        model = build_model(dict(img_size=8, patch_size=4, dim=16, num_heads=4, mlp_ratio=2,
                                 layer=2, time_step=2, num_classes=10), "cupy").train()
    model.cuda()
    monitor = GradientMonitor(model)
    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    monitor.begin(0, 0, scaler.get_scale())
    images, targets = batch()
    with torch.cuda.amp.autocast():
        loss = torch.nn.functional.cross_entropy(model(images.cuda()), targets.cuda())
    records = backward_step(loss, optimizer, monitor, scaler)
    assert all(row["valid"] for row in records)
    monitor.close()
    functional.reset_net(model)
