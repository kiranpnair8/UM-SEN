import copy
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("spikingjelly.clock_driven.functional")
from spikingjelly.clock_driven import functional

from experiments.gradient_flow.amp_safety import AMPSafety
from experiments.gradient_flow.gradient_monitor import GradientMonitor
from experiments.gradient_flow.intervene import same_state
from experiments.gradient_flow.online_runtime import OnlineRuntime, SafeOnlineScaler
from test_online_controller import model_batch, settings


def runtime(tmp_path, safety=None):
    model, batch = model_batch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    scaler = torch.amp.GradScaler("cpu", init_scale=128., growth_interval=1000)
    config = settings()
    if safety:
        config["amp_safety"] = safety
    owner = OnlineRuntime(model, optimizer, None, scaler, batch, "fixed", config, tmp_path, {})
    owner.adapter.epoch = 7
    return owner, model, optimizer, scaler, batch


def batch_step(owner, model, optimizer, batch):
    optimizer.zero_grad(set_to_none=True)
    model.train()
    try:
        logits = model(batch["images"])
        owner.adapter(torch.nn.functional.cross_entropy(logits, batch["targets"]), optimizer)
    finally:
        functional.reset_net(model)


@pytest.mark.parametrize("bad_name,block", [("block.0.attn.q_linear.weight", 0), ("head.weight", None)])
def test_single_native_amp_overflow_skips_step_updates_scale_and_excludes_statistics(tmp_path, bad_name, block):
    owner, model, optimizer, scaler, batch = runtime(tmp_path)
    name, parameter = next((n, p) for n, p in model.named_parameters() if n == bad_name)
    handle = parameter.register_hook(lambda grad: torch.full_like(grad, float("inf")))
    weights = copy.deepcopy(model.state_dict())
    try:
        batch_step(owner, model, optimizer, batch)
        assert same_state(weights, model.state_dict()) is False  # Training forward still updates BN, just like official AMP.
        for n, p in model.named_parameters():
            assert torch.equal(p, weights[n])
        assert not optimizer.state
        assert scaler.get_scale() == 64.
        assert owner.gradient_rows == []
        assert all(not r for r in owner.monitor.references)
        event = json.loads((tmp_path/"amp_events.jsonl").read_text())
        assert event["kind"] == "amp_overflow" and event["overflow"] and event["skipped"]
        assert event["epoch"] == 7 and event["batch"] == event["step"] == 0
        assert event["scale_before"] == 128 and event["scale_after"] == 64
        assert event["optimizer_updated"] is False and not event["statistics_included"]
        assert event["parameters"][0]["parameter"] == name and event["parameters"][0]["block_id"] == block
        assert all(b["valid"] is False for b in event["blocks"])
        handle.remove()
        handle = None
        batch_step(owner, model, optimizer, batch)
        assert len(owner.gradient_rows) == 4 and optimizer.state
        assert owner.adapter.step == 2 and owner.adapter.safety.optimizer_updates == 1
    finally:
        if handle is not None:
            handle.remove()
        owner.close()


def test_persistent_overflow_fails_only_after_native_skip_and_scale_backoff(tmp_path):
    owner, model, optimizer, scaler, batch = runtime(tmp_path, {"max_consecutive_overflows": 3})
    parameter = dict(model.named_parameters())["block.0.attn.q_linear.weight"]
    handle = parameter.register_hook(lambda grad: torch.full_like(grad, float("inf")))
    try:
        batch_step(owner, model, optimizer, batch)
        batch_step(owner, model, optimizer, batch)
        with pytest.raises(ValueError, match="Persistent AMP overflow"):
            batch_step(owner, model, optimizer, batch)
        assert scaler.get_scale() == 16.
        assert not optimizer.state and not owner.gradient_rows
        assert owner.adapter.safety.overflows == 3 and owner.adapter.step == 3
        events = [json.loads(line) for line in (tmp_path/"amp_events.jsonl").read_text().splitlines()]
        assert len(events) == 3 and events[-1]["persistence_failure"]
    finally:
        handle.remove()
        owner.close()


def test_true_nan_loss_fails_immediately_without_scaler_or_optimizer_update(tmp_path):
    owner, model, optimizer, scaler, batch = runtime(tmp_path)
    before = copy.deepcopy(scaler.state_dict())
    try:
        with pytest.raises(ValueError, match="Nonfinite training loss"):
            owner.adapter(torch.tensor(float("nan"), requires_grad=True), optimizer)
        assert scaler.state_dict() == before and not optimizer.state
        assert owner.adapter.step == owner.adapter.safety.overflows == 0
        assert json.loads((tmp_path/"amp_events.jsonl").read_text())["kind"] == "nonfinite_loss"
    finally:
        owner.close()


def test_invalid_monitor_with_finite_parameters_is_logged_separately_and_not_aggregated(tmp_path, monkeypatch):
    owner, model, optimizer, scaler, batch = runtime(tmp_path)
    original = owner.monitor.finish
    def invalid_finish(*args, **kwargs):
        owner.monitor.batch[2]["input"] = torch.tensor(float("inf"))
        return original(*args, **kwargs)
    monkeypatch.setattr(owner.monitor, "finish", invalid_finish)
    try:
        batch_step(owner, model, optimizer, batch)
        assert optimizer.state and scaler.get_scale() == 128
        assert owner.gradient_rows == [] and all(not r for r in owner.monitor.references)
        event = json.loads((tmp_path/"amp_events.jsonl").read_text())
        assert event["kind"] == "invalid_monitor_finite_parameters" and not event["overflow"]
        assert event["optimizer_updated"] is True and not event["skipped"]
        assert not event["parameters"] and event["blocks"][0]["block_id"] == 2
        monkeypatch.setattr(owner.monitor, "finish", original)
        batch_step(owner, model, optimizer, batch)
        assert len(owner.gradient_rows) == 4
    finally:
        owner.close()


def test_persistent_invalid_monitor_fails_as_diagnostic_not_amp_overflow(tmp_path, monkeypatch):
    owner, model, optimizer, scaler, batch = runtime(tmp_path, {"max_consecutive_invalid_diagnostics": 2})
    original = owner.monitor.finish
    def invalid_finish(*args, **kwargs):
        owner.monitor.batch[1]["input"] = torch.tensor(float("inf"))
        return original(*args, **kwargs)
    monkeypatch.setattr(owner.monitor, "finish", invalid_finish)
    try:
        batch_step(owner, model, optimizer, batch)
        with pytest.raises(ValueError, match="Persistent invalid diagnostics"):
            batch_step(owner, model, optimizer, batch)
        assert scaler.get_scale() == 128 and owner.adapter.safety.overflows == 0
        assert owner.adapter.safety.optimizer_updates == 2 and not owner.gradient_rows
    finally:
        owner.close()


def test_fp32_nonfinite_gradient_fails_immediately_with_parameter_name(tmp_path):
    owner, model, optimizer, scaler, batch = runtime(tmp_path)
    owner.adapter.native = None
    parameter = dict(model.named_parameters())["head.weight"]
    handle = parameter.register_hook(lambda grad: torch.full_like(grad, float("nan")))
    try:
        with pytest.raises(ValueError, match="Nonfinite FP32 parameter gradients"):
            batch_step(owner, model, optimizer, batch)
        event = json.loads((tmp_path/"amp_events.jsonl").read_text())
        assert event["kind"] == "nonfinite_fp32_gradients" and not event["optimizer_updated"]
        assert event["parameters"][0]["parameter"] == "head.weight" and event["parameters"][0]["nan"] > 0
        assert not optimizer.state
    finally:
        handle.remove()
        owner.close()


@pytest.mark.parametrize("diagnostics", [False, True])
def test_fixed_with_or_without_diagnostics_matches_inherited_native_amp_exactly(diagnostics):
    from timm.utils.cuda import NativeScaler
    model, batch = model_batch()
    baseline = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    inherited_optimizer = torch.optim.AdamW(baseline.parameters(), lr=.001)
    monitor = GradientMonitor(model, enabled=diagnostics)
    native = torch.amp.GradScaler("cpu", init_scale=128., growth_interval=2)
    original = object.__new__(NativeScaler)
    original._scaler = torch.amp.GradScaler("cpu", init_scale=128., growth_interval=2)
    events, statistics = [], []
    adapter = SafeOnlineScaler(monitor, native, statistics.extend, events.append)
    p = dict(model.named_parameters())["block.0.attn.q_linear.weight"]
    q = dict(baseline.named_parameters())["block.0.attn.q_linear.weight"]
    try:
        for step, overflow in enumerate([False, True, False, False]):
            handles = [x.register_hook(lambda grad: torch.full_like(grad, float("inf"))) for x in (p, q)] if overflow else []
            optimizer.zero_grad(set_to_none=True)
            inherited_optimizer.zero_grad(set_to_none=True)
            monitor.begin(0, step, native.get_scale())
            loss = torch.nn.functional.cross_entropy(model(batch["images"]), batch["targets"])
            other = torch.nn.functional.cross_entropy(baseline(batch["images"]), batch["targets"])
            adapter(loss, optimizer)
            original(other, inherited_optimizer)
            assert torch.equal(loss, other)
            assert same_state(model.state_dict(), baseline.state_dict())
            assert same_state(optimizer.state_dict(), inherited_optimizer.state_dict())
            assert native.state_dict() == original._scaler.state_dict()
            functional.reset_net(model)
            functional.reset_net(baseline)
            for handle in handles:
                handle.remove()
        assert len(events) == 1 and events[0]["skipped"]
        assert len(statistics) == (12 if diagnostics else 0)
    finally:
        monitor.close()


def test_persistence_counters_checkpoint_and_legacy_successful_prefix():
    safety = AMPSafety(dict(overflow_window=10, max_window_overflow_fraction=.2))
    for n in range(10):
        failure = safety.observe(overflow=n in (0, 3, 9), invalid_diagnostics=n in (0, 3, 9), updated=n not in (0, 3, 9))
    assert "rolling-window" in failure
    restored = AMPSafety(safety.settings)
    restored.load_state_dict(safety.state_dict())
    assert restored.state_dict() == safety.state_dict()
    legacy = AMPSafety()
    legacy.load_state_dict(None, successful_legacy_steps=2730)
    assert legacy.optimizer_updates == 2730 and not any(legacy.recent)
    assert legacy.overflows == legacy.invalid_diagnostics == 0


def test_legacy_amp_resume_preserves_epoch6_checkpoint_and_prefix(tmp_path, monkeypatch):
    from experiments.gradient_flow import amp_resume as migration
    root = tmp_path/"study"
    fixed = root/"fixed"
    fixed.mkdir(parents=True)
    identity = dict(source_commit=migration.LEGACY_COMMIT, method="fixed", synthetic=False,
                    training_signature="same-recipe", runtime={"torch": "same"}, batch_fingerprint="same-input")
    config = dict(online=settings(), layer=4)
    rows = [dict(epoch=epoch, controller=dict(alpha=[4.]*4, applied_step=[0.]*4),
        gradient=[dict(observations=390, grad_input_norm=1., grad_output_norm=1., parameter_grad_norm=1., G_l=0.) for _ in range(4)])
        for epoch in range(7)]
    saved = dict(epoch=6, optimizer={"full-state": True}, amp_scaler={"scale": 131072.},
        sage_training_state=dict(scheduler={"epoch": 7}, rng={"states": "present"}),
        gradient_flow_online=dict(identity=identity, rows=rows, policy=dict(settings=config["online"], alpha=[4.]*4)))
    path = fixed/"last.pth.tar"
    torch.save(saved, path)
    (fixed/"metrics.json").write_text(json.dumps(rows))
    (fixed/"provenance.json").write_text(json.dumps(dict(identity=identity)))
    before = {p.name: p.read_bytes() for p in fixed.iterdir()}
    source = dict(commit="approved-amp-fix")
    proposed = {**identity, "source_commit": source["commit"]}
    loader = lambda p: torch.load(p, weights_only=False)
    monkeypatch.setattr(migration, "verify_patch", lambda sha: ["experiments/gradient_flow/online_runtime.py"])
    with pytest.raises(ValueError, match="explicit"):
        migration.reconcile(root, proposed, config, source, False, loader)
    result, transition = migration.reconcile(root, proposed, config, source, True, loader)
    assert result == identity and transition["prefix_epochs"] == list(range(7))
    assert all((fixed/name).read_bytes() == contents for name, contents in before.items())
    assert Path(transition["protected_checkpoint"]).read_bytes() == before["last.pth.tar"]
    assert loader(path)["epoch"]+1 == 7  # Continue at epoch7, NOT restart.
    new_run = {**proposed, "method": "response_aware"}
    paired, _ = migration.reconcile(root, new_run, config, source, True, loader)
    assert paired == {**identity, "method": "response_aware"}
    with pytest.raises(ValueError, match="mismatch"):
        migration.reconcile(root, {**new_run, "training_signature": "different-recipe"}, config, source, True, loader)
