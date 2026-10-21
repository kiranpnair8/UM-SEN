import copy
import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
functional = pytest.importorskip("spikingjelly.clock_driven.functional")

from experiments.gradient_flow.alpha_scope import BlockAlphas
from experiments.gradient_flow.common import ROOT, build_model, load_config, seed_all
from experiments.gradient_flow.intervene import rng_snapshot, same_state
from experiments.gradient_flow.online_policy import OnlinePolicy
from experiments.gradient_flow.online_probe import OnlineProbe
from experiments.gradient_flow.online_runtime import OnlineRuntime, TrainingMetrics
from analysis.gradient_flow.online_summary import summarize_condition, feasibility_gate


def load_checkpoint(path):
    # Unit fixtures written immediately above by these tests, never CLI inputs.
    return torch.load(path, map_location="cpu", weights_only=False)


def settings():
    return load_config(ROOT/"configs/gradient_flow/controller_online.yml")["online"]


def model_batch():
    torch.set_num_threads(2)
    seed_all(42)
    model = build_model(dict(img_size=8, patch_size=4, dim=16, num_heads=4,
        mlp_ratio=2, layer=4, time_step=2, num_classes=10), "torch")
    return model, dict(images=torch.randn(2, 3, 8, 8), targets=torch.tensor([1, 2]))


def observation(G, response):
    return dict(G=G, Rhat=response)


def test_official_recipe_inherited_without_training_changes():
    baseline = load_config(ROOT/"spikformer/cifar10/cifar10.yml")
    online = load_config(ROOT/"configs/gradient_flow/controller_online.yml")
    for key, value in baseline.items():
        if key != "data_dir":
            assert online[key] == value, key
    assert online["epochs"] == 300 and online["online"]["epochs"] == 60
    assert online["fixed_alpha"] == online["sps_alpha"] == 4


def test_windowed_reference_and_epoch_update_schedule():
    s = settings()
    s.update(update_every_epochs=2, reference_window_epochs=4)
    policy = OnlinePolicy(1, "response_aware", s)
    first = policy.advance(0, observation([2.], [.2]))
    assert first["alpha"] == [4.] and first["G_ref"] == [2.]
    second = policy.advance(1, observation([2.1], None))
    assert second["alpha"] == [4.] and not second["update_due"]
    third = policy.advance(2, observation([2.2], [.2]))
    assert third["alpha"][0] < 4 and third["G_ref"] == [2.]
    policy.advance(3, observation([1.9], None))
    refresh = policy.advance(4, observation([3.], [.2]))
    assert refresh["G_ref"] == [3.] and refresh["error"] == [0.]
    assert not any(refresh["component_sign_crossing"])


def test_ema_reference_only_refreshes_at_window_boundary():
    s = settings()
    s["controller"].update(reference_mode="ema_window", reference_ema=.99)
    policy = OnlinePolicy(1, "fixed", s)
    policy.advance(0, observation([2.], None))
    for epoch in range(1, 5):
        assert policy.advance(epoch, observation([3.], None))["G_ref"] == [2.]
    assert policy.advance(5, observation([3.], None))["G_ref"] == pytest.approx([2.01])


def test_component_crossings_skip_zeros_and_do_not_cross_reference_windows():
    policy = OnlinePolicy(1, "fixed", settings())
    for epoch, G in enumerate([2., 2.1, 2., 1.9, 2.1, 1.9]):
        row = policy.advance(epoch, observation([G], None))
        if epoch == 3:
            assert row["component_sign_crossing"] == [True]
        if epoch == 4:
            assert row["oscillation_indicator"] and row["window_sign_crossings"] == [2]
    assert row["reference_refreshed"] and row["window_sign_crossings"] == [0]


def test_random_matching_bounds_and_resume_are_deterministic():
    s = settings()
    policy = OnlinePolicy(4, "random_matched", s)
    for epoch in range(3):
        source = dict(epoch=epoch, update_due=True, applied_step=[.1, -.2, .25, 0.])
        row = policy.advance(epoch, observation([1., 2., 3., 4.], [.1]*4), source)
        assert [abs(v) for v in row["applied_step"]] == pytest.approx([.1, .2, .25, 0.])
    state = policy.state_dict()
    restored = OnlinePolicy(4, "random_matched", s)
    restored.load_state_dict(state)
    source = dict(epoch=3, update_due=True, applied_step=[.2]*4)
    assert restored.advance(3, observation([1., 2., 3., 4.], [.1]*4), source) == policy.advance(3, observation([1., 2., 3., 4.], [.1]*4), source)
    policy.alpha = [s["controller"]["alpha_max"]]*4
    row = policy.advance(4, observation([1., 2., 3., 4.], [.1]*4), dict(epoch=4, update_due=True, applied_step=[.25]*4))
    assert row["applied_step"] == [-.25]*4
    with pytest.raises(ValueError, match="Missing matching"):
        policy.advance(5, observation([1., 2., 3., 4.], [.1]*4))


def test_changing_online_response_sign_is_tracked_without_wrong_sign_updates():
    s = settings()
    s["reference_window_epochs"] = 30
    s["controller"]["eta"] = .5
    policy = OnlinePolicy(1, "response_aware", s)
    policy.advance(0, observation([2.3], [-.1]))
    errors = []
    for epoch in range(1, 26):
        gain = -.1 if epoch < 13 else .1
        # Exogenous, smoothly changing weights shift the uncontrolled optimum.
        G = 2.3+gain*(policy.alpha[0]-4.)+.05
        row = policy.advance(epoch, observation([G], [gain]))
        new_error = gain*(policy.alpha[0]-4.)+.05
        assert abs(new_error) <= abs(row["error"][0])+1e-12
        assert row["error"][0]*gain*row["applied_step"][0] <= 1e-12
        assert 1 <= policy.alpha[0] <= 8
        errors.append(abs(new_error))
    assert policy.history[12]["response_sign"] == [-1]
    assert policy.history[13]["response_sign"] == [1]
    assert errors[11] < .001 and errors[-1] < .001


def test_online_probe_restores_optimizer_scheduler_scaler_and_all_model_state(monkeypatch):
    model, batch = model_batch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
    # Populate Adam moments, BN buffers and gradient buffers before calibration.
    model(batch["images"]).sum().backward()
    optimizer.step()
    functional.reset_net(model)
    model.eval()
    with torch.no_grad():
        model(batch["images"])
    class Scaler:
        def state_dict(self):
            return {"scale": 65536., "growth_tracker": 7}
    scaler = Scaler()
    before = dict(model=copy.deepcopy(model.state_dict()), optimizer=copy.deepcopy(optimizer.state_dict()),
        scheduler=copy.deepcopy(scheduler.state_dict()), scaler=scaler.state_dict(), rng=rng_snapshot(),
        modes=[m.training for m in model.modules()], alpha=BlockAlphas(model).snapshot(),
        gradients=[None if p.grad is None else p.grad.clone() for p in model.parameters()],
        neurons=[copy.deepcopy(m._memories) for m in model.modules() if hasattr(m, "_memories")])
    def forbidden(*a, **kw):
        raise AssertionError("No optimizer step permitted during calibration")
    monkeypatch.setattr(optimizer, "step", forbidden)
    result = OnlineProbe(model, batch, OnlinePolicy(4, "response_aware", settings()).config,
                         optimizer, scheduler, scaler)([4.]*4)
    assert result["Rhat"] is not None and result["raw_probe"]["optimizer_updates"] == 0
    assert same_state(before["model"], model.state_dict())
    assert same_state(before["optimizer"], optimizer.state_dict())
    assert same_state(before["scheduler"], scheduler.state_dict())
    assert before["scaler"] == scaler.state_dict() and same_state(before["rng"], rng_snapshot())
    assert before["modes"] == [m.training for m in model.modules()]
    assert before["alpha"] == BlockAlphas(model).snapshot()
    assert same_state(before["gradients"], [p.grad for p in model.parameters()])
    assert same_state(before["neurons"], [m._memories for m in model.modules() if hasattr(m, "_memories")])


def test_online_probe_detects_scheduler_mutation(monkeypatch):
    model, batch = model_batch()
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
    probe = OnlineProbe(model, batch, OnlinePolicy(4, "response_aware", settings()).config, optimizer, scheduler)
    def corrupted(*a, **kw):
        scheduler.last_epoch += 1
        return {}
    monkeypatch.setattr(probe.probe, "__call__", corrupted)
    # Special method lookup uses the class, not an instance attribute.
    monkeypatch.setattr(type(probe.probe), "__call__", corrupted)
    with pytest.raises(AssertionError, match="scheduler"):
        probe([4.]*4)


def test_fixed_instrumentation_matches_training_exactly_and_restores_runtime(tmp_path):
    model, batch = model_batch()
    baseline = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    other = torch.optim.AdamW(baseline.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
    runtime = OnlineRuntime(model, optimizer, scheduler, None, batch, "fixed", settings(), tmp_path, {"synthetic": True})
    try:
        for epoch in range(2):
            runtime.start_epoch(epoch)
            model.train()
            baseline.train()
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                other.zero_grad(set_to_none=True)
                out, ref = model(batch["images"]), baseline(batch["images"])
                loss = torch.nn.functional.cross_entropy(out, batch["targets"])
                ref_loss = torch.nn.functional.cross_entropy(ref, batch["targets"])
                runtime.adapter(loss, optimizer)
                ref_loss.backward()
                other.step()
                assert torch.equal(out, ref) and torch.equal(loss, ref_loss)
                assert same_state(model.state_dict(), baseline.state_dict())
                functional.reset_net(model)
                functional.reset_net(baseline)
            runtime.finish_epoch(dict(loss=float(loss.detach()), top1=0., top5=0.), dict(loss=1., top1=0., top5=0.))
        snapshot = runtime.state_dict()
        runtime.persist(complete=True)
        json.dumps(runtime.rows, allow_nan=False)
        assert all(r["alpha"] == [4.]*4 for r in runtime.policy.history)
        assert len(snapshot["rows"]) == 2
        for node in model.patch_embed.modules():
            if hasattr(node, "surrogate_function"):
                assert node.surrogate_function.alpha == 4
        assert (tmp_path/"metrics.csv").is_file() and (tmp_path/"summary.json").is_file()
    finally:
        runtime.close()


def test_nonfinite_loss_fails_before_optimizer_step(tmp_path, monkeypatch):
    model, batch = model_batch()
    optimizer = torch.optim.AdamW(model.parameters())
    runtime = OnlineRuntime(model, optimizer, None, None, batch, "fixed", settings(), tmp_path, {})
    def forbidden(*a, **kw):
        raise AssertionError("Must not step on nonfinite input")
    monkeypatch.setattr(optimizer, "step", forbidden)
    try:
        with pytest.raises(ValueError, match="Nonfinite training loss"):
            runtime.adapter(torch.tensor(float("nan"), requires_grad=True), optimizer)
    finally:
        runtime.close()


def test_soft_target_training_metrics_are_detached_and_do_not_change_loss():
    output = torch.tensor([[4., 3., 2., 1., 0., -1.]], requires_grad=True)
    target = torch.tensor([[.2, .8, 0., 0., 0., 0.]])
    loss_fn = lambda x, y: -(y*x.log_softmax(-1)).sum(-1).mean()
    metrics = TrainingMetrics(loss_fn)
    loss = metrics(output, target)
    assert torch.equal(loss, loss_fn(output, target))
    row = metrics.finish(float(loss.detach()))
    assert row["top1"] == pytest.approx(20.) and row["top5"] == pytest.approx(100.)
    json.dumps(row, allow_nan=False)


def test_scaled_activation_and_parameter_measurements_unscale_exactly_once(tmp_path):
    model, batch = model_batch()
    class CpuScale:
        def __init__(self):
            self.unscales = self.steps = self.updates = 0
        def get_scale(self):
            return 128.
        def state_dict(self):
            return dict(unscales=self.unscales, steps=self.steps, updates=self.updates)
        def scale(self, loss):
            return loss*128.
        def unscale_(self, optimizer):
            self.unscales += 1
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        p.grad.div_(128.)
        def step(self, optimizer):
            self.steps += 1
            optimizer.step()
        def update(self):
            self.updates += 1
    baseline = copy.deepcopy(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=.001)
    other = torch.optim.SGD(baseline.parameters(), lr=.001)
    scaled, unscaled = CpuScale(), None
    runtimes = [OnlineRuntime(m, opt, None, sc, batch, "fixed", settings(), tmp_path, {})
                for m, opt, sc in ((model, optimizer, scaled), (baseline, other, unscaled))]
    try:
        for runtime, m, opt in zip(runtimes, (model, baseline), (optimizer, other)):
            runtime.start_epoch(0)
            m.train()
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(m(batch["images"]), batch["targets"])
            runtime.adapter(loss, opt)
            functional.reset_net(m)
        assert scaled.unscales == scaled.steps == scaled.updates == 1
        assert same_state(model.state_dict(), baseline.state_dict())
        for a, b in zip(runtimes[0].gradient_rows, runtimes[1].gradient_rows):
            for key in ("grad_input_norm", "grad_output_norm", "parameter_grad_norm", "G_l"):
                assert a[key] == b[key]
    finally:
        for runtime in runtimes:
            runtime.close()


def test_interrupted_smoke_resume_matches_uninterrupted_weights_and_controller(tmp_path, monkeypatch):
    from experiments.gradient_flow.online_train import run_smoke
    config = load_config(ROOT/"configs/gradient_flow/controller_online.yml")
    _, batch = model_batch()
    args = SimpleNamespace(device="cpu", backend="torch", smoke_epochs=2, condition="response_aware")
    identity = {"synthetic": True}
    full, split = tmp_path/"full", tmp_path/"split"
    full.mkdir()
    split.mkdir()
    run_smoke(args, config, full, batch, identity, None, None)
    original_persist = OnlineRuntime.persist
    def interrupt_after_publish(self, complete=False):
        original_persist(self, complete)
        if len(self.rows) == 1:
            raise KeyboardInterrupt("synthetic preemption")
    monkeypatch.setattr(OnlineRuntime, "persist", interrupt_after_publish)
    with pytest.raises(KeyboardInterrupt):
        run_smoke(args, config, split, batch, identity, None, None)
    checkpoint = load_checkpoint(split/"last.pth.tar")
    assert checkpoint["epoch"] == 0
    monkeypatch.setattr(OnlineRuntime, "persist", original_persist)
    run_smoke(args, config, split, batch, identity, checkpoint, None)
    a, b = load_checkpoint(full/"last.pth.tar"), load_checkpoint(split/"last.pth.tar")
    for key in ("state_dict", "optimizer", "scheduler", "rng"):
        assert same_state(a[key], b[key])
    assert a["gradient_flow_online"]["policy"]["alpha"] == b["gradient_flow_online"]["policy"]["alpha"]
    assert a["gradient_flow_online"]["policy"]["rng"] == b["gradient_flow_online"]["policy"]["rng"]
    assert a["gradient_flow_online"]["step"] == b["gradient_flow_online"]["step"]


@pytest.mark.parametrize("backend", ["torch", "cupy"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA online AMP smoke needs a GPU")
def test_cuda_online_amp_smoke(tmp_path, backend):
    if backend == "cupy":
        pytest.importorskip("cupy")
    from experiments.gradient_flow.online_train import run_smoke
    config = load_config(ROOT/"configs/gradient_flow/controller_online.yml")
    _, batch = model_batch()
    args = SimpleNamespace(device="cuda", backend=backend, smoke_epochs=2, condition="response_aware")
    run_smoke(args, config, tmp_path, batch, {"synthetic": True}, None, None)
    saved = load_checkpoint(tmp_path/"last.pth.tar")
    assert len(saved["gradient_flow_online"]["rows"]) == 2


def test_smoke_checkpoint_resume_plotting_and_synthetic_gate(tmp_path):
    from experiments.gradient_flow.online_train import run_smoke
    from analysis.gradient_flow.plot_controller_online import plot
    config = load_config(ROOT/"configs/gradient_flow/controller_online.yml")
    batch = dict(images=torch.randn(2, 3, 8, 8, generator=torch.Generator().manual_seed(420)), targets=torch.tensor([1, 2]))
    identity_base = dict(synthetic=True, source_commit="synthetic-test", training_signature="test", batch_fingerprint="same-batch")
    for method in ("fixed", "response_aware", "random_matched"):
        directory = tmp_path/"results"/method
        directory.mkdir(parents=True)
        identity = dict(identity_base, method=method)
        (directory/"config.json").write_text(json.dumps(config))
        (directory/"provenance.json").write_text(json.dumps(dict(identity=identity)))
        args = SimpleNamespace(device="cpu", backend="torch", smoke_epochs=2, condition=method)
        matched = None if method != "random_matched" else json.loads((tmp_path/"results/response_aware/metrics.json").read_text())
        run_smoke(args, config, directory, batch, identity, None, matched)
        saved = load_checkpoint(directory/"last.pth.tar")
        assert len(saved["gradient_flow_online"]["rows"]) == 2
        before = copy.deepcopy(saved["state_dict"])
        # A complete resumed run performs no additional optimizer/controller update.
        run_smoke(args, config, directory, batch, identity, saved, matched)
        assert same_state(before, load_checkpoint(directory/"last.pth.tar")["state_dict"])
    result = plot(tmp_path/"results", tmp_path/"figures")
    assert result["gate"]["passed"] is False and not result["gate"]["checks"]["real_full_horizon"]
    assert (tmp_path/"results/summary.json").is_file()
    assert len(list((tmp_path/"figures").glob("*.png"))) >= 12


def test_official_epoch_wrapper_saver_and_interrupted_resume(tmp_path, monkeypatch):
    """Exercise the REAL official epoch/validation/checkpoint hooks on CPU tensors."""
    pytest.importorskip("timm.layers")
    import sys
    from experiments.gradient_flow.common import CIFAR
    if str(CIFAR) not in sys.path:
        sys.path.insert(0, str(CIFAR))
    import train as official
    from experiments.gradient_flow.online_train import run_official
    config = load_config(ROOT/"configs/gradient_flow/controller_online.yml")
    config["online"]["epochs"] = 2
    model, batch = model_batch()
    initial = copy.deepcopy(model.state_dict())
    destination = tmp_path/"official"
    destination.mkdir()
    args = SimpleNamespace(config="configs/gradient_flow/controller_online.yml", condition="response_aware")
    identity = {"synthetic": True}
    # Pretend timm's outer CLI main has just built model/loaders/scaler/saver;
    # the inner official train/validation loops and actual timm saver run below.
    parsed = SimpleNamespace(mixup_off_epoch=0, prefetcher=True, distributed=False,
        channels_last=False, clip_grad=None, clip_mode="norm", local_rank=1,
        world_size=1, log_interval=100, save_images=False, recovery_interval=0,
        tta=0, model="spikformer", amp=False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **kw: None)
    from timm.scheduler.cosine_lr import CosineLRScheduler
    monkeypatch.setattr(official, "create_scheduler", lambda args, opt: (CosineLRScheduler(opt, t_initial=300,
        warmup_t=20, warmup_lr_init=1e-5, lr_min=1e-5), 310))
    stop_after_one = [True]
    def outer_main():
        current, current_batch = model_batch()
        current.load_state_dict(initial)
        optimizer = torch.optim.AdamW(current.parameters(), lr=.0005)
        scheduler, stop = official.create_scheduler(parsed, optimizer)
        begin = 0
        if (destination/"last.pth.tar").exists():
            saved = load_checkpoint(destination/"last.pth.tar")
            current.load_state_dict(saved["state_dict"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["sage_training_state"]["scheduler"])
            begin = saved["epoch"]+1
        saver = official.experiment_state.state_saver_class(official.CheckpointSaver)(
            model=current, optimizer=optimizer, args=parsed, checkpoint_dir=str(destination),
            recovery_dir=str(destination), scheduler=scheduler, max_history=1)
        loader = [(current_batch["images"], current_batch["targets"])]*2
        for epoch in range(begin, stop):
            official.train_one_epoch(epoch, current, loader, optimizer, torch.nn.CrossEntropyLoss(), parsed,
                lr_scheduler=scheduler, saver=saver)
            metric = official.validate(current, loader, torch.nn.CrossEntropyLoss(), parsed)
            scheduler.step(epoch+1)
            saver.save_checkpoint(epoch, metric=metric["top1"])
            if stop_after_one[0]:
                return
    monkeypatch.setattr(official, "main", outer_main)
    with pytest.raises(RuntimeError, match="before requested"):
        run_official(args, config, destination, batch, identity, None, None)
    checkpoint = load_checkpoint(destination/"last.pth.tar")
    assert checkpoint["epoch"] == 0
    assert len(checkpoint["gradient_flow_online"]["rows"]) == 1
    assert checkpoint["sage_training_state"]["scheduler"] is not None
    stop_after_one[0] = False
    run_official(args, config, destination, batch, identity, checkpoint, None)
    final = load_checkpoint(destination/"last.pth.tar")
    assert final["epoch"] == 1 and len(final["gradient_flow_online"]["rows"]) == 2
    assert final["gradient_flow_online"]["step"] == 4
    assert json.loads((destination/"status.json").read_text())["status"] == "complete"


def test_prepare_refuses_provenance_drift_and_skips_only_consistent_complete_runs(tmp_path, monkeypatch):
    from experiments.gradient_flow import online_train as entry
    from experiments.gradient_flow import online_checkpoint as trusted
    config = load_config(ROOT/"configs/gradient_flow/controller_online.yml")
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    monkeypatch.setattr(trusted, "ROOT", tmp_path)
    source = dict(commit="synthetic-unit-test", dirty=False, torch="test", cuda=None, packages={"timm": "test"})
    monkeypatch.setattr(entry, "provenance", lambda: copy.deepcopy(source))
    args = SimpleNamespace(expected_commit=None, smoke=True, device="cpu", backend="torch", condition="fixed",
        output="results/gradient_flow/unit-only", smoke_epochs=2, resume=False)
    directory, batch, identity, _, _, complete = entry.prepare(args, config)
    assert not complete
    rows = [dict(epoch=epoch, controller={}) for epoch in range(2)]
    torch.save(dict(version=2, epoch=1, gradient_flow_online=dict(identity=identity, rows=rows)), directory/"last.pth.tar")
    (directory/"checkpoint_integrity.json").write_text(json.dumps(dict(identity=identity, epoch=1,
        sha256=trusted.digest(directory/"last.pth.tar"))))
    (directory/"model_best.pth.tar").write_bytes(b"unit-test-existence-marker-not-model")
    (directory/"metrics.json").write_text(json.dumps(rows))
    (directory/"controller_metrics.json").write_text(json.dumps([{}, {}]))
    (directory/"status.json").write_text(json.dumps(dict(status="complete", identity=identity, epochs=2)))
    args.resume = True
    assert entry.prepare(args, config)[-1] is True
    source["packages"]["timm"] = "changed"
    with pytest.raises(ValueError, match="provenance/config/commit"):
        entry.prepare(args, config)
    source["packages"]["timm"] = "test"
    (directory/"metrics.json").write_text(json.dumps(rows[:1]))
    with pytest.raises(ValueError, match="contiguous committed prefix"):
        entry.prepare(args, config)
