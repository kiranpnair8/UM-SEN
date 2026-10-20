import copy
import json
import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("spikingjelly.clock_driven.functional")

from experiments.gradient_flow.alpha_scope import BlockAlphas
from experiments.gradient_flow.common import ROOT, build_model, load_config
from experiments.gradient_flow.controller import ControllerConfig, LocalReference, ResponseController, apply_steps
from experiments.gradient_flow.controller_recovery import FrozenProbe, aggregate, recover_case, run_study
from experiments.gradient_flow.intervene import intervention, rng_snapshot, same_state
from test_training_states import synthetic_states


@pytest.mark.parametrize("gain", [-.08, .08])
def test_response_formula_and_sign_flip_reduces_scalar_error(gain):
    config = ControllerConfig()
    controller = ResponseController(config)
    # Nonzero pre-disturbance reference, deliberately NOT a G=0 objective.
    reference, alpha = [2.3], [5.]
    initial = abs(gain)
    for _ in range(12):
        observed = [reference[0] + gain*(alpha[0]-4.)]
        result = controller.update(alpha, observed, reference, [gain])
        expected = -config.eta*(observed[0]-reference[0])*gain/(gain*gain+config.regularizer)
        assert result["requested_step"][0] == pytest.approx(expected)
        assert abs(result["applied_step"][0]) <= config.max_step+1e-12
        alpha = result["alpha"]
    assert abs(gain*(alpha[0]-4.)) < initial*.1
    positive = controller.update([4.], [3.], [2.], [abs(gain)])["applied_step"][0]
    negative = controller.update([4.], [3.], [2.], [-abs(gain)])["applied_step"][0]
    assert positive < 0 < negative


@pytest.mark.parametrize("gain", [0., 1e-15, -1e-15, 1e8])
def test_small_response_and_steps_are_bounded(gain):
    result = ResponseController(ControllerConfig()).update([4.], [1e6], [0.], [gain])
    assert math.isfinite(result["alpha"][0])
    assert abs(result["applied_step"][0]) <= .25


def test_alpha_bounds_disabled_and_reference_windows():
    config = ControllerConfig()
    assert apply_steps([7.9], [100.], config)["alpha"] == [8.]
    assert apply_steps([1.1], [-100.], config)["alpha"] == [1.]
    with pytest.raises(ValueError, match="bounds"):
        apply_steps([9.], [0.], config)
    with pytest.raises(ValueError, match="finite"):
        ResponseController(config).update([4.], [float("nan")], [1.], [1.])
    assert ResponseController(ControllerConfig(enabled=False)).update([4.], [2.], [1.], [1.])["alpha"] == [4.]
    frozen = LocalReference([2.], config)
    assert frozen.refresh([100.]) == [2.]
    assert frozen.refresh([3.], window_boundary=True) == [3.]
    ema = LocalReference([2.], ControllerConfig(reference_mode="ema_window", reference_ema=.99))
    assert ema.refresh([100.]) == [2.]
    assert ema.refresh([3.], window_boundary=True) == pytest.approx([2.01])


@pytest.mark.parametrize("settings", [dict(eta=0), dict(regularizer=0), dict(delta=0),
                                    dict(alpha_min=9), dict(reference_ema=1), dict(max_step=-1)])
def test_invalid_controller_config_fails(settings):
    with pytest.raises(ValueError):
        ControllerConfig(**settings)


def linear_probe(gains):
    def observe(alpha, estimate=True):
        return dict(G=[1.+i+g*(a-4.) for i, (a, g) in enumerate(zip(alpha, gains))],
                    Rhat=list(gains) if estimate else None, loss=1., seconds=0.)
    return observe


@pytest.mark.parametrize("gain", [-.08, .08])
def test_deterministic_sign_flip_recovery_and_random_matching(gain):
    observe = linear_probe([gain]*4)
    config = ControllerConfig()
    first = recover_case(observe, [5., 3., 5., 3.], observe([4.]*4), config)
    second = recover_case(observe, [5., 3., 5., 3.], observe([4.]*4), config)
    # Exclude wall-clock timing from deterministic scientific comparisons.
    for result in (first, second):
        for method in result["methods"].values():
            for update in method["updates"]:
                update.pop("controller_seconds")
    assert first == second
    aware = first["methods"]["response_aware"]
    assert aware["final_normalized_error"] < .1
    assert aware["iterations_to_50_percent"] is not None
    assert aware["iterations_to_75_percent"] is not None
    assert all(all(u["descent_direction_consistent_with_estimated_response"]) for u in aware["updates"])
    assert first["methods"]["no_control"]["final_normalized_error"] == 1.
    if gain < 0:
        assert first["methods"]["fixed_sign"]["final_normalized_error"] > 1.
    matched = first["methods"]["random_matched"]["updates"]
    assert len(matched) == len(aware["updates"])
    assert all(u["magnitudes_exactly_matched"] for u in matched)
    json.dumps(first, allow_nan=False)


def tiny_model():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    return build_model(dict(img_size=8, patch_size=4, dim=16, num_heads=4,
        mlp_ratio=2, layer=4, time_step=2, num_classes=10), "torch")


def test_pooled_gate_allows_expected_positive_state_fixed_sign_ties():
    settings = load_config(ROOT / "configs/gradient_flow/controller_feasibility.yml")
    config = ControllerConfig(**settings["controller"])
    cases = []
    for state, gain in (("early", -.08), ("mid", .08), ("late", .12)):
        observe = linear_probe([gain]*4)
        for index in range(10):
            for disturbance, alpha in settings["disturbances"].items():
                cases.append(dict(recover_case(observe, alpha, observe([4.]*4), config, 42+index),
                                  state=state, disturbance=disturbance))
    summary = aggregate(cases, settings, synthetic=True)
    assert all(summary["reference_sign_reversal"])
    mid = summary["groups"]["mid"]["global_plus"]
    assert mid["response_aware"]["mean_final_ratio"] == mid["fixed_sign"]["mean_final_ratio"]
    pooled = summary["pooled_comparison"]
    assert pooled["response_aware"]["mean_final_ratio"] < pooled["fixed_sign"]["mean_final_ratio"]
    assert not summary["gate"]["passed"]  # Synthetic mechanics never authorizes real training.
    assert not any("fixed_sign" in failure for failure in summary["gate"]["failures"])


def test_centered_probes_restore_weights_optimizer_rng_alpha_modes_neurons(monkeypatch):
    model = tiny_model()
    batch = dict(images=torch.randn(2, 3, 8, 8), targets=torch.tensor([1, 2]))
    model.eval()
    with torch.no_grad():
        model(batch["images"])
    model.train()
    model.block[0].eval()
    optimizer = torch.optim.AdamW(model.parameters())
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    params, opt = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
    grads = [p.grad.clone() for p in model.parameters()]
    memories = [(m, copy.deepcopy(m._memories)) for m in model.modules() if hasattr(m, "_memories")]
    modes = [m.training for m in model.modules()]
    scope, rng = BlockAlphas(model), rng_snapshot()
    original = scope.snapshot()
    def forbidden(*args, **kwargs):
        raise AssertionError("Optimizer must never step during recovery probes")
    monkeypatch.setattr(optimizer, "step", forbidden)
    probe = FrozenProbe(model, batch, ControllerConfig(), optimizer)
    output = probe([3., 5., 3., 5.])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        repeat = probe([3., 5., 3., 5.])  # The probe explicitly disables inherited autocast.
    assert output["G"] == repeat["G"] and output["Rhat"] == repeat["Rhat"]
    assert same_state(params, model.state_dict()) and same_state(opt, optimizer.state_dict())
    assert all(torch.equal(p.grad, g) for p, g in zip(model.parameters(), grads))
    assert scope.snapshot() == original and same_state(rng_snapshot(), rng)
    assert all(same_state(m._memories, memory) for m, memory in memories)
    assert [m.training for m in model.modules()] == modes
    raw = output["raw_probe"]
    assert raw["optimizer_updates"] == 0 and raw["amp"] is False and raw["mode"] == "eval"
    for trial in raw["trials"]:
        expected = [3., 5., 3., 5.]
        expected[trial["target_block"]] = trial["alpha"]
        assert trial["block_alphas"] == expected
        assert all(a == 4. for a in trial["sps_alphas"].values())
    json.dumps(output, allow_nan=False)


def test_uniform_probe_legacy_behavior_and_exception_restoration():
    model = tiny_model()
    x, y = torch.randn(2, 3, 8, 8), torch.tensor([1, 2])
    legacy = intervention(model, x, y, [0], [3., 5.], mode="eval")
    extended = intervention(model, x, y, [0], [3., 5.], mode="eval", reference_alphas=[4.]*4)
    extended.pop("reference_alphas")
    extended.pop("alpha_values_by_block")
    assert legacy == extended
    original = copy.deepcopy(model.state_dict())
    scope = BlockAlphas(model)
    alphas = scope.snapshot()
    with pytest.raises(ValueError):
        intervention(model, x, y, [0], [3., 5.], mode="eval", reference_alphas=[3.]*4,
                     alpha_values_by_block={1: [2., 4.]})
    assert scope.snapshot() == alphas and same_state(original, model.state_dict())


def test_recovery_training_split_has_deterministic_eval_preprocessing(monkeypatch):
    from experiments.gradient_flow import make_probe_batch as helper
    config = load_config(ROOT / "configs/gradient_flow/cifar10_diagnostics.yml")
    calls = []
    class Dataset:
        def __init__(self, root, train, download, transform):
            calls.append((train, download))
            self.transform = transform
        def __len__(self):
            return 8
        def __getitem__(self, index):
            from PIL import Image
            return self.transform(Image.new("RGB", (32, 32), (index, 100, 200))), index % 10
    monkeypatch.setattr(helper, "CIFAR10", Dataset)
    first = list(helper.iter_batches("unused", config, 2, 2, split="train"))
    second = list(helper.iter_batches("unused", config, 2, 2, split="train"))
    assert calls == [(True, False)]*2
    for a, b in zip(first, second):
        assert a["split"] == "train" and torch.equal(a["images"], b["images"])
        assert a["targets"].dtype == torch.long
    assert first[0]["indices"] == [0, 1] and first[1]["indices"] == [2, 3]


def test_synthetic_three_state_recovery_smoke_resume_and_plots(tmp_path, monkeypatch):
    manifest = synthetic_states(tmp_path / "checkpoints")
    settings = load_config(ROOT / "configs/gradient_flow/controller_feasibility.yml")
    settings.update(num_batches=1, batch_size=2, disturbances=dict(global_plus=[5.]*4))
    settings["controller"]["max_iterations"] = 1
    batch = dict(images=torch.randn(2, 3, 8, 8), targets=torch.tensor([1, 2]),
                 indices=[0, 1], batch_index=0, synthetic=True)
    path = tmp_path / "recovery"
    result = run_study(manifest, [batch], settings, path, dict(commit="synthetic-only"),
                       backend="torch", device="cpu", synthetic=True)
    assert not result["gate"]["passed"]
    assert "Synthetic data cannot authorize online training" in result["gate"]["failures"]
    assert set(result["groups"]) == {"early", "mid", "late"}
    from analysis.gradient_flow.plot_controller_recovery import plot
    plot(path, tmp_path / "figures")
    assert (tmp_path / "figures/recovery_global_plus.png").is_file()
    assert (tmp_path / "figures/response_sign_by_state.pdf").is_file()
    assert (tmp_path / "figures/summary.txt").is_file()
    import experiments.gradient_flow.controller_recovery as shared
    monkeypatch.setattr(shared.FrozenProbe, "__call__", lambda *a, **k: pytest.fail("Valid recovery batches must skip"))
    assert run_study(manifest, [batch], settings, path, dict(commit="synthetic-only"), backend="torch",
                     device="cpu", resume=True, synthetic=True) == result
    (path / "early/batch_000/complete.json").write_text("corrupt")
    with pytest.raises(pytest.fail.Exception):
        run_study(manifest, [batch], settings, path, dict(commit="synthetic-only"), backend="torch",
                  device="cpu", resume=True, synthetic=True)
