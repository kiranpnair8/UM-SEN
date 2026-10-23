"""Synthetic-only method-design regression tests; no datasets are downloaded."""

import copy
import math
from types import SimpleNamespace

import pytest
import torch

from experiments.gradient_flow import method_ablation as study
from experiments.gradient_flow import calibration_spec as prior
from experiments.gradient_flow.common import ROOT, load_config, write_json
from experiments.gradient_flow.method_policy import MethodPolicy, MethodRuntime, VARIANTS
from experiments.gradient_flow.online_policy import OnlinePolicy
from experiments.gradient_flow.online_train import run_smoke
from analysis.gradient_flow import plot_controller_method_ablation as analysis


def configured(variant, deadband=.01):
    return study.resolve(study.declaration(), variant, deadband)


@pytest.mark.parametrize("variant", VARIANTS)
def test_exact_configs_keep_training_and_reference_unchanged(variant):
    base, config = study.declaration(), configured(variant)
    baseline = load_config(ROOT/prior.BASE)
    for key, value in baseline.items():
        if key not in ("output", "online"):
            assert base[key] == value
    expected = copy.deepcopy(base["online"])
    expected["controller"].update(eta=.1, max_step=.1)
    expected["method_design"] = dict(variant=variant, parameters=VARIANTS[variant], deadband=.01)
    assert config["online"] == expected
    assert config["online"]["controller"]["regularizer"] == 1e-4
    assert config["online"]["reference_window_epochs"] == 5
    assert config["online"]["controller"]["reference_mode"] == "frozen_window"
    assert config["seed"] == 42 and config["online"]["epochs"] == 60


def test_deep_only_block0_exactly_fixed_and_restored():
    settings = configured("deep_only")["online"]
    policy = MethodPolicy(4, "response_aware", settings)
    for epoch in range(60):
        row = policy.advance(epoch, dict(G=[.1*math.sin(epoch)]*4, Rhat=[.2]*4))
        assert row["alpha"][0] == row["alpha_before"][0] == 4. and row["applied_step"][0] == 0.
    assert any(r["applied_step"][1] != 0 for r in policy.history)
    restored = MethodPolicy(4, "response_aware", settings)
    restored.load_state_dict(policy.state_dict())
    assert restored.alpha[0] == 4.
    damaged = policy.state_dict()
    damaged["alpha"][0] = 3.99
    with pytest.raises(ValueError, match="block0"):
        restored.load_state_dict(damaged)


def test_restoring_force_sign_magnitude_and_clamp_order():
    policy = MethodPolicy(4, "response_aware", configured("regularized")["online"])
    result = policy.response_update([5., 3., 4., 4.], [-.2, 0., 0., 0.], [0.]*4, [.1]*4)
    assert result["restoring_step"] == pytest.approx([-.05, .05, 0., 0.])
    raw = .1*.2*.1/(.1**2+1e-4)
    assert result["requested_step"][0] == pytest.approx(raw-.05)
    assert result["applied_step"][0] == pytest.approx(.10)  # Not .05 (clip-then-restore).
    assert result["applied_step"][1] == pytest.approx(.05)


def test_deadband_inclusive_boundary_masks_entire_step_and_ignores_validation():
    policy = MethodPolicy(4, "response_aware", configured("deadband_regularized")["online"])
    row = policy.response_update([5.]*4, [.009, .01, .011, -.01], [0.]*4, [.2]*4)
    assert row["update_active"] == [False, True, True, True]
    assert row["applied_step"][0] == 0.  # Restoring force is suppressed too.
    assert all(row["applied_step"][b] != 0 for b in (1, 2, 3))
    a = MethodPolicy(4, "response_aware", configured("deadband_regularized")["online"])
    b = MethodPolicy(4, "response_aware", configured("deadband_regularized")["online"])
    for epoch in range(10):
        one = a.advance(epoch, dict(G=[epoch*.02]*4, Rhat=[.2]*4, validation_top1=0.))
        two = b.advance(epoch, dict(G=[epoch*.02]*4, Rhat=[.2]*4, validation_top1=100.))
        assert one == two


def test_deadband_is_single_predeclared_training_only_quantile():
    rows = [dict(epoch=i, validation=dict(top1=100.), controller=dict(reference_refreshed=i == 0,
        error=[0.] if i == 0 else [float(i)])) for i in range(21)]
    result = study.choose_deadband(rows)
    assert result["value"] == pytest.approx(5.75)
    changed = copy.deepcopy(rows)
    for r in changed:
        r["validation"]["top1"] = -999.
    assert study.choose_deadband(changed) == result
    assert result["rule"]["quantile"] == .25 and result["observations"] == 20
    with pytest.raises(ValueError, match="Too few"):
        study.choose_deadband(rows[:2])


def rows_fixture(config, method, commit, factor=1., variant=None):
    cls = MethodPolicy if variant else OnlinePolicy
    policy = cls(4, method, config["online"])
    rows = []
    for epoch in range(60):
        control = policy.advance(epoch, dict(G=[.02*(b+1)*math.sin(epoch/3) for b in range(4)], Rhat=[.3]*4))
        rows.append(dict(epoch=epoch, implementation_commit=commit, controller=control,
            train=dict(loss=2-epoch/100, top1=20.+epoch, top5=99.),
            validation=dict(loss=1., top1=60.+epoch/2, top5=99.),
            gradient=[dict(block_id=b, observations=2, G_l=factor*(b+1)*epoch/60,
                parameter_grad_norm=.1, parameter_grad_cv=.2, grad_input_norm=.1, grad_output_norm=.1,
                directional_consistency=.8, firing_rate=.3) for b in range(4)],
            epoch_seconds=100., controller_seconds=10., amp=dict(overflows=0, invalid_diagnostics=0, optimizer_updates=2)))
    return rows


def write_run(path, config, identity, rows):
    path.mkdir(parents=True)
    for name, payload in (("config.json", config), ("metrics.json", rows),
            ("provenance.json", dict(identity=identity)),
            ("status.json", dict(identity=identity, status="complete", epochs=60))):
        write_json(path/name, payload)


@pytest.fixture
def completed(tmp_path, monkeypatch):
    config, calibration = study.declaration(), prior.declaration()
    source = dict(commit="unit-method-commit", torch="unit", cuda="unit", packages={})
    signature = prior.training_signature(calibration)
    historical = dict(source_commit="d8721251988691af90a7e01d5e4ff12cdd4571a8", training_signature=signature,
        batch_fingerprint="unit-batch", runtime={k: source[k] for k in ("torch", "cuda", "packages")}, synthetic=False)
    for name in ("fixed", "response_aware"):
        cfg = prior.original_config(calibration)
        write_run(tmp_path/prior.ORIGINAL/name, cfg, dict(historical, method=name), rows_fixture(cfg, name, historical["source_commit"]))
    monkeypatch.setattr(prior, "ROOT", tmp_path)
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(analysis, "ROOT", tmp_path)
    monkeypatch.setattr(prior, "declaration", lambda data_dir=None: copy.deepcopy(calibration))
    original = prior.historical_sources(calibration)
    calibration_identity = dict(historical, source_commit="7fb359804367c3b44bc9e60d35421c6bf4e1041d", method="response_aware",
        study_config_hash="unit-calibration-hash", seed=42, dataset="torch/cifar10", amp_safety={}, backend="cupy", smoke_epochs=None)
    calroot = tmp_path/prior.OUTPUT
    calroot.mkdir(parents=True)
    write_json(calroot/"study_manifest.json", dict(identity=calibration_identity, config=calibration, sources=original))
    for name in ("conservative", "very_conservative"):
        cfg = prior.resolve(calibration, name)
        write_run(calroot/name, cfg, dict(calibration_identity, variant=name), rows_fixture(cfg, "response_aware", calibration_identity["source_commit"]))
    monkeypatch.setattr(study, "batch_fingerprint", lambda batch: "unit-batch")
    args = SimpleNamespace(backend="cupy", smoke=False)
    root = tmp_path/study.OUTPUT
    manifest = study.register(config, args, root, {}, source)
    for variant in VARIANTS:
        cfg = manifest["variant_configs"][variant]
        ident = dict(manifest["identity"], variant=variant, deadband=manifest["deadband"]["value"])
        write_run(root/variant, cfg, ident, rows_fixture(cfg, "response_aware", source["commit"], .5, variant))
    return root, config, manifest


def test_sources_untouched_and_all_summary_plots_gates(completed):
    root, _, manifest = completed
    before = {str(p): p.read_bytes() for name in (prior.ORIGINAL, prior.OUTPUT)
        for p in (analysis.ROOT/name).rglob("*") if p.is_file()}
    output = analysis.ROOT/analysis.OUTPUT
    result = analysis.plot(root, output)
    assert len(list(output.glob("*.png"))) == len(list(output.glob("*.pdf"))) == 8
    assert result["deep_only_block0_exactly_4"]
    for n in VARIANTS:
        assert result["relative_to_fixed"][n]["G_variance_reduction_fraction"] == pytest.approx(.75)
        assert "stabilization_per_accuracy_cost" not in result["relative_to_fixed"][n]
        assert result["gates"][n]["promising"]
    assert result["ranking"]["eligible_in_descriptive_order"][0] == "deep_only"
    assert result["ranking"]["final_sage_selected"] is None
    assert result["deadband"] == manifest["deadband"]
    assert before == {str(p): p.read_bytes() for name in (prior.ORIGINAL, prior.OUTPUT)
        for p in (analysis.ROOT/name).rglob("*") if p.is_file()}


@pytest.mark.parametrize("damage", ["incomplete", "config", "provenance", "deep_block0", "source", "deadband"])
def test_aggregation_fails_closed(completed, damage):
    root, _, manifest = completed
    if damage == "incomplete":
        p = root/"regularized/status.json"
        record = prior.read(p)
        record["status"] = "running"
    elif damage == "config":
        p = root/"regularized/config.json"
        record = prior.read(p)
        record["lr"] = 1.
    elif damage == "provenance":
        p = root/"regularized/provenance.json"
        record = prior.read(p)
        record["identity"]["source_commit"] = "wrong"
    elif damage == "deep_block0":
        p = root/"deep_only/metrics.json"
        record = prior.read(p)
        record[1]["controller"]["alpha"][0] = 3.9
    elif damage == "source":
        p = analysis.ROOT/prior.ORIGINAL/"response_aware/metrics.json"
        record = prior.read(p)
        record[1]["validation"]["top1"] += .1
    else:
        p = root/"study_manifest.json"
        record = prior.read(p)
        record["deadband"]["value"] *= 2
    write_json(p, record)
    with pytest.raises(ValueError):
        analysis.plot(root, analysis.ROOT/analysis.OUTPUT)
    assert not (root/"summary.json").exists()


def test_preparation_does_not_rewrite_or_change_numeric_deadband(completed):
    root, config, manifest = completed
    source = dict(commit="unit-method-commit", torch="unit", cuda="unit", packages={})
    args = SimpleNamespace(backend="cupy", smoke=False)
    before = (root/"study_manifest.json").read_bytes()
    assert study.register(config, args, root, {}, source)["deadband"] == manifest["deadband"]
    assert (root/"study_manifest.json").read_bytes() == before
    with pytest.raises(ValueError, match="preregistration"):
        study.register(config, args, root, {}, dict(source, commit="another"))


@pytest.mark.parametrize("variant", VARIANTS)
def test_cpu_smokes_shared_pipeline_and_exact_block0(tmp_path, variant):
    config = configured(variant)
    generator = torch.Generator().manual_seed(420)
    batch = dict(images=torch.randn(2, 3, 8, 8, generator=generator), targets=torch.tensor([1, 2]))
    args = SimpleNamespace(device="cpu", backend="torch", condition="response_aware", smoke_epochs=2)
    run_smoke(args, config, tmp_path, batch, {"synthetic": True}, None, None, runtime_class=MethodRuntime)
    rows = prior.read(tmp_path/"metrics.json")
    assert len(rows) == 2
    assert all(math.isfinite(v) for r in rows for v in r["controller"]["G"])
    if variant == "deep_only":
        assert all(r["controller"]["alpha"][0] == 4. for r in rows)
    saved = torch.load(tmp_path/"last.pth.tar", weights_only=False, map_location="cpu")
    restored = MethodPolicy(4, "response_aware", config["online"])
    restored.load_state_dict(saved["gradient_flow_online"]["policy"])
    assert restored.alpha == rows[-1]["controller"]["alpha"]


def test_parallel_locks_and_launcher_namespace(tmp_path):
    from experiments.gradient_flow.controller_calibration import lock
    with lock(tmp_path/"deep.lock"):
        with pytest.raises(ValueError, match="Another"):
            with lock(tmp_path/"deep.lock"):
                pytest.fail("Concurrent duplicate")
        with lock(tmp_path/"regularized.lock"):
            pass
    text = (ROOT/"jobs/gradient_flow/run_controller_method_ablation_condition.sh").read_text()
    assert "#SBATCH --nodelist" not in text and "--time=2-00:00:00" in text
    assert "--prepare-only" not in text and '--variant "${VARIANT}"' in text


def test_followups_blocked_without_review_and_configurable_without_rewrites(tmp_path, monkeypatch):
    from experiments.gradient_flow import controller_followup as future
    selection = dict(final_sage="deep_only", deadband=.02)
    a = future.resolved_config("cifar10", 43, selection)
    b = future.resolved_config("cifar100", 44, dict(selection, final_sage="regularized"))
    assert a["seed"] == 43 and b["seed"] == 44
    assert a["dataset"] == "torch/cifar10" and b["dataset"] == "torch/cifar100"
    assert b["num_classes"] == 100 and a["num_classes"] == 10
    assert a["online"]["method_design"]["variant"] == "deep_only"
    assert b["online"]["method_design"]["variant"] == "regularized"
    assert a["online"]["method_design"]["deadband"] == b["online"]["method_design"]["deadband"] == .02
    monkeypatch.setattr(future, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="approve FINAL_SAGE"):
        future.read_selection()
    assert not (tmp_path/"results").exists()


def test_cifar100_probe_uses_explicit_dataset_not_cifar10(monkeypatch):
    from experiments.gradient_flow import make_probe_batch as probe
    from experiments.gradient_flow.make_probe_batch import iter_batches
    from experiments.gradient_flow.controller_followup import resolved_config
    config = resolved_config("cifar100", 43, dict(final_sage="regularized", deadband=.02))
    calls = []
    class Dataset100:
        def __init__(self, root, train, download, transform):
            calls.append((root, train, download))
        def __len__(self):
            return 32
        def __getitem__(self, index):
            return torch.zeros(3, 32, 32), 68+index
    batch = next(iter_batches("unit-no-data", config, split="train", dataset_class=Dataset100))
    assert calls == [("unit-no-data", True, False)]
    assert batch["dataset"] == "cifar100" and int(batch["targets"].max()) == 99
    monkeypatch.setattr(probe, "CIFAR100", Dataset100)
    def wrong_dataset(*args, **kwargs):
        pytest.fail("CIFAR100 must never fall back to CIFAR10")
    monkeypatch.setattr(probe, "CIFAR10", wrong_dataset)
    automatic = next(iter_batches("unit-no-data", config, split="train"))
    assert int(automatic["targets"].max()) == 99


def test_gate_ranking_mechanical_accuracy_and_variance_order(completed):
    root, _, _ = completed
    _, result = analysis.load_results(root)
    s = result["conditions"]["regularized"]
    comparison = result["relative_to_fixed"]["regularized"]
    criteria = result["criteria"]
    assert analysis.method_gate(s, comparison, criteria)["promising"]
    assert not analysis.method_gate(dict(s, alpha_max=[8.1]*4), comparison, criteria)["promising"]
    assert not analysis.method_gate(s, dict(comparison, accuracy_delta_final_pp=-.501), criteria)["promising"]
    assert not analysis.method_gate(s, dict(comparison, G_variance_reduction_fraction=.249), criteria)["promising"]
    summaries, gates = copy.deepcopy(result["conditions"]), copy.deepcopy(result["gates"])
    comparisons = copy.deepcopy(result["relative_to_fixed"])
    gates["deep_only"]["checks"]["no_persistent_instability"] = False
    comparisons["regularized"]["G_variance_reduction_fraction"] = .5
    comparisons["deadband_regularized"]["G_variance_reduction_fraction"] = .6
    ranked = analysis.ranking(summaries, comparisons, gates)
    assert ranked["eligible_in_descriptive_order"] == ["deadband_regularized", "regularized"]
    assert "deep_only" in ranked["excluded"]
