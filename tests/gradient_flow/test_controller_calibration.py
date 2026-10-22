"""Synthetic fixtures only: no downloads, historical result claims or GPU training."""

import copy
from types import SimpleNamespace

import pytest
import torch

from experiments.gradient_flow import calibration_spec as spec
from experiments.gradient_flow import controller_calibration as entry
from experiments.gradient_flow.common import ROOT, load_config, write_json
from experiments.gradient_flow.multi_batch_intervene import digest_json
from experiments.gradient_flow.online_policy import OnlinePolicy
from analysis.gradient_flow.calibration_summary import calibration_gate, relative, summarize


@pytest.mark.parametrize("name,gain,mode,beta", [
    ("current", .25, "frozen_window", .99), ("conservative", .10, "frozen_window", .99),
    ("very_conservative", .05, "frozen_window", .99), ("anchored", .10, "ema_window", .90)])
def test_closed_config_inherits_every_nonexperimental_setting(name, gain, mode, beta):
    declared = spec.declaration()
    config = spec.resolve(declared, name)
    base = load_config(ROOT/spec.BASE)
    assert spec.original_config(config) == base
    c = config["online"]["controller"]
    assert c["eta"] == c["max_step"] == gain
    assert (c["reference_mode"], c["reference_ema"]) == (mode, beta)
    assert (c["alpha_min"], c["alpha_max"]) == (1, 8)
    assert config["fixed_alpha"] == config["sps_alpha"] == 4
    assert config["online"]["epochs"] == 60 and config["epochs"] == 300
    changed = {key for key, value in c.items() if value != base["online"]["controller"][key]}
    assert changed <= {"eta", "max_step", "reference_mode", "reference_ema"}
    assert spec.training_signature(config) == spec.training_signature(declared)


def test_no_extra_variants_or_undeclared_tuning():
    c = spec.declaration()
    with pytest.raises(ValueError, match="predeclared"):
        spec.resolve(c, "another_setting")
    c["calibration"]["variants"]["conservative"]["eta"] = .2
    with pytest.raises(ValueError, match="predeclared"):
        spec.resolve(c, "conservative")


def test_anchored_reference_equation_boundary_and_no_validation_information():
    config = spec.resolve(spec.declaration(), "anchored")
    a, b = [OnlinePolicy(4, "response_aware", config["online"]) for _ in range(2)]
    for epoch in range(11):
        observed = [2.+epoch/10]*4
        first = a.advance(epoch, dict(G=observed, Rhat=[.2]*4, validation_top1=0.))
        second = b.advance(epoch, dict(G=observed, Rhat=[.2]*4, validation_top1=100., test_labels=[9]))
        assert first == second
        expected = 2. if epoch < 5 else 2.05 if epoch < 10 else 2.145
        assert first["G_ref"] == pytest.approx([expected]*4)
        assert first["reference_refreshed"] == (epoch % 5 == 0)
    state = a.state_dict()
    restored = OnlinePolicy(4, "response_aware", config["online"])
    restored.load_state_dict(state)
    observation = dict(G=[4.]*4, Rhat=[.2]*4)
    assert restored.advance(11, observation) == a.advance(11, observation)


def fixture_rows(config, method="response_aware", factor=.5, commit="unit-calibration-commit"):
    policy = OnlinePolicy(4, method, config["online"])
    rows = []
    for epoch in range(60):
        G = [factor*(b+1)*(epoch/60) for b in range(4)]
        c = policy.advance(epoch, dict(G=G, Rhat=[.3]*4))
        rows.append(dict(epoch=epoch, implementation_commit=commit,
            train=dict(loss=2-epoch/100, top1=30.+epoch, top5=99.),
            validation=dict(loss=1., top1=60.+epoch/2, top5=99.), controller=c,
            gradient=[dict(block_id=b, observations=2, G_l=G[b], parameter_grad_norm=.1,
                parameter_grad_cv=.2, grad_input_norm=.1, grad_output_norm=.1,
                directional_consistency=.8, firing_rate=.3) for b in range(4)],
            epoch_seconds=100., controller_seconds=10.,
            amp=dict(overflows=int(epoch == 7), invalid_diagnostics=int(epoch == 7), optimizer_updates=2)))
    return rows


def write_condition(path, config, identity, rows):
    path.mkdir(parents=True)
    for name, payload in (("config.json", config), ("provenance.json", dict(identity=identity)),
            ("metrics.json", rows), ("status.json", dict(status="complete", epochs=len(rows), identity=identity))):
        write_json(path/name, payload)


@pytest.fixture
def completed_fixture(tmp_path, monkeypatch):
    from analysis.gradient_flow import plot_controller_calibration as plots
    config = spec.declaration()
    root = tmp_path/spec.OUTPUT
    legacy = "d8721251988691af90a7e01d5e4ff12cdd4571a8"
    identity = dict(source_commit="unit-calibration-commit", method="response_aware",
        training_signature=spec.training_signature(config), batch_fingerprint="unit-batch",
        runtime=dict(torch="unit", cuda="unit", packages={}), synthetic=False,
        study_config_hash=digest_json(config), seed=42, dataset="torch/cifar10", amp_safety={},
        backend="cupy", smoke_epochs=None)
    for name in ("fixed", "response_aware"):
        prior = {k: v for k, v in identity.items() if k in
            ("source_commit", "method", "training_signature", "batch_fingerprint", "runtime", "synthetic")}
        prior.update(source_commit=legacy, method=name)
        write_condition(tmp_path/spec.ORIGINAL/name, spec.original_config(config), prior,
            fixture_rows(config, name, 1., legacy))
    monkeypatch.setattr(spec, "ROOT", tmp_path)
    monkeypatch.setattr(plots, "ROOT", tmp_path)
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    sources = spec.historical_sources(config, identity)
    root.mkdir(parents=True)
    write_json(root/"study_manifest.json", dict(identity=identity, config=config, sources=sources, variants=spec.VARIANTS))
    (root/"current").mkdir()
    write_json(root/"current/reference.json", dict(condition="current", reused_from_completed_prior_experiment=True, **sources["current"]))
    for name in ("conservative", "very_conservative", "anchored"):
        resolved = spec.resolve(config, name)
        write_condition(root/name, resolved, dict(identity, variant=name), fixture_rows(resolved))
    return plots, root, config, identity


def test_current_sources_are_read_only_and_summary_matches_arithmetic(completed_fixture):
    plots, root, config, identity = completed_fixture
    prior = plots.ROOT/spec.ORIGINAL
    before = {str(p): p.read_bytes() for p in prior.rglob("*") if p.is_file()}
    data, result = plots.load_results(root)
    s = result["conditions"]["conservative"]
    assert len(data) == 5 and result["origins"]["current"]["reused"]
    assert result["origins"]["current"]["path"].endswith("response_aware")
    assert s["epochs"] == 60 and s["amp_overflow_skips"] == 1
    assert result["relative_to_fixed"]["conservative"]["G_variance_reduction_fraction"] == pytest.approx(.75)
    assert result["relative_to_fixed"]["conservative"]["stabilization_per_accuracy_cost"] == pytest.approx(7.5)
    assert s["mean_probe_G_variance"] == pytest.approx(s["mean_train_G_variance"])
    assert s["total_absolute_alpha_movement"] == pytest.approx(sum(s["cumulative_absolute_alpha_movement"]))
    assert before == {str(p): p.read_bytes() for p in prior.rglob("*") if p.is_file()}


@pytest.mark.parametrize("field,value", [("training_signature", "changed"), ("batch_fingerprint", "different"),
    ("source_commit", "another-implementation"), ("runtime", {}), ("study_config_hash", "changed")])
def test_provenance_comparison_rejects_changes(completed_fixture, field, value):
    plots, root, _, _ = completed_fixture
    path = root/"conservative/provenance.json"
    record = spec.read(path)
    record["identity"][field] = value
    write_json(path, record)
    status = spec.read(root/"conservative/status.json")
    status["identity"] = record["identity"]
    write_json(root/"conservative/status.json", status)
    with pytest.raises(ValueError, match="provenance"):
        plots.load_results(root)


def test_current_hash_mutation_refused(completed_fixture):
    plots, root, _, _ = completed_fixture
    path = plots.ROOT/spec.ORIGINAL/"response_aware/metrics.json"
    rows = spec.read(path)
    rows[0]["validation"]["top1"] += .01
    write_json(path, rows)
    with pytest.raises(ValueError, match="evidence changed"):
        plots.load_results(root)


def test_registration_is_immutable_and_current_never_starts_training(completed_fixture, monkeypatch):
    plots, root, config, _ = completed_fixture
    source = dict(commit="unit-calibration-commit", dirty=False, torch="unit", cuda="unit", packages={})
    monkeypatch.setattr(entry, "provenance", lambda: source)
    monkeypatch.setattr(entry, "batch_fingerprint", lambda batch: "unit-batch")
    monkeypatch.setattr(entry, "historical_sources", lambda config, proposed: spec.read(root/"study_manifest.json")["sources"])
    # Registration has its own complete identity, including inherited safety.
    initial = root/"study_manifest.json"
    previous = initial.read_bytes()
    args = SimpleNamespace(expected_commit=None, smoke=False, device="cuda", backend="cupy", smoke_epochs=2, variant="current")
    with pytest.raises(ValueError, match="preregistration/provenance"):
        entry.register(args, config, root, {})
    assert initial.read_bytes() == previous
    # CURRENT's CLI branch must return before any runner can be invoked.
    monkeypatch.setattr(entry, "declaration", lambda data: config)
    monkeypatch.setattr(entry, "output_root", lambda args: root)
    monkeypatch.setattr(entry, "fixed_batch", lambda *args: {})
    monkeypatch.setattr(entry, "register", lambda *args: ({}, source))
    def forbidden(*args):
        pytest.fail("CURRENT must be reference-only")
    monkeypatch.setattr(entry.online_train, "run_official", forbidden)
    monkeypatch.setattr(entry.online_train, "run_smoke", forbidden)
    monkeypatch.setattr("sys.argv", ["calibration", "--variant", "current"])
    entry.main()


def test_incomplete_variants_refuse_finalization_without_any_outputs(completed_fixture):
    plots, root, _, _ = completed_fixture
    path = root/"anchored/status.json"
    status = spec.read(path)
    status["status"] = "running"
    write_json(path, status)
    output = plots.ROOT/"figures/gradient_flow/controller_calibration_seed42"
    with pytest.raises(ValueError, match="Incomplete"):
        plots.plot(root, output)
    assert not output.exists() and not (root/"summary.json").exists()


def test_all_requested_figures_generated_without_touching_sources(completed_fixture):
    plots, root, _, _ = completed_fixture
    prior = plots.ROOT/spec.ORIGINAL
    before = {str(p): p.read_bytes() for p in prior.rglob("*") if p.is_file()}
    output = plots.ROOT/"figures/gradient_flow/controller_calibration_seed42"
    plots.plot(root, output)
    assert len(list(output.glob("*.png"))) == len(list(output.glob("*.pdf"))) == 10
    assert (output/"summary.txt").is_file() and (root/"summary.json").is_file()
    assert before == {str(p): p.read_bytes() for p in prior.rglob("*") if p.is_file()}


def test_gate_thresholds_and_undefined_variance_are_not_misleading():
    config = spec.resolve(spec.declaration(), "very_conservative")
    rows = fixture_rows(config)
    summary = summarize(rows, config["online"])
    fixed = copy.deepcopy(summary)
    fixed["mean_train_G_variance"] = summary["mean_train_G_variance"]/.6
    comparison = relative(summary, fixed, spec.CRITERIA)
    assert calibration_gate(summary, comparison, spec.CRITERIA, complete=True)["promising"]
    assert calibration_gate(summary, comparison, spec.CRITERIA, complete=True)["stronger_positive_indicator"]
    for key, value in (("epochs", 59), ("persistent_failure", True), ("bound_hit_fraction", .001),
            ("controller_overhead_fraction", .20)):
        altered = dict(summary, **{key: value})
        assert not calibration_gate(altered, comparison, spec.CRITERIA, complete=True)["promising"]
    for field, value in (("G_variance_reduction_fraction", .249), ("accuracy_delta_final_pp", -.501)):
        assert not calibration_gate(summary, dict(comparison, **{field: value}), spec.CRITERIA, complete=True)["promising"]
    assert not calibration_gate(summary, comparison, spec.CRITERIA, complete=True, synthetic=True)["promising"]
    assert not calibration_gate(summary, comparison, spec.CRITERIA, complete=False)["promising"]
    zero = relative(summary, dict(fixed, mean_train_G_variance=0.), spec.CRITERIA)
    assert zero["G_variance_reduction_fraction"] is zero["stabilization_per_accuracy_cost"] is None
    assert not calibration_gate(summary, zero, spec.CRITERIA, complete=True)["promising"]


def test_nonoverlap_and_parallel_per_variant_locks(tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    args = SimpleNamespace(output=None, smoke=False)
    assert entry.output_root(args) == tmp_path/spec.OUTPUT
    for output in (spec.ORIGINAL, spec.ORIGINAL+"/conservative", "results/gradient_flow/another"):
        with pytest.raises(ValueError, match="namespace"):
            entry.output_root(SimpleNamespace(output=output, smoke=False))
    with entry.lock(tmp_path/"conservative.lock"):
        with pytest.raises(ValueError, match="Another"):
            with entry.lock(tmp_path/"conservative.lock"):
                pytest.fail("Duplicate writer admitted")
        with entry.lock(tmp_path/"anchored.lock"):
            pass


@pytest.mark.parametrize("name", ["conservative", "very_conservative", "anchored"])
def test_cpu_smoke_uses_shared_runtime_and_resume_is_exact(tmp_path, name):
    from experiments.gradient_flow.online_train import run_smoke
    from experiments.gradient_flow.intervene import same_state
    config = spec.resolve(spec.declaration(), name)
    batch = entry.fixed_batch(SimpleNamespace(smoke=True), config)
    args = SimpleNamespace(device="cpu", backend="torch", smoke_epochs=2, condition="response_aware")
    identity = dict(synthetic=True, variant=name)
    run_smoke(args, config, tmp_path, batch, identity, None, None)
    # Locally generated unit fixture, not an arbitrary CLI checkpoint.
    saved = torch.load(tmp_path/"last.pth.tar", weights_only=False, map_location="cpu")
    state = copy.deepcopy(saved["state_dict"])
    run_smoke(args, config, tmp_path, batch, identity, saved, None)
    restored = torch.load(tmp_path/"last.pth.tar", weights_only=False, map_location="cpu")
    assert same_state(state, restored["state_dict"])
    assert len(saved["gradient_flow_online"]["rows"]) == 2
    assert saved["gradient_flow_online"]["policy"]["settings"] == config["online"]
    assert spec.read(tmp_path/"status.json")["status"] == "complete"
    assert spec.read(tmp_path/"checkpoint_integrity.json")["epoch"] == 1


def test_launcher_has_independent_selection_and_no_node_pin():
    text = (ROOT/"jobs/gradient_flow/run_controller_calibration_condition.sh").read_text()
    assert "#SBATCH --time=2-00:00:00" in text and "#SBATCH --partition=gpu" in text
    assert "#SBATCH --nodelist" not in text
    assert '--variant "${VARIANT}"' in text and "--resume" in text
    assert "random_matched" not in text


def protected_legacy_fixture(plots):
    """Match the actual pre-AMP writer: no epoch commit fields in the saved prefix."""
    path = plots.ROOT/spec.ORIGINAL/"fixed"
    rows = spec.read(path/"metrics.json")
    for row in rows[:7]:
        del row["implementation_commit"]
    write_json(path/"metrics.json", rows)
    artifacts = {}
    for name in ("metrics.json", "config.json", "provenance.json"):
        payload = rows[:7] if name == "metrics.json" else spec.read(path/name)
        protected = path/("amp_resume_origin_"+name)
        write_json(protected, payload)
        artifacts[name] = dict(path=str(protected), sha256=spec.digest(protected))
    marker = dict(legacy_commit="d8721251988691af90a7e01d5e4ff12cdd4571a8",
        implementation_commit="9a9c0a77773b8e962fe90f2558224033d1140c43",
        original_fixed_identity=spec.read(path/"provenance.json")["identity"],
        prefix_epochs=list(range(7)), prefix_metrics_sha256=artifacts["metrics.json"]["sha256"],
        protected_prefix_artifacts=artifacts)
    write_json(path.parent/"amp_compatibility.json", marker)
    return path, rows, marker


def test_protected_legacy_fixed_prefix_reused_without_backfilling(completed_fixture):
    plots, root, config, identity = completed_fixture
    path, rows, _ = protected_legacy_fixture(plots)
    prior = plots.ROOT/spec.ORIGINAL
    before = {str(p): p.read_bytes() for p in prior.rglob("*") if p.is_file()}
    sources = spec.historical_sources(config, identity)
    proof = sources["fixed"]["legacy_epoch_provenance"]
    assert proof["epochs_without_per_epoch_commit"] == list(range(7))
    assert "per-epoch commit was not logged" in proof["evidence"]
    manifest = spec.read(root/"study_manifest.json")
    manifest["sources"] = sources
    write_json(root/"study_manifest.json", manifest)
    write_json(root/"current/reference.json", dict(condition="current",
        reused_from_completed_prior_experiment=True, **sources["current"]))
    data, summary = plots.load_results(root)
    assert data["fixed"] == rows and all("implementation_commit" not in r for r in data["fixed"][:7])
    assert summary["origins"]["fixed"]["legacy_epoch_provenance"] == proof
    with pytest.raises(ValueError, match="Missing actual per-epoch"):
        spec.checked_directory(path)  # Strict default is NOT weakened.
    assert before == {str(p): p.read_bytes() for p in prior.rglob("*") if p.is_file()}


@pytest.mark.parametrize("damage", ["missing_marker", "changed_backup", "changed_prefix", "missing_later_epoch", "wrong_identity"])
def test_legacy_exception_fails_closed_on_invalid_evidence(completed_fixture, damage):
    plots, _, config, identity = completed_fixture
    path, rows, marker = protected_legacy_fixture(plots)
    marker_path = path.parent/"amp_compatibility.json"
    if damage == "missing_marker":
        marker_path.unlink()
    elif damage == "changed_backup":
        (path/"amp_resume_origin_metrics.json").write_text("[]")
    elif damage == "changed_prefix":
        rows[2]["validation"]["top1"] += .01
        write_json(path/"metrics.json", rows)
    elif damage == "missing_later_epoch":
        del rows[8]["implementation_commit"]
        write_json(path/"metrics.json", rows)
    else:
        marker["original_fixed_identity"]["training_signature"] = "wrong"
        write_json(marker_path, marker)
    with pytest.raises((ValueError, FileNotFoundError)):
        spec.historical_sources(config, identity)


def test_legacy_exception_never_applies_to_current_or_new_variants(completed_fixture):
    plots, root, config, identity = completed_fixture
    protected_legacy_fixture(plots)
    path = plots.ROOT/spec.ORIGINAL/"response_aware/metrics.json"
    rows = spec.read(path)
    del rows[0]["implementation_commit"]
    write_json(path, rows)
    with pytest.raises(ValueError, match="Unattested missing historical"):
        spec.historical_sources(config, identity)
    new = root/"anchored/metrics.json"
    rows = spec.read(new)
    del rows[0]["implementation_commit"]
    write_json(new, rows)
    with pytest.raises(ValueError, match="Missing actual per-epoch"):
        spec.checked_directory(new.parent)
