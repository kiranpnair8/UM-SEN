from argparse import Namespace
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("spikingjelly.clock_driven.functional")

from experiments.gradient_flow.common import ROOT, build_model, load_config
from experiments.gradient_flow.checkpoint_stages import (STAGES, audit, checkpoint_epoch, choose_stages,
    assess_historical, normalize_training_recipe, recipe_signature, scheduled_epochs, validate_baseline)
from experiments.gradient_flow.intervene import same_state
from experiments.gradient_flow.multi_batch_intervene import digest_file
from experiments.gradient_flow.multi_batch_stats import aggregate_measurements
from experiments.gradient_flow.state_response import compare_states, matrix_similarity, normalized_change
from experiments.gradient_flow.state_intervene import run_states, validate_manifest
from experiments.gradient_flow.train_milestones import milestone_saver_class
from test_multibatch import fake_measurements


def synthetic_states(path):
    """Contrived tiny checkpoint fixtures, NOT samples from a real training trajectory."""
    path.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    recipe = dict(img_size=8, patch_size=4, dim=16, num_heads=4, mlp_ratio=2, layer=4, time_step=4,
        num_classes=10, seed=42, model="spikformer", dataset="torch/cifar10", epochs=300,
        cooldown_epochs=0, sched="cosine", lr=.0005, min_lr=.00001, warmup_lr=.00001, warmup_epochs=20)
    model = build_model(recipe, "torch")
    entries = {}
    for label, epoch in zip(STAGES, (20, 150, 299)):
        with torch.no_grad():
            model.head.bias.add_(.01)
        checkpoint = path / (label+".pth.tar")
        torch.save(dict(state_dict=model.state_dict(), args=Namespace(**recipe, trajectory_id="SYNTHETIC-ONLY"),
                        epoch=epoch, version=2, metric=0.), checkpoint)
        entries[label] = dict(path=str(checkpoint.resolve()), sha256=digest_file(checkpoint), epoch=epoch, metric=0.)
    return dict(states=entries, training_seed=42, resolved_training_epochs=300, recipe=recipe,
        trajectory_directory=str(path.resolve()), trajectory_id="SYNTHETIC-ONLY", origin="regenerated",
        synthetic=True, trajectory_evidence="Synthetic correctness fixtures, not a scientific training trajectory")


def test_audit_actual_epochs_and_no_best_substitution(tmp_path):
    assert not audit(tmp_path / "unavailable")["available"]
    manifest = synthetic_states(tmp_path / "checkpoints")
    report = audit(tmp_path / "checkpoints")
    selected = choose_stages(report, manifest["recipe"], [20, 150, 299], "SYNTHETIC-ONLY")
    assert set(selected) == set(STAGES)
    report["checkpoints"] = [c for c in report["checkpoints"] if c["epoch"] != 299]
    best = copy.deepcopy(report["checkpoints"][0])
    best.update(path=str(tmp_path / "model_best.pth.tar"), epoch=262)
    report["checkpoints"].append(best)
    assert "late" not in choose_stages(report, manifest["recipe"], [20, 150, 299])
    assert checkpoint_epoch(dict(epoch=21, version=1)) == 20
    assert checkpoint_epoch(dict(epoch=20, version=2)) == 20
    assert checkpoint_epoch(dict(epoch=20)) is None
    assert checkpoint_epoch(dict(epoch=309, version=2)) == 309
    assert checkpoint_epoch(dict(epoch=310, version=1)) == 309
    recipe = load_config(ROOT / "spikformer/cifar10/cifar10.yml")
    recipe.update(seed=42)
    validate_baseline(recipe)
    assert scheduled_epochs(recipe) == 310
    assert recipe_signature(dict(recipe, output="a")) == recipe_signature(dict(recipe, output="b"))
    assert recipe_signature(dict(recipe, cutmix=.1)) != recipe_signature(dict(recipe, cutmix=.2))


def test_manifest_labels_duplicate_hash_and_trajectory_safety(tmp_path):
    manifest = synthetic_states(tmp_path / "checkpoints")
    validate_manifest(manifest)
    bad = copy.deepcopy(manifest)
    bad["states"]["mid"] = bad["states"]["early"]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_manifest(bad)
    bad = copy.deepcopy(manifest)
    bad["states"]["mid"]["epoch"] = 149
    with pytest.raises(ValueError, match="epoch"):
        validate_manifest(bad)
    bad = copy.deepcopy(manifest)
    bad["trajectory_id"] = "other-trajectory"
    with pytest.raises(ValueError, match="trajectory ID"):
        validate_manifest(bad)


def test_final_checkpoint_comparison_is_explicit(tmp_path, monkeypatch):
    import experiments.gradient_flow.checkpoint_stages as stages
    manifest = synthetic_states(tmp_path / "checkpoints")
    report = audit(tmp_path / "checkpoints")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    comparison = stages.compare_final_checkpoints(report, report, manifest["recipe"], 299, "unused")
    assert comparison["bitwise_equal"] is True
    assert comparison["maximum_absolute_weight_or_buffer_difference"] == 0
    assert comparison["prediction_comparison"] == "not measured: CUDA unavailable"


def test_paired_statistics_normalization_and_structural_mask():
    records = {}
    for label, shift in zip(STAGES, (0, 2, 3)):
        payloads = [fake_measurements(np.eye(4)*k + np.eye(4)*shift) for k in (1, 2, 3)]
        records[label] = dict(aggregate=aggregate_measurements(payloads, bootstrap_samples=100),
            batch_fingerprints=["b0", "b1", "b2"], G_reference=[[1.+shift]*4]*3)
    result = compare_states(records, dict(early=20, mid=150, late=299), bootstrap_samples=100)
    delta = result["differences"]["late_minus_early"]
    np.testing.assert_allclose(delta["matrix"]["mean"], 3*np.eye(4))
    np.testing.assert_allclose(delta["paired_diagonal_change"]["ci95"], [[3.]*4]*2)
    assert delta["paired_structural_change"]["mean_abs_diagonal"]["ci95"] == [3., 3.]
    assert delta["paired_structural_change"]["mean_abs_nearest_upstream"]["ci95"] == [0., 0.]
    normalized = normalized_change([0, 1e-10, 2], [1, 1, 5])
    assert normalized[:2] == [None, None] and normalized[2] == pytest.approx(1.5)
    a = np.eye(4)
    b = a + np.tril(np.ones((4, 4))*100, -1)
    assert matrix_similarity(a, b)["cosine"] == pytest.approx(1.)
    assert not np.asarray(matrix_similarity(a, b)["mask"])[3, 0]
    identical = compare_states({s: records["early"] for s in STAGES}, dict(early=20, mid=150, late=299), bootstrap_samples=100)
    assert np.count_nonzero(identical["differences"]["late_minus_early"]["matrix"]["mean"]) == 0
    bad = copy.deepcopy(records)
    bad["mid"]["batch_fingerprints"] = ["b1", "b0", "b2"]
    with pytest.raises(ValueError, match="correspondence"):
        compare_states(bad, dict(early=20, mid=150, late=299), bootstrap_samples=100)
    json.dumps(result, allow_nan=False)


def test_milestone_copy_preserves_full_state_and_rng(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "spikformer/cifar10"))
    from experiment_state import capture_rng, load_trusted_checkpoint, state_saver_class
    from timm.utils import CheckpointSaver
    import os
    import shutil
    if os.name == "nt":
        # This Windows test sandbox denies hardlinks; production timm is unchanged.
        monkeypatch.setattr(os, "link", shutil.copyfile)
    assert milestone_saver_class(CheckpointSaver, []) is CheckpointSaver
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    base = state_saver_class(CheckpointSaver)
    cls = milestone_saver_class(base, [20])
    saver = cls(model, optimizer, args=Namespace(model="synthetic", trajectory_id="SYNTHETIC-ONLY"),
        checkpoint_dir=str(tmp_path), scheduler=scheduler, max_history=1)
    before = capture_rng()
    params = copy.deepcopy(model.state_dict())
    lr = optimizer.param_groups[0]["lr"]
    saver.save_recovery(20, batch_idx=3)
    assert not (tmp_path / "checkpoint_epoch020.pth.tar").exists()
    saver.save_checkpoint(20, metric=.5)
    assert same_state(before, capture_rng())
    copied = load_trusted_checkpoint(tmp_path / "checkpoint_epoch020.pth.tar")
    last = load_trusted_checkpoint(tmp_path / "last.pth.tar")
    assert digest_file(tmp_path / "checkpoint_epoch020.pth.tar") == digest_file(tmp_path / "last.pth.tar")
    assert same_state(copied, last)
    assert same_state(params, model.state_dict()) and optimizer.param_groups[0]["lr"] == lr
    assert copied["epoch"] == 20 and same_state(copied["optimizer"], optimizer.state_dict())
    assert same_state(copied["sage_training_state"]["scheduler"], scheduler.state_dict())
    saver.save_checkpoint(21, metric=.6)
    assert (tmp_path / "checkpoint_epoch020.pth.tar").exists()
    assert not (tmp_path / "checkpoint_epoch021.pth.tar").exists()


def test_prepare_reuses_complete_historical_run_without_training(tmp_path, monkeypatch):
    import yaml
    import experiments.gradient_flow.checkpoint_stages as stages
    manifest = synthetic_states(tmp_path / "historical")
    (tmp_path / "historical/args.yaml").write_text(yaml.safe_dump(manifest["recipe"]))
    monkeypatch.setattr(stages, "provenance", lambda: dict(commit="synthetic-test", dirty=False))
    # Tiny fixtures replace only the real-backbone guard, not selection/audit logic.
    monkeypatch.setattr(stages, "validate_baseline", lambda recipe: None)
    def forbidden(*args, **kwargs):
        raise AssertionError("Complete historical targets must never retrain")
    monkeypatch.setattr(stages.subprocess, "run", forbidden)
    result = stages.prepare(tmp_path / "historical", tmp_path / "study", "unused", regenerate=True)
    assert result["origin"] == "historical"
    assert result["source_mode"] == "historical_complete"
    assert result["historical_rejection_reason"] is None
    validate_manifest(result)
    original_audit = stages.audit
    def missing_early(path):
        result = original_audit(path)
        result["checkpoints"] = [c for c in result["checkpoints"] if c.get("epoch") != 20]
        return result
    monkeypatch.setattr(stages, "audit", missing_early)
    with pytest.raises(ValueError, match="Explicit --regenerate-missing"):
        stages.prepare(tmp_path / "historical", tmp_path / "missing", "unused", regenerate=False)
    def conflicting_recipe(path):
        result = original_audit(path)
        next(c for c in result["checkpoints"] if c["epoch"] == 20)["args"]["mixup"] = .123
        return result
    monkeypatch.setattr(stages, "audit", conflicting_recipe)
    with pytest.raises(ValueError, match="manual review"):
        stages.prepare(tmp_path / "historical", tmp_path / "conflict", "unused", regenerate=True)


@pytest.mark.parametrize("late_evidence", ["provenance_only", "missing_metadata", "matching", "runtime_normalization"])
def test_lawrence_partial_collection_regenerates_complete_trajectory(tmp_path, monkeypatch, late_evidence):
    """Synthetic reproduction of the reported file collection, not real HPC metadata."""
    import yaml
    import experiments.gradient_flow.checkpoint_stages as stages
    manifest = synthetic_states(tmp_path / "historical")
    historical = tmp_path / "historical"
    recipe = dict(manifest["recipe"], cooldown_epochs=10, mixup=.5)
    if late_evidence == "runtime_normalization":
        recipe.update(amp=True, native_amp=False, apex_amp=False, no_prefetcher=False)
    (historical / "args.yaml").write_text(yaml.safe_dump(recipe))
    for p in historical.glob("*.pth.tar"):
        p.unlink()
    for epoch in (262, 283, 288, 295, 296, 298, 299, 305, 306, 308, 309):
        args = dict(recipe)
        if late_evidence == "runtime_normalization":
            args.update(native_amp=True, prefetcher=True)
        if epoch == 309 and late_evidence == "provenance_only":
            args["git_commit"] = "historical-source-unknown-to-args-yaml"
        if epoch == 309 and late_evidence == "missing_metadata":
            args.pop("mixup")
        name = "last.pth.tar" if epoch == 309 else f"checkpoint-{epoch}.pth.tar"
        torch.save(dict(state_dict={"synthetic": torch.tensor(float(epoch))}, args=Namespace(**args),
                        epoch=epoch, version=2, arch="spikformer", optimizer={}, amp_scaler={}, metric=0.), historical / name)
    import shutil
    shutil.copyfile(historical / "checkpoint-262.pth.tar", historical / "model_best.pth.tar")
    before = {p.name: digest_file(p) for p in historical.iterdir()}
    monkeypatch.setattr(stages, "provenance", lambda: dict(commit="synthetic-source-sha", dirty=False))
    monkeypatch.setattr(stages, "validate_baseline", lambda recipe: None)
    monkeypatch.setattr(stages, "compare_final_checkpoints", lambda *args: dict(status="synthetic test: no GPU"))
    calls = []
    def fake_train(command, **kwargs):
        calls.append(command)
        settings = yaml.safe_load(Path(command[command.index("-c")+1]).read_text())
        directory = Path(settings["output"]) / settings["experiment"]
        directory.mkdir(parents=True)
        for epoch in settings["milestone_checkpoints"]:
            torch.save(dict(state_dict={"synthetic": torch.tensor(float(epoch))}, args=Namespace(**settings),
                epoch=epoch, version=2, sage_training_state={"controller": None, "scheduler": {}, "rng": {}},
                optimizer={}, metric=0.), directory / f"checkpoint_epoch{epoch:03d}.pth.tar")
    monkeypatch.setattr(stages.subprocess, "run", fake_train)
    with pytest.raises(ValueError, match="Explicit --regenerate-missing"):
        stages.prepare(historical, tmp_path / "not_authorized", "unused", regenerate=False)
    assert not calls
    result = stages.prepare(historical, tmp_path / "study", "unused", regenerate=True)
    assert len(calls) == 1
    assert result["origin"] == "regenerated" and result["source_mode"] == "regenerated_complete_trajectory"
    assert result["historical_rejection_reason"]["missing_states"] == [20, 150]
    assert result["historical_rejection_reason"]["incompatible_target_epochs"] == (
        [] if late_evidence in ("matching", "runtime_normalization") else [309])
    assert {e["logical_epoch"] for e in result["states"].values()} == {20, 150, 309}
    assert len({e["sha256"] for e in result["states"].values()}) == 3
    assert {e["trajectory_id"] for e in result["states"].values()} == {result["trajectory_id"]}
    assert {e["recipe_sha256"] for e in result["states"].values()} == {result["recipe_sha256"]}
    assert all(e["seed"] == 42 and e["total_scheduled_epochs"] == 310 and
               e["git_commit"] == "synthetic-source-sha" for e in result["states"].values())
    assert all(Path(e["path"]).parent == (tmp_path / "study/trajectory/fixed_seed42").resolve()
               and e["physical_checkpoint"] == Path(e["path"]).name for e in result["states"].values())
    validate_manifest(result)
    assert {p.name: digest_file(p) for p in historical.iterdir()} == before
    assessment = json.loads((tmp_path / "study/historical_assessment.json").read_text())
    late = next(c for c in assessment["checkpoint_diagnoses"] if c["epoch"] == 309)
    assert Path(late["path"]).name == "last.pth.tar" and late["raw_epoch"] == 309 and late["version"] == 2
    assert assessment["epoch_numbering"]["scheduled_epoch_count"] == 310
    if late_evidence in ("provenance_only", "missing_metadata"):
        assert late["recipe_differences"]
        assert result["historical_rejection_reason"]["checkpoint_conflicts"]
    if late_evidence == "runtime_normalization":
        assert assessment["checkpoint_recipes_identical"]
        assert assessment["complete_regeneration_selected"]
        assert not late["recipe_differences"]
        assert {d["field"] for d in late["source_recipe_differences"]} == {"native_amp", "prefetcher"}
        assert set(assessment["runtime_normalization"]["applied_rules"]) == {"native_amp", "prefetcher"}
        assert assessment["source_recipe"]["native_amp"] is False and "prefetcher" not in assessment["source_recipe"]
        assert assessment["regeneration_effective_recipe"]["native_amp"] is True
        assert assessment["regeneration_effective_recipe"]["prefetcher"] is True
        assert assessment["regeneration_effective_recipe"]["amp"] is True
        assert assessment["regeneration_effective_recipe"]["no_prefetcher"] is False
        settings = yaml.safe_load((tmp_path / "study/regeneration.yml").read_text())
        assert settings["native_amp"] is True and settings["prefetcher"] is True
    # A restart skips training and keeps exactly the same primary checkpoint identities.
    assert stages.prepare(historical, tmp_path / "study", "unused", regenerate=True) == result
    assert len(calls) == 1


@pytest.mark.parametrize("field,value", [("lr", .1), ("seed", 43), ("cooldown_epochs", 10),
                                        ("fixed_alpha", 3.), ("dim", 32), ("unknown_training_option", 123)])
def test_internally_conflicting_recipe_still_blocks_regeneration(tmp_path, monkeypatch, field, value):
    import yaml
    import experiments.gradient_flow.checkpoint_stages as stages
    manifest = synthetic_states(tmp_path / "historical")
    historical = tmp_path / "historical"
    (historical / "args.yaml").write_text(yaml.safe_dump(manifest["recipe"]))
    checkpoint = stages.load_trusted_checkpoint(historical / "late.pth.tar")
    setattr(checkpoint["args"], field, value)
    torch.save(checkpoint, historical / "late.pth.tar")
    before = {p.name: digest_file(p) for p in historical.iterdir()}
    monkeypatch.setattr(stages, "validate_baseline", lambda recipe: None)
    def forbidden(*args, **kwargs):
        raise AssertionError("Ambiguous recipe must never train")
    monkeypatch.setattr(stages.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="internally ambiguous.*manual review"):
        stages.prepare(historical, tmp_path / "study", "unused", regenerate=True)
    report = json.loads((tmp_path / "study/historical_assessment.json").read_text())
    assert not report["recipe_unambiguous"]
    assert report["internal_conflicts"][0]["conflicts"][0]["field"] == field
    assert not (tmp_path / "study/trajectory_plan.json").exists()
    assert {p.name: digest_file(p) for p in historical.iterdir()} == before


def runtime_recipe_fixture():
    source = dict(model="spikformer", amp=True, native_amp=False, apex_amp=False, no_prefetcher=False, lr=.0005)
    checkpoints = [dict(path=name, args=dict(source, native_amp=True, prefetcher=True), scaler_keys=["amp_scaler"])
                   for name in ("checkpoint-262.pth.tar", "checkpoint-299.pth.tar", "last.pth.tar", "model_best.pth.tar")]
    return source, checkpoints


def test_verified_runtime_recipe_resolution_preserves_source_and_full_recipe():
    source, checkpoints = runtime_recipe_fixture()
    before = copy.deepcopy((source, checkpoints))
    result = normalize_training_recipe(source, checkpoints)
    assert not result["errors"]
    assert result["effective_recipe"] == dict(source, native_amp=True, prefetcher=True)
    assert (source, checkpoints) == before
    for field, rule in result["applied_rules"].items():
        assert "e01d1db535548a1e8505ff805673569c98107449" in rule["evidence"]
        assert rule["effective_runtime"] is True and len(rule["checkpoints"]) == 4
        assert all(c["args"][field] is True for c in checkpoints)
    assert result["applied_rules"]["native_amp"]["args_yaml"] is False
    assert not result["applied_rules"]["prefetcher"]["args_yaml_present"]
    assert len({json.dumps(recipe_signature(c["args"]), sort_keys=True) for c in checkpoints}) == 1


@pytest.mark.parametrize("field", ["native_amp", "prefetcher"])
def test_runtime_checkpoint_disagreement_fails(field):
    source, checkpoints = runtime_recipe_fixture()
    checkpoints[0]["args"][field] = False
    result = normalize_training_recipe(source, checkpoints)
    assert any(e["field"] == field and "unanimous" in e["reason"] for e in result["errors"])
    assessment = assess_historical(dict(checkpoints=checkpoints), source, [20, 150, 309])
    assert not assessment["recipe_unambiguous"]


@pytest.mark.parametrize("bad_evidence", ["amp_disabled", "apex_requested", "scaler_missing", "no_rule", "wrong_prefetcher"])
def test_runtime_discrepancy_without_verified_evidence_fails(monkeypatch, bad_evidence):
    import experiments.gradient_flow.checkpoint_stages as stages
    source, checkpoints = runtime_recipe_fixture()
    if bad_evidence == "amp_disabled":
        source["amp"] = False
        for c in checkpoints:
            c["args"]["amp"] = False
    elif bad_evidence == "apex_requested":
        source["apex_amp"] = True
        for c in checkpoints:
            c["args"]["apex_amp"] = True
    elif bad_evidence == "scaler_missing":
        checkpoints[0]["scaler_keys"] = []
    elif bad_evidence == "no_rule":
        monkeypatch.setattr(stages, "RUNTIME_RESOLUTION_EVIDENCE", {})
    else:
        for c in checkpoints:
            c["args"]["no_prefetcher"] = True
    result = normalize_training_recipe(source, checkpoints)
    assert result["errors"]
    assert not assess_historical(dict(checkpoints=checkpoints), source, [20, 150, 309])["recipe_unambiguous"]


def test_valid_runtime_normalization_does_not_mask_other_recipe_conflicts():
    source, checkpoints = runtime_recipe_fixture()
    checkpoints[0]["args"]["lr"] = .1
    assessment = assess_historical(dict(checkpoints=checkpoints), source, [20, 150, 309])
    assert set(assessment["runtime_normalization"]["applied_rules"]) == {"native_amp", "prefetcher"}
    assert not assessment["runtime_normalization"]["errors"]
    assert not assessment["recipe_unambiguous"]
    assert not assessment["checkpoint_recipes_identical"]
    assert assessment["internal_conflicts"][0]["conflicts"][0]["field"] == "lr"


def test_assessment_epoch_filename_and_duplicate_selection(tmp_path):
    manifest = synthetic_states(tmp_path / "historical")
    report = audit(tmp_path / "historical")
    for c, epoch in zip(sorted(report["checkpoints"], key=lambda c: c["epoch"]), (20, 150, 309)):
        c["epoch"] = epoch
    late = next(c for c in report["checkpoints"] if c["epoch"] == 309)
    late.update(path=str(tmp_path / "checkpoint-308.pth.tar"), raw_epoch=310, version=1)
    assessment = assess_historical(report, manifest["recipe"], [20, 150, 309])
    assert assessment["complete_and_compatible"]
    entry = next(c for c in assessment["checkpoint_diagnoses"] if c["epoch"] == 309)
    assert entry["raw_epoch"] == 310 and entry["epoch_semantics"] == "stored next epoch minus one"
    report["checkpoints"][1]["sha256"] = report["checkpoints"][0]["sha256"]
    with pytest.raises(ValueError, match="duplicate checkpoint content"):
        choose_stages(report, manifest["recipe"], [20, 150, 309])
    assert not assess_historical(report, manifest["recipe"], [20, 150, 309])["complete_and_compatible"]


def test_study_never_writes_inside_historical_directory(tmp_path):
    import experiments.gradient_flow.checkpoint_stages as stages
    historical = tmp_path / "historical"
    with pytest.raises(ValueError, match="historical artifacts are read-only"):
        stages.prepare(historical, historical / "study", "unused", regenerate=True)
    assert not historical.exists()


def test_three_state_smoke_resume_and_corruption(tmp_path, monkeypatch):
    manifest = synthetic_states(tmp_path / "checkpoints")
    generator = torch.Generator().manual_seed(42)
    batches = [dict(images=torch.randn(2, 3, 8, 8, generator=generator), targets=torch.tensor([i, i+1]),
                    batch_index=i, indices=[2*i, 2*i+1], synthetic=True) for i in range(2)]
    settings = dict(num_batches=2, batch_size=2, seed=42, alpha_values=[2, 3, 4, 5, 6],
        reference_alpha=4., zero_tolerance=1e-8, bootstrap_samples=100, backend="torch", device="cpu",
        synthetic=True)
    destination = tmp_path / "study"
    result = run_states(manifest, batches, destination, settings)
    assert all(np.asarray(result["per_state"][s]["aggregate"]["response_stats"]["mean"]).shape == (4, 4) for s in STAGES)
    assert len((destination / "aggregate/state_raw_measurements.jsonl").read_text().splitlines()) == 480
    from analysis.gradient_flow.plot_alpha_states import plot
    plot(result, json.loads((destination / "metadata.json").read_text()), tmp_path / "figures")
    assert (tmp_path / "figures/late_minus_early.png").exists()
    import experiments.gradient_flow.multi_batch_intervene as shared
    original = shared.intervention
    def forbidden(*args, **kwargs):
        raise AssertionError("Validated state batches must skip probes")
    monkeypatch.setattr(shared, "intervention", forbidden)
    assert run_states(manifest, batches, destination, settings, resume=True) == result
    (destination / "mid/aggregate/summary.json").write_text("corrupted")
    assert run_states(manifest, batches, destination, settings, resume=True) == result
    (destination / "mid/batches/batch_001/measurements.json").write_text("corrupted")
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(shared, "intervention", counted)
    assert run_states(manifest, batches, destination, settings, resume=True) == result
    assert len(calls) == 1
    (destination / "mid/batches/batch_001/complete.json").write_text(json.dumps(dict(status="running")))
    (destination / "status.json").write_text(json.dumps(dict(status="running")))
    assert run_states(manifest, batches, destination, settings, resume=True) == result
    assert len(calls) == 2
