import copy
import hashlib
import json
from pathlib import Path

import pytest

from analysis.gradient_flow.audit_controller_recovery import audit_gate, audit_oscillation, compare_values, rebuild


def completed_fixture():
    return json.loads((Path(__file__).parent / "fixtures/completed_recovery_gate.json").read_text())


def test_completed_lawrence_summary_fails_only_zero_oscillation_checks():
    fixture = completed_fixture()
    summary = fixture["summary"]
    details = audit_gate(summary, fixture["settings"])
    failed = {name: group["failed_checks"] for name, group in details.items() if group["failed_checks"]}
    assert failed == {"early/alternating": ["zero_oscillation_fraction"],
                      "late/alternating": ["zero_oscillation_fraction"]}
    assert summary["gate"]["passed"] is False
    assert summary["gate"]["failures"] == [name+": response-aware recovery/stability thresholds not met" for name in failed]
    for state, ratio, oscillation in (("early", .0643766046160492, .1), ("late", .048900192218208405, .2)):
        row = summary["groups"][state]["alternating"]["response_aware"]
        assert row["fraction_recovered"] == 1.
        assert row["mean_final_ratio"] == pytest.approx(ratio, abs=1e-15)
        assert row["oscillation_fraction"] == oscillation
        assert row["count"] == row["informative"] == 10
        assert row["bound_hit_fraction"] == row["diverged"] == 0


def test_actual_flagged_traces_match_definition_despite_monotonic_total_error():
    fixture = completed_fixture()
    reports = [audit_oscillation(c, fixture["settings"]["controller"]["zero_tolerance"]) for c in fixture["flagged_cases"]]
    assert [(r["state"], r["batch"]) for r in reports] == [("early", 6), ("late", 8), ("late", 9)]
    assert all(r["definition_matches"] and r["stored_flag"] and r["l2_nonincreasing"] for r in reports)
    assert [r["counts"] for r in reports] == [[2, 0, 0, 0], [2, 1, 0, 0], [2, 1, 0, 0]]
    assert [r["crossing_iteration_pairs"][0] for r in reports] == [
        [[0, 1], [6, 7]], [[3, 4], [10, 11]], [[3, 4], [11, 12]]]
    assert all(r["final_ratio"] < .07 for r in reports)


def test_numeric_comparison_tolerance_does_not_hide_gate_changes():
    fixture = completed_fixture()
    changed = copy.deepcopy(fixture["summary"])
    changed["groups"]["late"]["alternating"]["response_aware"]["mean_final_ratio"] += 1e-17
    assert all(d["within_tolerance"] for d in compare_values(fixture["summary"], changed))
    changed["gate"]["passed"] = True
    assert any(not d["within_tolerance"] for d in compare_values(fixture["summary"], changed))


def test_probe_free_rebuild_preserves_source_files(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from experiments.gradient_flow import controller_recovery as engine
    from experiments.gradient_flow.controller import ControllerConfig
    from experiments.gradient_flow.multi_batch_intervene import digest_json
    from test_controller import linear_probe
    settings = completed_fixture()["settings"]
    settings.update(num_batches=1, disturbances={"alternating": [3., 5., 3., 5.]})
    settings["controller"]["max_iterations"] = 2
    config = ControllerConfig(**settings["controller"])
    root = tmp_path / "original"
    root.mkdir()
    metadata = dict(settings=settings, source=dict(commit="SYNTHETIC-REGRESSION-ONLY"),
                    synthetic=True, batch_fingerprints=["synthetic-input"])
    (root/"metadata.json").write_text(json.dumps(metadata))
    (root/"status.json").write_text(json.dumps(dict(status="complete")))
    cases = []
    for state, gain in (("early", -.08), ("mid", .08), ("late", .12)):
        observe = linear_probe([gain]*4)
        case = dict(engine.recover_case(observe, [3., 5., 3., 5.], observe([4.]*4), config),
                    state=state, batch=0, disturbance="alternating", batch_fingerprint="synthetic-input")
        directory = root / state / "batch_000"
        directory.mkdir(parents=True)
        raw = json.dumps([case]).encode()
        (directory/"recovery.json").write_bytes(raw)
        (directory/"complete.json").write_text(json.dumps(dict(status="complete", signature=digest_json(metadata),
            sha256=hashlib.sha256(raw).hexdigest())))
        cases.append(case)
    (root/"summary.json").write_text(json.dumps(engine.aggregate(cases, settings, synthetic=True)))
    before = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*.json")}
    def forbidden(*args, **kwargs):
        raise AssertionError("Saved-output aggregation must never run models, backward or GPU probes")
    monkeypatch.setattr(engine.FrozenProbe, "__call__", forbidden)
    monkeypatch.setattr(engine, "build_model", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    report = rebuild(root, tmp_path / "rebuilt")
    assert report["cases"] == 3 and report["validated_batches"] == 3
    assert report["gpu_probes_run"] == 0 and report["gate_unchanged"]
    assert json.loads((tmp_path/"rebuilt/summary.json").read_text()) == json.loads((root/"summary.json").read_text())
    assert {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*.json")} == before
    with pytest.raises(ValueError, match="separate"):
        rebuild(root, root)
