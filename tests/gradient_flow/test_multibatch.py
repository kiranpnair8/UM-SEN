import copy
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("spikingjelly.clock_driven.functional")

from experiments.gradient_flow.common import ROOT, build_model, load_config
from experiments.gradient_flow.intervene import intervention
from experiments.gradient_flow.multi_batch_intervene import digest_file, run_batches
from experiments.gradient_flow.multi_batch_stats import aggregate_measurements, signs, structural_metrics, validate_measurements


def fake_measurements(matrix):
    """Analytical unit-test fixture, not an experimental result."""
    n = len(matrix)
    def trial(j=None, alpha=None):
        blocks = []
        for i in range(n):
            g = 1. + i if j is None else 1. + i + matrix[i][j] * (alpha-4)
            blocks.append(dict(block_id=i, G_l=float(g), grad_input_norm=1., grad_output_norm=1.,
                               parameter_grad_norm=1., firing_rate=.1, valid=True))
        return dict(target_block=j, alpha=alpha, loss=1., blocks=blocks,
                    block_alphas=[alpha if i == j else 4. for i in range(n)],
                    sps_alphas={"patch_embed.test_lif": 4.})
    return dict(block_ids=list(range(n)), alpha_values=[2, 3, 4, 5, 6], reference_alpha=4.,
                baseline=trial(), trials=[trial(j, a) for j in range(n) for a in (2, 3, 4, 5, 6)],
                mode="eval", amp=False, optimizer_updates=0,
                checks={key: True for key in ("reference_reproducible", "alpha_isolation",
                    "weights_exactly_unchanged", "restored_after_each_probe", "gradients_cleared_before_each_probe")})


def test_exact_aggregate_statistics_and_structure():
    a = np.array([[1., 2., 4., 8.], [0., 3., 5., 7.], [0., 0., 6., 9.], [0., 0., 0., 10.]])
    result = aggregate_measurements([fake_measurements(k*a) for k in (1, 2, 3)], bootstrap_samples=1000)
    stats = result["response_stats"]
    np.testing.assert_equal(stats["mean"], 2*a)
    np.testing.assert_equal(stats["std"], a)
    np.testing.assert_equal(stats["median"], 2*a)
    np.testing.assert_equal(stats["minimum"], a)
    np.testing.assert_equal(stats["maximum"], 3*a)
    assert np.asarray(stats["ci95"]).shape == (2, 4, 4)
    assert np.asarray(result["alpha_curves"]["mean"]).shape == (4, 4, 5)
    np.testing.assert_equal(result["response_sign_consistency"]["fraction_positive"], a > 0)
    per = result["structural_metrics"]["per_batch"][0]
    assert per["mean_abs_diagonal"] == 5
    assert per["mean_abs_off_diagonal"] == pytest.approx(35/12)
    assert per["ratio_all_off_diagonal"] == pytest.approx(12/7)
    assert per["ratio_nonzero_off_diagonal"] == pytest.approx(6/7)
    assert per["mean_abs_nearest_upstream"] == pytest.approx(16/3)
    assert per["mean_abs_distant_upstream"] == pytest.approx(19/3)
    assert per["mean_abs_downstream"] == 0
    assert result["structural_metrics"]["fraction_every_diagonal_positive"] == 1
    assert result["structural_metrics"]["per_block_fraction_diagonal_positive"] == [1.] * 4
    assert result["structural_metrics"]["fraction_nearest_larger_than_distant"] == 0
    assert result["alpha_curves"]["local_monotonicity"]["fraction_nonmonotonic"] == [[0.] * 4] * 4
    assert result == aggregate_measurements([fake_measurements(k*a) for k in (1, 2, 3)], bootstrap_samples=1000)
    json.dumps(result, allow_nan=False)


def test_tolerance_and_undefined_ratios(tmp_path):
    tol = 1e-8
    s = signs([-2*tol, -tol, 0, tol/2, tol, 2*tol], tol)
    assert s == dict(fraction_positive=1/6, fraction_negative=1/6, fraction_effectively_zero=4/6)
    m = structural_metrics(np.zeros((4, 4)), tol)
    assert m["ratio_all_off_diagonal"] is None and m["ratio_nonzero_off_diagonal"] is None
    result = aggregate_measurements([fake_measurements(np.zeros((4, 4)))] * 3, bootstrap_samples=100)
    assert result["structural_metrics"]["aggregate"]["ratio_nonzero_off_diagonal"]["count"] == 0
    assert result["structural_metrics"]["aggregate"]["ratio_all_off_diagonal"]["undefined_batches"] == 3
    from analysis.gradient_flow.plot_alpha_multibatch import plot
    plot(result, tmp_path)
    assert len(list(tmp_path.glob("*.png"))) == 10
    assert (tmp_path / "summary.txt").exists()
    single = aggregate_measurements([fake_measurements(np.zeros((4, 4)))], bootstrap_samples=100)
    assert single["response_stats"]["std"] is None and single["response_stats"]["ci95"] is None


def test_bad_observations_are_rejected():
    for modify in (lambda d: d["checks"].update(weights_exactly_unchanged=False),
                   lambda d: d["trials"].pop(),
                   lambda d: d["trials"][0]["blocks"][0].update(valid=False),
                   lambda d: d["trials"][2].update(loss=2.)):
        d = fake_measurements(np.eye(4))
        modify(d)
        with pytest.raises(ValueError):
            validate_measurements(d, 4, [2, 3, 4, 5, 6], 4)


def test_single_dataset_traversal_and_nonoverlapping_indices(monkeypatch):
    import experiments.gradient_flow.make_probe_batch as helper
    from PIL import Image
    calls = []
    class FakeCIFAR:
        def __init__(self, root, train, download, transform):
            calls.append((train, download))
            self.transform = transform
        def __len__(self):
            return 40
        def __getitem__(self, index):
            return self.transform(Image.new("RGB", (32, 32), (index, 100, 200))), index % 10
    monkeypatch.setattr(helper, "CIFAR10", FakeCIFAR)
    config = load_config(ROOT / "configs/gradient_flow/cifar10_diagnostics.yml")
    batches = list(helper.iter_batches("unused", config, 4, 3))
    assert calls == [(False, False)]
    assert [b["batch_index"] for b in batches] == [0, 1, 2]
    assert [b["indices"] for b in batches] == [list(range(0, 4)), list(range(4, 8)), list(range(8, 12))]
    again = list(helper.iter_batches("unused", config, 4, 3))
    assert all(torch.equal(a["images"], b["images"]) for a, b in zip(batches, again))
    assert not torch.equal(batches[0]["images"], batches[1]["images"])
    with pytest.raises(ValueError, match="exceed"):
        list(helper.iter_batches("unused", config, 32, 25))


def tiny_run():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    config = dict(img_size=8, patch_size=4, dim=16, num_heads=4, mlp_ratio=2, layer=4, time_step=4, num_classes=10)
    model = build_model(config, "torch").eval()
    generator = torch.Generator().manual_seed(42)
    batches = [dict(images=torch.randn(2, 3, 8, 8, generator=generator), targets=torch.tensor([i, i+1]),
                    batch_index=i, indices=[2*i, 2*i+1], synthetic=True) for i in range(3)]
    metadata = dict(protocol="SYNTHETIC CORRECTNESS ONLY", config=config, checkpoint=dict(path="synthetic"),
                    num_batches=3, batch_size=2, seed=42, alpha_values=[2, 3, 4, 5, 6], reference_alpha=4.,
                    block_count=4, zero_tolerance=1e-8, bootstrap_samples=100, backend="torch", device="cpu")
    return model, batches, metadata


def test_three_batch_smoke_resume_corruption_and_identity(tmp_path):
    model, batches, metadata = tiny_run()
    destination = tmp_path / "run"
    result = run_batches(model, batches, destination, metadata)
    assert result["batch_count"] == 3 and np.asarray(result["matrices"]).shape == (3, 4, 4)
    raw = (destination / "aggregate/raw_measurements.jsonl").read_text().splitlines()
    assert len(raw) == 240 and {json.loads(row)["batch_index"] for row in raw} == {0, 1, 2}
    original = digest_file(destination / "batches/batch_000/complete.json")
    def forbidden(*args, **kwargs):
        raise AssertionError("Validated batches must not rerun")
    assert run_batches(model, batches, destination, metadata, resume=True, probe=forbidden) == result
    (destination / "batches/batch_001/measurements.json").write_text("corrupted", encoding="utf-8")
    reruns = []
    def counted(*args, **kwargs):
        reruns.append(1)
        return intervention(*args, **kwargs)
    assert run_batches(model, batches, destination, metadata, resume=True, probe=counted) == result
    assert len(reruns) == 1 and digest_file(destination / "batches/batch_000/complete.json") == original
    marker_path = destination / "batches/batch_001/complete.json"
    marker = json.loads(marker_path.read_text())
    del marker["batch_fingerprint"]
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    assert run_batches(model, batches, destination, metadata, resume=True, probe=counted) == result
    assert len(reruns) == 2
    changed = copy.deepcopy(batches)
    changed[0]["images"] += .1
    with pytest.raises(ValueError, match="different inputs"):
        run_batches(model, changed, destination, metadata, resume=True, probe=forbidden)
    with pytest.raises(ValueError, match="identical metadata"):
        run_batches(model, batches, destination, dict(metadata, seed=43), resume=True)


def test_failures_propagate_and_completed_batches_survive(tmp_path):
    model, batches, metadata = tiny_run()
    calls = []
    def fail_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise AssertionError("injected reference/state failure")
        return intervention(*args, **kwargs)
    with pytest.raises(AssertionError, match="injected"):
        run_batches(model, batches, tmp_path / "failed", metadata, probe=fail_second)
    assert json.loads((tmp_path / "failed/status.json").read_text())["status"] == "failed"
    assert json.loads((tmp_path / "failed/batches/batch_000/complete.json").read_text())["status"] == "complete"
    assert (tmp_path / "failed/batches/batch_001/failure.json").exists()
    completed = digest_file(tmp_path / "failed/batches/batch_000/complete.json")
    reruns = []
    def retry(*args, **kwargs):
        reruns.append(1)
        return intervention(*args, **kwargs)
    result = run_batches(model, batches, tmp_path / "failed", metadata, resume=True, probe=retry)
    assert result["batch_count"] == 3 and len(reruns) == 2
    assert digest_file(tmp_path / "failed/batches/batch_000/complete.json") == completed
    batches[0]["indices"] = [0, 0]
    with pytest.raises(ValueError, match="non-overlapping"):
        run_batches(model, batches, tmp_path / "duplicate", metadata)
