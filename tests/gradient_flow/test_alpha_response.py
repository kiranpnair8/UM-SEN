import copy
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("spikingjelly.clock_driven.functional")

from experiments.gradient_flow.alpha_scope import BlockAlphas
from experiments.gradient_flow.common import build_model
from experiments.gradient_flow.intervene import intervention, local_response_matrix, save_flat_records


@pytest.fixture
def four_blocks():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    return build_model(dict(img_size=8, patch_size=4, dim=16, num_heads=4,
        mlp_ratio=2, layer=4, time_step=2, num_classes=10), "torch").eval()


def fixed_batch():
    return torch.randn(2, 3, 8, 8, generator=torch.Generator().manual_seed(42)), torch.tensor([1, 3])


def test_four_block_sweep_and_flat_output(four_blocks, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No optimizer may be constructed in FP32 probes")
    monkeypatch.setattr(torch.optim, "SGD", forbidden)
    weights = copy.deepcopy(four_blocks.state_dict())
    result = intervention(four_blocks, *fixed_batch(), range(4), [2, 3, 4, 5, 6], mode="eval")
    assert all(result["checks"].values())
    assert len(result["trials"]) == 20
    for trial in result["trials"]:
        assert len(trial["blocks"]) == 4
        assert trial["sps_alphas"] and set(trial["sps_alphas"].values()) == {4.}
        assert trial["block_alphas"] == [trial["alpha"] if i == trial["target_block"] else 4. for i in range(4)]
        if trial["alpha"] == 4:
            assert trial["loss"] == result["baseline"]["loss"]
            assert [b["G_l"] for b in trial["blocks"]] == [b["G_l"] for b in result["baseline"]["blocks"]]
    assert BlockAlphas(four_blocks).get() == [4.] * 4
    assert all(torch.equal(value, weights[key]) for key, value in four_blocks.state_dict().items())
    assert all(p.grad is None for p in four_blocks.parameters())
    assert all(node.v == 0 for node in four_blocks.modules() if hasattr(node, "surrogate_function"))
    matrix = local_response_matrix(result)
    assert len(matrix["matrix"]) == 4 and all(len(row) == 4 for row in matrix["matrix"])
    assert matrix["lower_alpha"] == 3 and matrix["upper_alpha"] == 5
    for i in range(4):
        for j in range(4):
            lo, hi = [next(t for t in result["trials"] if t["target_block"] == j and t["alpha"] == a)
                      for a in (3, 5)]
            assert matrix["matrix"][i][j] == (hi["blocks"][i]["G_l"] - lo["blocks"][i]["G_l"]) / 2
    save_flat_records(tmp_path, result, dict(checkpoint="synthetic", batch_file="fixed.pt", mode="eval",
                                           backend="torch", device="cpu"))
    rows = [json.loads(line) for line in (tmp_path / "raw_measurements.jsonl").read_text().splitlines()]
    assert len(rows) == 80
    assert len({(r["perturbed_block"], r["alpha"], r["measured_block"]) for r in rows}) == 80
    assert len((tmp_path / "raw_measurements.csv").read_text().splitlines()) == 81
    from analysis.gradient_flow.plot_alpha_response import plot
    plot(result, tmp_path)
    assert (tmp_path / "local_response_matrix.png").exists()
    assert "Diagonal responses" in (tmp_path / "summary.txt").read_text()


def test_restore_between_trials_and_exception(four_blocks):
    scope = BlockAlphas(four_blocks)
    scope.set(2, 3.)
    node = next(iter(scope.nodes[0].values()))
    node.v = torch.tensor(.2)
    four_blocks.block[1].train()
    modes = [m.training for m in four_blocks.modules()]
    for p in four_blocks.parameters():
        p.grad = torch.ones_like(p)
    seen = []
    def observe(module, inputs):
        seen.append(scope.get())
        assert all(p.grad is None for p in module.parameters())
        assert all(n.v == 0 for n in module.modules() if hasattr(n, "surrogate_function"))
        assert not any(m.training for m in module.modules())
        if len(seen) == 4:
            raise RuntimeError("injected probe failure")
    handle = four_blocks.register_forward_pre_hook(observe)
    try:
        with pytest.raises(RuntimeError, match="injected"):
            intervention(four_blocks, *fixed_batch(), [0], [2, 6], mode="eval")
    finally:
        handle.remove()
    assert seen == [[4.] * 4, [4.] * 4, [2., 4., 4., 4.], [6., 4., 4., 4.]]
    assert scope.get() == [4., 4., 3., 4.]
    assert float(node.v) == pytest.approx(.2)
    assert [m.training for m in four_blocks.modules()] == modes
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in four_blocks.parameters())
    with pytest.raises(ValueError, match="block_id"):
        intervention(four_blocks, *fixed_batch(), [4], [4], mode="eval")


def test_local_pair_requires_symmetry():
    with pytest.raises(ValueError, match="symmetric"):
        local_response_matrix(dict(reference_alpha=4, alpha_values=[2, 4, 5]))


def test_probe_batch_reuses_eval_transform_without_download(monkeypatch):
    from experiments.gradient_flow.common import ROOT, load_config
    from experiments.gradient_flow import make_probe_batch as helper
    config = load_config(ROOT / "configs/gradient_flow/cifar10_diagnostics.yml")
    calls = []
    class FakeCIFAR:
        def __init__(self, root, train, download, transform):
            calls.append((train, download))
            self.transform = transform
        def __len__(self):
            return 40
        def __getitem__(self, index):
            from PIL import Image
            return self.transform(Image.new("RGB", (32, 32), (index, 100, 200))), index % 10
    monkeypatch.setattr(helper, "CIFAR10", FakeCIFAR)
    a = helper.make_batch("unused", config)
    b = helper.make_batch("unused", config)
    assert calls == [(False, False)] * 2
    assert a["images"].shape == (32, 3, 32, 32)
    assert a["images"].dtype == torch.float32 and a["targets"].dtype == torch.long
    assert torch.equal(a["images"], b["images"]) and torch.equal(a["targets"], b["targets"])
    assert a["indices"] == list(range(32))
