import json
import pathlib
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
spikingjelly_functional = pytest.importorskip("spikingjelly.clock_driven.functional")
spikingjelly_neuron = pytest.importorskip("spikingjelly.clock_driven.neuron")
pytest.importorskip("timm")


CIFAR10_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(CIFAR10_DIR) not in sys.path:
    sys.path.insert(0, str(CIFAR10_DIR))

import attention_entropy_diagnostic as diagnostic
import model as refactored_model


def _torch_backend_lif_node(*args, **kwargs):
    kwargs["backend"] = "torch"
    return spikingjelly_neuron.MultiStepLIFNode(*args, **kwargs)


def _tiny_refactored_model(depths=2):
    original_lif_node = refactored_model.MultiStepLIFNode
    try:
        refactored_model.MultiStepLIFNode = _torch_backend_lif_node
        return refactored_model.Spikformer(
            img_size_h=8,
            img_size_w=8,
            patch_size=4,
            in_channels=3,
            num_classes=5,
            embed_dims=16,
            num_heads=4,
            mlp_ratios=2,
            depths=depths,
            sr_ratios=1,
            T=2,
        )
    finally:
        refactored_model.MultiStepLIFNode = original_lif_node


def _assert_finite_summary(summary):
    assert set(summary) == {"mean", "std", "min", "max", "num_records"}
    assert summary["num_records"] > 0
    for key in ("mean", "std", "min", "max"):
        assert torch.isfinite(torch.tensor(summary[key]))


def _assert_finite_topk_summary(summary):
    assert set(summary) == {"mean", "std", "min", "max", "num_records", "top_k"}
    assert summary["top_k"] == 3
    _assert_finite_summary({k: summary[k] for k in ("mean", "std", "min", "max", "num_records")})


def test_attention_entropy_diagnostic_stats_and_json_schema(tmp_path):
    torch.manual_seed(2468)
    model = _tiny_refactored_model(depths=2)
    model.eval()
    diagnostic.install_attention_metric_collectors(model, top_k=3)
    model.set_attention_entropy_collection(True)

    logits = model(torch.randn(2, 3, 8, 8))
    assert logits.shape == (2, 5)
    spikingjelly_functional.reset_net(model)

    attention_stats = diagnostic.collect_attention_epoch_stats(model)
    assert [block["block"] for block in attention_stats] == [0, 1]
    for block in attention_stats:
        assert set(block) == {
            "block",
            "raw_attention_scores",
            "temperature_entropy",
            "per_head_entropy",
            "attention_sparsity",
            "gini_impurity",
        }
        _assert_finite_summary(block["raw_attention_scores"])
        for temperature in ("0.25", "0.5", "1.0", "2.0"):
            entropy = block["temperature_entropy"][temperature]
            _assert_finite_summary(entropy)
            assert 0.0 <= entropy["min"] <= entropy["max"] <= 1.0 + 1e-6
            per_head = block["per_head_entropy"][temperature]
            assert [entry["head"] for entry in per_head] == [0, 1, 2, 3]
            for entry in per_head:
                _assert_finite_summary({k: entry[k] for k in ("mean", "std", "min", "max", "num_records")})
                assert 0.0 <= entry["min"] <= entry["max"] <= 1.0 + 1e-6
            _assert_finite_topk_summary(block["attention_sparsity"][temperature])
            assert 0.0 <= block["attention_sparsity"][temperature]["min"]
            assert block["attention_sparsity"][temperature]["max"] <= 1.0 + 1e-6
            _assert_finite_summary(block["gini_impurity"][temperature])

    args = SimpleNamespace(
        epochs=3,
        seed=2468,
        output_dir=str(tmp_path),
        entropy_log_interval=1,
        attention_top_k=3,
        data_dir="./data",
        batch_size=8,
        val_batch_size=8,
        workers=0,
        lr=5e-4,
        weight_decay=6e-2,
        time_step=2,
        layer=2,
        dim=16,
        num_heads=4,
        patch_size=4,
        mlp_ratio=2,
    )
    config = diagnostic.resolved_config(args, torch.device("cuda"))
    epoch_record = {
        "epoch": 0,
        "train": {"loss": 1.0, "accuracy": 0.25},
        "validation": {"loss": 1.5, "accuracy": 0.2},
        "attention": attention_stats,
    }
    output_path = tmp_path / "metrics.json"
    diagnostic.write_json(output_path, {"config": config, "epochs": [epoch_record]})

    payload = json.loads(output_path.read_text())
    assert set(payload) == {"config", "epochs"}
    assert payload["config"]["epochs"] == 3
    assert payload["config"]["surrogate_alpha"] == 4
    assert payload["config"]["backend"] == "cupy"
    assert payload["config"]["attention_top_k"] == 3
    assert payload["config"]["attention_temperatures"] == [0.25, 0.5, 1.0, 2.0]
    assert len(payload["epochs"]) == 1
    assert set(payload["epochs"][0]) == {"epoch", "train", "validation", "attention"}
    assert set(payload["epochs"][0]["train"]) == {"loss", "accuracy"}
    assert set(payload["epochs"][0]["validation"]) == {"loss", "accuracy"}
    assert len(payload["epochs"][0]["attention"]) == 2
