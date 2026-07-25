import pathlib
import sys
import types

import pytest

torch = pytest.importorskip("torch")
spikingjelly_surrogate = pytest.importorskip("spikingjelly.clock_driven.surrogate")
spikingjelly_functional = pytest.importorskip("spikingjelly.clock_driven.functional")
pytest.importorskip("timm")


CIFAR10_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(CIFAR10_DIR) not in sys.path:
    sys.path.insert(0, str(CIFAR10_DIR))

from model import Spikformer
from uncertainty_surrogate import make_surrogate_function


def _dense_threshold_input():
    near_threshold = torch.linspace(-1.0, 1.0, steps=2001, dtype=torch.float64)
    tight_threshold = torch.linspace(-0.05, 0.05, steps=2001, dtype=torch.float64)
    exact_threshold = torch.zeros(1, dtype=torch.float64)
    return torch.unique(torch.cat([near_threshold, tight_threshold, exact_threshold])).requires_grad_(True)


def test_make_surrogate_function_matches_spikingjelly_sigmoid_forward_and_backward():
    baseline_input = _dense_threshold_input()
    custom_input = baseline_input.detach().clone().requires_grad_(True)

    baseline = spikingjelly_surrogate.Sigmoid()
    custom = make_surrogate_function()

    baseline_output = baseline(baseline_input)
    custom_output = custom(custom_input)

    torch.testing.assert_close(custom_output, baseline_output, rtol=0.0, atol=0.0)

    grad_weight = torch.linspace(0.25, 1.25, steps=baseline_output.numel(), dtype=torch.float64)
    baseline_output.backward(grad_weight)
    custom_output.backward(grad_weight)

    torch.testing.assert_close(custom_input.grad, baseline_input.grad, rtol=0.0, atol=0.0)


def _load_original_model_module():
    source = (CIFAR10_DIR / "model.py").read_text()
    source = source.replace(
        "from timm.models.registry import register_model",
        "def register_model(fn):\n    return fn",
    )
    source = source.replace(
        "try:\n"
        "    from .uncertainty_surrogate import make_surrogate_function\n"
        "except ImportError:\n"
        "    from uncertainty_surrogate import make_surrogate_function\n",
        "",
    )
    source = source.replace(",\n                                        surrogate_function=make_surrogate_function()", "")
    source = source.replace(",\n                                      surrogate_function=make_surrogate_function()", "")
    source = source.replace(",\n                                         surrogate_function=make_surrogate_function()", "")
    source = source.replace(",\n                                          surrogate_function=make_surrogate_function()", "")
    source = source.replace(",\n                                        surrogate_function=make_surrogate_function()", "")

    module = types.ModuleType("original_cifar10_model")
    module.__file__ = str(CIFAR10_DIR / "model.py")
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def _portable_neuron_backend(model):
    for module in model.modules():
        if hasattr(module, "backend"):
            module.backend = "torch"


def _tiny_model(model_cls):
    return model_cls(
        img_size_h=8,
        img_size_w=8,
        patch_size=4,
        in_channels=3,
        num_classes=5,
        embed_dims=16,
        num_heads=4,
        mlp_ratios=2,
        depths=1,
        sr_ratios=1,
        T=2,
    )


def test_refactored_spikformer_matches_original_logits_loss_and_parameter_gradients():
    torch.manual_seed(1234)
    original_module = _load_original_model_module()
    original = _tiny_model(original_module.Spikformer)
    modified = _tiny_model(Spikformer)
    modified.load_state_dict(original.state_dict())

    _portable_neuron_backend(original)
    _portable_neuron_backend(modified)
    original.eval()
    modified.eval()

    input_batch = torch.randn(2, 3, 8, 8)
    target = torch.tensor([1, 3], dtype=torch.long)
    criterion = torch.nn.CrossEntropyLoss()

    original_logits = original(input_batch)
    modified_logits = modified(input_batch.clone())
    torch.testing.assert_close(modified_logits, original_logits, rtol=0.0, atol=0.0)

    original_loss = criterion(original_logits, target)
    modified_loss = criterion(modified_logits, target)
    torch.testing.assert_close(modified_loss, original_loss, rtol=0.0, atol=0.0)

    original.zero_grad(set_to_none=True)
    modified.zero_grad(set_to_none=True)
    original_loss.backward()
    modified_loss.backward()

    for (original_name, original_param), (modified_name, modified_param) in zip(
        original.named_parameters(), modified.named_parameters()
    ):
        assert modified_name == original_name
        if original_param.grad is None:
            assert modified_param.grad is None
            continue
        torch.testing.assert_close(modified_param.grad, original_param.grad, rtol=0.0, atol=0.0)

    spikingjelly_functional.reset_net(original)
    spikingjelly_functional.reset_net(modified)
