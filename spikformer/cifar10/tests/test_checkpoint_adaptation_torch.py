"""Real CPU tensor/checkpoint checks; skipped when PyTorch is unavailable."""

from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiment_state


def test_real_weights_only_and_fresh_training_state(tmp_path):
    source = torch.nn.Linear(3, 2)
    old_optimizer = torch.optim.AdamW(source.parameters(), lr=0.1)
    source(torch.ones(1, 3)).sum().backward()
    old_optimizer.step()
    path = tmp_path / "source.pth.tar"
    torch.save({"state_dict": source.state_dict(), "optimizer": old_optimizer.state_dict(),
                "epoch": 262, "metric": 94.81, "version": 2}, path)
    model = torch.nn.Linear(3, 2)
    result = experiment_state.initialize_model(model, experiment_state.load_trusted_checkpoint(path))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    assert not optimizer.state and scheduler.last_epoch == 0
    assert result["start_epoch"] == 0 and result["best_metric"] is None
    assert all(torch.equal(a, b) for a, b in zip(source.parameters(), model.parameters()))


def test_real_timm_resume(tmp_path):
    timm_models = pytest.importorskip("timm.models")
    cosine = pytest.importorskip("timm.scheduler.cosine_lr")
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = cosine.CosineLRScheduler(optimizer, t_initial=30, warmup_t=2, warmup_lr_init=1e-5)
    model(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    scheduler.step(8)
    saved_lr = optimizer.param_groups[0]["lr"]
    path = tmp_path / "resume.pth.tar"
    torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": 7, "version": 2,
                "sage_training_state": {"scheduler": scheduler.state_dict()}}, path)
    restored = torch.nn.Linear(3, 2)
    resumed_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01)
    # Match the trainer: scheduler construction before timm optimizer restore.
    resumed_scheduler = cosine.CosineLRScheduler(resumed_optimizer, t_initial=30, warmup_t=2, warmup_lr_init=1e-5)
    assert timm_models.resume_checkpoint(restored, str(path), optimizer=resumed_optimizer) == 8
    assert resumed_optimizer.state and resumed_optimizer.param_groups[0]["lr"] == saved_lr
    experiment_state.restore_extra_state(experiment_state.load_trusted_checkpoint(path), resumed_scheduler)
    assert resumed_optimizer.param_groups[0]["lr"] == saved_lr
    scheduler.step(9)
    resumed_scheduler.step(9)
    assert resumed_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), restored.parameters()))
