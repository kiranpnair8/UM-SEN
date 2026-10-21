"""No-step online calibration, delegating neuron/weight/RNG restoration to FrozenProbe."""

import copy

from .controller_recovery import FrozenProbe
from .intervene import same_state


class OnlineProbe:
    def __init__(self, model, batch, config, optimizer, scheduler=None, scaler=None):
        self.probe = FrozenProbe(model, batch, config, optimizer)
        self.scheduler, self.scaler = scheduler, scaler

    def __call__(self, alpha, estimate=True):
        owners = [self.probe.optimizer, self.scheduler, self.scaler]
        snapshots = [copy.deepcopy(owner.state_dict()) if owner is not None else None for owner in owners]
        try:
            # FrozenProbe already checks exact weights/BN/gradients/neuron memories,
            # per-module modes, RNG, alpha and optimizer state on every trial.
            result = self.probe(alpha, estimate=estimate)
        finally:
            for name, owner, before in zip(("optimizer", "scheduler", "AMP scaler"), owners, snapshots):
                if owner is not None and not same_state(before, owner.state_dict()):
                    raise AssertionError(name+" changed during no-step online probe")
        return result
