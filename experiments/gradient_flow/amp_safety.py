"""Native AMP semantics plus predeclared, checkpointed persistence guards."""

from collections import deque

import torch


DEFAULT_SAFETY = dict(max_consecutive_overflows=8, overflow_window=100,
    max_window_overflow_fraction=.20, max_consecutive_invalid_diagnostics=5)


class AMPSafety:
    def __init__(self, settings=None):
        self.settings = {**DEFAULT_SAFETY, **(settings or {})}
        s = self.settings
        if (set(s) != set(DEFAULT_SAFETY) or any(type(s[k]) is not int or s[k] < 1 for k in
                ("max_consecutive_overflows", "overflow_window", "max_consecutive_invalid_diagnostics")) or
                not 0 < s["max_window_overflow_fraction"] < 1):
            raise ValueError("Invalid AMP persistence settings")
        self.recent = deque(maxlen=s["overflow_window"])
        self.consecutive_overflows = self.consecutive_invalid_diagnostics = 0
        self.overflows = self.invalid_diagnostics = self.optimizer_updates = 0

    def observe(self, overflow, invalid_diagnostics, updated):
        self.recent.append(bool(overflow))
        self.consecutive_overflows = self.consecutive_overflows+1 if overflow else 0
        self.consecutive_invalid_diagnostics = self.consecutive_invalid_diagnostics+1 if invalid_diagnostics and not overflow else 0
        self.overflows += int(overflow)
        self.invalid_diagnostics += int(invalid_diagnostics)
        self.optimizer_updates += int(updated)
        s = self.settings
        if self.consecutive_overflows >= s["max_consecutive_overflows"]:
            return "Persistent AMP overflow: consecutive threshold reached"
        if len(self.recent) == s["overflow_window"] and sum(self.recent)/len(self.recent) > s["max_window_overflow_fraction"]:
            return "Persistent AMP overflow: rolling-window threshold exceeded"
        if self.consecutive_invalid_diagnostics >= s["max_consecutive_invalid_diagnostics"]:
            return "Persistent invalid diagnostics with finite parameter gradients (not a GradScaler overflow)"
        return None

    def state_dict(self):
        return dict(settings=self.settings, recent=list(self.recent),
            **{k: getattr(self, k) for k in ("consecutive_overflows", "consecutive_invalid_diagnostics",
                                           "overflows", "invalid_diagnostics", "optimizer_updates")})

    def load_state_dict(self, state, successful_legacy_steps=0):
        self.recent.clear()
        if state is None:
            # The old wrapper could publish ONLY all-finite/all-valid batches.
            self.recent.extend([False]*min(successful_legacy_steps, self.settings["overflow_window"]))
            self.optimizer_updates = successful_legacy_steps
            self.consecutive_overflows = self.consecutive_invalid_diagnostics = self.overflows = self.invalid_diagnostics = 0
            return
        if state["settings"] != self.settings:
            raise ValueError("AMP safety resume policy mismatch")
        self.recent.extend(state["recent"])
        for k in ("consecutive_overflows", "consecutive_invalid_diagnostics", "overflows", "invalid_diagnostics", "optimizer_updates"):
            setattr(self, k, state[k])


def nonfinite_parameters(model):
    failures = []
    for name, p in model.named_parameters():
        if p.grad is None or bool(torch.isfinite(p.grad).all()):
            continue
        g = p.grad.detach()
        failures.append(dict(parameter=name, block_id=int(name.split(".")[1]) if name.startswith("block.") else None,
            dtype=str(g.dtype), nan=int(torch.isnan(g).sum()), inf=int(torch.isinf(g).sum())))
    return failures


def invalid_blocks(records):
    return [{k: r[k] for k in ("block_id", "valid", "grad_input_norm", "grad_output_norm", "parameter_grad_norm", "invalid_reason")}
            for r in records if not r["valid"]]


def checked_optimizer_step(optimizer, scaler, expect_skip):
    """Observe actual step invocation; optimizer.step() returning None is NOT a skip flag."""
    if getattr(optimizer, "_step_supports_amp_scaling", False):
        raise ValueError("Custom/fused AMP optimizer step protocol is not validated here")
    calls = []
    handle = optimizer.register_step_post_hook(lambda *unused: calls.append(True))
    try:
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
    finally:
        handle.remove()
    if bool(calls) == bool(expect_skip):
        raise AssertionError("Native AMP optimizer skip/update did not match observed parameter finiteness")
    return bool(calls)
