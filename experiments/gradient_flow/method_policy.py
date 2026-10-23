"""Selective/restoring variants; inherited reference, estimator and diagnostics."""

import math

from .controller import apply_steps
from .online_policy import OnlinePolicy
from .online_runtime import OnlineRuntime


VARIANTS = {
    "deep_only": dict(eta=.10, max_step=.10, gamma=0., freeze_block0=True, use_deadband=False),
    "regularized": dict(eta=.10, max_step=.10, gamma=.05, freeze_block0=False, use_deadband=False),
    "deadband_regularized": dict(eta=.10, max_step=.10, gamma=.05, freeze_block0=False, use_deadband=True),
}


class MethodPolicy(OnlinePolicy):
    def __init__(self, blocks, method, settings, seed=42):
        super().__init__(blocks, method, settings, seed)
        design = settings["method_design"]
        self.variant = design["variant"]
        if self.variant not in VARIANTS or design["parameters"] != VARIANTS[self.variant]:
            raise ValueError("Unknown or modified method design")
        self.design = self.settings["method_design"]
        if (self.config.eta != .10 or self.config.max_step != .10 or
                self.config.reference_mode != "frozen_window" or settings["reference_window_epochs"] != 5 or
                self.config.alpha_min != 1 or self.config.alpha_max != 8 or
                not math.isfinite(design["deadband"]) or design["deadband"] <= 0):
            raise ValueError("Method-design protocol mismatch")

    def response_update(self, alpha, observed, reference, response):
        base = super().response_update(alpha, observed, reference, response)
        p = self.design["parameters"]
        # Use the UNCLIPPED inherited inverse step; add restoring force and mask
        # the entire update BEFORE the sole max-step clamp and alpha-bound clamp.
        restoring = [-p["gamma"]*(a-4.) for a in alpha]
        active = [(not p["freeze_block0"] or b != 0) and
            (not p["use_deadband"] or abs(e) >= self.design["deadband"])
            for b, e in enumerate(base["errors"])]
        requested = [r+s if enabled else 0. for r, s, enabled in
            zip(base["requested_step"], restoring, active)]
        return apply_steps(alpha, requested, self.config, errors=base["errors"], response=response,
            response_aware_raw_step=base["requested_step"], restoring_step=restoring,
            update_active=active, deadband=self.design["deadband"], gamma=p["gamma"])

    def advance(self, epoch, observation, matched=None):
        if self.design["parameters"]["freeze_block0"] and self.alpha[0] != 4.:
            raise ValueError("DEEP_ONLY block0 alpha must remain exactly 4")
        row = super().advance(epoch, observation, matched)
        if self.design["parameters"]["freeze_block0"] and row["alpha"][0] != 4.:
            raise AssertionError("DEEP_ONLY changed block0")
        return row

    def load_state_dict(self, state):
        super().load_state_dict(state)
        if self.design["parameters"]["freeze_block0"] and (self.alpha[0] != 4. or
                any(r["alpha"][0] != 4. or r["alpha_before"][0] != 4. for r in self.history)):
            raise ValueError("DEEP_ONLY restored a non-fixed block0")


class MethodRuntime(OnlineRuntime):
    policy_class = MethodPolicy
