"""Provisional response-aware disturbance recovery, NOT a finalized SAGE method."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class ControllerConfig:
    enabled: bool = True
    interval: int = 200  # Reserved for a future, separately validated online stage.
    delta: float = .5
    eta: float = .25
    regularizer: float = 1e-4
    alpha_min: float = 1.
    alpha_max: float = 8.
    max_step: float = .25
    reference_mode: str = "frozen_window"
    reference_ema: float = .99
    max_iterations: int = 12
    recovery_tolerance: float = .5
    max_abs_G: float = 50.
    max_gradient_norm: float = 1e6
    divergence_ratio: float = 10.
    zero_tolerance: float = 1e-10

    def __post_init__(self):
        numeric = [v for v in asdict(self).values() if isinstance(v, (int, float))]
        if not all(math.isfinite(v) for v in numeric):
            raise ValueError("Controller settings must be finite")
        if (not 0 < self.alpha_min < self.alpha_max or self.delta <= 0 or self.eta <= 0 or
                self.regularizer <= 0 or not 0 < self.max_step <= (self.alpha_max-self.alpha_min)/2 or
                self.max_iterations < 1 or self.interval < 1 or
                not 0 <= self.reference_ema < 1 or not 0 < self.recovery_tolerance < 1 or
                min(self.max_abs_G, self.max_gradient_norm, self.zero_tolerance) <= 0 or self.divergence_ratio <= 1 or
                self.reference_mode not in ("frozen_window", "ema_window")):
            raise ValueError("Invalid controller bounds, gain, reference or safety settings")


def finite_vector(values, size=None):
    result = [float(v) for v in values]
    if not result or (size is not None and len(result) != size) or not all(math.isfinite(v) for v in result):
        raise ValueError("Expected a nonempty, finite, correctly sized vector")
    return result


class LocalReference:
    """Never follows a disturbance inside a control window; no privileged target."""
    def __init__(self, values, config):
        self.values = finite_vector(values)
        self.config = config

    def get(self):
        return list(self.values)

    def refresh(self, observed, *, window_boundary=False):
        observed = finite_vector(observed, len(self.values))
        if window_boundary:
            beta = self.config.reference_ema if self.config.reference_mode == "ema_window" else 0.
            self.values = [beta*r + (1-beta)*g for r, g in zip(self.values, observed)]
        return self.get()


class ResponseController:
    def __init__(self, config):
        self.config = config

    def update(self, alphas, observed, reference, response, method="response_aware"):
        c = self.config
        alphas = finite_vector(alphas)
        observed, reference, response = [finite_vector(v, len(alphas)) for v in (observed, reference, response)]
        if any(not c.alpha_min <= a <= c.alpha_max for a in alphas):
            raise ValueError("Alpha outside configured bounds")
        if method not in ("response_aware", "fixed_sign"):
            raise ValueError("Unknown feedback method")
        errors = [g-r for g, r in zip(observed, reference)]
        gain = response if method == "response_aware" else [abs(r) for r in response]
        raw = [-c.eta*e*r/(r*r+c.regularizer) if c.enabled else 0. for e, r in zip(errors, gain)]
        return apply_steps(alphas, raw, c, errors=errors, response=response)


def apply_steps(alphas, requested, config, **metadata):
    alphas = finite_vector(alphas)
    requested = finite_vector(requested, len(alphas))
    if any(not config.alpha_min <= a <= config.alpha_max for a in alphas):
        raise ValueError("Alpha outside configured bounds")
    limited = [max(-config.max_step, min(config.max_step, s)) for s in requested]
    updated = [max(config.alpha_min, min(config.alpha_max, a+s)) for a, s in zip(alphas, limited)]
    return dict(alpha=updated, requested_step=requested, limited_step=limited,
        applied_step=[b-a for a, b in zip(alphas, updated)],
        bound_hit=[a in (config.alpha_min, config.alpha_max) for a in updated], **metadata)
