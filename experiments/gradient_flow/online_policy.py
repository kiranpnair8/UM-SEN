"""Epoch-boundary orchestration around the existing validated controller equations."""

import copy
import math
import random

from .controller import ControllerConfig, LocalReference, ResponseController, apply_steps, finite_vector


METHODS = ("fixed", "response_aware", "random_matched")


class OnlinePolicy:
    def __init__(self, blocks, method, settings, seed=42):
        if method not in METHODS:
            raise ValueError("Unknown online method")
        self.config = ControllerConfig(**settings["controller"])
        self.settings, self.method = copy.deepcopy(settings), method
        if (settings["update_every_epochs"] < 1 or settings["reference_window_epochs"] < 2 or
                settings["reference_window_epochs"] % settings["update_every_epochs"] or
                settings["tiny_response"] <= 0 or not self.config.alpha_min < 4 < self.config.alpha_max):
            raise ValueError("Invalid online reference/update schedule or alpha bounds")
        self.alpha = [4.] * blocks
        self.reference = None
        self.history = []
        self.rng = random.Random(seed + 1701)  # Never consume the training RNG.

    def due(self, epoch):
        return epoch % self.settings["update_every_epochs"] == 0

    def response_update(self, alpha, observed, reference, response):
        return ResponseController(self.config).update(alpha, observed, reference, response)

    def advance(self, epoch, observation, matched=None):
        if self.history and epoch <= self.history[-1]["epoch"]:
            raise ValueError("Controller epochs must increase")
        G = finite_vector(observation["G"], len(self.alpha))
        due = self.due(epoch)
        refresh = self.reference is None or epoch % self.settings["reference_window_epochs"] == 0
        if self.reference is None:
            self.reference = LocalReference(G, self.config)
        elif refresh:
            self.reference.refresh(G, window_boundary=True)
        reference = self.reference.get()
        error = [g-r for g, r in zip(G, reference)]
        response = None if observation["Rhat"] is None else finite_vector(observation["Rhat"], len(G))
        before = list(self.alpha)
        if self.method == "response_aware" and due:
            if response is None:
                raise ValueError("Scheduled control requires a measured response")
            update = self.response_update(before, G, reference, response)
        elif self.method == "random_matched" and due:
            if matched is None or matched["epoch"] != epoch or not matched["update_due"]:
                raise ValueError("Missing matching response-aware epoch")
            requested = []
            for alpha, step in zip(before, finite_vector(matched["applied_step"], len(before))):
                magnitude = abs(step)
                directions = [d for d in (-1, 1) if self.config.alpha_min <= alpha+d*magnitude <= self.config.alpha_max]
                if not directions:
                    raise ValueError("No feasible random direction with matched magnitude")
                requested.append(magnitude*self.rng.choice(directions))
            update = apply_steps(before, requested, self.config)
            update["matched_magnitudes"] = [abs(s) for s in matched["applied_step"]]
            if any(not math.isclose(abs(a), b, abs_tol=1e-12, rel_tol=1e-12)
                   for a, b in zip(update["applied_step"], update["matched_magnitudes"])):
                raise AssertionError("Random control magnitude budget mismatch")
        else:
            update = apply_steps(before, [0.]*len(G), self.config)
        self.alpha = update["alpha"]
        tol = self.config.zero_tolerance
        # Only compare errors within the same frozen-reference window. A refresh
        # changes the target and must not be counted as an error sign crossing.
        window = epoch // self.settings["reference_window_epochs"]
        previous = [r for r in self.history if r["window"] == window]
        last_nonzero = [next((r["error"][b] for r in reversed(previous) if abs(r["error"][b]) > tol), None)
                        for b in range(len(G))]
        crossings = [bool(p is not None and abs(e) > tol and (e > 0) != (p > 0))
                     for e, p in zip(error, last_nonzero)]
        crossing_counts = [sum(r["component_sign_crossing"][b] for r in previous)+int(crossings[b])
                           for b in range(len(G))]
        row = dict(epoch=epoch, window=window, update_due=due, reference_refreshed=refresh,
            alpha_before=before, G=G, G_ref=reference, error=error,
            Rhat=response, response_sign=None if response is None else
                [0 if abs(r) <= self.settings["tiny_response"] else (1 if r > 0 else -1) for r in response],
            tiny_response=None if response is None else [abs(r) <= self.settings["tiny_response"] for r in response],
            component_sign_crossing=crossings, window_sign_crossings=crossing_counts,
            oscillation_indicator=any(c >= 2 for c in crossing_counts),
            oscillation_definition="two component-error crossings within one fixed-reference window; descriptive only",
            update_count=0 if self.method == "fixed" else sum(r["update_due"] for r in self.history)+int(due),
            clipping=[abs(r-l) > tol or abs(l-a) > tol for r, l, a in
                zip(update["requested_step"], update["limited_step"], update["applied_step"])], **update)
        self.history.append(row)
        return row

    def state_dict(self):
        return copy.deepcopy(dict(method=self.method, settings=self.settings, alpha=self.alpha,
            reference=None if self.reference is None else self.reference.get(),
            history=self.history, rng=self.rng.getstate()))

    def load_state_dict(self, state):
        if state["method"] != self.method or state["settings"] != self.settings or len(state["alpha"]) != len(self.alpha):
            raise ValueError("Online policy resume mismatch")
        self.alpha = finite_vector(state["alpha"], len(self.alpha))
        if any(not self.config.alpha_min <= a <= self.config.alpha_max for a in self.alpha):
            raise ValueError("Restored alpha outside bounds")
        self.reference = None if state["reference"] is None else LocalReference(state["reference"], self.config)
        self.history = copy.deepcopy(state["history"])
        self.rng.setstate(state["rng"])
