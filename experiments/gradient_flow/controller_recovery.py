"""Frozen-state closed-loop feasibility. No optimizer updates or accuracy claims."""

import argparse
import copy
import gc
import json
import math
from pathlib import Path
import random
import time

import torch

from .common import ROOT, build_model, load_config, load_weights, provenance, seed_all, write_json
from .controller import ControllerConfig, LocalReference, ResponseController, apply_steps, finite_vector
from .intervene import intervention, same_state
from .make_probe_batch import iter_batches
from .multi_batch_intervene import batch_fingerprint, digest_file, digest_json
from .state_intervene import validate_manifest
from .checkpoint_stages import validate_baseline

METHODS = ("no_control", "fixed_sign", "response_aware", "random_matched")


class FrozenProbe:
    """Small adapter over the existing exact-restoration engine; FP32/eval only."""
    def __init__(self, model, batch, config, optimizer=None):
        self.model, self.config, self.optimizer = model, config, optimizer
        device = next(model.parameters()).device
        self.images, self.targets = batch["images"].to(device), batch["targets"].to(device)
        if self.images.dtype != torch.float32 or any(p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError("Frozen response probes require FP32 inputs and model parameters")
        self.calls, self.seconds = 0, 0.

    def __call__(self, alphas, estimate=True):
        c = self.config
        alphas = finite_vector(alphas, len(self.model.block))
        if any(not c.alpha_min <= a <= c.alpha_max for a in alphas):
            raise ValueError("Probe alpha outside configured bounds")
        delta = [min(c.delta, a-c.alpha_min, c.alpha_max-a) for a in alphas]
        if estimate and any(d <= c.zero_tolerance for d in delta):
            raise ValueError("Cannot form a symmetric bounded response probe at an alpha boundary")
        sweeps = ({j: [a-delta[j], a+delta[j]] for j, a in enumerate(alphas)} if estimate else {0: [alphas[0]]})
        before = None if self.optimizer is None else copy.deepcopy(self.optimizer.state_dict())
        if self.images.device.type == "cuda":
            torch.cuda.synchronize(self.images.device)
        tick = time.perf_counter()
        with torch.autocast(device_type=self.images.device.type, enabled=False):
            data = intervention(self.model, self.images, self.targets, list(sweeps), [alphas[0]],
                amp=False, mode="eval", reference_alphas=alphas, alpha_values_by_block=sweeps)
        if self.optimizer is not None and not same_state(before, self.optimizer.state_dict()):
            raise AssertionError("Optimizer state changed during no-step response probe")
        if self.images.device.type == "cuda":
            torch.cuda.synchronize(self.images.device)
        seconds = time.perf_counter()-tick
        self.calls += 1
        self.seconds += seconds
        for trial in [data["baseline"], *data["trials"]]:
            if trial["logits"] != data["baseline"]["logits"]:
                raise AssertionError("Alpha probe changed forward logits")
            for block in trial["blocks"]:
                if (not block["valid"] or not math.isfinite(trial["loss"]) or
                        abs(block["G_l"]) > c.max_abs_G or
                        any(block[k] is None or not math.isfinite(block[k]) or block[k] > c.max_gradient_norm
                            for k in ("grad_input_norm", "grad_output_norm", "parameter_grad_norm"))):
                    raise ValueError("Nonfinite/unsafe gradient state in frozen probe")
        gains = None
        if estimate:
            gains = []
            for j, a in enumerate(alphas):
                lo, hi = [t for t in data["trials"] if t["target_block"] == j]
                gains.append((hi["blocks"][j]["G_l"]-lo["blocks"][j]["G_l"])/(2*delta[j]))
            finite_vector(gains, len(alphas))
        def compact(trial):
            return {k: trial[k] for k in ("target_block", "alpha", "loss", "blocks", "block_alphas", "sps_alphas")}
        return dict(G=[b["G_l"] for b in data["baseline"]["blocks"]], Rhat=gains,
            loss=data["baseline"]["loss"], blocks=data["baseline"]["blocks"], seconds=seconds,
            raw_probe=dict(centers=alphas, delta=delta if estimate else None,
                baseline=compact(data["baseline"]), trials=[compact(t) for t in data["trials"]],
                checks=data["checks"], optimizer_updates=data["optimizer_updates"], mode="eval", amp=False))


def point(iteration, alpha, observation, reference, initial, config, update=None):
    values = finite_vector(observation["G"], len(alpha))
    error = [g-r for g, r in zip(values, reference)]
    if any(abs(g) > config.max_abs_G for g in values):
        raise ValueError("Gradient state exceeds safety threshold")
    norm = math.sqrt(sum(e*e for e in error))
    return dict(iteration=iteration, alpha=list(alpha), G=values, G_ref=list(reference), error=error,
        absolute_error=[abs(e) for e in error], error_l2=norm,
        normalized_error=norm/initial if initial > config.zero_tolerance else None,
        Rhat=observation["Rhat"], loss=observation["loss"], observation=observation,
        update=update, bound_hit=[a in (config.alpha_min, config.alpha_max) for a in alpha])


def episode(observe, initial_observation, initial_alpha, reference, config, method, rng, matched=None):
    controller = ResponseController(config)
    reference = LocalReference(reference, config)  # Frozen throughout this disturbance episode.
    alpha, observation = list(initial_alpha), initial_observation
    initial = math.sqrt(sum((g-r)**2 for g, r in zip(observation["G"], reference.get())))
    trace = [point(0, alpha, observation, reference.get(), initial, config)]
    updates, status = [], "complete"
    for iteration in range(config.max_iterations):
        if trace[-1]["normalized_error"] is not None and trace[-1]["normalized_error"] > config.divergence_ratio:
            status = "diverged"
            break
        if any(trace[-1]["bound_hit"]) and method in ("response_aware", "fixed_sign"):
            status = "stopped_at_bound"
            break
        control_tick = time.perf_counter()
        if method == "no_control":
            update = apply_steps(alpha, [0.]*len(alpha), config)
        elif method == "random_matched":
            if iteration >= len(matched):
                status = "matched_schedule_exhausted"
                break
            requested = [abs(s)*rng.choice([direction for direction in (-1, 1)
                if config.alpha_min <= a+abs(s)*direction <= config.alpha_max])
                for a, s in zip(alpha, matched[iteration]["applied_step"])]
            update = apply_steps(alpha, requested, config)
            update["matched_magnitudes"] = [abs(s) for s in matched[iteration]["applied_step"]]
            update["magnitudes_exactly_matched"] = all(math.isclose(abs(s), m, abs_tol=1e-12)
                for s, m in zip(update["applied_step"], update["matched_magnitudes"]))
        else:
            update = controller.update(alpha, observation["G"], reference.get(), observation["Rhat"], method)
            update["descent_direction_consistent_with_estimated_response"] = [e*r*s <= config.zero_tolerance
                for e, r, s in zip(update["errors"], update["response"], update["applied_step"])]
        update["controller_seconds"] = time.perf_counter()-control_tick
        alpha = update["alpha"]
        updates.append(update)
        # No-control is exactly the SAME frozen operating point; reuse that measured
        # observation explicitly rather than inventing repeated measurement timings.
        if method != "no_control":
            estimate = method in ("response_aware", "fixed_sign") and iteration+1 < config.max_iterations and not any(update["bound_hit"])
            observation = observe(alpha, estimate=estimate)
        trace.append(point(iteration+1, alpha, observation, reference.get(), initial, config, update))
    def first(fraction, block=None):
        start = initial if block is None else trace[0]["absolute_error"][block]
        if start <= config.zero_tolerance:
            return None
        return next((p["iteration"] for p in trace if (p["error_l2"] if block is None else
            p["absolute_error"][block]) <= fraction*start), None)
    crossings = []
    for block in range(len(alpha)):
        signs = [math.copysign(1, p["error"][block]) for p in trace if abs(p["error"][block]) > config.zero_tolerance]
        crossings.append(sum(a != b for a, b in zip(signs, signs[1:])))
    if trace[-1]["normalized_error"] is not None and trace[-1]["normalized_error"] > config.divergence_ratio:
        status = "diverged"
    if any(trace[-1]["bound_hit"]) and status == "complete" and method in ("response_aware", "fixed_sign"):
        status = "stopped_at_bound"
    return dict(method=method, status=status, trace=trace, updates=updates, initial_error=initial,
        final_residual=trace[-1]["error_l2"], final_normalized_error=trace[-1]["normalized_error"],
        iterations_to_50_percent=first(.5), iterations_to_75_percent=first(.25),
        per_block_iterations_to_50_percent=[first(.5, b) for b in range(len(alpha))],
        per_block_iterations_to_75_percent=[first(.25, b) for b in range(len(alpha))],
        error_sign_crossings=crossings, oscillation_indicator=any(n >= 2 for n in crossings),
        oscillation_definition="at least two nonzero error sign crossings in any block; descriptive, not a stability proof",
        alpha_bound_hit=any(any(p["bound_hit"]) for p in trace), optimizer_updates=0,
        no_control_observations_reused=method == "no_control")


def recover_case(observe, initial_alpha, reference_observation, config, seed=42):
    reference = finite_vector(reference_observation["G"], len(initial_alpha))
    disturbed = observe(initial_alpha, estimate=True)
    results = {}
    for method in ("response_aware", "no_control", "fixed_sign", "random_matched"):
        results[method] = episode(observe, disturbed, initial_alpha, reference, config, method,
            random.Random(seed), results.get("response_aware", {}).get("updates"))
    return dict(reference=reference_observation, disturbed_alpha=initial_alpha, methods=results)


def aggregate(cases, settings, synthetic=False):
    groups, signs = {}, {}
    for state in ("early", "mid", "late"):
        selected = [c for c in cases if c["state"] == state]
        if not selected:
            raise ValueError("Missing recovery state")
        calibrations = [c["reference"]["Rhat"] for c in selected]
        signs[state] = [sum(row[b] for row in calibrations)/len(calibrations) for b in range(len(calibrations[0]))]
        groups[state] = {}
        for disturbance in settings["disturbances"]:
            subset = [c for c in selected if c["disturbance"] == disturbance]
            groups[state][disturbance] = {}
            for method in METHODS:
                rows = [c["methods"][method] for c in subset]
                valid = [r["final_normalized_error"] for r in rows if r["final_normalized_error"] is not None]
                responses = [p["Rhat"] for r in rows for p in r["trace"] if p["Rhat"] is not None]
                checks = [v for r in rows for u in r["updates"]
                          for v in u.get("descent_direction_consistent_with_estimated_response", [])]
                groups[state][disturbance][method] = dict(count=len(rows), informative=len(valid),
                    mean_final_ratio=sum(valid)/len(valid) if valid else None,
                    fraction_recovered=sum(v <= settings["controller"]["recovery_tolerance"] for v in valid)/len(rows) if rows else 0.,
                    bound_hit_fraction=sum(r["alpha_bound_hit"] for r in rows)/len(rows) if rows else None,
                    oscillation_fraction=sum(r["oscillation_indicator"] for r in rows)/len(rows) if rows else None,
                    diverged=sum(r["status"] == "diverged" for r in rows),
                    mean_final_absolute_error_per_block=[sum(r["trace"][-1]["absolute_error"][b] for r in rows)/len(rows)
                        for b in range(len(signs[state]))] if rows else None,
                    response_positive_fraction=[sum(r[b] > 0 for r in responses)/len(responses)
                        for b in range(len(signs[state]))] if responses else None,
                    response_negative_fraction=[sum(r[b] < 0 for r in responses)/len(responses)
                        for b in range(len(signs[state]))] if responses else None,
                    update_direction_consistency_fraction=sum(checks)/len(checks) if checks else None)
    reversal = [signs["early"][b] < 0 < min(signs["mid"][b], signs["late"][b]) for b in range(len(signs["early"]))]
    failures = []
    gate = settings["gate"]
    if synthetic:
        failures.append("Synthetic data cannot authorize online training")
    if settings["num_batches"] < gate["min_batches"]:
        failures.append("Too few batches for the real-data feasibility gate")
    if gate["require_sign_reversal"] and not all(reversal):
        failures.append("Reference finite differences did not reproduce the early/mid/late sign reversal for all blocks")
    for state, disturbances in groups.items():
        for name, methods in disturbances.items():
            aware = methods["response_aware"]
            ratio = aware["mean_final_ratio"]
            if (aware["count"] != settings["num_batches"] or aware["informative"] != aware["count"] or
                    ratio is None or ratio > gate["max_mean_final_ratio"] or
                    aware["fraction_recovered"] < gate["min_recovered_fraction"] or aware["bound_hit_fraction"] != 0 or
                    aware["oscillation_fraction"] != 0 or aware["diverged"]):
                failures.append(f"{state}/{name}: response-aware recovery/stability thresholds not met")
    # Fixed-sign and response-aware are identical when every local gain is
    # positive. Demand a pooled advantage, not an impossible strict win in each
    # such state; keep statewise recovery/stability requirements above.
    pooled = {}
    for method in METHODS:
        values = [c["methods"][method]["final_normalized_error"] for c in cases]
        valid = [v for v in values if v is not None]
        pooled[method] = dict(count=len(values), informative=len(valid),
            mean_final_ratio=sum(valid)/len(valid) if valid else None)
    for baseline in ("no_control", "fixed_sign", "random_matched"):
        aware, other = pooled["response_aware"]["mean_final_ratio"], pooled[baseline]["mean_final_ratio"]
        if aware is None or other is None or aware >= other or pooled[baseline]["informative"] != len(cases):
            failures.append(f"Pooled cases: no complete mean-error advantage over {baseline}")
    return dict(protocol="provisional frozen-state disturbance recovery", groups=groups,
        reference_response_mean=signs, reference_sign_reversal=reversal, pooled_comparison=pooled,
        gate=dict(passed=not failures, failures=failures, criteria=gate),
        claims="Descriptive closed-loop feasibility only; no claim about optimal G or accuracy improvement")


def run_study(manifest, batches, settings, destination, source, backend="cupy", device="cuda", resume=False, synthetic=False):
    validate_manifest(manifest)
    if manifest["origin"] != "regenerated":
        raise ValueError("Recovery requires the verified regenerated same-trajectory checkpoints")
    if not synthetic and manifest.get("synthetic", False):
        raise ValueError("Synthetic checkpoint fixtures cannot authorize a real-data recovery study")
    config = ControllerConfig(**settings["controller"])
    if (settings["seed"] != 42 or settings["num_batches"] < 1 or settings["batch_size"] < 1 or
            not settings["disturbances"] or not config.alpha_min < settings["reference_alpha"] < config.alpha_max):
        raise ValueError("Require seed42, positive batch counts, disturbances and an interior reference alpha")
    for values in settings["disturbances"].values():
        values = finite_vector(values, manifest["recipe"]["layer"])
        if any(not config.alpha_min < a < config.alpha_max for a in values):
            raise ValueError("Initial disturbances must lie strictly inside probe bounds")
    gate = settings["gate"]
    if not 0 < gate["min_recovered_fraction"] <= 1 or not 0 < gate["max_mean_final_ratio"] < 1 or gate["min_batches"] < 10:
        raise ValueError("Invalid real-data recovery gate")
    batches, destination = list(batches), Path(destination)
    if len(batches) != settings["num_batches"]:
        raise ValueError("Unexpected deterministic batch count")
    for i, batch in enumerate(batches):
        if (not synthetic and (batch.get("split") != "train" or batch.get("synthetic", False))) or batch["indices"] != list(range(i*settings["batch_size"], (i+1)*settings["batch_size"])):
            raise ValueError("Recovery references require deterministic training-split batches, never held-out labels")
    identity = dict(manifest=manifest, settings=settings, source=source, backend=backend, device=device,
        mode="eval", amp=False, synthetic=synthetic,
        batch_fingerprints=[batch_fingerprint(b) for b in batches], split="synthetic" if synthetic else "train")
    identity = json.loads(json.dumps(identity))
    destination.mkdir(parents=True, exist_ok=True)
    metadata = destination / "metadata.json"
    if metadata.exists():
        if not resume or json.loads(metadata.read_text()) != identity:
            raise ValueError("Recovery resume requires identical metadata/checkpoints/batches/code")
    elif any(destination.iterdir()):
        raise ValueError("Refusing nonempty recovery output without metadata")
    else:
        write_json(metadata, identity)
    signature = digest_json(identity)
    records, tick = [], time.perf_counter()
    write_json(destination / "status.json", dict(status="running", signature=signature))
    try:
        for state in ("early", "mid", "late"):
            model = None
            try:
                model = build_model(manifest["recipe"], backend).to(device)
                load_weights(model, manifest["states"][state]["path"])
                for index, batch in enumerate(batches):
                    directory = destination / state / f"batch_{index:03d}"
                    directory.mkdir(parents=True, exist_ok=True)
                    path, marker = directory / "recovery.json", directory / "complete.json"
                    cached = None
                    if resume and marker.exists() and path.exists():
                        try:
                            done = json.loads(marker.read_text())
                        except (OSError, ValueError):
                            done = dict(status="corrupt", signature=signature)
                        if done.get("signature") != signature:
                            raise ValueError("Recovery completion marker belongs to a different protocol")
                        if done.get("status") == "complete" and done.get("sha256") == digest_file(path):
                            try:
                                cached = json.loads(path.read_text())
                            except (OSError, ValueError):
                                cached = None
                    if cached is None:
                        write_json(marker, dict(status="running", signature=signature))
                        probe = FrozenProbe(model, batch, config)
                        reference = probe([settings["reference_alpha"]]*len(model.block), estimate=True)
                        cached = []
                        for name, alphas in settings["disturbances"].items():
                            case = recover_case(probe, alphas, reference, config, settings["seed"]+index)
                            cached.append(dict(case, state=state, batch=index, disturbance=name,
                                batch_fingerprint=identity["batch_fingerprints"][index]))
                            write_json(directory / "progress.json", cached)
                        write_json(path, cached)
                        write_json(marker, dict(status="complete", signature=signature, sha256=digest_file(path),
                            calls=probe.calls, probe_seconds=probe.seconds))
                    records.extend(cached)
                    print(f"{state} batch {index+1}/{len(batches)}: recovery records verified; elapsed {time.perf_counter()-tick:.1f}s", flush=True)
            finally:
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        result = aggregate(records, settings, synthetic)
        write_json(destination / "summary.json", result)
        write_json(destination / "status.json", dict(status="complete", feasibility_passed=result["gate"]["passed"],
            elapsed_seconds=time.perf_counter()-tick, cases=len(records)))
        return result
    except Exception as error:
        write_json(destination / "status.json", dict(status="failed", error_type=type(error).__name__, error=str(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gradient_flow/controller_feasibility.yml")
    parser.add_argument("--manifest", default="results/gradient_flow/alpha_state_dependence_seed42/states.json")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="results/gradient_flow/controller_recovery_seed42")
    parser.add_argument("--backend", choices=("torch", "cupy"), default="cupy")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-pass", action="store_true", help="Exit nonzero when real-data feasibility criteria are unmet")
    args = parser.parse_args()
    source = provenance()
    if source.get("dirty") is not False:
        parser.error("Recovery requires a clean Git checkout")
    if args.backend == "cupy" and (torch.device(args.device).type != "cuda" or not torch.cuda.is_available()):
        parser.error("CuPy requires CUDA; no silent backend fallback")
    manifest = json.loads(Path(args.manifest).read_text())
    validate_baseline(manifest["recipe"])
    settings = load_config(ROOT / args.config)
    if settings["num_batches"] < 10:
        parser.error("Real frozen recovery requires at least 10 deterministic batches")
    seed_all(settings["seed"])
    destination = (ROOT / args.output).resolve()
    namespace = ROOT / "results/gradient_flow"
    if namespace.resolve() != namespace or not destination.is_relative_to(namespace) or destination == namespace:
        parser.error("Recovery outputs must use results/gradient_flow/<new-study>")
    if destination.is_relative_to(Path(manifest["trajectory_directory"]).resolve()):
        parser.error("Never write recovery output inside the checkpoint trajectory")
    result = run_study(manifest, iter_batches(args.data_dir, manifest["recipe"], settings["batch_size"],
        settings["num_batches"], split="train"), settings, destination, source, args.backend, args.device, args.resume)
    print("Frozen-state feasibility gate:", result["gate"], flush=True)
    if args.require_pass and not result["gate"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
