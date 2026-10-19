"""Descriptive batch-level statistics; no hypothesis tests or efficacy claims."""

import math

import numpy as np

from .intervene import local_response_matrix


def validate_measurements(data, block_count, alphas, reference):
    required_checks = ("reference_reproducible", "alpha_isolation", "weights_exactly_unchanged",
                       "restored_after_each_probe", "gradients_cleared_before_each_probe")
    if (data["optimizer_updates"] != 0 or data["amp"] or data["mode"] != "eval" or
            any(data["checks"].get(key) is not True for key in required_checks) or
            data["block_ids"] != list(range(block_count)) or data["alpha_values"] != list(alphas) or
            data["reference_alpha"] != reference):
        raise ValueError("Probe protocol or invariance check failed")
    expected = {(j, a) for j in range(block_count) for a in alphas}
    actual = [(t["target_block"], t["alpha"]) for t in data["trials"]]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Missing/duplicate block-alpha trials")
    baseline = data["baseline"]
    for trial in [baseline, *data["trials"]]:
        if not math.isfinite(trial["loss"]) or [b["block_id"] for b in trial["blocks"]] != list(range(block_count)):
            raise ValueError("Missing blocks or nonfinite loss")
        for block in trial["blocks"]:
            if not block["valid"] or any(block[k] is None or not math.isfinite(block[k]) for k in
                    ("G_l", "grad_input_norm", "grad_output_norm", "parameter_grad_norm", "firing_rate")):
                raise ValueError("Invalid/nonfinite block observation")
        expected_alpha = [reference] * block_count
        if trial["target_block"] is not None:
            expected_alpha[trial["target_block"]] = trial["alpha"]
        if (trial["block_alphas"] != expected_alpha or not trial["sps_alphas"] or
                any(a != 4. for a in trial["sps_alphas"].values())):
            raise ValueError("Alpha isolation/SPS invariant failed")
        if trial["alpha"] == reference:
            a = [trial["loss"], *[b["G_l"] for b in trial["blocks"]]]
            b = [baseline["loss"], *[b["G_l"] for b in baseline["blocks"]]]
            if not np.allclose(a, b, rtol=1e-6, atol=1e-8):
                raise ValueError("Reference reproducibility failed")
    matrix = local_response_matrix(data)
    if np.asarray(matrix["matrix"]).shape != (block_count, block_count):
        raise ValueError("Incorrect local matrix shape")
    return matrix


def summary(values, bootstrap_samples=10000, seed=42):
    """Percentile bootstrap CI of mean, resampling independent batch observations."""
    x = np.asarray(values, dtype=float)
    if len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("Statistics require nonempty finite observations")
    n = len(x)
    sd = np.std(x, axis=0, ddof=1) if n > 1 else np.full(x.shape[1:], np.nan)
    ci = None
    if n > 1:
        rng = np.random.default_rng(seed)
        # Chunk bootstrap draws to bound memory, including longer future experiments.
        estimates = []
        for start in range(0, bootstrap_samples, 256):
            indices = rng.integers(0, n, (min(256, bootstrap_samples - start), n))
            estimates.append(x[indices].mean(axis=1))
        ci = np.quantile(np.concatenate(estimates), [.025, .975], axis=0).tolist()
    return dict(count=n, mean=x.mean(axis=0).tolist(), std=sd.tolist() if n > 1 else None,
                median=np.median(x, axis=0).tolist(), minimum=x.min(axis=0).tolist(),
                maximum=x.max(axis=0).tolist(), ci95=ci)


def signs(values, tolerance):
    x = np.asarray(values, dtype=float)
    return dict(fraction_positive=(x > tolerance).mean(axis=0).tolist(),
                fraction_negative=(x < -tolerance).mean(axis=0).tolist(),
                fraction_effectively_zero=(np.abs(x) <= tolerance).mean(axis=0).tolist())


def structural_metrics(matrix, tolerance):
    n = len(matrix)
    diagonal = np.abs(np.diag(matrix))
    off = np.abs(matrix[~np.eye(n, dtype=bool)])
    nonzero = off[off > tolerance]
    def avg(x):
        return float(np.mean(x)) if len(x) else None
    diag, off_all, off_nz = avg(diagonal), avg(off), avg(nonzero)
    return dict(mean_abs_diagonal=diag, mean_abs_off_diagonal=off_all,
                mean_abs_nonzero_off_diagonal=off_nz, nonzero_off_diagonal_count=int(len(nonzero)),
                ratio_all_off_diagonal=diag / off_all if off_all is not None and off_all > tolerance else None,
                ratio_nonzero_off_diagonal=diag / off_nz if off_nz is not None else None,
                mean_abs_nearest_upstream=avg(np.abs(np.diag(matrix, 1))),
                mean_abs_distant_upstream=avg(np.abs(matrix[np.triu_indices(n, 2)])),
                mean_abs_downstream=avg(np.abs(matrix[np.tril_indices(n, -1)])),
                every_diagonal_positive=bool(np.all(np.diag(matrix) > 0)),
                every_downstream_effectively_zero=bool(np.all(np.abs(matrix[np.tril_indices(n, -1)]) <= tolerance)))


def aggregate_measurements(payloads, zero_tolerance=1e-8, bootstrap_samples=10000, seed=42):
    if not payloads or not math.isfinite(zero_tolerance) or zero_tolerance < 0 or bootstrap_samples < 100:
        raise ValueError("Need batches, finite nonnegative tolerance and >=100 bootstrap replicates")
    first = payloads[0]
    blocks, alphas, reference = first["block_ids"], first["alpha_values"], first["reference_alpha"]
    matrices = np.asarray([validate_measurements(p, len(blocks), alphas, reference)["matrix"] for p in payloads])
    curves = np.asarray([[[[next(t for t in p["trials"] if t["target_block"] == j and t["alpha"] == a)
                           ["blocks"][i]["G_l"] for a in alphas] for i in blocks] for j in blocks] for p in payloads])
    per_batch = [structural_metrics(matrix, zero_tolerance) for matrix in matrices]
    metrics = {}
    for key in per_batch[0]:
        values = [p[key] for p in per_batch if p[key] is not None]
        metrics[key] = summary(values, bootstrap_samples, seed) if values else dict(count=0)
        metrics[key]["undefined_batches"] = len(payloads) - len(values)
    near = sorted((a, index) for index, a in enumerate(alphas) if abs(a - reference) <= 1)
    monotonic = None
    if len(near) >= 3 and any(a == reference for a, _ in near):
        delta = np.diff(curves[..., [index for _, index in near]], axis=-1)
        nondecreasing = np.all(delta >= -zero_tolerance, axis=-1)
        nonincreasing = np.all(delta <= zero_tolerance, axis=-1)
        monotonic = dict(alphas=[a for a, _ in near],
            fraction_nondecreasing=nondecreasing.mean(axis=0).tolist(),
            fraction_nonincreasing=nonincreasing.mean(axis=0).tolist(),
            fraction_effectively_flat=np.all(np.abs(delta) <= zero_tolerance, axis=-1).mean(axis=0).tolist(),
            fraction_nonmonotonic=(~(nondecreasing | nonincreasing)).mean(axis=0).tolist())
    structural = dict(per_batch=per_batch, aggregate=metrics,
        fraction_every_diagonal_positive=float(np.mean([p["every_diagonal_positive"] for p in per_batch])),
        per_block_fraction_diagonal_positive=(np.diagonal(matrices, axis1=1, axis2=2) > 0).mean(axis=0).tolist(),
        fraction_diagonal_mean_larger_than_off_diagonal=float(np.mean([
            p["mean_abs_diagonal"] > p["mean_abs_off_diagonal"] for p in per_batch])),
        fraction_nearest_larger_than_distant=float(np.mean([
            p["mean_abs_nearest_upstream"] > p["mean_abs_distant_upstream"] for p in per_batch]))
            if len(blocks) > 2 else None,
        nearest_upstream_sign_consistency=signs(np.diagonal(matrices, offset=1, axis1=1, axis2=2), zero_tolerance))
    return dict(block_ids=blocks, alpha_values=alphas, reference_alpha=reference, batch_count=len(payloads),
        zero_tolerance=zero_tolerance, std_definition="sample SD (ddof=1); undefined for one batch",
        ci_method=dict(method="percentile bootstrap of batch mean; batch is resampling unit, not images",
                       replicates=bootstrap_samples, seed=seed, percentiles=[2.5, 97.5],
                       caveat="Descriptive interval conditional on this checkpoint and deterministic test subset; not a significance test"),
        response_stats=summary(matrices, bootstrap_samples, seed),
        response_sign_consistency=signs(matrices, zero_tolerance),
        structural_metrics=structural, matrices=matrices.tolist(),
        alpha_curves=dict(axes=["perturbed_block", "measured_block", "alpha"],
            mean=curves.mean(axis=0).tolist(), std=curves.std(axis=0, ddof=1).tolist() if len(curves) > 1 else None,
            local_monotonicity=monotonic),
        tolerance_definition="R: positive >tol, negative <-tol, effectively zero abs(R)<=tol; ratios undefined if denominator <=tol. G-curve increments use the same numeric tolerance in G units.")
