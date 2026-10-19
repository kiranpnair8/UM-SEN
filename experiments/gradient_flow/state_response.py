"""Paired whole-batch training-state comparisons; descriptive, not hypothesis tests."""

import numpy as np

from .checkpoint_stages import STAGES
from .multi_batch_stats import summary


def normalized_change(before, after, tolerance=1e-8, eps=1e-12):
    before, after = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    result = np.empty(before.shape, dtype=object)
    for index in np.ndindex(before.shape):
        result[index] = float((after[index]-before[index])/(abs(before[index])+eps)) if abs(before[index]) > tolerance else None
    return result.tolist()


def matrix_similarity(a, b):
    """Fixed topology mask i<=j, NOT an empirical mask selected from the data."""
    mask = np.triu(np.ones_like(a, dtype=bool))
    x, y = np.asarray(a)[mask], np.asarray(b)[mask]
    denominator = np.linalg.norm(x)*np.linalg.norm(y)
    return dict(mask=mask.tolist(), excluded="strict lower triangle i>j only; downstream responses reported separately",
                cosine=float(np.clip(np.dot(x, y)/denominator, -1, 1)) if denominator > 0 else None)


def compare_states(records, epochs, bootstrap_samples=10000, seed=42, tolerance=1e-8):
    if set(records) != set(STAGES):
        raise ValueError("Need exactly early, mid and late states")
    first = records["early"]
    for label in STAGES:
        r = records[label]
        if (r["batch_fingerprints"] != first["batch_fingerprints"] or
                len(r["batch_fingerprints"]) != r["aggregate"]["batch_count"] or
                r["aggregate"]["alpha_values"] != first["aggregate"]["alpha_values"] or
                r["aggregate"]["reference_alpha"] != first["aggregate"]["reference_alpha"] or
                r["aggregate"]["zero_tolerance"] != tolerance):
            raise ValueError("State comparisons require identical batch correspondence and probe settings")
    matrices = {s: np.asarray(records[s]["aggregate"]["matrices"], dtype=float) for s in STAGES}
    if any(m.shape != matrices["early"].shape or m.shape[1:] != (4, 4) for m in matrices.values()):
        raise ValueError("Need matched batches of 4x4 matrices")
    means = {s: m.mean(axis=0) for s, m in matrices.items()}
    differences = {}
    sign = lambda x: np.where(np.abs(x) <= tolerance, 0, np.sign(x))
    scalar_keys = ("mean_abs_diagonal", "mean_abs_nearest_upstream", "mean_abs_off_diagonal",
                   "mean_abs_distant_upstream", "mean_abs_downstream", "ratio_all_off_diagonal", "ratio_nonzero_off_diagonal")
    for before, after in (("early", "mid"), ("mid", "late"), ("early", "late")):
        delta = matrices[after]-matrices[before]
        scalars = {}
        for key in scalar_keys:
            a = records[before]["aggregate"]["structural_metrics"]["per_batch"]
            b = records[after]["aggregate"]["structural_metrics"]["per_batch"]
            values = [y[key]-x[key] for x, y in zip(a, b) if x[key] is not None and y[key] is not None]
            scalars[key] = summary(values, bootstrap_samples, seed) if values else dict(count=0)
            scalars[key]["undefined_pairs"] = len(a)-len(values)
        g_delta = np.asarray(records[after]["G_reference"])-np.asarray(records[before]["G_reference"])
        differences[after + "_minus_" + before] = dict(
            matrix=summary(delta, bootstrap_samples, seed), per_batch_matrix_differences=delta.tolist(),
            normalized_mean_matrix_change=normalized_change(means[before], means[after], tolerance),
            normalized_definition="(after-before)/(abs(before)+1e-12) only if abs(before)>effective-zero tolerance; otherwise null",
            paired_diagonal_change=summary(np.diagonal(delta, axis1=1, axis2=2), bootstrap_samples, seed),
            paired_structural_change=scalars, paired_G_reference_change=summary(g_delta, bootstrap_samples, seed),
            frobenius_norm_of_mean_difference=float(np.linalg.norm(means[after]-means[before])),
            paired_sign_unchanged_fraction=(sign(matrices[before]) == sign(matrices[after])).mean(axis=0).tolist(),
            upper_triangle_similarity=matrix_similarity(means[before], means[after]))
    trajectories = np.asarray([np.diag(means[s]) for s in STAGES])
    delta = np.diff(trajectories, axis=0)
    return dict(states=list(STAGES), epochs=epochs, batch_count=len(matrices["early"]),
        zero_tolerance=tolerance, bootstrap=dict(replicates=bootstrap_samples, seed=seed,
            method="percentile bootstrap of PAIRED whole-batch differences; identical batch indices resampled in each state",
            caveat="Descriptive intervals conditional on one training trajectory and deterministic test subset; not significance tests"),
        per_state=records, mean_matrices={s: m.tolist() for s, m in means.items()},
        diagonal_trajectories=trajectories.tolist(),
        diagonal_mean_trajectory_nondecreasing=np.all(delta >= -tolerance, axis=0).tolist(),
        diagonal_mean_trajectory_nonincreasing=np.all(delta <= tolerance, axis=0).tolist(),
        diagonal_mean_trajectory_nonmonotonic=(~(np.all(delta >= -tolerance, axis=0) | np.all(delta <= tolerance, axis=0))).tolist(),
        frobenius_norms={s: float(np.linalg.norm(m)) for s, m in means.items()}, differences=differences,
        mean_response_signs={s: sign(m).tolist() for s, m in means.items()},
        G_reference={s: summary(records[s]["G_reference"], bootstrap_samples, seed) for s in STAGES})
