"""Saved-data-only calibration arithmetic and fixed, predeclared criteria."""

from .online_summary import mean, summarize_condition


def summarize(rows, settings, persistent_failure=False):
    result = summarize_condition(rows, settings)
    blocks = len(result["alpha_final"])
    controllers = [r["controller"] for r in rows]
    movement = [sum(abs(r["applied_step"][b]) for r in controllers) for b in range(blocks)]
    deviation = [[abs(r["alpha"][b]-4.) for r in controllers] for b in range(blocks)]
    result.update(mean_train_G_variance=mean(result["train_G_variance"]),
        mean_probe_G_variance=mean(result["probe_G_variance"]),
        cumulative_absolute_alpha_movement=movement,
        total_absolute_alpha_movement=sum(movement),
        mean_absolute_alpha_deviation_from_4=mean([v for series in deviation for v in series]),
        max_absolute_alpha_deviation_from_4=max(v for series in deviation for v in series),
        per_block_mean_absolute_alpha_deviation_from_4=[mean(series) for series in deviation],
        response_sign_changes=[sum(a != b for a, b in zip(series, series[1:])) for series in
            ([r["response_sign"][b] for r in controllers if r["Rhat"] is not None and r["response_sign"][b] != 0]
                for b in range(blocks))],
        persistent_failure=bool(persistent_failure),
        alpha_initial=[4.]*blocks,
        update_magnitude_definition="absolute applied step per scheduled epoch and block; includes zero updates",
        pinned_definition="per-block fraction of scheduled observations exactly at either bound (not consecutive duration)")
    return result


def relative(variant, fixed, criteria):
    baseline = fixed["mean_train_G_variance"]
    reduction = 1-variant["mean_train_G_variance"]/baseline if baseline is not None and baseline > 0 else None
    delta = {"accuracy_delta_"+name+"_pp": variant[key]-fixed[key] for name, key in
        (("final", "final_validation_top1"), ("best", "best_validation_top1"), ("mean", "mean_validation_top1"))}
    return dict(**delta, G_variance_reduction_fraction=reduction,
        stabilization_per_accuracy_cost=None if reduction is None else
            reduction/max(abs(delta["accuracy_delta_mean_pp"]), criteria["tradeoff_accuracy_floor_pp"]))


def calibration_gate(summary, comparison, criteria, *, complete, synthetic=False):
    reduction = comparison["G_variance_reduction_fraction"]
    checks = dict(completed_60_real_epochs=complete and not synthetic and summary["epochs"] == criteria["epochs"],
        no_persistent_instability=not summary["persistent_failure"] and summary["nonfinite_or_state_failures"] == 0,
        no_alpha_bound_saturation=summary["bound_hit_fraction"] == 0 and all(v == 0 for v in summary["block_pinned_fraction"]),
        controller_overhead_below_20_percent=summary["controller_overhead_fraction"] is not None and
            summary["controller_overhead_fraction"] < criteria["max_overhead_fraction"],
        G_variance_reduced_at_least_25_percent=reduction is not None and reduction >= criteria["min_variance_reduction"],
        final_accuracy_within_half_pp=comparison["accuracy_delta_final_pp"] >= -criteria["max_final_accuracy_drop_pp"])
    strong = (all(checks.values()) and reduction >= criteria["strong_min_variance_reduction"] and
        comparison["accuracy_delta_final_pp"] >= -criteria["strong_max_accuracy_drop_pp"] and
        comparison["accuracy_delta_best_pp"] >= -criteria["strong_max_accuracy_drop_pp"])
    return dict(promising=all(checks.values()), stronger_positive_indicator=strong, checks=checks,
        failures=[key for key, value in checks.items() if not value],
        interpretation="Predeclared seed42 calibration criteria, not general superiority or publication claims. No automatic winner.")
