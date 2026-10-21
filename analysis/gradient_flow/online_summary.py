"""Predeclared descriptive online summaries; no model, probe, or training imports."""

import math
import statistics as stats


def mean(values):
    values = [v for v in values if v is not None]
    return stats.mean(values) if values else None


def variance(values):
    return stats.pvariance(values) if values else None


def distribution(values):
    return dict(count=len(values), mean=mean(values), median=stats.median(values) if values else None,
                min=min(values) if values else None, max=max(values) if values else None,
                std=stats.pstdev(values) if values else None)


def summarize_condition(rows, settings):
    if not rows:
        raise ValueError("No online epochs")
    blocks = len(rows[0]["controller"]["alpha"])
    controllers = [r["controller"] for r in rows]
    updates = [r for r in controllers if r["update_due"]]
    measured = [r for r in updates if r["Rhat"] is not None]
    best = max(rows, key=lambda r: r["validation"]["top1"])
    alpha = [[r["alpha"][b] for r in controllers] for b in range(blocks)]
    bound = [mean([float(r["bound_hit"][b]) for r in updates]) for b in range(blocks)]
    tiny = [float(v) for r in measured for v in r["tiny_response"]]
    signs = [[r["response_sign"][b] for r in measured] for b in range(blocks)]
    sign_changes = []
    for series in signs:
        nonzero = [v for v in series if v]
        sign_changes.append(mean([float(a != b) for a, b in zip(nonzero, nonzero[1:])]))
    norms = [[r["gradient"][b]["parameter_grad_norm"] for r in rows] for b in range(blocks)]
    total = sum(r["epoch_seconds"] for r in rows)
    overhead = sum(r["controller_seconds"] for r in rows)
    delta_loss = [b["train"]["loss"]-a["train"]["loss"] for a, b in zip(rows, rows[1:])]
    delta_acc = [b["validation"]["top1"]-a["validation"]["top1"] for a, b in zip(rows, rows[1:])]
    loss_tail = [r["train"]["loss"] for r in rows[-min(5, len(rows)):]]
    loss_head = [r["train"]["loss"] for r in rows[:min(5, len(rows))]]
    return dict(epochs=len(rows), final_validation_top1=rows[-1]["validation"]["top1"],
        best_validation_top1=best["validation"]["top1"], best_epoch=best["epoch"],
        final_train_loss=rows[-1]["train"]["loss"], mean_train_loss=mean([r["train"]["loss"] for r in rows]),
        mean_validation_top1=mean([r["validation"]["top1"] for r in rows]), total_epoch_seconds=total,
        controller_seconds=overhead, controller_overhead_fraction=overhead/total if total else None,
        alpha_final=controllers[-1]["alpha"], alpha_min=[min(v) for v in alpha], alpha_max=[max(v) for v in alpha],
        bound_hit_fraction=mean(bound), block_pinned_fraction=bound,
        oscillation_fraction=mean([float(r["oscillation_indicator"]) for r in updates]),
        large_oscillation_fraction=mean([float(r["oscillation_indicator"] and
            math.sqrt(sum(e*e for e in r["error"])) > settings["gate"]["large_error_l2"]) for r in updates]),
        absolute_Rhat=distribution([abs(v) for r in measured for v in r["Rhat"]]),
        tiny_response_fraction=mean(tiny), response_sign_distribution=[
            {str(s): series.count(s) for s in (-1, 0, 1)} for series in signs],
        response_sign_change_fraction=sign_changes,
        update_magnitude=distribution([abs(v) for r in updates for v in r["applied_step"]]),
        train_loss_variance=variance([r["train"]["loss"] for r in rows]),
        epoch_delta_train_loss_variance=variance(delta_loss),
        validation_accuracy_variance=variance([r["validation"]["top1"] for r in rows]),
        epoch_delta_validation_accuracy_variance=variance(delta_acc),
        gradient_norm_epoch_mean_cv=[stats.pstdev(v)/abs(mean(v)) if abs(mean(v)) > 1e-12 else None for v in norms],
        mean_within_epoch_gradient_norm_cv=[mean([r["gradient"][b]["parameter_grad_cv"] for r in rows]) for b in range(blocks)],
        train_G_variance=[variance([r["gradient"][b]["G_l"] for r in rows]) for b in range(blocks)],
        probe_G_variance=[variance([r["G"][b] for r in controllers]) for b in range(blocks)],
        loss_tail_head_ratio=mean(loss_tail)/max(mean(loss_head), 1e-12),
        nonfinite_or_state_failures=0,  # Only complete, assertion-validated epochs enter this function.
        amp_overflow_skips=sum(r.get("amp", {}).get("overflows", 0) for r in rows),
        invalid_diagnostic_batches=sum(r.get("amp", {}).get("invalid_diagnostics", 0) for r in rows),
        amp_interpretation="Recoverable GradScaler overflows are skipped updates, not model/state failures; invalid observations excluded",
        gradient_variance_definition="population variance of epoch means; within-epoch norm CV separately reported")


def feasibility_gate(summaries, statuses, settings, synthetic=False):
    s, fixed, c = summaries["response_aware"], summaries["fixed"], settings["gate"]
    checks = dict(
        real_full_horizon=not synthetic and all(r["epochs"] == settings["epochs"] for r in summaries.values()),
        all_completed=all(v == "complete" for v in statuses.values()),
        finite_and_restored=all(r["nonfinite_or_state_failures"] == 0 for r in summaries.values()),
        alpha_within_bounds=all(settings["controller"]["alpha_min"] <= a <= settings["controller"]["alpha_max"]
            for r in summaries.values() for a in r["alpha_min"]+r["alpha_max"]),
        bound_fraction=s["bound_hit_fraction"] is not None and s["bound_hit_fraction"] < c["max_bound_fraction"],
        no_block_pinned=all(v is not None and v <= c["max_block_pinned_fraction"] for v in s["block_pinned_fraction"]),
        final_accuracy=s["final_validation_top1"] >= fixed["final_validation_top1"]-c["max_final_accuracy_drop_pp"],
        response_measurable=s["tiny_response_fraction"] is not None and s["tiny_response_fraction"] < c["max_tiny_response_fraction"],
        overhead_reported=s["controller_overhead_fraction"] is not None,
        overhead_practical=s["controller_overhead_fraction"] is not None and s["controller_overhead_fraction"] < c["max_controller_overhead_fraction"],
        loss_stable=s["loss_tail_head_ratio"] <= c["max_train_loss_growth"],
        no_persistent_large_oscillation=s["large_oscillation_fraction"] is not None and
            s["large_oscillation_fraction"] < c["max_persistent_large_oscillation_fraction"])
    return dict(passed=all(checks.values()), checks=checks, failures=[k for k, ok in checks.items() if not ok],
        positive_indicators=dict(final_accuracy_above_fixed=s["final_validation_top1"] > fixed["final_validation_top1"],
            best_accuracy_above_fixed=s["best_validation_top1"] > fixed["best_validation_top1"],
            lower_mean_train_loss=s["mean_train_loss"] < fixed["mean_train_loss"],
            higher_mean_validation_accuracy=s["mean_validation_top1"] > fixed["mean_validation_top1"],
            lower_mean_G_variance=mean(s["train_G_variance"]) < mean(fixed["train_G_variance"])),
        interpretation="Single-seed online mechanics only. Not a final accuracy benchmark; no requirement to beat Fixed.")
