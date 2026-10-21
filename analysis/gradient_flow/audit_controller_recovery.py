"""Explain/rebuild completed recovery summaries from JSON only. Never run probes."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

from .plot_controller_recovery import ROOT, load_saved_records, plot


def explain_group(record, expected_batches, criteria):
    """Named diagnostics for the existing conjunctive gate; not a new policy."""
    ratio = record["mean_final_ratio"]
    checks = {
        "expected_batch_count": (record["count"], expected_batches, record["count"] == expected_batches),
        "all_batches_informative": (record["informative"], record["count"], record["informative"] == record["count"]),
        "mean_final_ratio_at_most": (ratio, criteria["max_mean_final_ratio"], ratio is not None and ratio <= criteria["max_mean_final_ratio"]),
        "fraction_recovered_at_least": (record["fraction_recovered"], criteria["min_recovered_fraction"], record["fraction_recovered"] >= criteria["min_recovered_fraction"]),
        "zero_bound_hit_fraction": (record["bound_hit_fraction"], 0., record["bound_hit_fraction"] == 0),
        "zero_oscillation_fraction": (record["oscillation_fraction"], 0., record["oscillation_fraction"] == 0),
        "zero_diverged": (record["diverged"], 0, record["diverged"] == 0),
    }
    return {name: dict(actual=a, required=b, passed=bool(ok)) for name, (a, b, ok) in checks.items()}


def audit_gate(summary, settings):
    groups = {}
    for state, disturbances in summary["groups"].items():
        for name, methods in disturbances.items():
            checks = explain_group(methods["response_aware"], settings["num_batches"], settings["gate"])
            groups[f"{state}/{name}"] = dict(checks=checks,
                failed_checks=[key for key, value in checks.items() if not value["passed"]])
    return groups


def audit_oscillation(case, tolerance):
    row = case["methods"]["response_aware"]
    counts, intervals = [], []
    for block in range(len(row["trace"][0]["error"])):
        points = [(p["iteration"], p["error"][block]) for p in row["trace"] if abs(p["error"][block]) > tolerance]
        crossings = [[a[0], b[0]] for a, b in zip(points, points[1:]) if (a[1] > 0) != (b[1] > 0)]
        counts.append(len(crossings))
        intervals.append(crossings)
    matches = counts == row["error_sign_crossings"] and any(c >= 2 for c in counts) == row["oscillation_indicator"]
    return dict(state=case["state"], batch=case["batch"], disturbance=case["disturbance"],
        counts=counts, crossing_iteration_pairs=intervals, stored_flag=row["oscillation_indicator"],
        definition_matches=matches, final_ratio=row["final_normalized_error"],
        l2_nonincreasing=all(b["error_l2"] <= a["error_l2"] for a, b in zip(row["trace"], row["trace"][1:])))


def compare_values(original, rebuilt, path=""):
    """Record exact differences, distinguishing summation roundoff from discrepancies."""
    differences = []
    if isinstance(original, dict) and isinstance(rebuilt, dict):
        for key in sorted(original.keys() | rebuilt.keys()):
            if key not in original or key not in rebuilt:
                differences.append(dict(path=path+"/"+key, within_tolerance=False, reason="missing key"))
            else:
                differences.extend(compare_values(original[key], rebuilt[key], path+"/"+key))
    elif isinstance(original, list) and isinstance(rebuilt, list) and len(original) == len(rebuilt):
        for i, (a, b) in enumerate(zip(original, rebuilt)):
            differences.extend(compare_values(a, b, path+"/"+str(i)))
    elif original != rebuilt:
        numeric = type(original) in (float, int) and type(rebuilt) in (float, int)
        differences.append(dict(path=path, original=original, rebuilt=rebuilt,
            within_tolerance=numeric and math.isclose(original, rebuilt, rel_tol=1e-12, abs_tol=1e-12)))
    return differences


def rebuild(run_dir, output_dir, figures_dir=None):
    # Import only the pure aggregation entry point. No model construction,
    # checkpoint load, backward pass or run_study call occurs in this path.
    from experiments.gradient_flow.controller_recovery import aggregate
    run_dir, output_dir = Path(run_dir).resolve(), Path(output_dir).resolve()
    if output_dir.is_relative_to(run_dir) or run_dir.is_relative_to(output_dir):
        raise ValueError("Audit output must be separate from the original result tree")
    if figures_dir is not None:
        figures_dir = Path(figures_dir).resolve()
        if figures_dir.is_relative_to(run_dir) or run_dir.is_relative_to(figures_dir):
            raise ValueError("Audit figures must be separate from the original result tree")
    metadata, original, cases = load_saved_records(run_dir)
    expected = 3*metadata["settings"]["num_batches"]*len(metadata["settings"]["disturbances"])
    identities = {(c["state"], c["batch"], c["disturbance"]) for c in cases}
    if len(cases) != expected or len(identities) != expected:
        raise ValueError("Missing or duplicate recovery cases")
    for case in cases:
        if (case["batch_fingerprint"] != metadata["batch_fingerprints"][case["batch"]] or
                case["disturbance"] not in metadata["settings"]["disturbances"]):
            raise ValueError("Saved case does not match input/protocol identity")
    rebuilt = aggregate(cases, metadata["settings"], metadata.get("synthetic", False))
    differences = compare_values(original, rebuilt)
    oscillation = [audit_oscillation(c, metadata["settings"]["controller"]["zero_tolerance"]) for c in cases]
    if any(not row["definition_matches"] for row in oscillation):
        raise ValueError("Stored oscillation flags do not match their documented definition; manual review required")
    if any(not row["within_tolerance"] for row in differences):
        raise ValueError("Rebuilt summary substantively differs from saved output; manual review required")
    report = dict(source_run=str(run_dir), original_summary_sha256=hashlib.sha256((run_dir/"summary.json").read_bytes()).hexdigest(),
        source_commit=metadata["source"].get("commit"), audit_python=sys.version,
        audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        aggregation_script_sha256=hashlib.sha256(Path(aggregate.__code__.co_filename).read_bytes()).hexdigest(),
        validated_batches=3*metadata["settings"]["num_batches"], cases=len(cases),
        optimizer_updates=0, gpu_probes_run=0, gate_unchanged=original["gate"] == rebuilt["gate"],
        numeric_comparison=dict(exactly_equal=not differences, differences=differences,
                                relative_tolerance=1e-12, absolute_tolerance=1e-12),
        groups=audit_gate(rebuilt, metadata["settings"]),
        flagged_episodes=[row for row in oscillation if row["stored_flag"]],
        conclusion="No gate bug demonstrated: recovery and zero-oscillation requirements are separate conjunctive checks; no criterion relaxed.")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir/"summary.json").write_text(json.dumps(rebuilt, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    (output_dir/"gate_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    lines = [report["conclusion"], "Gate remains: "+str(rebuilt["gate"]["passed"]),
             f"Recomputed from {len(cases)} saved cases; no GPU probes or optimizer updates.",
             "Floating-point tolerance applies ONLY to summary comparison, never to gate thresholds."]
    for name, group in report["groups"].items():
        if group["failed_checks"]:
            lines.append(name+": "+repr({k: group["checks"][k] for k in group["failed_checks"]}))
    for episode in report["flagged_episodes"]:
        lines.append(repr(episode))
    (output_dir/"gate_audit.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    if figures_dir is not None:
        plot(run_dir, figures_dir, output_dir/"summary.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="results/gradient_flow/controller_recovery_seed42")
    parser.add_argument("--output-dir", default="results/gradient_flow/controller_recovery_gate_audit_seed42")
    parser.add_argument("--figures-dir", default="figures/gradient_flow/controller_recovery_gate_audit_seed42")
    args = parser.parse_args()
    report = rebuild(ROOT/args.run_dir, ROOT/args.output_dir, ROOT/args.figures_dir)
    print(report["conclusion"])
    for name, group in report["groups"].items():
        if group["failed_checks"]:
            print(name, group["failed_checks"])


if __name__ == "__main__":
    main()
