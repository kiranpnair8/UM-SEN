# SAGE Training-State Alpha-Response Characterization

This is H4 characterization, not an adaptive controller or a test that G=0 is
optimal. No positive sign, state-dependent gain or efficacy result is assumed.
All probe mechanics and G remain in the existing intervention/multi-batch engine.

## Initial Audit and Authoritative Recipe

The local workspace inspected for this implementation does NOT contain
`results/umsen_official_recipe/baseline_seed42/`. Its HPC-only checkpoints,
`args.yaml`, summaries and provenance could not be inspected locally. Whether
regeneration is needed is therefore UNKNOWN until the Lawrence audit runs.
No missing checkpoint is fabricated, inferred from a filename, or interpolated.

`checkpoint_stages.py` inventories actual checkpoint dictionaries recursively,
reports hashes, raw/completed epoch semantics, embedded arguments, model keys,
optimizer/scaler/scheduler/RNG presence, best/current metrics, summary and provenance.
It uses historical `args.yaml` as authoritative, comparing full embedded recipes
(excluding explicit operational/output and logging/checkpoint-frequency fields).
Target-epoch files with conflicting recipe/provenance abort for manual review,
rather than triggering unnecessary automatic regeneration. Missing authoritative args aborts;
it does NOT silently fall back to the approximate recipe from a prompt.

Requested early/mid epochs are completed zero-based epochs 20 and 150. Late is
the final epoch of the resolved timm scheduler, not best-validation epoch. The
repository YAML currently says epochs=300 PLUS cooldown_epochs=10; the tested timm
schedule resolves to 310 epochs, so late would be 309, not 299. Actual HPC args and
installed scheduler determine the value and it is printed/stored everywhere.
Version2 timm stores the completed epoch; version1 stores the next epoch, so the
audit subtracts one. Unversioned epochs are not guessed/automatically selected.

Historical targets are reused only when all three actual epochs exist under the
same original run directory with consistent seed/recipe and trajectory IDs if
present. Different state paths/hashes are mandatory. `model_best` is eligible ONLY
if its actual completed epoch matches the requested target; it is never a fallback
for final. Legacy same-directory provenance is evidence of one run, not a
cryptographic proof of initial weights/data order; this limitation is recorded.
Conflicting IDs/recipes abort or prevent selection. Ambiguous provenance needs
manual review before attributing any result to the original optimization trajectory.

## Missing Historical Milestones

The job explicitly authorizes `--regenerate-missing`. If any target is unavailable,
ALL three primary states come from ONE new from-scratch Fixed-SG trajectory under
this study's own `trajectory/fixed_seed42/`, never a mixture with historical late.
No historical checkpoint is moved, overwritten, or used to initialize regeneration.

The existing official CIFAR trainer, loaders, augmentation, optimizer, scheduler,
loss, AMP and checkpoint/resume implementation are reused. Regeneration retains
the authoritative recipe and seed42, adds deterministic mode, fixed transformer
alpha4/SPS4, new output paths, and a recorded trajectory ID. Deterministic mode
follows the existing validated single-process seeding/loader setup (including no
persistent workers). It is an explicitly NEW trajectory, NOT a claim to reproduce
historical bitwise weights or historical data ordering. Recipe differences are
recorded, and all primary checkpoints are internally from that same new trajectory.

`train_milestones.py` is an isolated wrapper. No shared trainer or SAGE-v2 source is
edited. With no milestone epochs the saver class is returned unchanged. At selected
epoch boundaries it copies the enriched full checkpoint before timm publishes last:
`checkpoint_epoch020.pth.tar`, `checkpoint_epoch150.pth.tar`, and the resolved final
epoch file. Recovery/minibatch saves are not milestones. Copies draw no RNG, perform
no optimizer/scheduler steps, and retain model, optimizer, scaler, scheduler, best
metric, RNG/loaders and args as saved by the existing full-state saver. Unit tests
verify RNG, parameters and LR unchanged and payload equality with `last.pth.tar`.
Milestones are independent copies and survive top-k checkpoint pruning.

Resubmission resumes only this regeneration's `last.pth.tar`, with its matching
trajectory ID and epoch-boundary RNG/scheduler/optimizer/scaler state. It does not
overwrite restored per-group LR. No warm-start, no-resume-opt, start-epoch override,
multi-epochs loader, EMA or distributed run is allowed in this wrapper. Missing
already-past milestones abort for manual review/new study rather than inventing
them from later weights. Interrupted training never proceeds to probes until all
targets exist. If the job ends at the two-day limit, resubmit unchanged.

When regeneration completes, `trajectory_validation.json` compares configuration,
seed, final/best validation summaries, target-final weight/buffer equality and, on
CUDA when both target-final checkpoints exist, fixed-test-batch logits/predictions/
loss. Discrepancies are recorded without asserting equivalence. CUDA/CuPy and
historical environment nondeterminism prevent a blanket bitwise guarantee. This
comparison does NOT authorize mixing old/new checkpoints.

## Matched State Probes and Statistics

One deterministic traversal creates the 25 non-overlapping test batches (size32,
indices0..799, seed42, no augmentation). The exact tensor batches are reused in all
states. Their indices and content fingerprints are in root metadata and independently
checked against every per-state completed batch. Each input is also saved per-state.
All states use eval/FP32/CUDA/CuPy, alpha2,3,4,5,6, reference4, SPS4, no optimizer
steps, and R_ij=(G_i(5)-G_i(3))/2 (central finite difference, not exact derivative).
All raw curves, invariance checks, per-state statistics/plots and safe batch resume
semantics from the previous multi-batch protocol remain active.

Comparisons retain early/mid/late means, SDs, CIs, sign consistency, raw curves,
G(alpha4), diagonal trajectories and all structural ratios. Raw paired R changes
are retained for mid-early, late-mid, late-early. Paired whole-batch percentile
bootstrap intervals (10,000 draws, seed42) are computed from batchwise DIFFERENCES,
including every matrix/diagonal entry, mean absolute diagonal and nearest-upstream
changes. Ratios use only pairs with defined denominators and report exclusions.
These intervals are descriptive conditional on one training trajectory and a
deterministic test subset, NOT automatic significance tests.

The effective-zero tolerance remains 1e-8. Normalized mean change is
(after-before)/(abs(before)+1e-12), ONLY when abs(before)>1e-8; otherwise null.
Raw changes always remain available. Frobenius norms cover the full mean matrices
and their differences. Optional cosine similarity uses the FIXED i<=j mask, excluding
the structural lower triangle rather than selecting favorable entries from data;
downstream responses are still reported separately. Mean-sign and paired sign-change
fractions are descriptive. Sampled nonmonotonic diagonal trajectories refer to the
three measured mean operating points, not a fitted training dynamics model.

## Paths and Lawrence Job

All output is separate from prior single/multi-batch runs:

```text
results/gradient_flow/alpha_state_dependence_seed42/
    checkpoint_inventory.json, checkpoint_inventory_latest.json
    trajectory_plan.json, regeneration.yml (only if needed)
    trajectory/fixed_seed42/          # new training artifacts only if necessary
    trajectory_validation.json       # only after regeneration completes
    states.json                      # only after actual three checkpoints verify
    probes/
        metadata.json, status.json
        early/, mid/, late/          # existing full multi-batch output schema
        aggregate/
            summary.json, state_response_{mean,std,ci95}.json
            state_differences.json, state_structural_metrics.json
            state_raw_measurements.{csv,jsonl}
figures/gradient_flow/alpha_state_dependence_seed42/
```

Per-state/batch outputs are hash-validated and skipped on resume; corrupted/incomplete
outputs are rerun through the original engine. All states must complete before
combined aggregation/figures. Source/settings/manifest changes require a new study
namespace; do not run concurrent writers. Working source must be clean. The plan
locks the selected historical vs regenerated origin across restarts.
The SLURM launcher additionally requires `flock` and holds a repository-local
advisory lock for its entire workflow; duplicate concurrent submissions fail
without writing trajectory data. The lock is automatically released on exit.

The launcher explicitly requests ONE GPU on gpu006, partition gpu, 16 CPUs, 64G,
maximum two days, and the verified existing snn environment. It checks hostname and
dirty status, prints hardware/Git/Python details, audits first, conditionally trains,
probes, analyzes and prints elapsed time. No account/QoS/module is invented.

```bash
# From the clean SAGE repository root after pulling the validation branch:
mkdir -p logs/gradient_flow
JOBID=$(sbatch --parsable jobs/gradient_flow/run_alpha_state_dependence_cifar10.sh)
squeue -j "$JOBID"
tail -f "logs/gradient_flow/alpha_states_${JOBID}.log"
```

Resubmit the same command to resume. `RUN_DIR`, `DATA_DIR`, `HISTORICAL_DIR`,
`SNN_ENV` and `FIGURE_DIR` can be exported at submission. The script never writes
inside the historical baseline directory. Common limits are shared across mean
heatmaps, and separately across difference heatmaps. Matplotlib exports PNG/PDF
and a descriptive summary; no controller or H4-confirmation claim is generated.
At the user-reported ~205s/25-batch V100 probe rate, three states would be roughly
615s plus audit/I/O/analysis when NO regeneration is needed. Retraining cost is not
estimated without the HPC timing evidence. No training/full study is launched locally.
