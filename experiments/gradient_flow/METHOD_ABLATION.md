# Selective / Regularized Method Ablation

Exactly three seed42 CIFAR10 designs, 60 epochs each, on the same Spikformer
4-384/T4 backbone. This remains a prefix of the inherited 300-epoch cosine
recipe, not a new 60-epoch schedule. Architecture, SPS alpha=4, training data,
optimizer, augmentation/loss, AMP skip/persistence policy, probe batch,
finite-difference response estimator, epoch-boundary timing and five-epoch frozen
reference are unchanged. Alpha bounds remain [1,8]. No existing experiment is
rerun, moved, relabeled or edited.

| Design | Controlled blocks | eta/max_step | restoring gamma | deadband |
|---|---|---|---|---|
| deep_only | 1,2,3 (block0 exactly 4) | .10/.10 | 0 | none |
| regularized | all | .10/.10 | .05 | none |
| deadband_regularized | all, conditional | .10/.10 | .05 | one frozen global value |

The inherited inverse step is `u=-eta*e*R/(R*R+1e-4)`. The `1e-4` inverse
denominator damping is unchanged; "no regularization" in DEEP_ONLY means no
restoring force, not removal of this numerical damping. First compute this
UNCLIPPED step. Then add `-gamma*(alpha-4)`. Then mask the entire requested step
for block0 in DEEP_ONLY or when `abs(e)<deadband` in DEADBAND_REGULARIZED.
Equality at the threshold is ACTIVE. Finally apply the sole max-step clamp,
then the alpha-bound clamp. Restoration is suppressed inside the deadband too.
The epoch0 unperturbed reference initializes at alpha4. All four responses are
still measured, including frozen block0, preserving the original probe budget.

## Deadband Preregistration (Before Any GPU Run)

The completed raw error statistics are HPC-only; no numeric empirical deadband
is fabricated in this checkout. The tracked config predeclares one deterministic
rule: the linearly interpolated 25th percentile of positive absolute CURRENT
training-probe errors, pooled across blocks/epochs, excluding reference-refresh
epochs. Boundary zeros are structural, not a useful empirical small-error scale.
The lower quartile deliberately suppresses small errors without selecting a
threshold from validation accuracy. This is a scale heuristic, not an estimated
noise confidence interval. No calibration-variant outcome is used to tune it.

An explicit preparation command validates all reference hashes/configs/identities,
computes this ONE numeric value, prints its count/min/max/rationale, and freezes
it in `study_manifest.json` and all resolved variant configs BEFORE jobs start.
All three designs share that declaration; only C uses the mask. Runs recheck the
same frozen value and source hashes; no per-run/per-block refit is allowed.
Changing the rule/number after seeing GPU outcomes is outside this preregistered
study. Synthetic smoke uses a clearly labeled .01 fixture threshold instead and
cannot enter real aggregation.

References are completed Fixed/CURRENT plus conservative/very_conservative.
They are read-only JSON evidence. The known legacy Fixed epoch0-6 provenance
exception still requires protected-prefix/compatibility hashes. Anchored is not
rerun and is not required by this comparison. No historical checkpoint is loaded.

## Lawrence Commands

After pulling the validation branch, activate `snn` in UM-SEN and run:

```bash
mkdir -p logs/gradient_flow
python -m experiments.gradient_flow.method_ablation --prepare-only
# REVIEW the printed frozen deadband and manifest before submitting:
sbatch --export=ALL,VARIANT=deep_only jobs/gradient_flow/run_controller_method_ablation_condition.sh
sbatch --export=ALL,VARIANT=regularized jobs/gradient_flow/run_controller_method_ablation_condition.sh
sbatch --export=ALL,VARIANT=deadband_regularized jobs/gradient_flow/run_controller_method_ablation_condition.sh
```

Use the original `--data-dir` spelling for preparation; export `DATA_DIR` with
that same spelling for jobs if not `data`. Each has an independent directory,
status and lock in `results/gradient_flow/controller_method_ablation_seed42/`.
The three can run simultaneously; training never writes the shared manifest or
other condition's outputs. The default runtime/policy extension hooks exactly
preserve the original Fixed/CURRENT/calibration paths. Jobs use the existing gpu
partition/snn environment, one GPU, 16 CPUs,64G, <=2 days; no node is hardcoded.
Expected time is approximately the previously observed CURRENT ~1.72 hours per
condition, plus checkpoint I/O and hardware/alpha-kernel-cache variation. No new
probe budget is added. Pause resumes the exact saved states; failed runs require
manual review. Changing implementation commit after preparation is refused.

CPU smoke for each design (2 epochs x 2 batches, synthetic only):

```bash
python -m experiments.gradient_flow.method_ablation --variant deep_only --smoke --device cpu --backend torch
python -m experiments.gradient_flow.method_ablation --variant regularized --smoke --device cpu --backend torch
python -m experiments.gradient_flow.method_ablation --variant deadband_regularized --smoke --device cpu --backend torch
# Optional allocated GPU smoke; no real training dataset involved:
python -m experiments.gradient_flow.method_ablation --variant deadband_regularized --smoke --device cuda --backend cupy --output results/gradient_flow/controller_method_ablation_cuda_smoke_seed42
```

After all THREE real conditions complete:

```bash
python -m analysis.gradient_flow.plot_controller_method_ablation
```

This writes summary JSON/text and eight PNG/PDF figure types in the new result
root and `figures/gradient_flow/controller_method_ablation_seed42/`. Incomplete
conditions cannot finalize. G variance is population variance across epoch-mean
G, NOT across minibatches. Mean block variance is unweighted. Existing logging
definitions (response signs/tiny threshold, within-window oscillation, AMP skips,
invalid observations, controller/epoch time) are retained. Alpha movement sums
absolute applied steps; post-update alpha deviations are relative to4. DEEP_ONLY
checks block0 exactly4 in every row, at update time, and on state restoration.

Targets remain >=25% variance reduction and final top1 >=Fixed-0.5pp; strong
target >=40%, final AND best >=Fixed-0.25pp. Mechanical eligibility additionally
requires complete finite/restored training, no persistent failure, no bound-hit
observations, alpha in [1,8], and overhead<20%. Rank accuracy-eligible mechanically
stable designs by decreasing variance reduction; exact ties prefer simpler then
lower overhead. No combined score is invented, and ranking does NOT choose a
FINAL_SAGE automatically. This is seed42-only evidence.

## Prepared Follow-ups: DO NOT Launch Until Review

`controller_followup.yml` and `controller_followup.py` prepare seeds42/43/44,
Fixed vs configurable FINAL_SAGE, for CIFAR10 and CIFAR100. Horizon is explicitly
60 epochs, retaining the inherited schedule. CIFAR100 uses the existing CIFAR
trainer with 100 classes and real CIFAR100 probe data, NOT CIFAR10 probe data.
Its preprocessing/recipe initially inherit CIFAR10, as in the existing repo's
CIFAR100 comparison wrapper; it is NOT a published CIFAR100 baseline reproduction.
That dataset recipe requires explicit acknowledgment before approval. These
prepared choices can be reviewed before selection, not silently changed mid-run.
The selected deadband is carried unchanged to new seeds/datasets, never refitted.

After the complete ablation has been scientifically reviewed (NOT now):

```bash
python -m experiments.gradient_flow.controller_followup --approve-selection --final-sage CHOSEN_VARIANT --reviewed-by YOUR_NAME --acknowledge-cifar100-recipe
sbatch --export=ALL,DATASET=cifar10,FINAL_SAGE=CHOSEN_VARIANT jobs/gradient_flow/run_controller_followup.sh
sbatch --export=ALL,DATASET=cifar100,FINAL_SAGE=CHOSEN_VARIANT jobs/gradient_flow/run_controller_followup.sh
```

Each array maps tasks0..5 to Fixed42,Final42,Fixed43,Final43,Fixed44,Final44.
Without an explicit approved selection, every follow-up refuses BEFORE loaders
or outputs are created. Source summary, manifest, config, selection hashes and
actual commit are pinned. Outputs are isolated under
`results/gradient_flow/controller_followup_cifar10/seed42/...` (analogously100).
Paired seed/dataset conditions share the exact training/probe recipe. Optional
`--summarize --dataset cifar10` (or100) reports mean/sample std only after all six
runs complete. No selection file or follow-up result is generated in this task.
