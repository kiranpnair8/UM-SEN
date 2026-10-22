# Controller Calibration (Seed 42 Only)

This is a closed, predeclared four-condition ablation, not a hyperparameter
search. No automatic winner is chosen. The model, SPS alpha=4, training recipe,
60-epoch prefix of the original 300-epoch schedule, train-split FP32/eval probe,
response estimator, update timing, alpha bounds [1,8], AMP persistence limits,
diagnostics and checkpoint pipeline are inherited unchanged.

| Variant | eta | max_step | Reference |
|---|---:|---:|---|
| current | 0.25 | 0.25 | Frozen five-epoch windows; existing result only |
| conservative | 0.10 | 0.10 | Frozen five-epoch windows |
| very_conservative | 0.05 | 0.05 | Frozen five-epoch windows |
| anchored | 0.10 | 0.10 | Persistent EMA reference, beta=0.90, five-epoch boundaries |

The existing `OnlinePolicy`/`LocalReference` already implement both reference
modes. Anchored starts at the first valid unperturbed G observation (epoch 0,
alpha=4), stays fixed inside a window, and updates at epochs 5,10,...:
`G_ref = 0.90 * G_ref + 0.10 * G_observed_boundary`. The boundary observation is
at the then-current alpha before that epoch's update, exactly as in the original
windowed policy. No validation metrics/labels or future results enter this rule.
Anchoring does not imply that G=0 is a target.

## Evidence And Isolation

`configs/gradient_flow/controller_calibration.yml` declares all four variants and
criteria. Code rejects changes outside this declaration. Before any real run,
the runner checks complete 60-epoch Fixed and response-aware source directories,
resolved original configs, original training signature, runtime package versions,
probe fingerprint and approved historical implementation commits. Missing or
incompatible sources fail closed; CURRENT is never silently rerun or copied.

The original `results/gradient_flow/controller_online_seed42/` is read-only.
Only metrics/config/status/provenance/event JSON are read; historical checkpoints,
compatibility markers and protected prefix files are never deserialized or written.
The existing legacy identity is recorded as historical, not relabeled with this
implementation. A new registration records actual implementation SHA, config hash,
training signature, runtime, batch fingerprint, safety settings, and source file
hashes. `current/reference.json` points to the completed result. Hashes are
rechecked before final aggregation. Existing per-epoch implementation SHAs remain
visible, including the legacy all-finite prefix and reviewed AMP corrections.

The original writer did not log per-epoch implementation commits. The known
Fixed epoch0-6 prefix is accepted ONLY when the original AMP compatibility marker,
protected JSON artifact hashes, original identity/config, and exact prefix rows
all agree. Missing provenance outside those seven Fixed rows is rejected. The
manifest records this as run-level/protected-prefix evidence, NOT as invented
per-epoch commits. Historical rows, markers and backups are never modified;
new calibration rows retain the strict per-epoch commit requirement.

New results use only `results/gradient_flow/controller_calibration_seed42/`.
The new root has registration and per-variant locks; identical variants cannot
run concurrently, but B/C/D can. Resume requires the same clean source SHA,
variant/config/runtime/probe identity and checkpoint-integrity evidence. It uses
the existing scoped trusted full-state resume, never an unsafe loading fallback.
Only completed variants are skipped. Failed variants need manual review; paused
variants resume from their saved epoch. Original online entry points/configs are
not redirected into calibration, and their legacy continuation restrictions are
not relaxed by this experiment.

## Commands On Lawrence

After pulling the validation branch, activate `snn` and create the SLURM log
directory before submission. Use the same data path spelling as the original
study's config. With an allocated GPU, optional preregistration checks source
compatibility and prints the observed CURRENT runtime, without building a model:

```bash
mkdir -p logs/gradient_flow
python -m experiments.gradient_flow.controller_calibration --register-only
sbatch --export=ALL,VARIANT=conservative jobs/gradient_flow/run_controller_calibration_condition.sh
sbatch --export=ALL,VARIANT=very_conservative jobs/gradient_flow/run_controller_calibration_condition.sh
sbatch --export=ALL,VARIANT=anchored jobs/gradient_flow/run_controller_calibration_condition.sh
```

No node is hardcoded. Each requests one GPU, 16 CPUs, 64G and at most two days,
using existing Lawrence partition/environment conventions. Node availability is
controlled by SLURM; add `--nodelist=...` only when needed. Do not change configs
or source between jobs. Logs include the variant/job ID. The original `online`
jobs must not be relaunched from this new calibration commit.

Tiny synthetic checks (NOT CIFAR evidence; two epochs x two batches):

```bash
python -m experiments.gradient_flow.controller_calibration --variant conservative --smoke --device cpu --backend torch
python -m experiments.gradient_flow.controller_calibration --variant very_conservative --smoke --device cpu --backend torch
python -m experiments.gradient_flow.controller_calibration --variant anchored --smoke --device cpu --backend torch
# On an allocated Lawrence GPU, same synthetic check with native AMP/CuPy:
python -m experiments.gradient_flow.controller_calibration --variant anchored --smoke --device cuda --backend cupy --output results/gradient_flow/controller_calibration_cuda_smoke_seed42
```

Each smoke directory also refuses restart/overwrite; use `--resume` only with
the same implementation/config/runtime. Synthetic data cannot pass real criteria
or be substituted for CURRENT. No smoke writes to the completed study.

After ALL B/C/D complete:

```bash
python -m analysis.gradient_flow.plot_controller_calibration
```

The command refuses incomplete/provenance-mismatched variants BEFORE publishing
any aggregate summary. It writes summary JSON/text in the calibration result root
and `figures/gradient_flow/controller_calibration_seed42/`, with PNG (300 DPI)/PDF
figures; it never runs probes or training.

## Measurement And Interpretation

`train_G_variance` is population variance **across epoch-mean G**, per block;
`probe_G_variance` is variance across epoch-boundary probe G. Neither is
within-minibatch gradient variance. The cross-block mean is an unweighted mean
of these per-block variances. Alpha movement sums absolute applied steps;
deviations use all post-controller epoch alphas relative to 4. The initial alpha
is explicitly recorded as 4. Update distributions include scheduled zero steps.
Response sign changes ignore tiny/zero signs, matching the inherited summary.
Pinned fraction is the fraction of scheduled observations at a bound, per block,
not duration of consecutive saturation. Oscillations retain the existing
within-window error-crossing definition; reference updates change windows.

Recoverable AMP skips are counted, not called persistent instability. Persistent
failure events or incomplete runs cannot be labeled promising. Criteria are:
60 completed real epochs; no persistent instability; **zero bound-hit observations**
(a conservative definition of no saturation); controller time / epoch wall time
<20%; at least 25% epoch-level train G variance reduction; final top1 no worse
than Fixed minus 0.5 pp. Stronger indicator additionally requires >=40% reduction
and both final and best top1 no worse than Fixed minus 0.25 pp. Criteria do not
require beating Fixed. Zero Fixed variance makes reduction undefined and fails
the variance criterion rather than fabricating an improvement.

`stabilization_per_accuracy_cost` divides reduction by
`max(abs(mean_validation_accuracy_delta_pp),0.1)`; descriptive only, never an
objective or selection rule. This seed42 ablation cannot establish superiority.
Essentially tied results should favor the simpler/more conservative policy for
the next multiseed study. No thresholds or extra variants should be added after
seeing results. Wall time should be estimated from the recorded CURRENT run,
not invented from synthetic smoke; checkpoint I/O is excluded from epoch timing.
