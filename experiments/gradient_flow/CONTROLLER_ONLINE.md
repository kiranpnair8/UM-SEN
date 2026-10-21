# Short Online-Training Feasibility

Question: does response-aware regulation remain mechanically stable as weights
change, relative to Fixed alpha and matched random alpha directions? This is a
single-seed optimization-dynamics experiment, not a paper-scale benchmark or an
accuracy-improvement claim. The frozen recovery results/gates are unchanged.

## Protocol

- CIFAR-10, seed 42, from scratch in all conditions; identical initialization.
- Released CIFAR Spikformer: T=4, four blocks, dim384, 12 heads, patch4, MLP4.
- Reuse `spikformer/cifar10/train.py` model construction, loaders, RandAugment,
  Random Erasing, Mixup, smoothing/SoftTargetCrossEntropy, AdamW, native AMP,
  clipping setting, validation, LR scheduler and timm checkpoint saver.
- Exactly epochs 0--59 of the inherited 300-epoch training recipe (20-epoch
  warmup, 10-epoch cooldown configured). **Do not compress cosine decay to 60.**
  The official batch128, LR0.0005, min/warmup LR0.00001, weight decay0.06,
  Mixup0.5, CutMix0, smoothing0.1 and other augmentation defaults are inherited.
- Deterministic DataLoader generators and nonpersistent workers use the existing
  official paired-seed path. Custom CuPy kernels are not guaranteed bitwise
  deterministic across machines. No second dataset/seed/architecture is run.
- Transformer alpha starts at4; SPS alpha stays4; alpha is not trainable.

## Controller and reference

`OnlinePolicy` calls the EXISTING `ResponseController`, `LocalReference` and
`apply_steps`; it does not reimplement their control equation. Defaults are in
`configs/gradient_flow/controller_online.yml`: delta0.5, eta0.25, inverse-response
regularizer0.0001, max step0.25, bounds[1,8]. Only the measured diagonal local
response is used; cross-block coupling is not inverted.

At the START of each epoch (one opportunity/epoch, never every minibatch):

1. Use the first sequential32 **training** samples with deterministic official
   eval preprocessing, FP32/eval and hard training targets. This is not a new
   train/validation split: these samples remain in the unchanged training set.
   The same probe batch/fingerprint is used across conditions and epochs.
2. For response-aware/random conditions, use `OnlineProbe -> FrozenProbe ->
   intervention` to measure G and symmetric per-block responses. At interior
   points each calibration uses10 FP32 forward/backwards: two reproducibility
   checks and eight +/- trials. No optimizer updates occur. Near bounds use the
   existing symmetric bounded delta and record its actual value; a point exactly
   at a bound cannot be calibrated and fails closed, rather than silently using
   a one-sided derivative. Fixed uses three observation-only passes, no response.
3. At epochs0,5,10,... refresh the frozen reference to the CURRENT training-probe
   G; hold that reference for the next five epochs. In particular, epoch0 has zero
   error and no alpha change. This is **not a G=0 target**. The implemented optional
   `ema_window` mode applies the existing beta0.99 only at window boundaries;
   it does not instantly track each observation.
4. Apply one existing regularized response-aware update; use the resulting alpha
   for every training batch in this epoch. Persist both alpha-before and alpha-after.

RANDOM_MATCHED runs AFTER RESPONSE_AWARE and replays its SAME epoch/per-block
absolute **applied** step magnitudes. A private Python RNG (seed42+1701) chooses
independent signs from directions that fit within the same bounds. At a bound
only the feasible direction is eligible; log this constraint rather than claiming
unconstrained directions. Magnitudes must match within1e-12 or abort. No accuracy,
validation labels, future checkpoint or future reference is consulted. Reading the
completed schedule is only for the paired magnitude-budget control. Its source
metrics hash and exact config/commit/batch identities are recorded.

The fixed condition executes no alpha updates (`update_count=0`), but the same
passive training instrumentation and restored training-probe observations. A
scheduled opportunity with zero error still counts as an attempted update for
the other two conditions. This convention is also used for bound fractions.

## State and AMP

The existing intervention engine restores parameters, BN buffers, all per-module
modes, neuron memories, gradients, alpha and Python/NumPy/Torch/CUDA RNG after
each trial; exact runtime comparisons are retained. `OnlineProbe` additionally
asserts optimizer, scheduler and native AMP scaler state equality. External
training hooks are inactive during probes. All probes explicitly disable AMP.

The training adapter scales/backwards, unscales ONCE, measures gradients BEFORE
clipping/step, checks finiteness/safety, then performs the unchanged clip and
optimizer/scaler updates. Activation hooks divide norms by the forward loss
scale. G uses the existing epsilon definition. No gradient is replaced by hooks.
Neurons reset after every official train/validation batch and every probe.

Per-batch observations are detached Python numbers, reduced to epoch means/CVs,
then cleared. Only checkpoint diagnostic EMA references retain detached gradient
tensors. Training top1/top5 are **expected target mass in top-k** under official
Mixup/smoothing, not hard-label CIFAR accuracy; validation top1/top5 use hard labels.

## Outputs and continuation

New namespace: `results/gradient_flow/controller_online_seed42/{fixed,response_aware,random_matched}/`.
Each condition writes config/resolved arguments/provenance, metrics JSON/CSV,
controller metrics (including raw calibration), timing, summary and status, plus
official `last.pth.tar` and `model_best.pth.tar`. Metric histories contain no tensors.
The final checkpoint is `last.pth.tar`; checkpoints are saved every epoch.

The existing timm+repository saver retains optimizer/scaler/epoch/best/scheduler
and loader RNG. `gradient_flow_online` adds policy/private RNG/reference/history,
monitor EMA/step and detached epoch records. Optimizer LR is NOT overwritten on
resume. Restore is epoch-boundary only; a partial epoch restarts from the last
published checkpoint. Checkpoint history is authoritative after interruption.
Source commit, resolved configuration, backend, method and batch fingerprint must
match exactly; no legacy checkpoint initialization is silently substituted.
Completed conditions are skipped only with matching provenance and output files.

On a SLURM walltime warning, Python requests a stop before the next epoch. A
checkpoint already published for the previous epoch is retained. Exit75 means
paused, not permission to tune/retry automatically. A hard kill still resumes
from the last fully published epoch. No automatic SLURM submission is implemented.

## Predeclared descriptive gate

`online_summary.py` reads only complete, assertion-validated records. The committed
config fixes all thresholds BEFORE the real run:

- All three real conditions completed60 epochs, no NaN/Inf/restoration failures.
- Alpha remains in configured bounds; response-aware overall update/block bound
  fraction <0.20, each block pinned for at most25% of update opportunities.
- Response-aware final validation accuracy is within1.0 percentage point of Fixed
  or higher. Beating Fixed is NOT required.
- Tiny response means abs(Rhat)<=0.0001. Fewer than50% of measured response
  components may be tiny; controller overhead must be reported and <20%.
- Mean last-five/first-five epoch train loss ratio <=2.0.
- Fewer than25% of update opportunities have BOTH a repeated within-window
  component-error crossing (>=2) AND error L2>1.0. These are descriptive crossings
  in a changing-weight system, not a proof of closed-loop instability. Reference
  refreshes do not count as crossings. The old frozen-study gate is NOT changed.

Summaries separately report epoch-level loss/accuracy variance, variance of their
epoch-to-epoch differences, block gradient-norm CV (epoch means and within-epoch),
G variance, response sign distribution/changes, tiny-response/bound fractions,
absolute response and update-magnitude distributions. Positive indicators (mean
loss, mean/best/final validation top1, G variance) do not change the hard gate.
Synthetic smoke results cannot pass this real-experiment gate.

NaN/Inf, unsafe gradient norms/G, alpha/state/provenance violations abort before
further optimizer updates where detected. A catastrophic run stops the launcher;
no hyperparameter tuning or replacement run is launched automatically.

## Commands

CPU plumbing smoke (no downloads, tiny model, two batches x two epochs):

```bash
python -m experiments.gradient_flow.online_train --smoke --condition response_aware --device cpu --backend torch
```

Use a NEW `--output results/gradient_flow/<smoke-id>` if rerunning with changed
code/config, or `--resume` only for identical provenance. To test plotting, run
fixed, response_aware, random_matched sequentially under the SAME smoke root,
then point the plotting command at that root. For a GPU smoke use explicit
`--device cuda --backend cupy`; real defaults never silently fall back to Torch.

Lawrence (already synchronized validation branch, clean tree):

```bash
git switch gradient-flow-stage1-validation
git pull --ff-only origin gradient-flow-stage1-validation
mkdir -p logs/gradient_flow
sbatch jobs/gradient_flow/run_controller_online_cifar10.sh
```

The launcher uses the existing gpu partition/gpu002/snn environment conventions,
one GPU,16 CPUs,64G, max48hours, no invented account/QoS. Resubmit the SAME command
after a walltime stop; matching completed conditions skip, partial ones resume.
Each condition has its own stdout/stderr log, plus the launcher log under
`logs/gradient_flow/`. `DATA_DIR`, `SNN_ENV`, `RUN_DIR`, `FIGURE_DIR` can override
runtime locations; do not change RUN_DIR just to hide a failure.

Saved-output-only analysis:

```bash
python -m analysis.gradient_flow.plot_controller_online
```

Produces PNG/PDF curves, response-sign heatmap, bound/crossing diagnostics and a
compact panel under `figures/gradient_flow/controller_online_seed42/`, and
summary JSON/TXT both there and at the new results root. It refuses mixed
config/commit/batch identities and incomplete conditions. No previous experiment
directory is read or overwritten by this new launcher.

## Runtime Evidence and Limitations

The supplied completed frozen audit (source e07e5ea) contains1080 response-aware
calibrations: mean2.905s, median3.097s, range1.645--3.910s. Its90 terminal
observation-only calls average0.978s. This suggests about174s calibration time
over60 online epochs per adaptive/random condition, plus new online optimizer
snapshot/assertion costs and kernel compilation; Fixed observations about59s.
These are measured frozen-probe timings, NOT measured online epoch timings.

The supplied official-training excerpt reports3.740s/batch x390 batches, or
24.31minutes/epoch before validation. At that conservative rate,60 epochs cost
about24.3hours per condition, or73hours for all three, plus common passive
instrumentation/checkpoint I/O. That excerpt came from historical UM-SEN and is
NOT a controlled Fixed or verified current-V100 throughput measurement. Local
HPC-only complete official logs are unavailable; a tighter estimate is unsupported.
The 48-hour SLURM cap is therefore respected with resumable allocations, NOT a
promise all three conditions fit in one submission. Use the first real epochs'
logged epoch/controller time to assess completion, without retuning the protocol.

Probe overhead is separately timed; passive diagnostic overhead is common to all
conditions and included in total epoch time. Epoch timing includes train,
validation and control, but excludes checkpoint I/O; SLURM timestamps give total
job walltime. New alpha values can generate new CuPy kernels: use the project
cache directory, not a quota-limited home cache. GPU smoke is still required on
Lawrence when no local CUDA device is available.

This design tests only short online feasibility with a small reused training
probe and diagonal feedback. It cannot establish generalization across seeds,
datasets, architectures, optimal G targets, full-Jacobian control or SOTA accuracy.
Optional learnable alpha is deliberately omitted; no SAGE behavior is modified.
