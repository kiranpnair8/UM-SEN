# Provisional Closed-Loop Feasibility

This is method-design infrastructure, not a finalized SAGE algorithm or a claim
of accuracy improvement. It does not modify the shared trainer, model or SAGE-v2.

## Stage Gate

Only frozen-state recovery is implemented here. The real early/mid/late recovery
study has NOT been run locally. Online integration and the 60-epoch comparison
are intentionally deferred until that study passes and is reviewed. A synthetic
smoke can NEVER authorize training. `RUN_ONLINE=1` fails closed; it does not start
a hidden or placeholder trainer. There is no 310-epoch controller campaign.

## Protocol

The existing regenerated states manifest supplies the exact early20/mid150/late309
checkpoints from one Fixed-SG trajectory. Hashes, recipe and trajectory IDs are
verified with the existing manifest validator. No best-checkpoint substitution.
Ten sequential, non-overlapping TRAINING-split CIFAR-10 batches of size32 use the
existing evaluation transform, with no augmentation or download. This avoids
using validation/test labels for controller references. Existing H1-H4 probe
defaults still use the test split and are not changed.

Every batch measures a fresh pre-disturbance reference at alpha4, then separately
applies global+1, global-1 and [3,5,3,5] disturbances. SPS alpha remains4. Reference
G is frozen for the entire episode. No future checkpoints, oracle alpha or G=0
target is used. Checkpoint states are separate experiments, never reference inputs
to a controller at an earlier state. Configured additional single-block disturbance
vectors are supported without a code change.
Recovery alpha trajectories are controller state passed as each probe's current
center; the caller model alpha is restored between observations, not left changed.

`intervention()` accepts optional per-block centers/sweeps; its legacy defaults
and exact restoration checks are preserved. All recovery probes use eval/FP32,
no scaler and NO optimizer updates. At each response-aware/fixed-sign iteration,
each block is independently perturbed about its CURRENT alpha, holding every
other block at its current alpha. Symmetric delta is at most0.5, reduced only to
stay within bounds; the actual endpoints/delta and all-block measurements are
saved. No symmetric estimate is fabricated at a boundary: that feedback episode
stops and reports its bound hit. Caller parameters, buffers, BN, modes, neuron
memories, gradients and RNG are restored by the validated engine. An optional
optimizer state check is tested; no optimizer is constructed by the frozen study.
Forward logits must also match exactly across alpha probes.

The provisional update is
`-eta * (G-G_ref) * Rhat / (Rhat**2 + regularizer)`.
Defaults: eta0.25, regularizer1e-4, maximum step0.25, alpha[1,8], 12 iterations.
These are explicit initial feasibility choices, not values tuned for accuracy.
Zero response yields zero update, not division by zero. Updates are sign-aware;
every applied response-aware step logs whether e*Rhat*step is nonpositive. Group
summaries include response-sign fractions, update-direction agreement, and final
per-block absolute errors. Reused no-control observations are not independent gain
estimates; their identical signs are descriptive only.
`interval=200` is reserved for a FUTURE online integration and has no effect here.

Reference helpers support frozen-window refresh or beta0.99 EMA ONLY at explicit
window boundaries. They never track a disturbance inside a frozen recovery window.
The frozen study performs no refresh; the EMA helper is unit-tested, not presented
as an already validated online reference policy.

## Comparisons and Measurements

No control holds the disturbed state fixed and explicitly reuses its deterministic
measurement. Fixed-sign uses abs(Rhat) in the same regularized formula, incorrectly
assuming a positive response. Random matched replays the exact per-block applied
magnitudes and number of response-aware updates, choosing random FEASIBLE signs
with a local seed. Boundary conditioning is disclosed, not called unbiased random
directions. Matching flags and all requested/applied steps are saved. Random trials
do not estimate unused response gains; absent Rhat values are null, not zero.

Each trace saves G/reference, alpha, local finite-difference response, raw endpoints,
L2/per-block errors, normalized L2 error, loss and existing block gradient/firing
statistics, bound hits and update timings. Terminal outcomes do not estimate an
unused response gain. Initial error below1e-10 is uninformative (ratio=null).
50%/75% first-passage iteration counts are saved globally and per block.
Oscillation is a DESCRIPTIVE indicator: two or more nonzero error-sign crossings
in any block. Divergence means error ratio>10; unsafe abs(G)>50, gradient norm>1e6,
NaN/Inf or failed restoration aborts and records failure. No failure is suppressed.

The provisional real-data gate requires at least10 batches and, in EVERY
state/disturbance, at least80% reaching50% error reduction, mean final ratio<=0.5,
no response-aware oscillation/bound/divergence, and a lower POOLED mean normalized
residual than every baseline across all cases. Fixed-sign may correctly tie the
response-aware controller in positive-response states; no impossible strict
statewise advantage is required there. It also verifies the reference early-negative/mid-late-positive mean
response reversal in all blocks. This is a predeclared feasibility gate, not a
significance test, universal stability guarantee or paper-performance conclusion.
Failure leaves all measurements available and forbids progression to training.

## Outputs and Resume

Use `results/gradient_flow/controller_recovery_seed42/` and separate corresponding
figures. Metadata includes Git/environment, exact checkpoint manifest, settings,
training split and input fingerprints. Each state/batch completion marker hashes
its raw recovery JSON. Identical validated batches skip on resume; corrupt or
incomplete batches recompute. Changed code/settings/checkpoints/inputs require a
NEW output namespace. Existing H1-H4 output directories are never touched.
Actual probe calls/time are recorded once per batch; controller computation time
is separate. Do not sum repeated no-control observation timing fields.

Figures use only saved outputs and matplotlib: error/method comparison, per-block
alpha, Rhat, G/reference, and checkpoint sign heatmap, all PNG/PDF. Error curves
show mean +/- population SD; stopped trajectories are not extrapolated, so later
means may use fewer batches. Raw traces and descriptive summary remain available.

## Lawrence

For completed-output gate audits, use `analysis.gradient_flow.audit_controller_recovery`.
It verifies raw JSON hashes, rebuilds summary/figures without loading checkpoints
or running GPU probes, and names each failed conjunctive condition. Defaults use
NEW `controller_recovery_gate_audit_seed42` result/figure namespaces; originals
are not overwritten. The uploaded completed run failed only the zero-oscillation
condition in early/alternating (1/10 batches) and late/alternating (2/10 batches).
The flags match the declared two-component-error-crossings rule even though
total L2 error decreases monotonically in those episodes. No arithmetic bug was
found and no gate criterion was relaxed. See the provenance-backed test fixture.

```bash
python -m analysis.gradient_flow.audit_controller_recovery
```

Current project convention is gpu002, partition gpu, one GPU, 16CPUs, 64G and
maximum two days. No account/QoS/module is invented. The launcher activates the
existing snn environment, prints provenance, locks the writer, runs recovery and
plots, then exits NONZERO if the real-data gate fails. It never launches training.

```bash
mkdir -p logs/gradient_flow
sbatch jobs/gradient_flow/run_controller_recovery_cifar10.sh
```

Interactive inside an ALREADY sufficient gpu002 allocation:

```bash
python -m experiments.gradient_flow.controller_recovery --backend cupy --device cuda --resume
python -m analysis.gradient_flow.plot_controller_recovery
```

Runtime is measured in per-batch markers. Approximately 1 reference calibration
+3 disturbances x [1 initial calibration +2 feedback methods x11 intermediate
calibrations +14 terminal/random observations], each calibration10 FP32 passes
and observation3 passes: about826 passes per batch, or24,780 passes across
30 state/batch pairs. This is roughly1,126 of the earlier22-pass H3 sweeps.
Using the user-reported205s/25 H3 batches gives a rough2.6hours, excluding
additional JSON/I/O/kernel-cache and model-state-copy costs. This is an estimate,
not a measured V100 controller runtime; no training time is included. CuPy may
compile kernels for many distinct alpha values, so compilation/cache costs can
dominate that estimate and must be profiled on Lawrence.
