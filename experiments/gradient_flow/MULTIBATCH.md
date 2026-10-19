# SAGE Multi-Batch Alpha-Response Validation

This is a fixed-checkpoint, evaluation-only experiment, not training or an adaptive
controller. It reuses `intervention()` and its reference, alpha-isolation, weight,
neuron, BatchNorm, gradient, mode and RNG restoration checks. Caller state is now
also checked exactly after each restoration. No optimizer is constructed in FP32.
SPS alpha stays 4 during every probe; caller alpha is restored afterward.

## Protocol

The default is the existing CIFAR-10 Spikformer: T4, four blocks, dim384, 12 heads,
patch4, MLP ratio4. One checkpoint is loaded once. A single sequential traversal
of the real CIFAR-10 test split supplies 25 non-overlapping batches of 32 samples:
batch 0 uses indices 0..31, batch 1 uses 32..63, ..., batch 24 uses 768..799.
Seed42 is recorded and used for RNG initialization, not for resampling indices.
The existing evaluation transform factory/normalization is reused; there is no
augmentation, download, AMP, optimizer step, or silent backend switch.

Every batch sweeps alpha=2,3,4,5,6 independently in every block, with all other
transformer alphas=4. Two baseline reference passes plus the four alpha=4 trials
are checked for reproducibility (rtol1e-6, atol1e-8). Each sweep has 20 interventions
and 80 block observations. All raw G_i(alpha_j) values are retained. The local
matrix uses the nearest symmetric measured pair around the reference: (G_i(5) -
G_i(3))/2 by default. It is a finite difference at step1, not an exact derivative.

## Outputs and Resume

Default output: `results/gradient_flow/alpha_multibatch_seed42/` (ignored by Git).

```text
metadata.json                 # exact settings, resolved config, checkpoint hash,
                              # code/dirty status, library versions, GPU, dataset
status.json                   # running/failed/complete; validated batch count
batches/batch_000/             # through batch_024/
    batch.pt                  # exact preprocessed tensors, hard labels, indices
    measurements.json         # raw sweep, invariance checks, duration, fingerprint
    local_response_matrix.json
    raw_measurements.jsonl
    raw_measurements.csv
    complete.json             # written LAST; identity and all artifact SHA256 hashes
    failure.json              # only on a caught failed attempt; retained on retry
aggregate/
    summary.json              # all descriptive statistics and per-batch matrices
    response_{mean,std,median,minimum,maximum,ci95}.json
    response_sign_consistency.json
    structural_metrics.json
    alpha_curves.json
    raw_measurements.{jsonl,csv}
```

Without `--resume`, existing output is refused. Resume requires identical metadata,
including source commit/dirty flag, checkpoint content hash, preprocessing/config,
seed, batch counts, numerical/statistical settings and runtime versions/device.
The deterministic traversal recreates inputs; tensor fingerprints must match.
Only hash-checked, structurally valid completed batches are skipped. Incomplete or
corrupt batch artifacts are recomputed in that batch directory. A valid completed
batch with DIFFERENT input identity fails instead of mixing data. Raw files and
completion markers use atomic replacement where applicable; the completion marker
is written last. Any probe invariant/reference failure is recorded and re-raised;
completed earlier batches remain intact. Aggregate files are rebuilt only after
ALL requested batches validate. Analysis refuses a non-complete run, even if old
aggregate files are present. Do not run concurrent writers into the same run path.
The research CLI and launcher refuse a dirty checkout; do not change code/environment
mid-run. Use a new namespace for a different protocol. Synthetic unit/smoke tests
exercise the orchestration function directly and are explicitly labeled synthetic.

## Statistics

For each matrix entry: mean, sample SD (ddof1), median, min, max, and a 95% percentile
bootstrap CI of the mean. The default is 10,000 bootstrap resamples of whole batches,
seed42; samples are not treated as 800 independent images for this interval. For a
single batch, SD and CI are undefined/null. This interval is descriptive, conditional
on one checkpoint and this deterministic test subset, not a significance test or
uncertainty over training seeds. No automatic significance or controllability claim.

Effective-zero tolerance defaults to 1e-8 in response units: positive means R>tol,
negative R<-tol, zero abs(R)<=tol. Strict R_ii>0 fractions are ALSO reported as
requested, separately from tolerance-based sign fractions. The summary includes
mean absolute diagonal, all off-diagonal, effectively-nonzero off-diagonal,
nearest-upstream (j=i+1), distant-upstream (j>i+1), and downstream (i>j) values.
Both diagonal/off-diagonal ratios are per-batch ratios, then summarized across
batches, not ratios of pooled means. Denominators <=tol are undefined/null and
excluded with explicit undefined counts; nonzero-only denominators exclude abs(R)
<=tol entries. Ratios are not clipped. Also reported: strict diagonal-positive
fractions, diagonal-vs-off-diagonal comparisons, nearest-vs-distant comparisons,
all-downstream-zero fractions and nearest-upstream sign consistency.

Raw alpha curves report mean +/- sample SD across batches for each perturbed/measured
pair. Sampled local monotonicity uses alpha within reference +/-1 (3,4,5 here), with
the same numerical tolerance applied to G increments. Nondecreasing/nonincreasing
fractions overlap for effectively flat curves; nonlinear behavior is not discarded.

## Lawrence

The launcher follows existing project conventions: partition `gpu`, generic one-GPU
allocation, 16 CPU cores, 64G, maximum two days, existing `snn` environment, shared
repository data, repository-local CuPy cache, no invented account/QoS/module. A V100
32GB is sufficient; select a known V100 node at submission if necessary rather than
inventing a GPU type label. The script refuses dirty tracked/untracked source.
SLURM opens its logs before Bash runs, so create the log directory before submission.

```bash
# From the clean SAGE repository root, on gradient-flow-stage1-validation:
mkdir -p logs/gradient_flow
JOBID=$(sbatch --parsable jobs/gradient_flow/run_alpha_multibatch_cifar10.sh)
squeue -j "$JOBID"
tail -f "logs/gradient_flow/alpha_multibatch_${JOBID}.log"

# Resubmit the SAME job to resume the same output; completed validated batches skip.
# To override locations: sbatch --export=ALL,DATA_DIR=...,CHECKPOINT=...,RUN_DIR=...,FIGURE_DIR=... ...
python -m analysis.gradient_flow.plot_alpha_multibatch \
  --run-dir results/gradient_flow/alpha_multibatch_seed42 \
  --output-dir figures/gradient_flow/alpha_multibatch_seed42
```

Results are under the default run path above; PNG/PDF figures and descriptive
`summary.txt` are under `figures/gradient_flow/alpha_multibatch_seed42/`. The plotter
uses only saved outputs and matplotlib, not seaborn. Figures may be regenerated in
that run's figure directory. Existing single-batch result paths are never touched.

Timing printed per batch includes the probe and validation; ETA is a rough estimate
from fresh batches in the current invocation, excluding file/plot overhead. First
CuPy compilation can make the first batch slower than later batches. No V100 runtime
is asserted without a measured V100 batch. Tiny CPU synthetic smoke timings are
correctness measurements only, not GPU runtime estimates or real-CIFAR evidence.
