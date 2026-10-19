# Gradient-Flow Research

See [multi-batch causal validation](MULTIBATCH.md) for the fixed-checkpoint,
25-batch SAGE evaluation protocol, resume rules, statistics, and Lawrence launcher.

This namespace implements the first passive diagnostic stage for block-specific
surrogate gradients and backward optimization dynamics in Spiking Transformers.
There is no adaptive feedback controller or finalized method name. Shared model,
surrogate and SAGE implementation files are unchanged.

Shared models and existing training pipelines remain under `spikformer/`.
SAGE source, configurations, experiment documentation and historical outputs must
remain independent of this track. Do not activate an existing adaptive controller
implicitly in future causal diagnostic experiments.

| Path | Purpose | Tracked |
| --- | --- | --- |
| `configs/gradient_flow/` | Reviewed configurations | Yes |
| `experiments/gradient_flow/` | Experiment code, protocols and provenance documentation | Yes |
| `analysis/gradient_flow/` | Analysis and visualization source | Yes |
| `jobs/gradient_flow/` | Reviewed launchers | Yes |
| `checkpoints/gradient_flow/` | Optional standalone runtime checkpoint assets | No |
| `logs/gradient_flow/` | Runtime logs | No |
| `results/gradient_flow/` | Runtime metrics and run-local checkpoints | No |
| `figures/gradient_flow/` | Generated figures | No |

Future run directories should distinguish dataset, protocol, T, seed and run ID.
Use an explicit protocol identifier once reviewed; this document does not select
an algorithm. A checkpoint saved inside a run's result directory need not be
duplicated into `checkpoints/`.

Git preserves these source namespaces through their READMEs, not empty runtime
directories. From the repository root, local or HPC Bash users may create only
the empty runtime containers with:

```bash
mkdir -p checkpoints/{sage_v2,gradient_flow} logs/{sage_v2,gradient_flow} \
  results/{sage_v2,gradient_flow} figures/{sage_v2,gradient_flow}
```

This command does not launch experiments, copy assets, relocate historical files,
or create placeholder metrics/checkpoints. Existing content is left untouched.
See [the organization inventory](../REPOSITORY_LAYOUT.md).

## Measurements

Blocks are identified by zero-based indices in `model.block`. Each block's input
is x_l and output is x_(l+1), after both residual additions. Tensor gradient hooks
read their backward cotangents, return None, and keep only detached scalar norms.
The input/output gradient tensors themselves are not retained in metric histories.

G_l = log((||dL/dx_l||_2 + eps) / (||dL/dx_(l+1)||_2 + eps)). Norms include all
time, batch, token and channel coordinates. G measures amplification along the
observed gradient direction, not a Jacobian condition number or gradient quality.
The per-block parameter norm is sqrt(sum_p ||p.grad||_2^2), including weights and
biases with gradients. Unused LayerNorm parameters with grad=None are omitted.

Directional consistency is cosine(current parameter gradient, PREVIOUS EMA).
Only after calculating cosine is the EMA updated as beta*reference+(1-beta)*grad.
The first reference is initialized to the first valid gradient, not an artificial
zero vector; first/zero-norm/missing-reference cosine is null, never called
"gradient correctness". References are detached and checkpointable; no raw tensor
history is accumulated. A missing/nonfinite block prevents that step's EMA update.

Firing rate is sum of emitted spikes divided by total output elements over all
actually executed LIF modules within the block. It is not an unweighted average of
module means. The seven executed CIFAR block neurons are included; SPS is excluded.
Alpha is one non-trainable scalar per block; SPS remains 4. Get/set/snapshot/restore
live in `alpha_scope.py`. Shared-surrogate aliases or a competing SAGE controller
are rejected instead of silently changing another block's alpha.

## AMP and state

For native CUDA AMP, capture GradScaler.get_scale() BEFORE forward. Activation
hooks observe scaled gradients; reduce their detached FP32 norm and divide by
that exact scale BEFORE adding eps. Native backward then calls unscale_(optimizer)
exactly once. Parameter norms and directional statistics are read post-unscale,
pre-clip and pre-step. GradScaler.step sees the already-unscaled state and does not
unscale twice. Clipping/optimizer/scaler update otherwise follow the trainer's
original order. Invalid steps have null G/cosine and do not update the EMA; native
GradScaler still determines whether its optimizer step is skipped. FP32 retains
the baseline optimizer's nonfinite behavior; diagnostics never repair gradients.

The training entry point wraps the existing CIFAR train/evaluation loops, loader,
augmentations, loss, optimizer, scheduler and checkpoint saver. It imports no SAGE
wrapper, does not monkey-patch surrogate math, and preserves default alpha=4.
The original loop resets neuron state after each batch. A passive scaler adapter
adds the observation point; disabled diagnostics use the original scaler/step path.
Use `--disable-diagnostics` for an instrument-off control.

Stage 1 intentionally supports eager single-GPU native AMP or FP32, no DDP, Apex,
model EMA, TorchScript or diagnostic resume. Weights-only `--init-checkpoint` is
available through the existing trainer, with explicit non-EMA key precedence.
The diagnostic EMA is saved separately as `diagnostic_state.pt`; future resume
integration must restore it and step numbering, so resume is rejected for now.

## Training and smoke commands

Run modules from the repository root in an environment containing the existing
Spikformer dependencies plus NumPy, PyYAML, matplotlib, seaborn and pytest for tests.
Keep the HPC's working CUDA/SpikingJelly stack; CPU smoke validation does not certify
an arbitrary CUDA version combination. Local CPU tests used PyTorch 2.10.0+cpu,
torchvision 0.25.0+cpu, timm 0.5.4 and SpikingJelly 0.0.0.0.12.

```bash
# Real CIFAR loader/recipe; at most 3 train and 3 evaluation batches, one epoch:
python -m experiments.gradient_flow.train \
  -c configs/gradient_flow/cifar10_diagnostics.yml -data-dir data \
  --experiment cifar10_diagnostic_seed42

# Same bounded experiment, explicit block settings (not an adaptive controller):
python -m experiments.gradient_flow.train \
  -c configs/gradient_flow/cifar10_diagnostics.yml -data-dir data \
  --block-alphas 4 4 4 4 --experiment cifar10_explicit_alpha_seed42

# Separate synthetic CPU correctness smoke, full 4-block/384-dim/T4 backbone:
python -m experiments.gradient_flow.smoke --batches 3 \
  --output results/gradient_flow/smoke_cpu
```

The training config retains the full inherited schedule (300 plus 10 cooldown
epochs as resolved by timm); diagnostic budgets limit execution, not the schedule.
Zero for a diagnostic limit removes that limit; do not do so without an explicit
long-run decision. Limited evaluation accuracy is NOT a full-CIFAR result. At least
two training batches are required by the existing trainer's progress formatter.

The CPU smoke uses synthetic data and CE/AdamW for correctness, not the full
augmentation/Mixup/AMP data recipe. It writes a clearly labeled synthetic batch
and checkpoint only for exercising the intervention tools, not paper experiments.

## Controlled interventions

Provide a trusted torch minibatch file with `images` [B,3,H,W] and hard `targets`
[B]. It must already contain the fixed preprocessing you want to study. No random
augmentation or Mixup is performed during probes; all probes use the same CE loss.
These are conditional gradient measurements, not validation accuracy evaluations.

```bash
python -m experiments.gradient_flow.intervene \
  --checkpoint results/umsen_official_recipe/baseline_seed42/model_best.pth.tar \
  --batch results/gradient_flow/probe_batch.pt --block 1 --alphas 2 3 4 5 6 \
  --output results/gradient_flow/single_block_probe

python -m experiments.gradient_flow.intervene \
  --checkpoint results/umsen_official_recipe/baseline_seed42/model_best.pth.tar \
  --batch results/gradient_flow/probe_batch.pt --matrix --alphas 3.9 4 4.1 \
  --output results/gradient_flow/response_matrix

# Existing early/mid/late checkpoints; no training or checkpoint fabrication:
python -m experiments.gradient_flow.intervene \
  --checkpoints early=checkpoints/gradient_flow/early.pth.tar \
    mid=checkpoints/gradient_flow/mid.pth.tar late=checkpoints/gradient_flow/late.pth.tar \
  --batch results/gradient_flow/probe_batch.pt --matrix --alphas 3.9 4 4.1 \
  --output results/gradient_flow/checkpoint_response_comparison
```

The example minibatch/early/mid/late files must be supplied; they are not created
by these docs. Block IDs are zero-based. Default probe backend/device is CuPy/CUDA;
use `--backend torch --device cpu` for CPU. Default probe precision is FP32; `--amp`
enables native CUDA AMP. `--mode eval` is default; `--mode train` is a different,
explicit BatchNorm condition and is recorded in output.

All blocks start at reference alpha=4 (override with --reference-alpha); each trial
changes only its target block. Every forward restores identical weights and BN
buffers, resets all neurons, clears gradients and restores Python/NumPy/Torch/CUDA
RNG. Only in AMP, an SGD object with LR=0 exists solely as an unscale target: its
step method is NEVER called. FP32 constructs no optimizer. The caller's original weights, gradients, alphas,
module modes, neuron memories and RNG are restored in a finally block. Call probes
at batch boundaries, not while retaining a previous autograd graph.

A baseline pass establishes a fixed parameter-gradient reference. Each alpha trial
reloads that same reference; alpha order cannot contaminate directional consistency
through an evolving EMA. This is explicitly labeled baseline-reference consistency
in probe output, distinct from the temporal training EMA.

For every checkpoint, measurements.json stores baseline and raw sweep records for
ALL blocks, loss/logits, source checkpoint key/path/hash, fixed-batch hash, mode,
precision and zero optimizer updates. Architecture compatibility is strict. The
response matrix has rows i=observed block and columns j=perturbed block:
R_ij = (G_i(alpha_max at j)-G_i(alpha_min at j))/(alpha_max-alpha_min).
This is a secant; only a small symmetric sweep estimates a local central derivative.
Do not interpret a broad [2,6] sweep as a precise derivative at alpha=4. Raw sweep
measurements are retained even when the endpoint matrix is used. Matrix JSON and
PNG/PDF heatmaps are saved alongside the checkpoint's measurements.

## Logging, plotting and validation

All experiment outputs are protected under `results/gradient_flow/<run-id>/`.
Existing destinations and paths outside that namespace are refused. The official
trainer's ordinary checkpoints/args.yaml/summary.csv coexist with streamed
`steps.jsonl`, `epochs.json`, `diagnostic_config.json` and `diagnostic_state.pt`.
Step records include epoch, step, block_id, alpha_l, grad_input_norm,
grad_output_norm, G_l, parameter_grad_norm, directional_consistency, firing_rate,
validity and loss scale. Epoch summaries are means of valid STEP values (including
mean log-ratios, not the log of mean norms), with explicit observation counts.

```bash
python -m analysis.gradient_flow.plot_diagnostics \
  --metrics results/gradient_flow/cifar10_diagnostic_seed42/epochs.json \
  --output-dir figures/gradient_flow/cifar10_diagnostic_seed42
python -m pytest tests/gradient_flow spikformer/cifar10/tests -q
```

Tests cover disabled/enabled equivalence, observer-only hooks, scalar/detached
logging, alpha isolation and its actual surrogate backward, neuron resets, exact
intervention state/RNG restoration, repeatability, loss-scale correction, one
unscale call, pre-clip norms, invalid-step handling and response-matrix structure.
CUDA AMP testing is skipped when no CUDA GPU is available. Synthetic response
matrices/figures are correctness artifacts, not evidence of CIFAR accuracy gains,
predictable feedback behavior or a completed new research method.

## First fixed-checkpoint causal experiment

Use eval mode and FP32 explicitly, reference transformer alpha=4, and sweep
2,3,4,5,6. SPS is forced to 4 during each probe and its caller state is restored.
All alphas, weights, BatchNorm buffers, neuron memories, modes, gradients and RNG
are restored after EVERY probe, even on error. Gradients are cleared before every
forward. FP32 probes construct no optimizer; optional legacy AMP support uses an
optimizer only to unscale, never step. Repeated reference passes must agree in loss
and G (rtol=1e-6, atol=1e-8); parameter invariance is checked exactly.

From the repository root, with existing local CIFAR-10 test files and a trusted
checkpoint (no dataset download or optimizer updates):

```bash
python -m experiments.gradient_flow.make_probe_batch --data-dir data \
  --batch-size 32 --seed 42 \
  --output results/gradient_flow/probe_batches/cifar10_eval_bs32_seed42.pt

CKPT=results/umsen_official_recipe/baseline_seed42/model_best.pth.tar
BATCH=results/gradient_flow/probe_batches/cifar10_eval_bs32_seed42.pt
python -m experiments.gradient_flow.intervene \
  --config configs/gradient_flow/cifar10_diagnostics.yml \
  --checkpoint "$CKPT" --batch "$BATCH" --block 0 \
  --alphas 2 3 4 5 6 --reference-alpha 4 --mode eval --device cuda --backend cupy \
  --output results/gradient_flow/alpha_block0_seed42
python -m experiments.gradient_flow.intervene \
  --config configs/gradient_flow/cifar10_diagnostics.yml \
  --checkpoint "$CKPT" --batch "$BATCH" --matrix \
  --alphas 2 3 4 5 6 --reference-alpha 4 --mode eval --device cuda --backend cupy \
  --output results/gradient_flow/alpha_matrix_seed42
python -m analysis.gradient_flow.plot_alpha_response \
  --measurements results/gradient_flow/alpha_matrix_seed42/checkpoint/measurements.json \
  --output-dir figures/gradient_flow/alpha_matrix_seed42
```

The batch helper selects test indices 0..31 in deterministic order (seed is recorded,
not a random sample selector) and reuses the project's eval transform factory,
including configured normalization. Schema: `images` FP32 [B,3,32,32], `targets`
int64 [B], plus provenance. Batch and output paths must be new; nothing is overwritten.

Each checkpoint output directory contains nested `measurements.json`, flat
`raw_measurements.jsonl` and `.csv` (perturbed block x alpha x measured block), and
`metadata.json`. `response_matrix.json` retains the historical endpoint secant
(alpha=2/6 for this sweep). SEPARATELY, `local_response_matrix.json` uses the nearest
symmetric measured pair around reference alpha: (G_i(5)-G_i(3))/2 here. It is a
finite-difference estimate at step size 1, not an exact derivative. If no symmetric
pair exists, a warning is emitted and raw/endpoint outputs are still retained.
The existing endpoint heatmap is retained and labeled as a secant; generate the
new local-response plots/summary with the saved-output analysis command above.

The matplotlib-only plotter exports PNG/PDF G-versus-alpha curves and a labeled
local-response heatmap. Its summary lists diagonal responses, strongest absolute
off-diagonal response, mean absolute diagonal/off-diagonal ratio when defined,
and sampled monotonicity within reference +/-1. These are descriptive measurements,
not claims that G=0 is optimal or that alpha is an effective feedback controller.
