# SAGE-v2 / IJCNN 2027 Experiments

This is the new development namespace. It does not define a finalized algorithm,
training recipe, or completed experiment.

| Path | Purpose | Version controlled |
| --- | --- | --- |
| `configs/sage_v2/` | Reviewed experiment configurations | Yes |
| `jobs/sage_v2/` | New SLURM launchers | Yes |
| `experiments/sage_v2/manifests/` | Experiment and provenance manifests | Yes |
| `results/sage_v2/` | Generated runtime results and checkpoints | No |
| `logs/sage_v2/` | Generated runtime logs | No |
| `figures/sage_v2/` | Generated figures | No |

No runtime directories or placeholder results are created by this restructuring.
Future run directories should distinguish dataset, method, simulation T, seed,
and run ID, for example
`results/sage_v2/<dataset>/<method>/T<T>/seed<seed>/<run_id>/`.

CogMI-era results must never be automatically mixed into SAGE-v2 aggregates.
Future analysis must select this namespace explicitly and verify run identity
and completion. Historical runs are documented in
[the archive manifest](../../archive/cogmi2026/manifests/README.md).

The three historical Fixed-SG best CIFAR-10 checkpoints are protected candidate
initialization assets. Their use requires checking existence, checkpoint
compatibility, and provenance, then recording the original path and hash in the
new run manifest. Using a historical checkpoint does not turn its old metrics
into SAGE-v2 results.

Existing CIFAR training code remains the reference implementation. Historical
diagnostic and mechanism scripts remain in their original locations to preserve
imports and reproducibility. They are not the new experimental pipeline.

## Checkpoint Adaptation Infrastructure

This is an experimental development protocol, not the sole evidence proposed
for the final SAGE-v2 paper. No new SAGE algorithm or completed run is claimed.

**From-scratch training:** no checkpoint option. **Resume:** `--resume` uses
timm's model/optimizer/scaler loading and version-dependent next-epoch rule;
the new `sage_training_state` additionally restores scheduler, best metric,
controller state, and RNG/DataLoader generators where present. `--no-resume-opt`
keeps optimizer/scaler/scheduler fresh but still resumes epoch/controller/RNG.
Explicit `--start-epoch` retains its historical override semantics.

**Initialization/adaptation:** `--init-checkpoint` selects non-EMA weights in
deterministic order: `state_dict` > `model_state_dict` > `model` > raw state
mapping. `--init-use-ema` explicitly switches to `state_dict_ema` > `model_ema`;
missing requested weights produce an error, never implicit EMA fallback.
The selected key is logged and recorded in provenance. Loading strips a
`module.` prefix (rejecting key collisions) and enforces strict
architecture/key compatibility. It does not load optimizer, scheduler, scaler,
epoch, best metric, controller, or RNG. Epoch starts at 0. Source path, SHA-256,
available epoch/metric, dirty Git status/commit, architecture, and resolved
parameters go into `provenance.json`. Loading uses trusted pickle-compatible
checkpoints only; never load an untrusted checkpoint.

### Historical Checkpoint Schema Review

The official trainer delegates save/load to `timm.utils.CheckpointSaver` and
`timm.models.resume_checkpoint`, with native/Apex scalers from timm. Expected
official keys from those implementations are:

| Component | Key | Notes |
| --- | --- | --- |
| Model | `state_dict` | Non-EMA weights; optional DDP `module.` prefix |
| Optimizer | `optimizer` | Includes state and per-group LR/initial LR where present |
| Native AMP | `amp_scaler` | Optional, when native scaler enabled |
| Apex AMP | `amp` | Optional alternative, not a native-scaler alias |
| Epoch | `epoch`, `version` | Version >1 resumes at saved epoch+1 |
| Current metric | `metric` | Optional checkpoint validation metric, not inherently best-so-far |
| EMA | `state_dict_ema` | Optional model EMA snapshot |
| Metadata | `arch`, `args` | Original architecture label and argparse namespace |

Historical official files do not explicitly persist saver `best_metric` or
`best_epoch`; new files save these under `sage_training_state`, along with the
scheduler and controller. An isolated `last.pth.tar` metric is not sufficient to
recover historical best-so-far. The separate simplified 100-epoch runner uses
`model_state_dict`, `optimizer_state_dict`, `epoch`, `metrics`, `config`,
`config_name`, and `seed`, with `controller_state`/`records` in latest files;
this is not the official resume schema. Initialization supports its weights,
not automatic restoration of its different training state.

The installed HPC timm version and actual protected checkpoint bytes are not
available here; this review establishes the expected delegated schema, not an
independent verification of those files.

### Alpha Scope and Resume LR

`--fixed-alpha` controls `block.<i>.attn.{q_lif,k_lif,v_lif,attn_lif,proj_lif}`
and `block.<i>.mlp.{fc1_lif,fc2_lif}`: seven neurons per transformer block,
28 for the four-block model. `--sps-alpha` separately controls
`patch_embed.{proj_lif,proj_lif1,proj_lif2,proj_lif3,rpe_lif}`: five neurons.
Both default to 4, preserving historical Fixed-SG behavior. Block alpha=3/5
does not change SPS unless explicitly requested. The classifier has no LIF
surrogate. No model/neuron/surrogate equations are changed.

On resume, construct the scheduler before timm restores optimizer state,
because scheduler constructors may initialize/reset group LR (including
warmup). Then timm restores the optimizer, including each group's saved LR.
Loading saved scheduler state does not step/recompute LR. No redundant manual
LR overwrite is performed. Legacy checkpoints lacking scheduler state retain
the historical `step(start_epoch)` reconstruction; `--no-resume-opt` likewise
uses reconstruction rather than claiming complete optimizer/scheduler resume.

Initialization automatically enables SAGE-v2 output guards. Both output root and
resolved run path must be inside this checkout's `results/sage_v2/`, including
symlink resolution. Existing nonempty runs require resume, and a resume file must
be inside the requested run. No historical checkpoint is written or copied.
The old `--initial-checkpoint` flag was not consumed by this trainer; it now
raises a clear error directing callers to the explicit initialization mode.

The optional `train_umsen.py --umsen` wrapper retains its entropy, EMA, running
statistics, centering, dead zone, mapping, warmup, and one-batch delay. Controller
JSON now distinguishes `applied_alpha` (used in the last forward/backward) from
`candidate_alpha` (computed for a future batch), with warmup status. The legacy
`alpha` summary is retained as a candidate-value alias. No expensive diagnostic
hooks are added; the existing attention callback and structured controller state
remain available for future instrumentation.

New checkpoints retain timm's format and add continuation state by enriching its
temporary checkpoint before last/best/periodic publication. This requires an
additional CPU load/write per checkpoint; no entropy/forward overhead is added.
Old files remain loadable but warn when missing state: neither exact historical
controller history nor historical best accuracy can be reconstructed from an
isolated old checkpoint. The periodic-retention list is not restored. Recovery
checkpoints retain the historical epoch-restart behavior, not exact minibatch
continuation. Complete continuation claims are limited to epoch boundaries.

`--deterministic` seeds Python/NumPy/Torch/CUDA, supplies separate seeded
train/eval DataLoader generators, disables persistent workers and cuDNN
benchmarking, and enables deterministic algorithms with warnings rather than
silently failing unsupported torch operations. Workers retain the existing
Python/Torch DataLoader initialization plus NumPy reseeding. CuPy custom kernels
are not audited by PyTorch's checks; bitwise GPU determinism is not guaranteed.
SAGE-v2 continuation is currently single-process. Avoid multi-epoch loaders in
paired mode. Matching checkpoint, seed, workers, recipe, and hardware is required
for closely paired branches. Real GPU/CuPy validation is still required.

### Direct Pilot Commands

From the repository root in the SNN environment:

```bash
ROOT="$PWD"
export PYTHONPATH="$ROOT/spikformer/cifar10:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=42
cd "$ROOT/spikformer/cifar10"

# Smoke test: one full epoch, its own output directory.
python train.py -c "$ROOT/configs/sage_v2/fixed_cifar10_adaptation.yml" --init-checkpoint "$ROOT/results/umsen_official_recipe/baseline_seed42/model_best.pth.tar" -data-dir "$ROOT/data" --epochs 1 --output "$ROOT/results/sage_v2/cifar10/fixed/T4/seed42" --experiment lr1e-5_smoke

# Two independent 30-epoch pilots (same W0 and seed; fresh state each time).
python train.py -c "$ROOT/configs/sage_v2/fixed_cifar10_adaptation.yml" --init-checkpoint "$ROOT/results/umsen_official_recipe/baseline_seed42/model_best.pth.tar" -data-dir "$ROOT/data" --lr 1e-5 --epochs 30 --output "$ROOT/results/sage_v2/cifar10/fixed/T4/seed42" --experiment lr1e-5_pilot
python train.py -c "$ROOT/configs/sage_v2/fixed_cifar10_adaptation.yml" --init-checkpoint "$ROOT/results/umsen_official_recipe/baseline_seed42/model_best.pth.tar" -data-dir "$ROOT/data" --lr 5e-5 --epochs 30 --output "$ROOT/results/sage_v2/cifar10/fixed/T4/seed42" --experiment lr5e-5_pilot
```

Before a longer run, check smoke output for `args.yaml`, `provenance.json`,
`summary.csv`, `last.pth.tar`, and `model_best.pth.tar`, then inspect the actual
source checkpoint metadata, effective alpha, resolved schedule, and warnings.
