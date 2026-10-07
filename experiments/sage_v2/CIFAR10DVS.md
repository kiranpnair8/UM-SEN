# CIFAR10-DVS SAGE T=10

This is an unrun, single-seed experiment path, not a result or a finalized SAGE-v2
algorithm. No static CIFAR/CogMI outputs are read or changed.

This experiment follows the authors' **released-code defaults**, with the explicit
T=10 and seed=42 overrides below, not every hyperparameter written in the paper.
It must not be described as an exact reproduction of the paper's written protocol.

## Known paper/code discrepancies

| Setting | Paper, Section 4.2 | Authors' released code and this experiment |
| --- | --- | --- |
| AdamW base learning rate | 0.1 | 0.001 |
| SSA attention scaling | Learnable scaling parameter | Fixed scalar 0.25 |

Sources: [paper, page 8](https://arxiv.org/pdf/2209.15425#page=8),
[released DVS trainer](https://github.com/ZK-Zhou/spikformer/blob/main/cifar10dvs/train.py),
and [released DVS model](https://github.com/ZK-Zhou/spikformer/blob/main/cifar10dvs/model.py).
The released README invokes `python train.py` without overriding LR. There is no
verified explanation here for the discrepancies or evidence establishing which LR
produced the published results. We deliberately retain LR=0.001 and SSA scale=0.25;
the latter is distinct from the auxiliary entropy temperature (also 0.25).

## Existing recipe, traced from source

`spikformer/cifar10dvs/model.py` defines Spikformer with two transformer blocks,
256-dimensional embeddings, 16 heads, MLP ratio 4, and ten classes. The two-channel
128x128 input passes through four SPS pooling stages to an 8x8 (64-token) grid.
LIF nodes use tau=2, detach_reset=True and the production CuPy backend; the attention
output neuron has threshold 0.5, other thresholds default to 1. The input's temporal
axis supplies T; the model does not synthesize/repeat static images.

`train.load_data` uses SpikingJelly CIFAR10DVS frame integration with
`frames_number=T, split_by='number'`. It takes the first ceil(90%) of each class
for training and the remainder for testing without shuffling the split. For the
complete 10,000-example dataset this is 9,000/1,000; actual counts are saved in the
resolved config. There is no additional validation set. Selection of the best
checkpoint uses test accuracy, following the legacy code, and should be disclosed.
Event counts are cast to float, not RGB-normalized or resized.

Defaults inherited from the original DVS parser:

| Setting | Value |
| --- | --- |
| Batch / workers | 16 / 4; train drop_last=True, test False |
| Optimizer | timm AdamW, LR 0.001, weight decay 0.06, epsilon 1e-8, default betas |
| Schedule | timm cosine, 96 schedule epochs + 10 cooldown = 106 total |
| LR warmup | 10 epochs, warmup LR 1e-5; minimum LR 1e-5 |
| Augmentation | Per-example horizontal flip 0.5 and SNNAugmentWide, coherent across frames |
| SNNAugmentWide | Identity, shear, translations, rotation, or cutout |
| Mixup | alpha 0.5, probability 0.5, batch mode; disabled at epoch index 75 |
| CutMix / smoothing | 0 / 0.1 |
| Loss | SoftTargetCrossEntropy for training; CrossEntropyLoss for evaluation |
| Precision | CUDA AMP training; evaluation in float32 |
| Clipping / early stop | Neither |
| Neuron reset | After every training/evaluation batch |

The legacy DVS launcher defaults to T=16 and seed 2021. This experiment deliberately
uses T=10 and seed 42. It otherwise follows the released repository recipe. The
two blocks, dimension 256, 16 heads, batch 16 and 106 total epochs agree with the
paper, subject to the explicit paper/code discrepancies above. The original
code's scheduler argument is 96, not 106: the new runner's `--epochs` denotes TOTAL
epochs and subtracts cooldown before constructing the same scheduler. Smoke mode
does not shorten or retune that scheduler. It runs four training and four test
batches in epoch zero; its accuracy is not a full-test-set or paper result.

The old main passes `drop_block_rate` into a constructor that does not accept it;
newer timm also injects unsupported factory metadata. The new runner calls the
existing local model factory directly with the same supported arguments. It also
avoids the legacy unused import-time TensorBoard writer. These are orchestration
fixes, not model or controller changes. Existing unused model parameters/modules
(including `res_lif`, normalization layers and position embeddings) remain intact.

## Controller insertion and scope

The static CIFAR controller was extracted unchanged into
`spikformer/cifar10/sage_controller.py`; `train_umsen.py` re-exports it. DVS SSA now
has a default no-op observer immediately after `attn = q @ k.transpose(-2, -1)`.
Its tensor is `[T, B, 16, 64, 64]`; the last dimension is the key-token distribution.
Production attention has no softmax, and `(attn @ v) * 0.25` remains unchanged.
Unlike static CIFAR's pre-scaled attention tensor, DVS scores here are raw q-k
products. No extra scaling was introduced to disguise this backbone difference.

The shared controller detaches attention, applies auxiliary softmax(attn / 0.25)
over keys, and divides entropy by log(64). It averages over time, batch and query
tokens, then computes population standard deviation across heads. EMA beta is
0.95 with first-observation initialization. Welford running statistics normalize
the EMA using sample variance and epsilon 1e-6; the current observation participates
in those statistics. Original z is centered across both blocks. The unchanged
dead zone |centered z| < 0.25 gives alpha=4; otherwise alpha=4+0.5*tanh(centered z),
clipped to [3,5]. The entire first epoch AND at least 100 steps use applied alpha=4.
Statistics and candidates still update during warmup. Each batch uses the previous
candidate, observes current attention during forward, and computes the next batch's
candidate before backward. Evaluation does not update the controller.

Each block shares one alpha across q/k/v, attention-output and projection neurons,
the two MLP neurons, and the registered but unused res_lif (8 registered, 7 executed
LIF nodes per block). All five SPS LIF nodes remain fixed at 4. The new runner gives
each LIF an independent deepcopy of its existing Sigmoid surrogate to prevent
shared constructor-default objects from leaking alpha between blocks or into SPS.
No neuron forward equations, model weights, or attention values change.
Although alpha is assigned to all eight registered block nodes, `res_lif` is not
called in the active forward path and contributes no surrogate gradient. Only the
seven actually executed transformer-block LIF nodes per block influence training.

Epoch records contain detached Python values for raw/EMA dispersion, original and
centered z, applied/candidate alpha, and warmup flags. No expensive profiler is
added. Existing loop timing/memory console messages are preserved.

## Commands and outputs

Run these from the repository root on the HPC, after reviewing the changes:

```bash
mkdir -p logs/sage_v2
sbatch --export=ALL,T=10,SEED=42,SMOKE_TEST=1 jobs/sage_v2/run_sage_cifar10dvs.sbatch
sbatch --export=ALL,T=10,SEED=42 jobs/sage_v2/run_sage_cifar10dvs.sbatch
```

Override EPOCHS=106, BATCH_SIZE=16, LR=0.001, DATA_DIR, or SNN_ENV through the same
export mechanism. DATA_DIR defaults to `<repo>/data/CIFAR10DVS`. No dataset download
or cache creation has been performed by this implementation task.

The job uses the existing gpu partition, one GPU, 16 CPUs, 64G host RAM, and a
two-day limit. The repository does not establish which node/GRES label is a V100;
to prefer a confirmed V100, add `--nodelist=<confirmed-V100-node>` to submission.
It does not assume gpu004/gpu005 has a particular GPU model. No jobs were submitted.

Full outputs: `results/sage_v2/cifar10dvs/sage/T10/seed42/full_<job-id>/`.
Smoke outputs use `smoke_<job-id>/`. Existing directories and destinations
outside the DVS SAGE-v2 namespace are rejected. Saved files are `config.json`,
`metrics.json`, `umsen_controller.json`, `model_best.pth`, and `last.pth`. Checkpoints
include model, optimizer, scheduler, scaler, epoch, best accuracy and controller
state. This minimal fresh-run entry point does not add checkpoint adaptation or
claim resume support. Accuracy is percent; training accuracy uses the argmax of
Mixup targets, as in the existing loop. All logs are under `logs/sage_v2/`.

GPU peak memory and total runtime are unknown without a real run. At the default
batch/T, one raw FP32 attention tensor alone is about 40 MiB per block, not an
estimate of total training memory; SPS activations and backward storage dominate
additional memory. There is no local DVS runtime evidence to promise completion
within two days. Smoke mode is intended to verify the installed HPC stack first.
