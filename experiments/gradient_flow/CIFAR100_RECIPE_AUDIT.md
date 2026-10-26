# CIFAR100 Recipe Audit and Isolated Preparation

Audit baseline: commit 7cd92a8d7402957083c855823dca906d5ac681b6. No active HPC
job, output, checkpoint, approval, tracked shared source or config is modified.
The active HPC filesystem is not accessible from the development workspace;
execution structure below is from the actual launcher, not a live progress claim.

## Native Recipe Search

Tracked configs, scripts, jobs, Spikformer sources and README were searched for
CIFAR100/CIFAR-100. There is no standalone CIFAR100 training YAML or model folder.
The historical scripts/run_surrogate_comparison.py dataset_specs entry selects
spikformer/cifar10/train.py and cifar10.yml, overriding only dataset/classes.
The prepared controller_followup.py does the same. Therefore the available recipe
is a repository-native CIFAR10-to-CIFAR100 adaptation, NOT a verified official
CIFAR100 reproduction. No claim is made here about an unaudited upstream repo.

## Actual Existing CIFAR100 Recipe

| Component | Actual resolved behavior |
|---|---|
| Dataset | timm create_dataset('torch/cifar100'), torchvision CIFAR100 fine labels (100, not 20 superclasses) |
| Splits | training=True 50,000; training=False official test 10,000 used as validation every epoch; no separate held-out validation split |
| Download | timm1.0.28 create_dataset defaults download=False; training call does not override it; probe also download=False |
| Backbone | Spikformer-4-384, four blocks, 384 dim, 12 heads, patch4, MLP ratio4, T4, RGB32x32 |
| Head | Linear(384,100); classes propagate to model and Mixup; not an accidental 10-class classifier |
| Neurons | inherited MultiStepLIF/Sigmoid, CuPy production backend; SPS alpha4 |
| Training geometry | RandomResizedCropAndInterpolation to32, scale=(1,1), ratio=(1,1), bicubic; no CIFAR padding-4 random-crop recipe |
| Flips/color | hflip .5, vflip0; color_jitter0; RandAugment takes precedence over color jitter |
| RandAugment | rand-m9-n1-mstd0.4-inc1, existing local aa_snn implementation |
| Eval geometry | resize32 and center-crop32 (crop_pct1), bicubic; deterministic |
| Mean | [0.4914,0.4822,0.4465] inherited CIFAR10 values |
| Std | [0.2470,0.2435,0.2616] inherited CIFAR10 values |
| Normalization | PrefetchLoader: (uint8.float -255*mean)/(255*std); CPU probe uses equivalent ToTensor/Normalize |
| Random Erasing | .25 probability, const mode, one region; training prefetcher only; none in validation/probe |
| Mixup | alpha .5, probability1, batch mode; off_epoch200 (never disabled in this 60-epoch prefix) |
| CutMix | alpha0, minmaxNone: disabled; switch_prob.5 is inactive for CutMix |
| Training loss | SoftTargetCrossEntropy; smoothing .1 through FastCollateMixup with num_classes100 |
| Eval loss | ordinary CrossEntropyLoss on hard targets |
| Sampler | single process, shuffled RandomSampler training; sequential evaluation; no distributed/repeated-augmentation sampler |
| Batch | train128/drop_lastTrue: 390 batches,49,920 examples/epoch; eval128/drop_lastFalse:79 batches,all10,000 examples |
| Workers |16, seeded generator per seed; persistent_workers=False in deterministic mode; pin_memFalse by parser default |
| Optimizer | timm create_optimizer AdamW, LR5e-4, weight_decay.06; default betas(.9,.999),eps1e-8; no LR batch rescaling |
| Weight decay groups | timm defaults exclude bias/1D parameters and model-specified exclusions from decay |
| Schedule | cosine configured for300 epochs +10 cooldown; wrapper stops at60, not a cosine retuned to60 |
| Warmup |20 epochs from1e-5; cosine minimum1e-5; no LR noise configured |
| AMP | native AMP autocast/GradScaler; inherited overflow skip/backoff and persistence guards; no clipping (clip_gradNone) |
| Training | no freezing/pretraining/initialization checkpoint/EMA; fresh seed42/43/44, deterministic settings |
| Eval | eval/no_grad, AMP model forward, sample-weighted top1/top5; neurons reset after every batch; no TTA |
| Epoch horizon |60, zero-based0..59; no early stopping |
| Checkpoints | timm best by validation top1, last, history1; optimizer/scaler plus scheduler/best/RNG and online policy/monitor state |
| Logs | resolved config/args, per-epoch metrics/controller/AMP/timing, actual implementation commit and integrity hashes |

Training top1 is expected smoothed/Mixup target mass, NOT ordinary hard-label
training accuracy. Validation top1 is ordinary hard-label accuracy over all test
examples. There is no independent final-test-only evaluation in this protocol.

The dataset and head were already correct. What was unverified/inherited was
normalization and every other CIFAR10 recipe choice, particularly crop policy,
augmentation, optimization and truncated300-epoch schedule. CIFAR10 normalization
on CIFAR100 is not a software error by itself, but must not be called a verified
dataset-specific/published recipe. No guessed replacement constants are introduced.
If CIFAR100-specific normalization or a full schedule is desired, it needs a new
explicit reviewed protocol, applied identically to Fixed and SAGE, before launch.

## New Isolated Preparation

Only five NEW files are added: this audit, the CIFAR100-specific config, entry
point, launcher and tests. No file used by the active CIFAR10 entry point is edited.
The new config makes inherited mean/std explicit and validates that all remaining
training fields still match controller_online.yml. Preparation requires explicit
acknowledgment of this adaptation. It reads ONLY the completed method-ablation
JSON and original immutable deep_only selection, not active CIFAR10 outputs.

The parent selection stays attributed to7cd92a8d7402957083c855823dca906d5ac681b6.
The new CIFAR100 approval/rows/checkpoints record the NEW actual implementation
commit separately, plus parent selection commit/hash and recipe/runtime/probe
fingerprints. Existing CIFAR10 approval is neither rewritten nor copied as a
new-commit approval. Deep-only is provisional: the supplied ~6.55% mean G-variance
reduction does not meet the predeclared25% stabilization target; no claim of
method success is fabricated and the running study is not changed.

Each seed/method has its own lock, directory, last/best checkpoints and status
under results/gradient_flow/cifar100_deep_only_followup/seed42/{fixed,final_sage}
(analogously43/44). Resume authorizes only the exact local checkpoint after
approval/config/identity/epoch/hash validation; full loading has no unsafe
fallback and generic loaders are untouched. Completed runs skip; failed runs
require manual review. The shared official loop/runtime is reused unmodified.

## Active CIFAR10 Execution Structure

run_controller_followup.sh is a SIX-TASK SLURM ARRAY, not a shell loop over six
runs. Mapping:0 Fixed42,1 SAGE42,2 Fixed43,3 SAGE43,4 Fixed44,5 SAGE44.
Every task requests one GPU/16CPUs/64G and runs ONE training process. Independent
tasks are eligible to overlap. The submitted --nodelist=gpu006 restricts every
task to that node; if it has only one allocatable GPU, tasks queue one-at-a-time.
The launcher does not guarantee six simultaneous GPUs and no live node capacity
or remaining runtime can be inferred from the script. Do not restart that array.

## Commands: Separate Checkout, NO Launch During Preparation

DO NOT git pull/switch/checkout the active CIFAR10 working tree while its pending
or running array tasks still depend on it: its selection pins the original HEAD.
Use fetch plus a detached worktree (NEW_SHA is the preparation commit):

    cd /home/rizk_lab/shared/kiran/UM-SEN
    git fetch origin gradient-flow-stage1-validation
    git worktree add --detach ../UM-SEN-cifar100 NEW_SHA
    cd ../UM-SEN-cifar100
    mkdir -p logs/gradient_flow
    python -m experiments.gradient_flow.cifar100_followup --show-config

After reviewing the explicit inherited recipe, prepare metadata ONLY:

    python -m experiments.gradient_flow.cifar100_followup --prepare --source-root ../UM-SEN --reviewed-by kiran --acknowledge-inherited-normalization --data-dir /path/to/existing/cifar100/cache

The data must already exist; this does not download it or launch training.
Only after explicit launch approval, six independent tasks can be submitted with
sbatch --array=0,2,4 (Fixed42,43,44 individually) and --array=1,3,5 (SAGE42,43,44
individually) against jobs/gradient_flow/run_cifar100_deep_only_followup.sh.
Alternatively its default array0-5 submits all six independently. No node is
pinned by the file. --exclude=gpu006 can keep future CIFAR100 jobs off the active
CIFAR10 node. Up to six GPUs may be used concurrently if scheduling/quotas, CPU
and RAM permit; this is not a claim of six currently available GPUs.

Local validation is CPU/unit/config only. No HPC job is queried, launched,
resumed, stopped or modified, and gpu006 is not used for validation.
