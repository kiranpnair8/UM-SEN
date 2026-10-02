# SAGE-v2 Configurations

`fixed_cifar10_adaptation.yml` is a development pilot, not a validated final
training strategy. It inherits `spikformer/cifar10/cifar10.yml` using a relative
`base_config`, preserving architecture, AdamW, weight decay, batch size,
RandAugment, Mixup, Random Erasing, smoothing, loss selection, and AMP.

Overrides: 30 adaptation epochs, seed 42, T=4, fixed alpha=4, cosine scheduling,
no warmup/cooldown, LR=1e-5 (CLI may select 5e-5), and minimum LR=1e-7. These are
pilot choices rather than scientific conclusions. All parameters remain trainable.
No SAGE-v2 algorithm has been selected or implemented.

The CLI overrides both inherited and local YAML values. Checkpoint/data/output
paths must be supplied explicitly. Resolved `args.yaml` and `provenance.json`
are saved only when a real run starts, under its ignored runtime directory.

- From scratch: no initialization/resume option; historical defaults remain.
- Resume: `--resume` continues available training state.
- Adaptation: `--init-checkpoint` loads weights only, with epoch 0 and fresh
  optimizer, scheduler, AMP scaler, best metric, RNG progression, and controller.

Never combine initialization and resume. `--fixed-alpha` targets transformer
blocks only (SSA and MLP neurons). SPS/input-stage neurons use the separate
`--sps-alpha`, default 4. Changing block alpha alone never changes SPS alpha.
The unchanged adaptive controller requires the historical block reference 4.

See [experiment conventions](../../experiments/sage_v2/README.md).
