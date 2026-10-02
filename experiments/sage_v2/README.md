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
