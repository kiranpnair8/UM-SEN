# Historical CogMI 2026 Provenance

[historical_runs.yaml](historical_runs.yaml) records the historical HPC run
summaries supplied by the researcher. These are external historical records,
not newly executed experiments or independently verified filesystem findings.
The known repository source reference is
`e01d1db535548a1e8505ff805673569c98107449`; exact producing HPC source versions
and any local differences still require verification.

HPC results, checkpoints, datasets, and logs remain at their existing paths.
Nothing has been copied, moved, deleted, or fabricated. Historical CogMI metrics
must not be automatically included in SAGE-v2 / IJCNN 2027 aggregates.

## Historical source retained in place

| Files | Historical role |
| --- | --- |
| `analysis/run_umsen_mechanism_test.py` | Simplified mechanism tests and shuffled controls |
| `analysis/run_umsen_multiseed.py` | Simplified five-epoch multiseed comparison |
| `analysis/run_umsen_100ep.py` | Simplified 100-epoch experiment and resume |
| `analysis/analyze_umsen_multiseed.py`, `analysis/analyze_block1_controller.py` | Seed/controller analysis |
| `spikformer/cifar10/attention_entropy_diagnostic.py`, `analysis/plot_attention_uncertainty.py` | Passive entropy-temperature, sparsity, Gini, and per-head diagnostics |
| `analysis/plot_surrogate_table_bars.py` | Manually supplied paper-table plotting; not run-derived provenance |
| `scripts/run_surrogate_comparison.py`, `scripts/plot_surrogate_comparison.py` | Historical comparison orchestration and plotting, including unsupported-method checks |
| Existing files directly under `jobs/` | Historical launchers, including official-recipe training and resume |
| `spikformer/imagenet/`, `spikformer/cifar10dvs/` | Separate historical backbone/training/evaluation pipelines |

The simplified experiments are distinct from the official-recipe runs. The
multiseed and 100-epoch runners import the mechanism runner; preserve that bundle
and its relative layout. Active CIFAR source, compatibility helpers, configs,
tests, original images, and licensing remain in place.

## Protected checkpoint policy

Protect best and last checkpoints, original configuration, summary CSV/JSON,
controller history, all job/resume logs, and producing source/environment data.
`umsen_seed44` is incomplete and must not be treated as a completed run.

Periodic or recovery checkpoints are only candidates for a later manual pruning
review. Before pruning, verify retained files are readable, record hashes and
epoch/metric metadata, and preserve every checkpoint needed for an intermediate
reported result, recovery, or unique provenance. Similar filenames and sizes do
not establish duplication. No checkpoint pruning is authorized by this cleanup.

Future archived bulk artifacts, if explicitly supplied and reviewed, may use
`archive/cogmi2026/artifacts/` (ignored). Keep manifests outside that directory.
