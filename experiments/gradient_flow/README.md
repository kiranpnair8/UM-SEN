# Gradient-Flow Research

This namespace is reserved for research on block-specific surrogate gradients and
backward optimization dynamics in Spiking Transformers. There is no finalized
controller or method name, and no diagnostics or training implementation yet.

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
