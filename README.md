# UM-SEN Research Repository

Two parallel research tracks share the existing implementations in `spikformer/`:

| Track | Configurations | Experiments | Analysis | Jobs |
| --- | --- | --- | --- | --- |
| SAGE | `configs/sage_v2/` | `experiments/sage_v2/` | Existing historical material in `analysis/` | `jobs/sage_v2/` |
| Gradient-flow research | `configs/gradient_flow/` | `experiments/gradient_flow/` | `analysis/gradient_flow/` | `jobs/gradient_flow/` |

The gradient-flow track currently contains documentation only. No diagnostic,
surrogate controller, new training strategy, or completed result is implemented
or implied by these directories.

Generated checkpoints, logs, results and figures use the corresponding ignored
`<kind>/<track>/` directories. Existing jobs may keep checkpoints inside their
result directories; this organization does not change those established paths.
Datasets remain under root-level `data/` or an explicitly supplied runtime path.
Root-level `docs/` and `paper/`, if present on another clone, stay at the root.

See [the organization inventory](experiments/REPOSITORY_LAYOUT.md) for before/after
trees, classification, preserved historical references and clone setup commands.
Existing SAGE commands remain in [the SAGE-v2 documentation](experiments/sage_v2/README.md)
and [the DVS experiment notes](experiments/sage_v2/CIFAR10DVS.md).

Do not mix historical CogMI, SAGE-v2 and gradient-flow metrics automatically.
Historical provenance remains in `archive/cogmi2026/`; preserving an old artifact
does not establish that it is a validated result for either new research track.
