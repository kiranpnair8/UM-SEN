# Parallel Research Track Organization

## Scope and safety

Inspected local repository baseline: `07f97ab8d5df9149c6ac7455e071f84b347e6263`.
This is a source organization inventory, not an experiment result or an HPC
filesystem audit. The initial tracked working tree was clean.

No source/config/job was moved, renamed or deleted. No historical path reference,
model/controller equation, training behavior or dependency was changed. SAGE-v2
files were already namespaced; older scripts stay in place because imports and
historical output paths are part of their reproducibility context.

## Tree before

```text
UM-SEN/
  .gitignore
  spikformer/                         shared and existing dataset-specific source
  configs/sage_v2/
    README.md
    fixed_cifar10_adaptation.yml
    sage_cifar10dvs_t10.yml
  experiments/sage_v2/
    README.md
    CIFAR10DVS.md
    manifests/README.md
  jobs/
    <8 historical launchers listed below>
    sage_v2/
      README.md
      run_fixed_adaptation.sbatch
      run_sage_cifar10dvs.sbatch
  analysis/<7 historical scripts listed below>
  scripts/<2 historical scripts listed below>
  archive/cogmi2026/
    manifests/{README.md,historical_runs.yaml}
    source_snapshots/README.md
```

`checkpoints/`, `figures/`, `logs/`, `old_logs/`, `output/`, `results/`, `data/`,
`docs/` and `paper/` were absent locally. Their possible contents on the HPC are
unknown and were not inferred, moved or recreated here.

## Tree after

```text
UM-SEN/
  README.md                          new tracked navigation
  .gitignore                         adds old_logs/ exclusion
  spikformer/                        unchanged, at root
  configs/
    sage_v2/                         unchanged
    gradient_flow/README.md           new tracked namespace
  experiments/
    REPOSITORY_LAYOUT.md              this tracked inventory
    sage_v2/                         unchanged
    gradient_flow/README.md           new tracked namespace
  analysis/
    <existing 7 scripts>              unchanged paths
    gradient_flow/README.md           new tracked namespace
  jobs/
    <existing 8 historical launchers> unchanged paths
    sage_v2/                         unchanged
    gradient_flow/README.md           new tracked namespace
  scripts/                           unchanged
  archive/                           unchanged, at root
  checkpoints/{sage_v2,gradient_flow}/ empty, ignored local runtime containers
  logs/{sage_v2,gradient_flow}/        empty, ignored local runtime containers
  results/{sage_v2,gradient_flow}/     empty, ignored local runtime containers
  figures/{sage_v2,gradient_flow}/     empty, ignored local runtime containers
```

`data/`, `docs/` and `paper/` remain root-level conventions if present on other
clones; no empty versions were fabricated. `old_logs/` and `output/` remain absent
locally. Empty ignored directories are not represented by Git/GitHub.

## Classification of all existing items in the inspected source directories

Directory classifications are containers; each existing leaf file is listed.
Historical code may implement SAGE-specific logic while its primary organizational
classification remains legacy/archive. Classification does not authorize a move.

| Existing item | Classification | Action / reason |
| --- | --- | --- |
| `configs/` | Shared infrastructure | Retain parallel track container |
| `configs/sage_v2/` | SAGE-specific | Already correctly namespaced |
| `configs/sage_v2/README.md` | SAGE-specific | Keep |
| `configs/sage_v2/fixed_cifar10_adaptation.yml` | SAGE-specific | Fixed-SG pilot within the SAGE-v2 track; keep |
| `configs/sage_v2/sage_cifar10dvs_t10.yml` | SAGE-specific | Keep |
| `experiments/` | Shared infrastructure | Retain parallel track container |
| `experiments/sage_v2/` | SAGE-specific | Keep |
| `experiments/sage_v2/README.md` | SAGE-specific | Keep |
| `experiments/sage_v2/CIFAR10DVS.md` | SAGE-specific | Keep |
| `experiments/sage_v2/manifests/` | SAGE-specific | Keep |
| `experiments/sage_v2/manifests/README.md` | SAGE-specific | Keep tracked provenance conventions |
| `analysis/` | Shared infrastructure | Mixed historical code and new track container; keep |
| `analysis/analyze_block1_controller.py` | Legacy/archive | Historical SAGE analysis; preserve input paths |
| `analysis/analyze_umsen_multiseed.py` | Legacy/archive | Historical SAGE analysis; preserve input paths |
| `analysis/plot_attention_uncertainty.py` | Legacy/archive | Historical entropy analysis; preserve input paths |
| `analysis/plot_surrogate_table_bars.py` | Legacy/archive | Manually supplied table plotting, not run-derived provenance |
| `analysis/run_umsen_100ep.py` | Legacy/archive | Imports mechanism runner; preserve relative layout |
| `analysis/run_umsen_mechanism_test.py` | Legacy/archive | Unique simplified controller/shuffled-control code; keep |
| `analysis/run_umsen_multiseed.py` | Legacy/archive | Imports mechanism runner; preserve relative layout |
| `scripts/` | Legacy/archive | Existing comparison bundle; no relocation |
| `scripts/run_surrogate_comparison.py` | Legacy/archive | Existing trainer paths/unsupported-method checks; keep |
| `scripts/plot_surrogate_comparison.py` | Legacy/archive | Existing comparison result namespace; keep |
| `jobs/` | Shared infrastructure | Historical launchers plus track namespaces; keep |
| `jobs/resume_umsen_100ep.sbatch` | Legacy/archive | Historical continuation path; keep |
| `jobs/run_attention_entropy_diagnostic_20ep.sh` | Legacy/archive | Historical diagnostic output path; keep |
| `jobs/run_surrogate_comparison.sbatch` | Legacy/archive | Historical comparison runner path; keep |
| `jobs/run_umsen_100ep.sbatch` | Legacy/archive | Historical checkpoint/result paths; keep |
| `jobs/run_umsen_controller_debug.sbatch` | Legacy/archive | Historical mechanism debug path; keep |
| `jobs/run_umsen_mechanism_test.sbatch` | Legacy/archive | Historical mechanism runner path; keep |
| `jobs/run_umsen_multiseed.sbatch` | Legacy/archive | Historical multiseed runner path; keep |
| `jobs/run_umsen_official_recipe.sbatch` | Legacy/archive | Official-recipe historical SAGE run/resume paths; keep |
| `jobs/sage_v2/` | SAGE-specific | Already correctly namespaced |
| `jobs/sage_v2/README.md` | SAGE-specific | Keep |
| `jobs/sage_v2/run_fixed_adaptation.sbatch` | SAGE-specific | Protected historical initialization reference; keep |
| `jobs/sage_v2/run_sage_cifar10dvs.sbatch` | SAGE-specific | Keep |

Additional root items remain: `.gitignore` is shared infrastructure; `spikformer/`
contains shared backbones plus SAGE/dataset-specific modules, all retained in
place; `archive/cogmi2026/` and its three existing documentation/manifest files are
legacy/archive. No existing item was found whose classification required a source
move. Contents of absent HPC runtime directories are **unclear**, not presumed
regenerable or safe to relocate.

## Runtime artifacts and preserved path usage

None of these artifacts exists in this workspace. This table records source
references, not a claim that any particular run is present or complete.

| Established runtime path | Scripts/configs referencing it | Organization decision |
| --- | --- | --- |
| `results/umsen_official_recipe/` | `jobs/run_umsen_official_recipe.sbatch`; `jobs/sage_v2/run_fixed_adaptation.sbatch`; historical manifest and SAGE-v2 documentation | Protected external history; no move or reference edits |
| `results/umsen_100ep/` | `analysis/run_umsen_100ep.py`; `jobs/run_umsen_100ep.sbatch`; `jobs/resume_umsen_100ep.sbatch` | Preserve checkpoint subdirectory and continuation paths |
| `results/umsen_mechanism_test/` | `analysis/run_umsen_mechanism_test.py`; mechanism/debug jobs | Preserve historical paths |
| `results/umsen_multiseed/` | `analysis/run_umsen_multiseed.py`; both multiseed/controller analysis scripts; multiseed job | Preserve historical paths |
| `results/attention_entropy_diagnostic_20ep/` | Entropy diagnostic job and `analysis/plot_attention_uncertainty.py` | Preserve historical paths |
| `results/surrogate_comparison/` | Both comparison scripts and comparison job | Preserve historical paths |
| `figures/umsen_multiseed/` | Both multiseed/controller analysis scripts | Preserve historical paths |
| `figures/attention_entropy_diagnostic_20ep/` | `analysis/plot_attention_uncertainty.py` | Preserve historical paths |
| `figures/surrogate_comparison/` | `scripts/plot_surrogate_comparison.py` | Preserve historical paths |
| `figures/surrogate_table_bars/` | `analysis/plot_surrogate_table_bars.py` | Preserve supplied-table provenance |
| `logs/` directly | All eight historical launchers | Preserve log identity/job correspondence |
| `output/train/` | Default output fallback in static CIFAR/ImageNet training | Preserve legacy defaults |
| `output/attention_entropy_diagnostic/` | Default in the passive CIFAR diagnostic | Preserve legacy default |
| Relative `logs` / TensorBoard suffixes | Legacy DVS trainer defaults | Preserve legacy behavior |
| `results/sage_v2/`, `logs/sage_v2/` | SAGE-v2 configs/jobs; CIFAR output guard; DVS SAGE runner | Keep current namespaced usage |
| Root `checkpoints/`, `old_logs/` | No active root-specific default located in this pass; simplified checkpoints live under result directories | No artifacts available; no migration authorized |

For future loose SAGE-v2 artifacts, `checkpoints/sage_v2/`, `logs/sage_v2/`,
`results/sage_v2/` and `figures/sage_v2/` are appropriate destinations only after
inventory, producing-source/config verification, checkpoint hashes, reference
search and migration approval. Historical CogMI artifacts are not automatically
SAGE-v2 artifacts; a verified future archive destination may instead be
`archive/cogmi2026/artifacts/`. Retain a tracked original-to-archive manifest before
any approved migration. Do not duplicate or move checkpoints merely for symmetry.

Protect historical `baseline_seed42`, `baseline_seed43`, `baseline_seed44` best and
last checkpoints; preserve SAGE histories and the incomplete `umsen_seed44` as-is.
Do not run historical launchers as part of organization or aggregate historical
metrics into a new track.

### old_logs/ and output/ review

- `old_logs/`: absent locally. If found on HPC, leave it in place until job IDs,
  source revisions and run associations are mapped. Verified historical logs can
  later be archived with a path manifest. Do not rename or delete it based only
  on the directory name; its contents remain unclear. Newly ignore this runtime
  directory to prevent accidental tracking, without removing files.
- `output/`: absent locally and still referenced by trainer/diagnostic defaults.
  Keep its convention now. Later classify each run before selectively archiving;
  do not rename the whole directory or change existing defaults. Eventual deletion
  requires evidence that individual outputs are regenerable and carry no unique
  provenance. No deletion or move is authorized in this pass.

## Reference and ignore policy

Existing executable/config path references changed: **none**. Only new README
navigation and conventions were added. Existing source imports, YAML inheritance,
historical launch commands and protected checkpoint references remain intact.
Legacy machine-specific defaults were not rewritten; new namespaces contain no
machine-specific absolute configuration paths or executable configs.

`.gitignore` already excludes datasets, weights, checkpoints, logs, results,
figures, output, Python/CuPy caches and SLURM logs. Add only `old_logs/`. Do not
blanket-ignore YAML/JSON, research manifests or source. Empty runtime directories
have no tracked placeholders, result claims or fabricated checkpoint files.

## Clone consistency and later commit

Tracked README files make the four new source directories reproducible through
Git on local/GitHub/HPC clones. Ignored data and runtime artifacts are neither
synchronized by Git nor removed by a pull. The empty runtime layout can be made
on any Bash-based clone from its root using:

```bash
mkdir -p checkpoints/{sage_v2,gradient_flow} logs/{sage_v2,gradient_flow} \
  results/{sage_v2,gradient_flow} figures/{sage_v2,gradient_flow}
```

After review, stage only the organization files, commit and push:

```bash
git status --short
git add -- .gitignore README.md experiments/REPOSITORY_LAYOUT.md \
  configs/gradient_flow/README.md experiments/gradient_flow/README.md \
  analysis/gradient_flow/README.md jobs/gradient_flow/README.md
git diff --cached --check
git diff --cached --stat
git commit -m "chore: organize parallel SAGE and gradient-flow research tracks"
git push origin main
```

On the HPC, enter the existing UM-SEN repository root, inspect local changes,
then use `git pull --ff-only origin main` and the runtime mkdir command above.
If the pull refuses a dirty/divergent branch, preserve local work and resolve it
before retrying; do not reset, clean, delete outputs or force-push. No commit,
push, HPC sync or training is performed by this organization pass.
