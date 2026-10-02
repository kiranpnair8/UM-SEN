# SAGE-v2 Launchers

Store new reviewed SLURM launchers here. Future jobs
must use `results/sage_v2/` and `logs/sage_v2/`, never the historical output paths.
Walltime must not exceed `2-00:00:00`.

Existing jobs in the parent directory are historical CogMI launchers. They keep
their original paths and behavior for provenance; they are not SAGE-v2 jobs.

See [experiment conventions](../../experiments/sage_v2/README.md).

## Fixed-SG initialization pilot

From the repository root, after reviewing the environment/data paths:

```bash
# One full epoch to check load, outputs, validation, and checkpoint writing:
sbatch --export=ALL,SMOKE_TEST=1,LR=1e-5 jobs/sage_v2/run_fixed_adaptation.sbatch
# Separate 30-epoch submissions (one LR per job):
sbatch --export=ALL,LR=1e-5 jobs/sage_v2/run_fixed_adaptation.sbatch
sbatch --export=ALL,LR=5e-5 jobs/sage_v2/run_fixed_adaptation.sbatch
```

These commands are instructions only; no jobs have been submitted.
The launcher requests gpu005, one GPU, 64G, 16 CPUs, and at most two days.
It activates `/home/rizk_lab/shared/kiran/envs/snn`, reads the protected
`baseline_seed42/model_best.pth.tar`, and writes distinct run IDs for each LR
under `results/sage_v2/cifar10/fixed/T4/seed42/`. Each submission runs exactly
one LR (default 1e-5). Both output directory and stdout/stderr log include
`lr<LR>_<pilot-or-smoke>_<job-ID>`; logs live under `logs/sage_v2/`.
CuPy's cache is redirected away from the home quota.
Smoke runs have their own IDs and must not be aggregated as completed pilots.

The pilot does not auto-resume or overwrite a prior run. To continue a pilot,
use its own `last.pth.tar` with `--resume --sage-v2` and the original run ID,
without `--init-checkpoint`, preserving the pilot's config and seed.
