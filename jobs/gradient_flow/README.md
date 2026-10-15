# Gradient-Flow Jobs

Reserved for future reviewed gradient-flow launchers. No job script is created or
submitted in this organization pass.

Use repository-relative configuration paths and runtime environment/CLI overrides
for machine-specific dataset and environment locations. Submit from the repository
root, isolate runtime output under this track's namespaces, and retain the
project's maximum walltime of `2-00:00:00`.

If SLURM output/error directives refer to `logs/gradient_flow/`, that directory must
exist before submission: SLURM opens its log before the script can create it.
Existing SAGE and historical launchers remain unchanged in their original paths.
