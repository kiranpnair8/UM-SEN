# Gradient-Flow Jobs

`run_alpha_multibatch_cifar10.sh` runs the fixed-checkpoint SAGE 25-batch alpha
response validation and saved-output analysis, with safe resume. It launches no
training or adaptive controller. See
[the protocol](../../experiments/gradient_flow/MULTIBATCH.md).

Use repository-relative configuration paths and runtime environment/CLI overrides
for machine-specific dataset and environment locations. Submit from the repository
root, isolate runtime output under this track's namespaces, and retain the
project's maximum walltime of `2-00:00:00`.

If SLURM output/error directives refer to `logs/gradient_flow/`, that directory must
exist before submission: SLURM opens its log before the script can create it.
Existing SAGE and historical launchers remain unchanged in their original paths.
