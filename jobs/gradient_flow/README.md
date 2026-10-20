# Gradient-Flow Jobs

`run_controller_recovery_cifar10.sh` pins gpu002 and runs only the provisional
frozen-state recovery study and figures. It fails the job if the real-data gate
does not pass. Online training remains deferred; see
[the feasibility protocol](../../experiments/gradient_flow/CONTROLLER_FEASIBILITY.md).

`run_alpha_state_dependence_cifar10.sh` explicitly pins gpu002. It audits historical
milestones, conditionally regenerates one separate Fixed-SG trajectory, and reuses
the multi-batch probe/analysis tooling across early/mid/late. See
[the state protocol](../../experiments/gradient_flow/TRAINING_STATES.md).

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
