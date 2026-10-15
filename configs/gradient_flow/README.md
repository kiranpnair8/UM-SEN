# Gradient-Flow Configurations

`cifar10_diagnostics.yml` inherits the existing CIFAR-10 recipe, overrides the data
and output paths with repository-relative paths, and enables passive diagnostics.
Defaults are at most three train/evaluation batches and one epoch; the inherited
full optimizer/LR schedule is not shortened. No adaptive controller is present.

`diagnostic_beta` controls the parameter-gradient EMA reference (default 0.95).
`diagnostic_eps` is the transmission ratio epsilon (default 1e-12). The first
directional-consistency observation is null. `--block-alphas` accepts one positive
non-trainable scalar per transformer block; default transformer/SPS alpha is 4.

Keep configuration paths repository-relative. Resolve dataset, environment and
machine-specific locations at runtime through CLI arguments or environment
variables; do not commit local/HPC absolute paths or credentials.

Future configurations may reuse the shared Spikformer implementation and reviewed
training recipes, but must record their exact inheritance and resolved settings.
Do not edit SAGE configurations merely to select this research track.
