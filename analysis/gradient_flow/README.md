# Gradient-Flow Analysis

`plot_diagnostics.py` plots block-by-epoch heatmaps and per-block temporal curves
for gradient transmission, directional consistency, parameter-gradient L2 norm
and firing rate. It also renders finite-difference response matrices. It exports
300-DPI PNG and PDF files and masks missing/invalid observations.

```bash
python -m analysis.gradient_flow.plot_diagnostics \
  --metrics results/gradient_flow/cifar10_diagnostic_seed42/epochs.json \
  --output-dir figures/gradient_flow/cifar10_diagnostic_seed42
```

Epoch metrics are means over valid step observations, with explicit counts; the
plotter does not pool datasets, checkpoints or research tracks automatically.

Future analysis must explicitly select `results/gradient_flow/`, verify run
identity/completion, and write generated figures to `figures/gradient_flow/`.
Keep reusable analysis code tracked; generated figures and summaries stay ignored.
Existing files in the parent `analysis/` directory retain their historical SAGE
paths and meanings. Do not aggregate their old results into this track.
