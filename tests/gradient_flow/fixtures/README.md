# Completed Recovery Gate Regression

`completed_recovery_gate.json` contains the actual completed Lawrence summary
provided by the user, the run settings, and a minimal projection of the three
flagged alternating/response-aware traces. It is deliberate test data, not a new
experimental run or an initialization asset. It contains no checkpoints, image
batches, HPC absolute paths, or fabricated experimental values.

Source run commit: `e07e5ea8676a62b6dc0d101d51e7d81a7e9859fb`.
The fixture records SHA256 of the uploaded archive and original summary.
All 30 recovery-file hashes were validated during the audit; all 90 saved cases
reproduced the gate decision. Minor cross-interpreter floating-point summation
differences (maximum approximately4.44e-16) did not alter any predicate.

The regression deliberately retains failure: early/alternating oscillation
fraction0.1 and late/alternating0.2 violate the declared zero-oscillation gate,
despite complete recovery coverage and small final errors. Component reference
crossings can coexist with monotonically decreasing total L2 error. Treating
that warning differently would require a separately declared criterion revision,
not an arithmetic fix to these results.
