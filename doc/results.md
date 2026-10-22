# Results status

The supported attention architecture awaits a clean training run. There is no selected attention checkpoint or matching accuracy, AUROC, F1, ECE or latency benchmark in `models/`. The earlier architecture screen informed the user's choice and is not reported as the outcome of this fresh run.

The previous MLP checkpoint and all matching published metrics/figures are preserved together in ignored `retired/mlp-fusion/`. They are not metrics for the attention model. Local code/asset tests verify behavior, not predictive quality.

After training, report selection results for the selected checkpoint, fit temperature on independent calibration groups, and evaluate the raw and scaled variants separately. Bind every report to checkpoint, dataset, temperature and prediction hashes. Include class support, per-task accuracy/F1/AUROC, NLL, Brier, ECE and signal-use checks. Mark unsupported classes and already-inspected historical test data explicitly. Measure latency through uncached-signal public inference on identified hardware.
