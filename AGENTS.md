# Project conventions

- Use established machine-learning terminology. The supported head applies question/candidate self-attention, then signal cross-attention, followed by a shared MLP candidate scorer with Hadamard product features.
- Keep supported model, input, training and inference code in `paya/`, with tests in `tests/`.
- Keep historical code and experiment tools in ignored `retired/`; local shell helpers stay in ignored `.temp/`.
- Preserve archived checkpoint identities and predictions. The accepted attention architecture will be trained cleanly; do not promote the screening weights or attach historical MLP metrics to it. Report metrics only for their matching checkpoint variant.

## Experiment workflow

- Develop proposed architecture and training changes in `experiments/`, with tests in `tests/`.
- Implement and verify locally first; wait for the user to provide a GPU before running research experiments.
- Promote experimental code into `paya/` only after controlled results establish sound behavior and useful performance. Then move superseded code to ignored `retired/` rather than maintaining legacy branches.
- This project is unreleased: do not bump the package version for these changes. Bind experiments and checkpoints by configuration, source, data and asset hashes.
- Use descriptive operation names. Keep the accepted attention computation as the only supported head. Historical MLP implementations, fallbacks, comparison tools and reports belong in ignored `retired/`. NormWear is outside the active work.
- Fine-tuning initializes compatible head weights with a fresh optimizer and clears calibration. Recovery resumes an identical run. Select on held-out data, recalibrate on independent groups and evaluate every intended signal/domain before making accuracy or ECE claims.
- Signal type and sampling rate are processing inputs only. Already processed samples bypass additional cleaning/resampling. Do not add physical sensor metadata features, fixed ending text or an extra answer slot.
