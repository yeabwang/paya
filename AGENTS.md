# Project conventions

- Use established machine-learning terminology. The selected head is an MLP fusion head with Hadamard product features.
- Keep supported model, input, training and inference code in `paya/`, with tests in `tests/`.
- Keep historical code and experiment tools in ignored `retired/`; local shell helpers stay in ignored `.temp/`.
- Preserve the selected checkpoint identity and predictions. Report metrics only for their matching checkpoint variant.
