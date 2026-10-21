# User preferences

- Use established machine-learning terminology in explanations, reports, and new names. Describe custom combinations by their standard components; do not present implementation labels as recognized or novel ML architectures. Refer to the current decision head as an MLP fusion head with element-wise (Hadamard) product features. Existing code/config identifiers may remain for compatibility.
- Put experiment runners, audit, analysis and postprocessing tools in `scripts/`; keep reusable model and public input code in `paya/`. Keep machine-specific `.sh` helpers in ignored `.temp/`.
- Current research boundary: complete the small experiments and report outcomes, then stop for the user's review before expanding to any full run. The requested final NormWear encoder comparison is included in that review boundary.
