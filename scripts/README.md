# Experiment tools

Run these modules from the repository root with `python -m scripts.<module>`.
Reusable model, inference, input and checkpoint code belongs in `paya/`.
Experiment runners, audits, selection analysis and postprocessing belong here.
Machine-specific shell helpers remain in ignored `.temp/`.

| Module | Purpose |
| --- | --- |
| `coverage` | Audit QA consistency and independent subject/recording coverage |
| `group_split` | Create group-stratified role manifests without copying waveforms |
| `development_bank` | Build verified frozen-encoder feature banks |
| `development` | Historical controlled architecture and optimization experiments |
| `finetune` | Recoverable, short checkpoint-continuation comparisons |
| `finetune_controls` | Replay sampling, regularization, PCGrad and retention checks |
| `selection_metrics` | Selection metrics with canonical answer ordering |
| `postprocess` | Checkpoint validation, calibration, evaluation and latency reports |
| `finetune_integration` | Small fitting and public-API checkpoint reload diagnostic |
| `finetune_review` | Evaluate final pilot weights and predefined interpolations |
| `fetch_finetune_reports` | Retrieve small reports through the secure local helper |
| `finetune_support` | Audit label support in the exact training/monitor samples |
| `verify_finetune_tracking` | Read back online W&B histories for completed pilots |
| `report_finetune` | Generate the review report from aggregate outputs |
| `normwear_audit` | Inspect pinned public source/configuration without weights |

For example: `python -m scripts.finetune --plan config/finetune-conflict.json`.
Plans bind source hashes, inputs and parent checkpoints. Moving code changes the
binding: do not resume a historical run using a new revision. Keep its committed
revision available. Completed outputs remain valid and must not be overwritten.

The current experiment phase stops after small comparisons and reporting. Full
training, including the requested NormWear comparison, awaits the user's review.
