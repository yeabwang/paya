# Experiments

The release is **Paya full-003, step 4,797**, selected by NLL on 196,707 selection rows. It has the lowest recorded full-selection NLL among the completed comparable development and full-data candidates. Tiny fitting checks and scores on smaller selection subsets are not ranked against full-corpus results.

| Stage | Outcome |
| --- | --- |
| Small fitting and architecture/loss screens | Verified learning, candidate alignment and frozen-encoder behavior; informed larger controls. |
| Expanded attention/MLP controls | Full-selection NLL ranged from 0.78627 to 0.81303; none supersedes the final full-data checkpoint. |
| Full-002 attention baseline | Selected NLL 1.01766; historical test accuracy 76.61%, AUROC 0.60189, ECE 0.07258. |
| Full-003 MLP, label smoothing 0.05 | Selected NLL 0.76180; historical test accuracy 76.86%, AUROC 0.67465, ECE 0.02115. |
| Eight checkpoint-continuation pilots | All retained the original parent; no branch passed the selection/retention criteria. |

Full-003 trained a fresh supervised head on 1,637,289 retained QA rows for three epochs, batch 512, totaling 9,594 updates. AdamW used learning rate 0.0003 and weight decay 0.01; label smoothing was 0.05, without class weighting. Selection occurred every 1,599 updates. The selected checkpoint is halfway through epoch two, rather than the last update.

Frozen FP32 signal/text features were cached with checksums and recoverable progress. Sampling visited every training row once per epoch. Existing train/test membership and the subject/recording/waveform separation were preserved. The supplied validation corpus was divided before training into selection and independent calibration roles.

A global temperature of 1.04831 improved calibration-set NLL slightly but worsened historical test NLL, Brier and ECE. The release uses the raw checkpoint with temperature 1. Earlier benchmark findings informed development, so the test comparison is retrospective. Architecture, smoothing and data-access changes were combined in the final run; aggregate improvements cannot be attributed to architecture alone.

Training revision: `48754f3b86cb0c29e04673acfefd6f2f6418f55e`. Historical implementations, configurations and detailed experiment outputs remain in ignored `retired/2026-10-21/` and Git history. See [fine-tuning outcomes](finetune.md) and [release results](results.md).
