# Fine-tuning

Eight pilots continued the same selected full-003 checkpoint. Each used 400 updates, batch 512, seed 42, a fixed 64,000-row training pool and a fixed 16,384-row selection monitor. Final weights were evaluated on all 196,707 selection rows. Encoders and model architecture were unchanged.

Recovery tasks were AF, blood pressure, respiratory rate and sleep breathing. Replay reserved 25% of draws for the other tasks. Default continuation learning rate was 0.00003; label smoothing 0.05, AdamW weight decay 0.01 and gradient clipping 1 were retained.

| Experiment | Final selection NLL | Recovery-task mean F1 change |
| --- | ---: | ---: |
| All tasks | 0.76874 | -0.00361 |
| Weak tasks only | 0.78603 | -0.00262 |
| Replay | 0.77638 | -0.00836 |
| Replay + L2-SP 0.1 | 0.77637 | -0.00836 |
| Replay + distillation 0.1 | 0.77462 | -0.00801 |
| Replay, learning rate 0.00001 | 0.77168 | -0.00470 |
| Replay + PCGrad | 0.77709 | -0.00760 |
| Replay + L2-SP 10 | 0.77577 | -0.00825 |

The parent NLL was **0.76180**. Every pilot selected update zero, and the retained head tensors equal the parent exactly. Weak-task-only training improved AF F1 slightly while degrading respiratory/sleep-breathing and several HRV results. Replay, regularization, distillation, lower learning rate and PCGrad did not meet the joint objective. Three predefined parent/replay weight interpolations also failed. No fine-tuned candidate replaces the release.

A plateau controller halved learning rate after two unsuccessful checks. Stopping patience was five checks; the four-check pilot budget ended before early stopping could trigger. The 16-example fitting diagnostic reached 16/16 and passed public-API reload; this verifies wiring, not generalization. All eight online W&B histories were verified through update 400.
