# Fine-tuning experiments

Eight bounded continuation experiments tested whether the selected head could improve weaker tasks without degrading the others. **No candidate qualified for replacement.** The release remains the original full-003 checkpoint at update 4,797 and temperature 1. All experiments retained the same frozen encoders and the same [MLP fusion architecture](architecture.md#decision-head).

## Parent checkpoint and task groups

Parent checkpoint SHA-256:

```text
461b939b19ef59bfa05725c69438b2e8332e031b2586aeab9234fe848c633eba
```

The recovery group contained **AF detection, blood pressure, respiratory rate and sleep breathing**. The other nine tasks formed the protected group. These roles governed sampling, distillation and retention checks; they were not additional inputs to the model.

The motivating question was a stability/plasticity tradeoff: can extra optimization recover weak-task performance while preserving the parent's stronger behavior? A weak-task-only control measured the cost of omitting other tasks. Replay, parameter regularization, distillation and gradient projection tested different ways of preserving them.

## Controlled protocol

| Setting | Value |
| --- | --- |
| Parent / trainable parameters | Same selected raw checkpoint / head only |
| Training pool | Fixed 64,000-row sample from native train; all 52 task–answer pairs present |
| Monitor | Fixed 16,384 selection rows |
| Final comparison | All 196,707 selection rows |
| Budget | 400 updates × 512 examples = 204,800 draws per branch |
| Sampling | Independently sampled batches; rows can recur across updates |
| Seed | 42; fixed pool constructed with seed 20261021 |
| Optimizer | AdamW; learning rate 0.00003 unless specified; weight decay 0.01 |
| Loss / dropout / clipping | Label smoothing 0.05 / dropout 0.1 / global gradient norm 1 |
| Monitoring | Every 100 updates; checkpoint ranking by task-mean NLL |

The monitor covered 50 of the 52 training task–answer pairs; severe sleep breathing and moderate hypoxemia were absent. Final comparisons used full selection, whose own class-coverage limitations still apply. The calibration and test roles were not used to select a continuation branch.

Replay allocated **75% of draws to the four recovery tasks and 25% to the nine protected tasks**, with task-balanced sampling within each group. Within each draw pool, a batch sampled without replacement when enough rows were available, otherwise with replacement; batches did not exhaust the pool before the next update. Task balancing is not class balancing: the original class prevalence within each sampled task remains relevant. The all-task control used the full pool; weak-task-only training omitted protected-task replay.

## Objectives and controls

All branches retained the supervised, real-candidate label-smoothed cross-entropy. Additional controls were tested separately:

| Branch | Change relative to its control |
| --- | --- |
| All tasks | Continue ordinary supervised training on the complete pool |
| Weak tasks only | Restrict supervised training to the four recovery tasks |
| Replay | Mix recovery and protected tasks at 75:25 |
| Replay + L2-SP, 0.1 | Add normalized squared distance to the parent's head parameters |
| Replay + distillation, 0.1 | Match parent output probabilities on protected examples while retaining ground-truth replay |
| Replay, lower rate | Reduce learning rate from 0.00003 to 0.00001 |
| Replay + PCGrad | Project conflicting task gradients before applying the update |
| Replay + L2-SP, 10 | Increase the parameter-preservation coefficient |

L2-SP used `λ × ||θ − θ_parent||² / max(||θ_parent||², 1e-12)`. The normalization makes the recorded coefficients interpretable relative to the parent parameter scale; AdamW weight decay remains a separate term.

Distillation used `λ × T² × KL(p_parent,T || p_student,T)` on protected examples, with **T = 2** and **λ = 0.1**. This training temperature is unrelated to inference calibration; the released model's inference temperature remains 1.

PCGrad used deterministic task order and projected a task gradient when its dot product with another task's gradient was negative. A preceding three-repeat, 32-examples-per-task audit found 62.82% negative task-gradient pairs. That observation motivated the control; it does not establish gradient conflict as the cause of poor generalization.

A plateau controller halved the learning rate after two unsuccessful checks. Early-stopping patience was five checks. The 400-update budget contains only four post-start checks, so **early stopping did not trigger** in these pilots. It should not be credited with the outcome.

## Checkpoint selection and promotion

The research selector minimized task-mean NLL on the fixed monitor. A separate eligible-checkpoint selector required improvement in that objective with no retention failures. Promotion additionally required a meaningful recovery-group gain. The combined criteria were:

1. Recovery-task mean macro F1 improvement of at least **0.02 absolute**.
2. No decline in mean task macro AUROC across all tasks.
3. No decline in protected-task accuracy or macro F1 under the recorded checks.
4. No worsening of pooled NLL or Brier score.

Only checkpoints satisfying the retention checks were eligible to replace the parent. Every branch's best and eligible update remained **zero**. Eligible saved head tensors matched the parent exactly. The table below evaluates each branch's **final update-400 weights**, including unsuccessful weights; it does not describe the retained eligible checkpoint.

## Final-weight results on full selection

AUROC and F1 are equal-weight means across task macro values; accuracy, NLL, Brier and ECE pool QA rows. Recovery ΔF1 is the change in mean F1 across the four recovery tasks. Lower probability errors are better.

| Branch | Accuracy | Macro F1 | Macro AUROC | NLL | Brier | ECE | Recovery ΔF1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Parent** | **69.5735%** | **0.39028** | **0.72725** | **0.76180** | **0.42943** | **0.07911** | — |
| All tasks | 69.2594% | 0.38589 | 0.72675 | 0.76874 | 0.43220 | 0.09048 | −0.00361 |
| Weak tasks only | 69.1729% | 0.39035 | 0.72098 | 0.78603 | 0.44054 | 0.09226 | −0.00262 |
| Replay | 69.3097% | 0.38893 | 0.71679 | 0.77638 | 0.43559 | 0.09166 | −0.00836 |
| Replay + L2-SP 0.1 | 69.3087% | 0.38893 | 0.71680 | 0.77637 | 0.43559 | 0.09165 | −0.00836 |
| Replay + distillation 0.1 | 69.3895% | 0.38908 | 0.71875 | 0.77462 | 0.43471 | 0.08900 | −0.00801 |
| Replay, rate 0.00001 | 69.3859% | 0.38829 | 0.72292 | 0.77168 | 0.43426 | 0.09135 | −0.00470 |
| Replay + PCGrad | 69.3620% | 0.38936 | 0.71648 | 0.77709 | 0.43545 | 0.09453 | −0.00760 |
| Replay + L2-SP 10 | 69.3183% | 0.38924 | 0.71777 | 0.77577 | 0.43548 | 0.09156 | −0.00825 |

Weak-task-only training produced a very small overall macro-F1 increase, but recovery-group F1 decreased and NLL, Brier, ECE and AUROC worsened. Its small AF improvement did not offset declines in respiratory/sleep-breathing and other tasks. It did not meet the joint objective.

Three predefined parent/replay weight interpolations, `θ = (1−α)θ_parent + αθ_replay` for `α ∈ {0.25, 0.5, 0.75}`, also failed the retention checks. At α = 0.25, NLL was 0.76439, Brier 0.43049 and recovery ΔF1 −0.00163. No interpolated model was promoted.

## Verification and interpretation

A separate 16-example fitting diagnostic covered all 13 tasks, reached **16/16 correct**, and reduced NLL from 0.84478 to 0.03242. Fresh-feature agreement and public-API checkpoint reload passed. This establishes integration and fitting capability, not held-out accuracy.

Interruption/recovery checks covered optimizer state, dropout/RNG state, replay/controller state and PCGrad, with identical recovered head tensors on the checked paths. All eight online W&B histories were verified through update 400. The parent checkpoint remained unchanged.

These results reject promotion of the tested configurations under this budget and protocol. They do not show that all possible fine-tuning is ineffective. No larger continuation training, new split or encoder replacement is represented by these results. Historical plans, per-update histories and detailed audits remain in ignored `retired/2026-10-21/`; the tracked source contains the supported released model.
