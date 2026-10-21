# Experiments

The selected release is **Paya full-003, update 4,797, temperature 1**. It achieved the lowest recorded full-selection NLL among the completed comparable development and full-data candidates. Small fitting checks and scores on selection subsets are not ranked as full-corpus results. This document records the training protocol, architectural comparison and evidence behind that selection; continuation outcomes are recorded in [Checkpoint continuation experiments](#checkpoint-continuation-experiments).

## Data and evaluation protocol

Training uses processed, normalized, single-channel PPG segments: **10 seconds, 125 Hz, 1,250 samples**. Questions and correct answers come from the supplied row-level QA content. Actual answer strings are preserved, including labels beyond the descriptive dataset guide. Targets are indices into each row's candidate list; probabilities and class-sensitive metrics are aligned by answer text, not by a presumed fixed candidate position.

Preparation excluded 3,482 all-zero signals and 335,253 rows whose subject/recording identities could not be verified for split isolation. These are documented training exclusions, not repaired or relabeled examples. An unverifiable identity is a provenance limitation; it does not by itself show an incorrect waveform or a mapping error. Retained waveforms and QA answers were not changed to address minority-class performance.

| Role | QA rows | Purpose |
| --- | ---: | --- |
| Train | 1,637,289 | Optimize head parameters |
| Selection | 196,707 | Compare configurations and select checkpoints |
| Calibration | 21,342 | Fit temperature after checkpoint selection |
| Test | 323,944 | Report the historical benchmark |

Native train/test membership was retained. The supplied validation split was subdivided into selection and independent calibration roles before training. Connected subject, recording and waveform identities remain together; cross-role overlap checks passed. Audit identifiers support grouping and traceability and are not learned model inputs. Multiple QA rows may describe the same waveform, so row counts are not counts of independent patients or recordings.

Development used train and selection data. Calibration was separate from optimization and checkpoint selection. The test had already been inspected in the earlier baseline and informed the broader research direction; all test comparisons here are **retrospective historical comparisons**, not a new unseen benchmark. No statistical significance is inferred from small aggregate differences.

## Architecture comparison

Both completed full runs used the same frozen signal/text encoders and the same public `state`, `question`, `criteria` interface.

| Property | Full-002 baseline | Selected full-003 |
| --- | --- | --- |
| Learned head | Attention-based fusion | MLP fusion with Hadamard product features |
| Hidden width | 256 | 128 |
| Attention layers / heads in the decision head | 2 / 4 | None |
| Trainable parameters | 2,567,681 | 264,065 |
| Signal input to the head | One global 512-D vector | One global 512-D vector |
| Text input | Separate 768-D question/candidate vectors | Same |

The earlier head used text self-attention and cross-attention to the global signal token. With one unmasked signal key, cross-attention's softmax weight is necessarily one: query/key projections cannot select between different signal positions. This does not make the earlier network incapable of learning; its value projections and subsequent nonlinear layers still use the signal. The compact head instead explicitly combines projected signal/question and context/candidate features. Its [complete layer specification](architecture.md#decision-head) documents the retained implementation.

The head parameter count fell by **89.72%**. That is a parameter-count comparison, not a claim that all full-run gains were caused by architecture alone. Label smoothing and data-access/optimization arrangements also changed during development.

## Development stages

### Integration and memorization

A deliberately balanced diagnostic used 208 training and 101 selection examples. Frozen encoders and candidate alignment were checked before comparing heads. The attention baseline, an attention variant with signal LayerNorm and the compact MLP all reached 100% training accuracy within 400 updates; the MLP reached it by update 100. Late selection NLL exceeded 5 despite successful memorization. These experiments established that the pipeline could learn the supplied labels and exposed overfitting; none qualified as a generalization result or release candidate.

### Matched small-data screens

Six pilots used identical 8,192 training and 4,096 selection rows, seed 42, batch 256 and 800 updates, evaluating every 200 updates. The head checkpoint selector was minimum selection NLL. Pilot draws were uniform with replacement from the fixed training pool. Frozen FP32 feature/text banks, candidate masks and target indices were checked for mapping alignment and file integrity.

Class-weighted variants used training-only square-root inverse class frequency, clipped at 4 before task-wise normalization to unit mean. Label smoothing was 0.05 over real candidates only. All values below are from the pilot selection subset.

| Pilot | Accuracy | NLL | Brier | ECE | Task macro F1 | Task macro AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Attention, cross-entropy | 65.58% | 0.91071 | 0.50733 | 0.07926 | 0.21266 | 0.56617 |
| Attention, weighted | 62.62% | 0.90941 | 0.51949 | 0.06759 | 0.24487 | 0.63104 |
| Attention, smoothed | 65.60% | 0.89056 | 0.50305 | 0.05969 | 0.21418 | 0.55691 |
| Attention, weighted + smoothed | 62.94% | 0.89861 | 0.51677 | 0.06493 | 0.24166 | 0.62478 |
| MLP, cross-entropy | 65.06% | 0.84230 | 0.47643 | 0.06087 | 0.27150 | 0.61407 |
| MLP, weighted + smoothed | 61.08% | 0.88825 | 0.50963 | 0.02633 | 0.30490 | 0.60730 |

Weighting sometimes improved class-sensitive metrics while reducing accuracy or worsening probability scores. Minimum NLL and maximum F1 could occur at different updates; a favorable metric from an unselected checkpoint was not substituted for the selected checkpoint's result.

### Expanded controls and refinement

Promising controls expanded to 102,345 training rows, batch 512 and 1,600 updates. A fixed 16,384-row selection monitor chose checkpoints every 400 updates; selected weights were then evaluated on **all 196,707 selection rows**. These remain development results.

| Expanded configuration | Accuracy | NLL | Brier | ECE | Task macro F1 | Task macro AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full-002 reference checkpoint | 67.12% | 1.01766 | 0.49039 | 0.09868 | 0.26318 | 0.63589 |
| Attention, cross-entropy | 68.37% | 0.80803 | 0.45195 | 0.10195 | 0.33156 | 0.68711 |
| Attention, weighted + smoothed | 66.88% | 0.78627 | 0.44779 | 0.03186 | 0.36954 | 0.68515 |
| MLP, cross-entropy | 68.43% | 0.81303 | 0.45140 | 0.09089 | 0.35053 | 0.70535 |
| MLP, smoothed | 68.40% | 0.78821 | 0.44595 | 0.06921 | 0.35013 | 0.70003 |
| MLP, weighted + smoothed | 65.83% | 0.78893 | 0.44999 | 0.03488 | 0.35584 | 0.69633 |

The reference had a different training budget; it is a useful checkpoint comparator, not a matched estimate of architecture's isolated effect. Among the compact refinements, smoothing without class weighting preserved substantially more accuracy than weighting while improving NLL over ordinary cross-entropy. That configuration was nominated before full-data expansion.

Mean-signal ablations also tested waveform sensitivity. Expanded attention cross-entropy accuracy fell from 68.37% to 44.78%, and weighted/smoothed attention from 66.88% to 35.42%. These checks show dependence on waveform features; they do not establish that every learned dependence is physiologically valid.

## Full-003 training recipe

| Setting | Recorded value |
| --- | --- |
| Trainable modules | MLP fusion head only; encoders frozen in FP32 |
| Training data | 1,637,289 QA rows; each visited once per epoch |
| Epochs / batch | 3 / 512 |
| Updates | 3,198 per epoch; 9,594 total |
| Optimizer | AdamW, learning rate 0.0003, weight decay 0.01 |
| Supervised objective | Cross-entropy with 0.05 label smoothing over real candidates |
| Dropout / gradient clipping | 0.1 / global norm 1 |
| Class weighting | None |
| Seed | 42 |
| Checkpoint selection | Lowest full-selection NLL, checked every 1,599 updates |
| Selected update | 4,797, halfway through epoch two |

All three epochs completed. Selection retained the best checkpoint rather than the final one; the full run did not use early stopping. Weight decay, dropout, label smoothing and frozen encoders were already part of its regularization. The later continuation pilots added separate preservation and stopping controls; see [Checkpoint continuation experiments](#checkpoint-continuation-experiments).

Frozen-feature preparation used checksummed, recoverable chunks and a disk-backed SQLite feature cache. The data loader bounds individual blocks rather than loading the complete waveform corpus into RAM. Experimental recovery state preserved optimizer state, RNG state, logical update, best weights and online W&B identity. Scratch interruption/resume checks reproduced final head tensors exactly. Online metrics were enabled; the GPU's local checkpoint files were authoritative.

## Full-run outcome

Both columns use the same 323,944 historical test rows and **uncalibrated** probabilities.

| Metric | Full-002 | Full-003 |
| --- | ---: | ---: |
| Accuracy | 76.6117% | **76.8599%** |
| Mean task macro F1 | 0.26758 | **0.37885** |
| Mean task macro AUROC | 0.60189 | **0.67465** |
| NLL | 0.74045 | **0.65148** |
| Brier | 0.38164 | **0.34587** |
| ECE, 15 bins | 0.07258 | **0.02115** |

The accuracy difference is **0.2482 percentage points**, or 804 additional correct QA answers. Accuracy improved in five tasks and macro F1 in twelve. Respiratory-rate accuracy fell from 50.83% to 40.18%; sleep-breathing accuracy fell from 60.96% to 54.45%. Minority-class failures remain despite stronger aggregate scores. Per-task release metrics and exact latency conditions are in [Results](results.md).

A separate global temperature fit slightly improved calibration-role NLL but worsened historical test NLL, Brier and ECE. The release therefore reports the original checkpoint at temperature 1. Eight subsequent continuation pilots did not meet the joint improvement/retention criteria; none replaced the selected model. No alternative encoder is part of this release.

## Reproducibility and scope

The selected configuration is [config/paya.yaml](../config/paya.yaml); training/evaluation revision is `48754f3b86cb0c29e04673acfefd6f2f6418f55e`. [Checkpoint identity](../models/checkpoint.json), [runtime versions](../models/runtime.json) and [exact release metrics](metrics.json) are retained in the tracked repository.

Historical experiment runners, plans and detailed outputs are archived locally under ignored `retired/2026-10-21/` and in Git history. They are not shipped as supported source. The supported `paya.train` entry point shares the model/input/loss implementation, but its generic loop is not a bit-exact replay of the archived full-feature-bank runner and its 1,599-update selection cadence. Reproducing that historical run requires its recorded revision, data memberships and experiment bindings; the selected config alone is insufficient.

## Checkpoint continuation experiments

Eight bounded continuation experiments tested whether the selected head could improve weaker tasks without degrading the others. **No candidate qualified for replacement.** The release remains the original full-003 checkpoint at update 4,797 and temperature 1. All experiments retained the same frozen encoders and the same [MLP fusion architecture](architecture.md#decision-head).

### Parent checkpoint and task groups

Parent checkpoint SHA-256:

```text
461b939b19ef59bfa05725c69438b2e8332e031b2586aeab9234fe848c633eba
```

The recovery group contained **AF detection, blood pressure, respiratory rate and sleep breathing**. The other nine tasks formed the protected group. These roles governed sampling, distillation and retention checks; they were not additional inputs to the model.

The motivating question was a stability/plasticity tradeoff: can extra optimization recover weak-task performance while preserving the parent's stronger behavior? A weak-task-only control measured the cost of omitting other tasks. Replay, parameter regularization, distillation and gradient projection tested different ways of preserving them.

### Controlled protocol

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

### Objectives and controls

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

### Checkpoint selection and promotion

The research selector minimized task-mean NLL on the fixed monitor. A separate eligible-checkpoint selector required improvement in that objective with no retention failures. Promotion additionally required a meaningful recovery-group gain. The combined criteria were:

1. Recovery-task mean macro F1 improvement of at least **0.02 absolute**.
2. No decline in mean task macro AUROC across all tasks.
3. No decline in protected-task accuracy or macro F1 under the recorded checks.
4. No worsening of pooled NLL or Brier score.

Only checkpoints satisfying the retention checks were eligible to replace the parent. Every branch's best and eligible update remained **zero**. Eligible saved head tensors matched the parent exactly. The table below evaluates each branch's **final update-400 weights**, including unsuccessful weights; it does not describe the retained eligible checkpoint.

### Final-weight results on full selection

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

### Verification and interpretation

A separate 16-example fitting diagnostic covered all 13 tasks, reached **16/16 correct**, and reduced NLL from 0.84478 to 0.03242. Fresh-feature agreement and public-API checkpoint reload passed. This establishes integration and fitting capability, not held-out accuracy.

Interruption/recovery checks covered optimizer state, dropout/RNG state, replay/controller state and PCGrad, with identical recovered head tensors on the checked paths. All eight online W&B histories were verified through update 400. The parent checkpoint remained unchanged.

These results reject promotion of the tested configurations under this budget and protocol. They do not show that all possible fine-tuning is ineffective. No larger continuation training, new split or encoder replacement is represented by these results. Historical plans, per-update histories and detailed audits remain in ignored `retired/2026-10-21/`; the tracked source contains the supported released model.
