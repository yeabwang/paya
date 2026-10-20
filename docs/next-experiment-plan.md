# Paya: research and proposed next experiments

2026-10-21. Status: research proposal for review. No training, split changes, dataset downloads, or instance changes were started for this proposal.

Recommendation: keep the current MLP fusion head and frozen encoders for the first experiments. Test targeted fine-tuning with replay and explicit preservation controls. Develop representative evaluation roles in a separately versioned protocol. Measure task interference before adopting gradient manipulation. Calibrate only after checkpoint selection, and optimize latency in a separate controlled comparison.

The user selected validation coverage, multitask training, calibration, and data/label audits as priorities. Temporal-token architecture changes are outside the first experiment. The terminology preference in [AGENTS.md](../AGENTS.md) applies throughout.

## Evidence from our completed runs

The [full-003 report](reports/paya-full-003.md), its [machine-readable snapshot](reports/paya-full-003-metrics.json), and the [experiment ledger](experiments.md) are the local evidence. Existing data audit summaries under `.temp/reports/new-gpu/` were inspected; their counts describe QA rows, not independent patients.

| Observation | Implication for the next experiment |
| --- | --- |
| Best full-003 checkpoint is step 4,797; the last update was 9,594 | Start a new fine-tuning run from the selected raw checkpoint, preserving the original run. Do not assume the final recovery checkpoint is the best model. |
| Macro F1 and AUROC improved, but accuracy improved in only 5/13 tasks | Evaluate all tasks and protect existing performance explicitly. |
| Respiratory-rate accuracy fell 50.83% to 40.18%; sleep breathing fell 60.96% to 54.45% | Make these recovery targets. Their cause remains unproven. |
| AF recall is 241/4,789 = 5.03%; AF AUROC is approximately 0.579 | Measure ranking, positive recall, precision and false positives. A threshold change alone would not establish improved discrimination. |
| All 89 historical-test hypoxemia examples were missed | Audit labels, independent support and signal information before promising a loss-function fix. |
| Severe sleep breathing has no inherited validation rows; moderate sleep breathing has no historical test rows | Neither optimization nor resampling creates independent evidence for absent classes. |
| Calibration has no AF, activity or stress rows | Global temperature fitting did not validate those tasks. |
| Calibration SpO2 consists of 236 normal rows only; sleep breathing consists of 3,067 mild rows only; respiratory rate has 223 normal and 13 tachypnea rows | Task presence is insufficient: class and independent-group coverage must also be audited. |
| AF prevalence is 3,063/67,964 = 4.51% in train, 13,083/44,108 = 29.66% in native validation, and 4,789/32,560 = 14.71% in historical test | These are distribution differences, not proof of pure label shift. Do not estimate corrections from test frequencies. |
| Temperature 1.0483 improved fitting-set NLL slightly but worsened historical-test NLL/Brier/ECE | Keep an identity calibration baseline and assess transfer beyond the calibration fitting set. |

The matched architecture experiment had mixed results: MLP versus attention improved F1/AUROC/ECE, barely changed accuracy/Brier, and slightly worsened NLL. Full-run gains also include label smoothing and changed sampling order. They cannot be attributed to architecture alone.

## 1. Data and outcome audit before optimization

Create a coverage report for every task/answer/subset using QA counts, unique waveforms, recordings, and connected subject groups. Report how much support is concentrated in the largest groups. Thousands of paraphrased QA rows or adjacent windows from one subject do not establish thousands of independent examples.

Check exact and near-duplicate windows across roles, including overlapping segments where timestamps exist. Preserve connected components of known subject, recording and waveform identities. Flag uncertain aliases rather than claiming that hashes prove all possible patient identities are known.

Audit question/answer/candidate alignment, exclusions, and any contradictory answers for the same waveform and task. Distinguish genuine contradictions from different tasks or different definitions. Keep supplied QA labels unchanged during this audit; model disagreement is not a relabeling rule. Keep noisy waveforms that are valid signal-quality targets. An opaque supplied category such as `unknown_0` is not automatically unusable data.

For breathing, BP, oxygen and HRV, trace whether the reference label describes the actual ten-second interval, a longer recording, or the subject. Where provenance cannot resolve the distinction, document that uncertainty. This is a task-definition audit, not permission to invent replacement answers.

Add development-only diagnostic baselines: majority class, a per-task regularized logistic regression on frozen signal features, and signal-ablation checks for the shared head. A successful simple probe would suggest useful information is available but insufficiently exploited. A failed probe would not prove that the representation contains no information. Diagnostic task labels are not added to public model inputs.

## 2. Keep two evaluation protocols separate

### Protocol A: fine-tune the existing selected checkpoint

Keep the existing 1,637,289 train, 196,707 selection, 21,342 calibration and 323,944 historical-test memberships. Fine-tune using train only; select using selection only. The existing calibration role remains independent of the supervised weights/selection process, subject to its documented coverage limits.

This is the most direct and least expensive test of the user's checkpoint proposal. It can measure recovery on available selection classes. It cannot validate severe sleep breathing or class-specific calibration for classes absent from those roles.

Do not move examples from old train or old selection into a new calibration set and call that set independent for this checkpoint. Its supervised training or checkpoint selection already used those examples. The reference test has informed research direction, so comparisons against it remain retrospective.

Adaptive reuse of holdouts can produce optimistic evaluation, even without gradient updates on the holdout. Our response is to declare protocol limits and reserve additional data, rather than claiming that repeated test inspection remains unseen evaluation. [Holdout reuse research](https://arxiv.org/abs/1506.02629)

### Protocol B: representative roles for newly trained supervised heads

First determine feasibility from group-level coverage. Keep the historical test unchanged. Partition eligible non-test connected groups into train, selection, calibration and a locked confirmation role. A provisional row-count target is 80/10/5/5, subordinate to group isolation and task/class coverage. Report achieved counts and distribution shifts; do not force exact ratios by splitting patients.

Stratify using task/answer and subset counts at the connected-component level. One component can contain several tasks and answers; this is a constrained multi-label allocation problem. A standard grouped stratification implementation is a starting point, not a substitute for checking every constraint. Some classes may have too few groups to populate all roles honestly. [Grouped stratification documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)

Freeze the assignment and hashes before any supervised training for this protocol. Train a fresh MLP fusion head with the same frozen encoders. Reuse verified frozen features, not a supervised head already exposed to newly reserved groups. All fine-tuning controls under Protocol B start from the same Protocol-B baseline checkpoint. Teacher predictions and parameter anchors must also come from that baseline, not full-003.

A confirmation role taken from old training data is not unseen for the old model. Evaluate matched newly trained Protocol-B models on it; do not present an old-model comparison there as independent. It is a prospective confirmation role for these new supervised runs, not an external cohort or proof against every form of pretraining overlap. External data would provide stronger evidence after an overlap audit.

Represent splits as versioned manifests/indices into existing data. Preserve canonical QA content and subset identity. No second waveform dataset copy is needed.

## 3. Targeted fine-tuning experiment

Primary recovery tasks: respiratory rate, sleep breathing, AF and blood pressure. Evaluate all thirteen tasks throughout. Oxygen is included in the audit and diagnostics, with targeted optimization conditional on adequate independent supervision and clarified label meaning.

Start Protocol A from `/root/autodl-tmp/paya/runs/paya-full-003/training/best.pt`. Create new child run directories and W&B runs; preserve the parent and its predictions. Reset temperature to one. Initialize a new optimizer because this is a new fine-tuning objective, not crash recovery of the original optimization. Save the new optimizer, sampler, scheduler, early-stopping and RNG states for exact recovery.

Keep the head architecture, encoders, preparation, candidate vocabulary, batch size and initial weights identical across branches. Task identities are used for sampling/reporting only. Do not add them to the public inference inputs.

| Branch | Change relative to its control | Question answered |
| --- | --- | --- |
| Unchanged checkpoint | No updates | What must every continuation beat or preserve? |
| All-task continued training | Natural row-proportion sampling | Does additional low-rate training alone explain a gain? |
| Weak-task-only fine-tuning | Sample only the four recovery tasks | Does focused training recover them, and what is forgotten? |
| Fine-tuning with replay | Initially 75% recovery tasks, 25% other-task training examples | Does rehearsal retain stronger capabilities? |
| Replay plus L2-SP | Same replay batches plus an anchor toward starting weights | Does constraining parameter drift improve retention? |
| Replay plus knowledge distillation | Same replay batches plus starting-model prediction matching on protected-task replay examples | Does functional preservation help beyond label replay? |

The replay fraction is a proposed starting choice, not a published optimum. Balance exposure across recovery tasks initially, retaining natural class proportions within each task. Draw replay across the other tasks with recorded sampling probabilities and exposure caps for repeated waveforms/groups. Report resulting class/group proportions. Do not silently combine class oversampling and inverse-frequency weights, which can amplify the same correction twice.

Compare branches at equal total optimizer updates and batch size. Report weak-task exposure and repeat counts separately: weak-only necessarily sees more weak-task rows at equal compute. If that difference explains the outcome, perform a matched weak-exposure check on finalists.

Replay and distillation have evidence from continual-learning benchmarks, but those settings differ from our shared physiological QA model. Their role here is a hypothesis about avoiding forgetting, not a guarantee. [Dark Experience Replay](https://arxiv.org/abs/2004.07211)

L2-SP penalizes distance from the starting weights, whereas AdamW weight decay shrinks toward zero. Test it as a separate regularization branch. The source evidence comes from CNN transfer learning, so strength and benefit must be established for our MLP. [L2-SP research](https://proceedings.mlr.press/v80/li18a.html)

For distillation, retain ground-truth replay loss and align candidate logits by exact answer text. The teacher remains frozen, with raw probabilities and temperature handling recorded. Do not constrain weak-task mistakes to remain unchanged. A smoothed teacher can transfer less useful logit information, which is another reason to compare distillation with replay alone rather than assume superiority. [Label smoothing and distillation](https://arxiv.org/abs/1906.02629)

A low-cost preservation check is parameter interpolation between the parent and one shortlisted fine-tuned head: `theta = (1-alpha)*theta_parent + alpha*theta_finetuned`. Evaluate at most three additional development candidates, `alpha=0.25/0.5/0.75`, without extra training. Each produces one ordinary MLP, with the same inference graph. The literature supports weight averaging in related fine-tuning settings, not a guarantee for our supervised physiological model. Require identical parameterization and a shared parent; do not average the incompatible attention and MLP heads, or assume arbitrary independently initialized networks can be averaged. Reset calibration after interpolation and apply all retention checks. [Weight interpolation research](https://arxiv.org/abs/2109.01903), [weight-averaging research](https://arxiv.org/abs/2203.05482)

## 4. Regularization, learning-rate control and stopping

Use the existing dropout 0.10, smoothing 0.05 and clipping norm 1.0 as the controlled starting recipe. Preserve weight decay 0.01 for the basic branches; L2-SP is declared as an additional penalty in its branch. Avoid changing all these settings simultaneously.

Proposed initial fine-tuning learning rate: `3e-5`, versus the previous `3e-4`. Compare `1e-5` on the replay branch if the initial rate causes instability or rapid forgetting. These are proposed search values, not research-derived optima. Starting from a selected checkpoint justifies testing smaller updates.

Add validation-based learning-rate reduction, for example a factor of 0.5 after two unsuccessful checks, and early stopping after five unsuccessful checks with at least one check following a reduction. Count checks explicitly; do not confuse half-epoch checks with epochs. Save an initial step-zero candidate so continued training can be rejected entirely. Set a maximum update budget regardless of stopping. [PyTorch scheduler documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html)

Monitor the mean of task-level, unsmoothed selection NLLs so task prevalence does not dominate the stopping statistic. Retain pooled NLL and per-task results separately. A lower task-mean NLL is not sufficient to promote a checkpoint: apply the retention checks below. Fix the policy before the first comparison and use it for every branch.

Validation results are deterministic conditional on a checkpoint and data, but trajectories vary. A patience rule does not prove generalization. Preserve learning curves and repeat finalists with different sampling/dropout seeds.

Test recovery of scheduler counters, best qualifying checkpoint, early-stopping counters, replay RNG and parent checkpoint binding. A resumed run must follow the same future update sequence. The existing tests verify recovery for the old runner; new control state requires its own interruption/resumption check.

## 5. Diagnose multitask interference before adding a remedy

On fixed training probe batches, estimate per-task gradient norms and pairwise cosine similarities at the starting checkpoint and during continuation. Repeat over batches; a single negative cosine is not a diagnosis. Compare shared-head gradients, not frozen encoder parameters.

If directional conflicts repeatedly coincide with recovery/retention tradeoffs, compare Projected Conflicting Gradients (PCGrad) against the same replay branch. If magnitude domination is the main finding, compare GradNorm instead. Start with one of these conditional branches; do not combine both immediately. They add training overhead and change optimization, so record matched updates and elapsed compute. [PCGrad](https://arxiv.org/abs/2001.06782), [GradNorm](https://arxiv.org/abs/1711.02257)

These papers motivate diagnostics and candidate treatments. They do not establish that our RR/SDB regressions are caused by gradient conflict, or that eliminating observed conflict guarantees improvement.

## 6. Class imbalance and probability quality

Global square-root class weighting already cost accuracy in our previous cycle. The next experiment starts with task allocation and replay, not a repeat of that blanket intervention.

If minority recall still collapses after coverage and training diagnostics, test one controlled class-balancing method on the affected task. Candidates include capped class sampling or training-time logit adjustment using training priors. Preserve an unadjusted cross-entropy control. Report how the changed objective relates to natural-prevalence probability estimation. [Logit-adjustment research](https://arxiv.org/abs/2007.07314)

Training, selection and test can differ in both prevalence and conditional signal distribution. Do not assume label shift alone, estimate target priors from historical-test labels, or tune a correction to the observed test distribution. Include subset/group analyses to distinguish easy source cues from generalizable signal information.

Focal loss is a conditional later experiment. It can emphasize difficult examples, including mislabeled or uninformative ones. Its raw outputs are not guaranteed to estimate class-posterior probabilities because focal loss is not strictly proper. This matters when NLL/Brier/ECE are joint goals. [Focal-loss probability analysis](https://arxiv.org/abs/2011.09172)

Keep argmax as the public decision rule. Precision/recall operating points can be reported as diagnostics, with thresholds selected on development data only, without silently changing the choice/probability contract.

## 7. Independent calibration after selecting a checkpoint

Fit calibration only after supervised training and checkpoint selection are frozen. Compare identity (`T=1`) with global temperature scaling first. Under Protocol A, retain all coverage warnings; do not claim a fitted global temperature establishes calibration for absent tasks/classes. Dataset shift can undermine post-hoc calibration. [Temperature scaling](https://arxiv.org/abs/1706.04599), [uncertainty under shift](https://arxiv.org/abs/1906.02530)

Under Protocol B, use grouped internal cross-validation within the calibration role to choose the method and regularization, then refit the chosen method on that role. Keep the confirmation role locked until the method, checkpoint, candidate mapping and evaluation rules are fixed.

Next candidates are regularized group-wise temperature scaling and, only with adequate independent class support, vector scaling or Dirichlet calibration. Shrink flexible maps toward identity or a shared temperature. Do not fit unconstrained class biases when classes are absent. Eligibility depends on independent-group support and held-out calibration behavior, not only QA count. [Dirichlet calibration](https://arxiv.org/abs/1910.12656)

The current API provides no explicit task identifier. Consequently, task-specific calibration cannot silently depend on hidden dataset labels. A compatible first grouping uses exact candidate-vocabulary sets available in the request, with canonical ordering and restoration to the user's order. HRV tasks sharing `low/normal/high` would share that map. An unknown vocabulary or unsupported group falls back to the preregistered identity/shared policy. Any richer task recognition needs a separately validated design.

Positive scalar temperature scaling preserves the within-request argmax and ordering. It does not repair missed-class decisions. A fixed binary-task temperature also preserves AUROC/AP ranking; task-varying temperatures can alter rankings when aggregating across tasks. Vector/Dirichlet maps can change decisions and must be evaluated for discrimination as well as calibration.

Choose the calibration method from held-out calibration results, not from test results or fitting-set improvement alone. Keep raw and calibrated predictions and report both even if calibration fails to transfer.

## 8. Data suitability and additional-data candidates

For SpO2, direct optical measurement uses wavelength-dependent information. Our normalized single-channel input does not contain the two-wavelength ratio used by conventional pulse oximetry. A classifier might exploit correlations, but this is a reason to audit generalization and paired reference labels, not evidence that every oxygen category is impossible or incorrect. [Primary optical measurement study](https://pmc.ncbi.nlm.nih.gov/articles/PMC4029673/)

For HRV, avoid a blanket claim that ten seconds is invalid. A 3,387-person study found stronger short-duration agreement for RMSSD than SDNN. Its actual recording methods used finger arterial-pressure interval measurements; its application discussion includes routine ECG. It does not validate our PPG-derived categorical thresholds or every recording condition. [Primary HRV duration study](https://pmc.ncbi.nlm.nih.gov/articles/PMC4586373/)

For sleep breathing, determine whether a subject-level severity label is being predicted from each window, rather than assuming it means a window-level event. For RR, inspect reference timing, duration and class definitions. Preserve the supplied labels while documenting what the task actually measures.

Potential additions to audit, rather than automatically ingest:

| Candidate | Useful evidence | Required checks before use |
| --- | --- | --- |
| [BIDMC PPG and Respiration](https://physionet.org/content/bidmc/1.0.0/) | 53 eight-minute recordings, 125 Hz PPG, respiratory references and manual breath annotations | Existing-dataset overlap, patient identity, reference timing, label definitions and attribution license. It is MIMIC-derived, so it cannot be assumed external to our data. |
| [MIMIC PERform, version 2.0](https://zenodo.org/records/15906524) | AF/non-AF recordings from 35 adults; added 32-second RR training/testing collections from 681/167 adult patients | Existing waveform/patient overlap and each supplied license. RR references are algorithm-derived from respiratory signals; they are not automatically manual gold-standard annotations. A 32-second reference is not automatically a ten-second label. |

These are targeted source candidates, not guaranteed new independent cohorts. Do not manufacture extra rare-class independence by oversampling, generate replacement QA labels, or assume that more rows from the same patients solve missing coverage. Additional modalities, longer contexts, new encoders or temporal-token heads remain later experiments if diagnostics support them.

## 9. Metrics and acceptance criteria

Report accuracy, macro F1, per-class support/precision/recall, confusion matrices, supported-class OvR AUROC, NLL, multiclass Brier and ECE for every task. Preserve legacy all-vocabulary macro F1 for comparability, and show supported-class summaries separately when roles omit a class.

Add average precision (AP) and precision-recall curves, especially for AF and rare classes. Define AP explicitly rather than mixing it with trapezoidal PR area. Compare AP on the same evaluation population because prevalence affects precision-recall results. [Primary precision-recall study](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432)

Retain ECE with 15 fixed bins for historical comparison, but add class-wise reliability and an equal-mass calibration summary. ECE is sensitive to sample size and binning and is not the sole selection target. [Calibration-estimator research](https://proceedings.mlr.press/v151/roelofs22a.html)

For finalists, compute paired uncertainty intervals by resampling connected subject groups, carrying all their recordings/waveforms/QA rows together. Use approximately 2,000 paired replicates and report insufficient-class replicates rather than replacing undefined AUROC with zero. Also report unique-waveform and group-weighted summaries alongside row-weighted results. This is an application of the cluster-resampling principle, not a claim that a mixed-effects-model bootstrap theorem directly proves our metric intervals. [Cluster bootstrap methods](https://journal.r-project.org/articles/RJ-2023-015/)

Proposed promotion rules, fixed before experiments:

- Recovery-task mean macro F1 improves by at least 0.02 absolute on full selection; report every recovery task separately, including RR/SDB accuracy and AF AP/positive recall. The 0.02 threshold is an engineering target, not a clinical threshold or a literature-derived guarantee.
- Mean task AUROC does not decline. Missing/unsupported classes cannot satisfy a success claim.
- No observed mean accuracy or macro-F1 regression in any of the other tasks across the finalist seeds. The first screen allows no loss; any later noninferiority tolerance must be declared explicitly rather than introduced after seeing results.
- Pooled NLL and Brier do not worsen on selection. ECE remains a diagnostic constraint with uncertainty, not a noiseless scalar to optimize.
- A zero-recall class is not called fixed on the basis of one newly correct example. Give support and uncertainty; require confirmation where feasible.
- Paired latency does not regress materially beyond observed benchmark variability. Claim a latency improvement only when a separate measured implementation change supports it.

Apply checkpoint eligibility checks first, then select eligible checkpoints by task-mean selection NLL; if no updated checkpoint qualifies, retain the unchanged parent. Track a separate research result for tradeoff candidates without promoting them. Bootstrap intervals and repeated seeds do not remove bias from adaptive selection; the locked confirmation role is still needed.

On confirmation, report point estimates and intervals. Wide intervals mean retention is unconfirmed, not proven. If confirmation exposes a failure, reject promotion, document it, and treat that confirmation role as now inspected. No silent repeated tuning against it.

## 10. Latency and resource experiment

Fine-tuning the same architecture does not inherently reduce inference latency. Preserve a separate performance comparison so learning improvements are not credited with unrelated speedups.

Profile native preparation, CPU/device transfers, frozen text encoding, frozen signal encoding, decision scoring and calibration. Measure warm text and cold text separately. Use synchronized CUDA timing, interleaved same-input cases and enough requests to estimate p50/p95/p99. Exclude profiler overhead from final timings. [PyTorch profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html), [CUDA timing semantics](https://docs.pytorch.org/docs/main/notes/cuda.html)

Try FP32 compilation of supported fixed-shape paths only if profiling suggests dispatch/kernel overhead. Record compilation cost and graph recompilations; an initial compile slowdown must not be hidden by reporting only warm inference. Validate probability agreement and end-to-end results against eager execution. [Compilation tutorial](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)

Mixed precision and quantization are deferred until profiling justifies them. They can change frozen features and therefore require numerical and cache-identity checks rather than silent reuse of FP32 identities. Training feature caching is not permitted in the public uncached-signal latency measurement.

Reuse the verified frozen-feature stores, allocate bank indices rather than waveform copies, retain GPU headroom, and monitor both allocation and compute activity. Large memory allocation is not itself a utilization objective. Benchmark first; do not promise a runtime or cost improvement from papers measured on other hardware.

## 11. Bounded execution order after review

1. Audit coverage, labels, known overlaps and Protocol-B feasibility. Verify the parent checkpoint and reproduce its available selection metrics. No model change yet.
2. Validate new sampler, regularizer, scheduler, stopping, candidate alignment and checkpoint recovery on a 16-32-example integration case. Treat memorization as wiring verification only.
3. Screen all-task continuation, weak-task-only fine-tuning and replay on a fixed approximately 64,000-row training pool, initially 400 updates, batch 512, seed 42. Add the L2-SP and distillation branches if replay needs additional preservation. Use a fixed representative approximately 16,000-row selection monitor and a separately reported rare-class diagnostic panel; do not mix an oversampled panel into natural-prevalence metrics. Include every available class where group support allows, with explicit gaps. Keep the training pool and monitoring rows identical across comparable branches. The 400-update screen has a hard cap; early-stopping behavior is exercised over longer expanded budgets as well.
4. Allow at most three additional short comparisons: a smaller learning rate, a regularizer-strength sensitivity check, and PCGrad or GradNorm if the gradient audit justifies one. No unrestricted Cartesian hyperparameter search.
5. Expand at most two candidate recipes plus the all-task continuation control to the fixed approximately 102,000-row development pool, up to 1,600 updates, seeds 42/43/44, evaluating on full selection. These seeds measure continuation variability from one parent, not variability of the original parent training.
6. Nominate one recipe before historical-test evaluation. If representative calibration and missing-class confirmation are required, carry that recipe into Protocol B: train the fresh baseline and apply the continuation recipe only from its own selected checkpoint. Compare matched Protocol-B baseline/continuation pairs, initially one seed, with two additional pairs only if the first pair passes screening. Estimate the GPU budget from measured pilot time before expansion.
7. Select checkpoints, fit the chosen calibration policy, unlock confirmation once, and run paired latency. Score historical test once for the nominated recipe as a clearly labeled retrospective comparison. Keep native test untouched by training and hyperparameter selection.
8. Deliver the report and wait for user review. A failed promotion is a valid experiment outcome; preserve the parent and all evidence.

Every stage records code/config/input hashes, parent checkpoint hash, sampler probabilities, unique exposures, seeds, learning-rate/stopping histories, raw/calibrated metrics, elapsed time and online W&B links. Commit implementation changes separately from results, retain failed experiments, and give short reports between stages. All shell helpers stay under `.temp`. Raw datasets and model checkpoints remain on the GPU; retrieve only small reports. Create a new monitor only when an approved full run actually starts.

The first proposed action after review is the data/coverage audit followed by the controlled checkpoint fine-tuning comparison. The proposal does not authorize replacing the current model based solely on aggregate accuracy or a literature result.
