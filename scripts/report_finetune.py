"""Build the small-experiment report from retrieved aggregate outputs."""
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / '.temp/reports/next'
    def read(name):
        return json.loads((source / name).read_text(encoding='utf-8'))
    review, details = read('review.json'), read('details.json')
    coverage, roles = read('coverage.json'), read('role-report.json')
    integration = read('integration.json')
    baseline = review['reference']
    assert len(review['last_weights']) == len(details['experiments']) == 8
    assert all(v['best_step'] == v['eligible_step'] == 0 for v in details['experiments'].values())
    assert details['parent_unchanged']
    assert all(v['eligible_head_equals_parent'] for v in details['experiments'].values())
    names = {'all-task':'All-task continuation', 'weak-only':'Weak-task-only continuation',
             'replay':'25% replay', 'replay-anchor':'Replay + L2-SP, coefficient 0.1',
             'replay-distillation':'Replay + distillation, coefficient 0.1',
             'replay-low-rate':'Replay, learning rate 1e-5',
             'replay-pcgrad':'Replay + PCGrad',
             'replay-strong-anchor':'Replay + L2-SP, coefficient 10'}
    def line(label, value):
        p = value['pooled']
        return (f"| {label} | {100*p['accuracy']:.3f}% | {value['task_macro_f1']:.5f} | "
                f"{value['task_macro_auroc']:.5f} | {p['nll']:.5f} | {p['brier']:.5f} | {p['ece_15']:.5f} |")
    table = ['| Final trained weights | Accuracy | Macro F1 | AUROC | NLL | Brier | ECE |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |', line('Unchanged parent', baseline)]
    for key, value in review['last_weights'].items():
        table.append(line(names[key], value['metrics']))
    recovery = ['| Experiment | Change in recovery-task mean F1 | Failed retention checks | Selected update |',
                '| --- | ---: | ---: | ---: |']
    for key, value in review['last_weights'].items():
        recovery.append(f"| {names[key]} | {value['recovery_f1_gain']:+.5f} | {len(value['retention_failures'])} | 0 |")
    per_task = ['| Task | Parent accuracy | Weak-only accuracy | Parent F1 | Weak-only F1 |',
                '| --- | ---: | ---: | ---: | ---: |']
    for task, before in baseline['per_task'].items():
        after = review['last_weights']['weak-only']['metrics']['per_task'][task]
        per_task.append(f"| {task} | {100*before['accuracy']:.2f}% | {100*after['accuracy']:.2f}% | "
                        f"{before['macro_f1']:.4f} | {after['macro_f1']:.4f} |")
    role_table = ['| Draft role | QA rows | Connected groups | Task/answer pairs |',
                  '| --- | ---: | ---: | ---: |']
    for role, value in roles['roles'].items():
        pairs = sum(len(a) for a in value['class_coverage'].values())
        role_table.append(f"| {role} | {value['rows']:,} | {value['groups']:,} | {pairs} |")
    tracking = '\n'.join(f"- [{names[key]}]({value['wandb_url']})" for key, value in details['experiments'].items())
    online = read('online.json') if (source/'online.json').exists() else {}
    support = read('sample-support.json') if (source/'sample-support.json').exists() else {}
    online_text = ('All eight W&B runs were read back through the online API; each is finished and has '
                   'loss points through update 400.' if len(online) == 8 and all(
                       v.get('state') == 'finished' and v.get('maximum_sampled_step', 0) == 400
                       for v in online.values()) else 'Online W&B logging was enabled; verification details are retained in the metrics receipt.')
    support_text = ''
    if support:
        support_text = '\nFixed-pool coverage: '+ '; '.join(
            f"{role}: {data['rows']:,} QA rows, {sum(len(a) for a in data['task_answer_rows'].values())} observed task/answer pairs, "
            f"{len(data['missing_task_answers'])} missing pairs" for role, data in support.items() if isinstance(data, dict) and 'rows' in data)+'.\n'
    output = root / 'docs/reports'
    output.mkdir(parents=True, exist_ok=True)
    receipt = {'phase':'complete_small_experiments_stopped_before_expansion',
        'parent_sha256':details['parent_sha256'],'parent_unchanged':True,
        'full_training_started':False,'test_scored_this_cycle':False,
        'selection_baseline':baseline,'final_weights':review['last_weights'],
        'interpolation':review['interpolation'],'details':details['experiments'],
        'integration':integration,'coverage':coverage,'draft_roles':roles,
        'sample_support':support,'online_tracking':online}
    (output/'paya-finetuning-001-metrics.json').write_text(json.dumps(receipt,indent=2,allow_nan=False),encoding='utf-8')
    text = f'''# Paya checkpoint-continuation experiment report

Eight bounded continuation pilots completed. None qualifies for expansion or
replacement. Every branch selected update 0: the original parent. All eight
eligible heads are tensor-identical to that parent, whose file SHA-256 is
`{details['parent_sha256']}`. No larger development run, new full run or NormWear
training had started at this stage. The monitoring timer remains paused. This
report completes the pilot stage before the separately authorized NormWear work;
that separate experiment must not alter or replace the current main model.

## Protocol and implementation

These are Protocol A comparisons on unchanged train/selection roles. Training
uses the same deterministic 64,000-row sample; selection monitoring uses the same
16,384 rows, with final comparisons over all 196,707 selection rows. Each branch
uses 400 updates, batch 512 and seed 42, giving 204,800 row draws. Replay is
task-balanced within the recovery/protected pools; it is not class balancing.
The recovery tasks are respiratory rate, sleep breathing, AF and blood pressure.
Rows repeat under sampling; draws are not independent examples or epochs.

The frozen signal/text encoders and the 264,065-parameter MLP fusion head with
Hadamard product features are unchanged. This cycle changes optimization and
sampling, not the inference architecture. Dropout is 0.1, label smoothing 0.05,
AdamW weight decay 0.01 and gradient clipping 1. Initial learning rate is 3e-5
except the declared 1e-5 branch. Replay assigns 75% of draws to recovery tasks
and 25% to other tasks. Weak-only assigns no replay.

L2-SP penalizes squared deviation from the parent, normalized by its squared
parameter norm. Distillation adds a temperature-2 KL penalty on protected-task
outputs alongside labeled replay. PCGrad projects opposing task gradients while
retaining their batch-frequency weighting. The mean pairwise gradient cosine was
negative for 62.82% of task pairs over three training-batch probes; that motivated
this diagnostic and does not prove causal harmful interference.

Checkpoints are evaluated every 100 updates. A plateau halves the learning rate
after two unsuccessful checks; stopping patience is five checks. The 400-update
cap allows four checks, so no pilot actually triggers patience-based stopping.
The controller's stopping/recovery behavior is covered by tests. Every branch
keeps both the best research checkpoint and a checkpoint satisfying retention.
The monitored objective is mean per-task NLL. Since no updated checkpoint beats
the initial objective, even the research selection remains the parent.

Implementation revisions: audit `c302d53`, first controls/pilots `f9b013e`,
preservation/new-role support `0a12b0f`, final two controls and reorganization
`3be95f2`. Per-file source/config/input bindings are preserved in the JSON receipt.
The final GPU suite passed **182 unit tests**, with one pretrained integration
test deselected. The focused local suite passed 25 tests. Forced recovery
reproduced dropout, optimizer, sampler and controller outcomes exactly, including
the PCGrad branch. Historical pretrained integration and this cycle's actual
native-feature/public-API checks are separate from that unit-suite result.

## Final weights versus parent

These rows evaluate each branch's actual update-400 weights, not its selected
update-0 copy. Accuracy/NLL/Brier/ECE pool QA rows. F1 averages all candidate
classes within each task, then tasks equally; an absent class has F1 zero.
AUROC averages supported one-versus-rest classes within each task, then tasks;
undefined classes remain null. ECE uses 15 equal-width confidence bins. These
are selection results, not new test results.

{chr(10).join(table)}

{chr(10).join(recovery)}

Failed-check counts are metric checks, not numbers of damaged tasks. They cover
protected-task accuracy/F1, pooled NLL/Brier and task-mean AUROC. A positive
recovery mean F1 gain of at least 0.02 was also required. No branch attains it.
The full per-task metrics, confusion matrices, AP, recalls and failed-check
names are retained in the accompanying machine-readable receipt.

## What the results show

Weak-task-only continuation provides no safe recovery: AF F1 rises slightly,
but blood-pressure, respiratory-rate and sleep-breathing F1 decline. Several
protected HRV tasks also decline. Its tiny overall macro-F1 increase is outweighed
by lower AUROC and worse probability errors. The following directly addresses
whether training only the weaker tasks hurts the others:

{chr(10).join(per_task)}

Replay reduces some drift relative to weak-only but does not satisfy retention.
Distillation and the smaller learning rate reduce some probability-error damage
relative to replay, while remaining worse than the parent. Normalized L2-SP at
0.1 barely changes replay; coefficient 10 still fails. PCGrad costs more updates'
computation and fails to recover the target metrics. These outcomes apply to the
tested coefficients, sample and budget, not every possible implementation of
these methods.

Three predefined parent/replay weight interpolations at 0.25, 0.5 and 0.75 also
fail. Even 0.25 gives NLL 0.76439, Brier 0.43049 and recovery mean F1 change
-0.00163. There is no promising finalist to justify a multi-seed expansion.
No fresh calibration, historical-test scoring or latency comparison was performed
for rejected candidates. Fine-tuning unchanged weights/architecture supplies no
evidence of faster inference. Prior full-run results remain in
[the full-003 report](paya-full-003.md).

## Wiring and data evidence

The native 16-example diagnostic spans all 13 tasks, matches fresh frozen
features to the bank, and learns 16/16. NLL falls from 0.84478 to 0.03242.
The saved checkpoint reloads through the public API with matching probabilities.
This is memorization/wiring verification, not a generalization benchmark.

The metadata audit covered 1,855,338 non-test QA rows and {coverage['independent_components']:,}
connected components. Mapped answers agree with the supplied QA provenance;
there are no conflicting answers for an identical waveform/task in this audit.
That verifies mapping consistency, not physiological ground truth. Known
subject/recording/waveform identities stay disjoint; undisclosed aliases and
near-duplicate temporal overlap remain limitations.

The rarest severe sleep-breathing and severe-hypoxemia labels each span only
six independent groups across the available non-test pool. Many waveform rows
do not replace independent subjects. Existing selection has no severe
sleep-breathing positives and only one moderate-hypoxemia positive, limiting
the current comparisons for those labels.
{support_text}
## Draft group-stratified roles: not used for model fitting

Coverage feasibility was established and metadata-only role manifests were
prepared. They reference existing rows without copying waveform data; historical
test membership is unchanged. All 13 tasks and 52 task/answer pairs occur in each
draft role. The achieved fractions are approximately 76.98/10.38/6.60/6.03,
because independent-group class coverage takes priority over exact 80/10/5/5.

{chr(10).join(role_table)}

This is not a cure for rare independent support. Severe hypoxemia has only
8 selection, 5 calibration and 9 confirmation positives, each from one group;
severe sleep breathing has one independent group in each held-out role too.
The current parent was trained on some newly reserved groups. Any experiment
using these roles must train a fresh supervised baseline and cannot treat the
old parent as independent. Confirmation predictions remain uncomputed/locked.
No pilot above used these new roles.

## Tracking, artifacts and decision

{online_text} Cloud model-artifact uploads are disabled; local GPU checkpoints
and recovery files remain authoritative. Only small aggregate reports were
retrieved locally. No datasets or checkpoints were downloaded or published.

{tracking}

Active experiment, audit, analysis and postprocessing tools now live in
`scripts/`; reusable model/input/checkpoint code stays in `paya/`. Shell helpers
stay in ignored `.temp/`. Input hashes and the parent checkpoint were verified
after all eight runs. At completion the GPU was idle, with 31 GB of data-disk
space available. No automatic full-run launch remains active.

Recommendation: retain the current model. The results do not support simply
continuing longer on the weak tasks, and do not establish that a new split or
encoder will improve generalization. Review the separate fresh-baseline and
encoder-comparison design before spending a larger training budget. NormWear
will be isolated with its own configuration, head and results; replacement is
the user's decision regardless of its scores. Raw multimodal data/QA preparation
is a later phase and is not mixed into these eight comparisons.

Receipt: [all metrics and audit details](paya-finetuning-001-metrics.json).
'''
    (output/'paya-finetuning-001.md').write_text(text,encoding='utf-8')
    print(json.dumps({'report':'docs/reports/paya-finetuning-001.md','experiments':8,
                      'selected_parent':True,'full_run_started':False}))


if __name__ == '__main__':
    main()
