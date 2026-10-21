"""Reproducible task sampling, retention checks and fine-tuning penalties."""

from collections import defaultdict

import numpy as np
import torch

RECOVERY_TASKS = ('rr_category', 'sdb_label', 'af_label', 'blood_pressure_category')


class ReplaySampler:
    """Stateless per-step sampling; dropout RNG is never consumed by sampling."""

    def __init__(self, rows, batch_size=512, seed=42, replay=None, recovery=RECOVERY_TASKS):
        self.rows, self.batch_size, self.seed, self.replay = rows, batch_size, seed, replay
        self.pools = defaultdict(list)
        for i, row in enumerate(rows): self.pools[row['task']].append(i)
        self.pools = {t:np.array(v) for t,v in self.pools.items()}
        self.weak = sorted(set(recovery)&self.pools.keys())
        self.protected = sorted(self.pools.keys()-set(recovery))
        if replay is not None and not 0 <= replay <= 1:
            raise ValueError('Replay fraction must lie in [0,1]')
        if replay is not None and (not self.weak or (replay and not self.protected)):
            raise ValueError('Selected task pools cannot provide the requested replay')

    def at(self, step):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, step]))
        if self.replay is None:
            return rng.choice(len(self.rows), self.batch_size, replace=len(self.rows)<self.batch_size)
        def draw(tasks, count):
            if not count: return []
            tasks = [tasks[i] for i in rng.permutation(len(tasks))]
            sizes = np.full(len(tasks),count//len(tasks)); sizes[:count%len(tasks)]+=1
            return [rng.choice(self.pools[t],int(n),replace=len(self.pools[t])<n) for t,n in zip(tasks,sizes) if n]
        n = round(self.batch_size*self.replay)
        parts = draw(self.weak,self.batch_size-n)+draw(self.protected,n)
        values = np.concatenate(parts); rng.shuffle(values)
        return values


def distillation_loss(student, teacher, mask, temperature=2.):
    if temperature <= 0: raise ValueError('Distillation temperature must be positive')
    a = (teacher/temperature).log_softmax(-1).masked_fill(~mask,0)
    b = (student/temperature).log_softmax(-1).masked_fill(~mask,0)
    p = (teacher/temperature).softmax(-1).masked_fill(~mask,0)
    return (p*(a-b)).sum(-1).mean()*temperature**2


def anchor_loss(head, initial):
    numerator = sum((value-initial[name]).square().sum() for name,value in head.named_parameters())
    denominator = sum(value.square().sum() for value in initial.values()).clamp_min(1e-12)
    return numerator/denominator


def retention_failures(candidate, reference, recovery=RECOVERY_TASKS):
    failures = []
    for task, baseline in reference['per_task'].items():
        if task in recovery: continue
        for metric in ['accuracy','macro_f1']:
            if candidate['per_task'][task][metric] < baseline[metric]-1e-12:
                failures.append(task+'/'+metric)
    for metric in ['nll','brier']:
        if candidate['pooled'][metric] > reference['pooled'][metric]+1e-12:
            failures.append('pooled/'+metric)
    if candidate['task_macro_auroc'] is not None and reference['task_macro_auroc'] is not None:
        if candidate['task_macro_auroc'] < reference['task_macro_auroc']-1e-12:
            failures.append('task_macro_auroc')
    return failures


def recovery_f1(result, recovery=RECOVERY_TASKS):
    values = [v['macro_f1'] for t,v in result['per_task'].items() if t in recovery]
    return float(np.mean(values)) if values else None


class PlateauControl:
    def __init__(self, best, minimum=1e-4, reduce_after=2, stop_after=5):
        self.best, self.minimum = best, minimum
        self.reduce_after, self.stop_after = reduce_after, stop_after
        self.bad = self.since_reduction = self.checks = 0

    def update(self, score, optimizer):
        self.checks += 1
        if score < self.best-self.minimum:
            self.best, self.bad, self.since_reduction = score, 0, 0
        else:
            self.bad += 1; self.since_reduction += 1
            if self.since_reduction >= self.reduce_after:
                for group in optimizer.param_groups: group['lr'] = max(group['lr']*.5,1e-7)
                self.since_reduction = 0
        return self.bad >= self.stop_after

    def state_dict(self): return vars(self).copy()

    def load_state_dict(self, state): self.__dict__.update(state)
