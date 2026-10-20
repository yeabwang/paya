"""Controlled head-only experiments over verified frozen feature banks."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import yaml

from .atomic import atomic_output
from .classification import classification_metrics
from .evaluate import metrics
from .head import DecisionHead
from .model import PayaModel
from .provenance import fingerprint, sha256
from .tracking import open_run


class InteractionHead(nn.Module):
    """Experimental shared candidate scorer with explicit signal/text products."""

    def __init__(self, width=128, dropout=0.1):
        super().__init__()
        self.config = dict(kind="interaction", width=width, dropout=dropout)
        self.signal = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, width), nn.GELU())
        self.text = nn.Sequential(nn.Linear(768, width), nn.LayerNorm(width))
        self.context = nn.Sequential(nn.Linear(width * 3, width), nn.GELU(), nn.Dropout(dropout))
        self.score = nn.Sequential(nn.Linear(width * 3, width), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(width, 1))

    def forward(self, signal, signal_mask, question, criteria, criteria_mask):
        if signal.shape[1:] != (1, 512) or not signal_mask.all():
            raise ValueError("Interaction head requires one unpadded global signal token")
        x, q, c = self.signal(signal[:, 0]), self.text(question), self.text(criteria)
        context = self.context(torch.cat([x, q, x * q], -1))[:, None].expand_as(c)
        z = self.score(torch.cat([context, c, context * c], -1)).squeeze(-1)
        return z.masked_fill(~criteria_mask, -torch.inf)


def class_weights(rows, power=0.5, maximum=4.0):
    """Use training-only task/answer frequencies; normalize within each task."""
    counts = {}
    for row in rows:
        key = row['task'], row['criteria'][row['target']]
        counts[key] = counts.get(key, 0) + 1
    task_total, task_classes = {}, {}
    for (task, _), count in counts.items():
        task_total[task] = task_total.get(task, 0) + count
        task_classes[task] = task_classes.get(task, 0) + 1
    raw = {key: min(maximum, (task_total[key[0]] / (task_classes[key[0]] * count)) ** power)
           for key, count in counts.items()}
    normalizers = {task: sum(raw[key] * count for key, count in counts.items() if key[0] == task)
                   / total for task, total in task_total.items()}
    return np.array([raw[row['task'], row['criteria'][row['target']]] / normalizers[row['task']]
                     for row in rows], dtype=np.float32)


def smoothed_choice_loss(logits, targets, mask, smoothing=0.0):
    """Uniform smoothing covers real candidates only, never padded -inf logits."""
    if not 0 <= smoothing < 1:
        raise ValueError("Smoothing must be in [0,1)")
    logp = logits.log_softmax(-1)
    nll = -logp.gather(1, targets[:, None]).squeeze(1)
    smooth = -logp.masked_fill(~mask, 0).sum(-1) / mask.sum(-1)
    return (1 - smoothing) * nll + smoothing * smooth


class Bank:
    def __init__(self, path, limit=None, seed=20261021, device="cuda"):
        self.path = Path(path)
        manifest = json.loads((self.path / 'manifest.json').read_text())
        for name, digest in manifest['files'].items():
            if sha256(self.path / name) != digest:
                raise ValueError("Development bank checksum differs")
        self.manifest = manifest
        indices = np.random.default_rng(seed).permutation(manifest['rows'])
        indices = indices[:limit] if limit else np.arange(manifest['rows'])
        self.indices = indices
        wanted = set(map(int, indices))
        metadata = {}
        with (self.path / 'rows.jsonl').open() as source:
            for index, line in enumerate(source):
                if index in wanted:
                    metadata[index] = json.loads(line)
        self.rows = [metadata[int(index)] for index in indices]
        self.sha256 = fingerprint({'bank': manifest['binding'], 'indices': indices.tolist()})
        self.device = device
        def load(name):
            return np.load(self.path / (name + '.npy'), mmap_mode='r')
        self.signal = torch.tensor(np.asarray(load('signals')[indices]), device=device)[:, None]
        self.question = torch.tensor(np.asarray(load('questions')), device=device)
        self.criteria = torch.tensor(np.asarray(load('criteria')), device=device)
        self.qi = torch.tensor(np.asarray(load('question_indices')[indices]), device=device).long()
        self.ci = torch.tensor(np.asarray(load('criteria_indices')[indices]), device=device).long()
        self.targets = torch.tensor(np.asarray(load('targets')[indices]), device=device).long()

    def batch(self, indices, ablation=None):
        signal = self.signal[indices]
        if ablation == 'mean':
            signal = self.signal.mean(0, keepdim=True).expand_as(signal)
        c = self.ci[indices]
        return signal, torch.ones(len(signal), 1, dtype=torch.bool, device=self.device), \
            self.question[self.qi[indices]], self.criteria[c.clamp_min(0)], c >= 0


@torch.no_grad()
def evaluate(head, bank, ablation=None):
    head.eval()
    records = []
    for start in range(0, len(bank.rows), 1024):
        indices = torch.arange(start, min(start + 1024, len(bank.rows)), device=bank.device)
        z = head(*bank.batch(indices, ablation)).cpu()
        for row, values in zip(bank.rows[start:start + len(z)], z):
            records.append({'logits': values[:len(row['criteria'])].tolist(),
                            'criteria': row['criteria'], 'target': row['target']})
    per_task = {}
    for task in sorted(bank.manifest['tasks']):
        values = [v for v, row in zip(records, bank.rows) if row['task'] == task]
        if values:
            per_task[task] = {**metrics(values), **classification_metrics(values)}
    aucs = [v['macro_auroc_ovr'] for v in per_task.values() if v['macro_auroc_ovr'] is not None]
    return {'pooled': metrics(records), 'per_task': per_task,
            'task_macro_f1': sum(v['macro_f1'] for v in per_task.values()) / len(per_task),
            'task_macro_auroc': sum(aucs) / len(aucs) if aucs else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    torch.set_num_threads(8)
    root = Path(plan['directory']); root.mkdir(parents=True, exist_ok=True)
    train = Bank('cache/paya-development/train', plan.get('train_limit'))
    selection = Bank('cache/paya-development/selection', plan.get('selection_limit'))
    config = yaml.safe_load(Path('config/paya-fast.yaml').read_text())
    model = PayaModel.from_assets(config['assets'], config['preparation'], config['head'], 'cuda')
    reference = root / 'reference.json'
    if not reference.exists():
        from .checkpoint import load_checkpoint
        load_checkpoint('runs/paya-full-002/training/best.pt', model)
        with atomic_output(reference, 'w') as out:
            json.dump(evaluate(model.head, selection), out, indent=2, allow_nan=False)
    for experiment in plan['experiments']:
        destination = root / experiment['name']; destination.mkdir(exist_ok=True)
        if (destination / 'result.json').exists():
            continue
        seed = experiment.get('seed', 42)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); random.seed(seed)
        head = (InteractionHead() if experiment['head'] == 'interaction'
                else DecisionHead(**config['head'])).cuda()
        weights = torch.tensor(class_weights(train.rows) if experiment.get('weighted')
                               else np.ones(len(train.rows), dtype=np.float32), device='cuda')
        optimizer = torch.optim.AdamW(head.parameters(), lr=experiment.get('lr', 3e-4), weight_decay=.01)
        binding = fingerprint({'plan': plan, 'experiment': experiment,
                               'train': train.sha256, 'selection': selection.sha256})
        resume = destination / 'last.pt'
        begin, best, best_state, history = 0, float('inf'), None, []
        previous = None
        if resume.exists():
            saved = torch.load(resume, map_location='cpu', weights_only=True)
            if saved['binding'] != binding:
                raise ValueError("Experiment resume bindings differ")
            head.load_state_dict(saved['head']); optimizer.load_state_dict(saved['optimizer'])
            begin, best, best_state, history = saved['step'], saved['best'], saved['best_state'], saved['history']
            previous = saved['wandb']
        settings = deepcopy(config)
        settings['wandb'].update(name='paya-' + experiment['name'], log_models=False)
        settings['experiment'] = {**experiment, 'plan': plan, 'binding': binding}
        proxies = {name: SimpleNamespace(rows=bank.rows, sha256=bank.sha256)
                   for name, bank in [('train', train), ('selection', selection)]}
        started = perf_counter()
        with open_run(settings, destination, model, proxies, previous) as run:
            if resume.exists():
                torch.set_rng_state(saved['rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
            for step in range(begin + 1, plan['steps'] + 1):
                head.train()
                indices = torch.randint(len(train.rows), (plan.get('batch_size', 256),), device='cuda')
                batch = train.batch(indices)
                z = head(*batch)
                losses = smoothed_choice_loss(z, train.targets[indices], batch[-1], experiment.get('smoothing', 0))
                loss = (losses * weights[indices]).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite experimental loss")
                optimizer.zero_grad(set_to_none=True); loss.backward()
                norm = nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Nonfinite experimental gradient")
                optimizer.step()
                if step % 20 == 0:
                    run.log({'step': step, 'train/loss': float(loss.detach()), 'train/gradient_norm': float(norm)})
                if step % plan['evaluate_every'] == 0 or step == plan['steps']:
                    result = evaluate(head, selection)
                    history.append({'step': step, 'selection': result})
                    if result['pooled']['nll'] < best:
                        best = result['pooled']['nll']
                        best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
                    run.log({'step': step, **{f'development/{key}': value for key, value in result['pooled'].items() if key != 'reliability'},
                             'development/task_macro_f1': result['task_macro_f1'],
                             'development/task_macro_auroc': result['task_macro_auroc']})
                    print(json.dumps({'experiment': experiment['name'], 'step': step,
                        **{k: v for k, v in result['pooled'].items() if k != 'reliability'},
                        'macro_f1': result['task_macro_f1'], 'auroc': result['task_macro_auroc']}), flush=True)
                if step % 100 == 0 or step == plan['steps']:
                    with atomic_output(resume) as out:
                        torch.save({'binding': binding, 'head': head.state_dict(), 'optimizer': optimizer.state_dict(),
                            'step': step, 'best': best, 'best_state': best_state, 'history': history,
                            'rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                            'wandb': {'id': run.id, 'entity': run.entity, 'project': run.project, 'mode': 'online'}}, out)
            head.load_state_dict(best_state)
            result = {'config': experiment, 'binding': binding, 'train_rows': len(train.rows),
                'selection_rows': len(selection.rows), 'seconds': perf_counter() - started,
                'parameters': sum(p.numel() for p in head.parameters()), 'history': history,
                'selected_by': 'lowest selection NLL among evaluated steps',
                'selection': evaluate(head, selection), 'mean_signal_ablation': evaluate(head, selection, 'mean'),
                'wandb': run.url}
            with atomic_output(destination / 'best.pt') as out:
                torch.save({'binding': binding, 'config': experiment, 'head': best_state}, out)
            with atomic_output(destination / 'result.json', 'w') as out:
                json.dump(result, out, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
