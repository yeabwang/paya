"""Controlled head-only experiments over verified frozen feature banks."""

import argparse
from copy import deepcopy
import json
import os
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
from .head import DecisionHead, InteractionHead
from .model import PayaModel
from .provenance import fingerprint, sha256
from .tracking import open_run
from .train import choice_loss


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
    return choice_loss(logits,targets,mask,smoothing,reduction='none')


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
        if limit:
            wanted = set(map(int, indices)); metadata = {}
            with (self.path / 'rows.jsonl').open() as source:
                for index, line in enumerate(source):
                    if index in wanted: metadata[index] = json.loads(line)
            self.rows = [metadata[int(index)] for index in indices]
        else:
            with (self.path / 'rows.jsonl').open() as source:
                self.rows = [json.loads(line) for line in source]
            if len(self.rows) != manifest['rows']:
                raise ValueError("Bank row count differs")
        self.sha256 = fingerprint({'bank': manifest['binding'], 'indices': indices.tolist()})
        self.device = device
        def load(name):
            return np.load(self.path / (name + '.npy'), mmap_mode='r')
        if str(device).startswith('cuda'):
            estimated = len(indices) * (512 * 4 + 256)
            free, _ = torch.cuda.mem_get_info()
            if estimated > free * .7:
                raise ValueError("Feature bank exceeds the reserved GPU memory budget; reduce the development sample")
        def selected(name):
            value=load(name)
            return np.asarray(value[indices] if limit else value)
        self.signal = torch.tensor(selected('signals'), device=device)[:, None]
        self.question = torch.tensor(np.asarray(load('questions')), device=device)
        self.criteria = torch.tensor(np.asarray(load('criteria')), device=device)
        self.qi = torch.tensor(selected('question_indices'), device=device).long()
        self.ci = torch.tensor(selected('criteria_indices'), device=device).long()
        self.targets = torch.tensor(selected('targets'), device=device).long()

    def batch(self, indices, ablation=None):
        signal = self.signal[indices]
        if ablation == 'mean':
            signal = self.signal.mean(0, keepdim=True).expand_as(signal)
        c = self.ci[indices]
        return signal, torch.ones(len(signal), 1, dtype=torch.bool, device=self.device), \
            self.question[self.qi[indices]], self.criteria[c.clamp_min(0)], c >= 0


class EpochBatches:
    """Deterministic no-replacement batches, recoverable from a logical step."""
    def __init__(self, count, batch_size, seed, device='cpu'):
        self.count,self.batch_size,self.seed,self.device=count,batch_size,seed,device
        self.steps_per_epoch=(count+batch_size-1)//batch_size
        self.epoch=None;self.order=None

    def at(self, step):
        epoch,offset=divmod(step,self.steps_per_epoch)
        if epoch!=self.epoch:
            generator=torch.Generator(device=self.device).manual_seed(self.seed+epoch)
            self.order=torch.randperm(self.count,generator=generator,device=self.device)
            self.epoch=epoch
        return self.order[offset*self.batch_size:(offset+1)*self.batch_size]


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
    code = {str(path.relative_to(Path(__file__).parent)): sha256(path)
            for path in sorted(Path(__file__).parent.rglob('*.py'))}
    torch.set_num_threads(8)
    root = Path(plan['directory']); root.mkdir(parents=True, exist_ok=True)
    train = Bank(plan.get('bank_train','cache/paya-development/train'), plan.get('train_limit'))
    selection_path=plan.get('bank_selection','cache/paya-development/selection')
    selection = Bank(selection_path, plan.get('selection_limit'))
    full_selection = Bank(selection_path) if plan.get('final_full_selection') else None
    if plan.get('epochs'):
        plan['steps']=((len(train.rows)+plan['batch_size']-1)//plan['batch_size'])*plan['epochs']
    config = yaml.safe_load(Path(plan.get('base_config','config/paya-fast.yaml')).read_text())
    reference_head=yaml.safe_load(Path('config/paya-fast.yaml').read_text())['head']
    model = PayaModel.from_assets(config['assets'], config['preparation'], reference_head, 'cuda')
    reference = root / 'reference.json'
    if not reference.exists():
        from .checkpoint import load_checkpoint
        load_checkpoint('runs/paya-full-002/training/best.pt', model)
        with atomic_output(reference, 'w') as out:
            json.dump(evaluate(model.head, selection), out, indent=2, allow_nan=False)
    if full_selection is not None and not (root / 'reference-full.json').exists():
        from .checkpoint import load_checkpoint
        load_checkpoint('runs/paya-full-002/training/best.pt', model)
        with atomic_output(root / 'reference-full.json', 'w') as out:
            json.dump(evaluate(model.head, full_selection), out, indent=2, allow_nan=False)
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
        sampler=EpochBatches(len(train.rows),plan.get('batch_size',256),seed,'cuda') if plan.get('epochs') else None
        binding = fingerprint({'plan': plan, 'experiment': experiment, 'code': code,
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
        settings['head']=head.config
        settings['batch_size']=plan.get('batch_size',256)
        settings['optim'].update(lr=experiment.get('lr',3e-4),label_smoothing=experiment.get('smoothing',0),
            optimizer_steps=plan['steps'],sampling='epoch-no-replacement' if sampler else 'uniform-with-replacement')
        settings['experiment'] = {**experiment, 'plan': plan, 'binding': binding, 'code_sha256': code}
        model.head = head
        proxies = {name: SimpleNamespace(rows=bank.rows, sha256=bank.sha256)
                   for name, bank in [('train', train), ('selection', selection)]}
        started = perf_counter()
        with open_run(settings, destination, model, proxies, previous) as run:
            if resume.exists():
                torch.set_rng_state(saved['rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
            for step in range(begin + 1, plan['steps'] + 1):
                head.train()
                indices = sampler.at(step-1) if sampler else torch.randint(len(train.rows), (plan.get('batch_size', 256),), device='cuda')
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
                    with atomic_output(root/'status.json','w') as out:
                        json.dump({'phase':'training','experiment':experiment['name'],'step':step,
                            'code_revision':os.environ.get('PAYA_CODE_REVISION','source-hashes-in-result'),
                            'total_steps':plan['steps'],'epochs':plan.get('epochs'),
                            'completed_epochs':step//sampler.steps_per_epoch if sampler else None,
                            'latest_selection':history[-1] if history else None,'wandb_url':run.url,
                            'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/1024**2},out)
            head.load_state_dict(best_state)
            result = {'config': experiment, 'binding': binding, 'code_sha256': code, 'train_rows': len(train.rows),
                'selection_rows': len(selection.rows), 'seconds': perf_counter() - started,
                'parameters': sum(p.numel() for p in head.parameters()), 'history': history,
                'selected_by': 'lowest selection NLL among evaluated steps',
                'selection': evaluate(head, selection), 'mean_signal_ablation': evaluate(head, selection, 'mean'),
                'wandb': run.url}
            if full_selection is not None:
                result['full_selection'] = evaluate(head, full_selection)
                result['full_selection_mean_signal_ablation'] = evaluate(head, full_selection, 'mean')
                run.log({'step': plan['steps'],
                    **{f'full_selection/{k}': v for k, v in result['full_selection']['pooled'].items() if k != 'reliability'},
                    'full_selection/task_macro_f1': result['full_selection']['task_macro_f1'],
                    'full_selection/task_macro_auroc': result['full_selection']['task_macro_auroc']})
            with atomic_output(destination / 'best.pt') as out:
                torch.save({'binding': binding, 'config': experiment, 'head': best_state}, out)
            with atomic_output(destination / 'result.json', 'w') as out:
                json.dump(result, out, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
