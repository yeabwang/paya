"""Controlled, recoverable continuation of an existing selected MLP head."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from .atomic import atomic_output
from .checkpoint import load_checkpoint, save_checkpoint
from .data import ChoiceDataset, assert_disjoint
from .development import Bank, verify_bank_identity
from .finetune_controls import (ReplaySampler, PlateauControl, RECOVERY_TASKS,
                                anchor_loss, distillation_loss, retention_failures, recovery_f1)
from .model import PayaModel
from .provenance import fingerprint, sha256
from .selection_metrics import score_logits
from .tracking import open_run, run_identity
from .train import choice_loss


def write(path, value):
    with atomic_output(path, 'w') as out: json.dump(value, out, indent=2, allow_nan=False)


class BankView:
    def __init__(self, bank, count=None, positions=None):
        self.bank = bank
        self.positions = np.asarray(positions if positions is not None else np.arange(count or len(bank.rows)))
        self.rows = [bank.rows[int(i)] for i in self.positions]
        self.manifest, self.device = bank.manifest, bank.device
        self.targets = bank.targets[torch.as_tensor(self.positions,device=bank.device)]
        self.sha256 = fingerprint({'source':bank.sha256,'positions':self.positions.tolist()})

    def batch(self, indices):
        positions = torch.as_tensor(self.positions,device=self.device)[indices]
        return self.bank.batch(positions)


@torch.no_grad()
def predict_logits(head, bank, batch_size=2048):
    head.eval(); width = max(len(r['criteria']) for r in bank.rows)
    logits = np.full((len(bank.rows),width),-np.inf,dtype=np.float32)
    for start in range(0,len(bank.rows),batch_size):
        ids = torch.arange(start,min(start+batch_size,len(bank.rows)),device=bank.device)
        values = head(*bank.batch(ids)).cpu().numpy()
        logits[start:start+len(ids),:values.shape[1]] = values
    return logits


def evaluate(head, bank): return score_logits(bank.rows,predict_logits(head,bank))


def update_step(head, teacher, bank, indices, optimizer, initial, experiment):
    head.train(); indices = torch.as_tensor(indices,device=bank.device)
    batch = bank.batch(indices); z = head(*batch)
    data_loss = choice_loss(z,bank.targets[indices],batch[-1],experiment.get('smoothing',.05))
    penalty = z.new_zeros(())
    if experiment.get('anchor',0): penalty = penalty+experiment['anchor']*anchor_loss(head,initial)
    if experiment.get('distill',0):
        protected = torch.tensor([bank.rows[int(i)]['task'] not in RECOVERY_TASKS for i in indices.cpu()],
                                 dtype=torch.bool,device=bank.device)
        if protected.any():
            with torch.no_grad(): old = teacher(*batch)
            penalty = penalty+experiment['distill']*distillation_loss(z[protected],old[protected],batch[-1][protected])
    loss = data_loss+penalty
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite fine-tuning loss')
    optimizer.zero_grad(set_to_none=True); loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
    if not torch.isfinite(norm): raise FloatingPointError('Nonfinite fine-tuning gradients')
    optimizer.step()
    return dict(loss=float(loss.detach()),data_loss=float(data_loss.detach()),penalty=float(penalty.detach()),
                gradient_norm=float(norm),learning_rate=optimizer.param_groups[0]['lr'])


def cpu_state(head): return {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}


def gradient_probe(head, bank, repeats=3):
    tasks = sorted({r['task'] for r in bank.rows}); vectors = []; norms = []; head.eval()
    pools = {t:np.array([i for i,r in enumerate(bank.rows) if r['task']==t]) for t in tasks}
    for repeat in range(repeats):
        gradients = []
        for i,task in enumerate(tasks):
            ids = np.random.default_rng(8100+repeat*len(tasks)+i).choice(pools[task],32,replace=len(pools[task])<32)
            batch = bank.batch(torch.as_tensor(ids,device=bank.device))
            loss = choice_loss(head(*batch),bank.targets[torch.as_tensor(ids,device=bank.device)],batch[-1],.05)
            values = torch.autograd.grad(loss,tuple(head.parameters()))
            gradients.append(torch.cat([v.flatten() for v in values]).detach())
        matrix = torch.stack(gradients); size = matrix.norm(dim=1)
        cosine = (matrix@matrix.T)/(size[:,None]*size[None,:]).clamp_min(1e-20)
        vectors.append(cosine.cpu().numpy()); norms.append(size.cpu().numpy())
    mean = np.mean(vectors,axis=0)
    return dict(tasks=tasks,mean_cosine=mean.tolist(),mean_gradient_norm=np.mean(norms,axis=0).tolist(),
        negative_pair_fraction=float((mean[np.triu_indices(len(tasks),1)]<0).mean()),
        interpretation='Repeated training-batch diagnostic; negative cosines alone do not prove harmful interference.')


def run_experiment(model, parent, train, selection, full_selection, config, plan, experiment, root, code):
    directory = root/experiment['name']; directory.mkdir(parents=True,exist_ok=True)
    binding = fingerprint(dict(plan=plan,experiment=experiment,train=train.sha256,selection=selection.sha256,
                               parent=sha256(plan['parent']),code=code,config=config))
    result_path = directory/'result.json'
    if result_path.exists():
        value = json.loads(result_path.read_text())
        if value['binding']!=binding: raise ValueError('Completed fine-tuning experiment binding differs')
        return value
    seed = experiment.get('seed',42); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    head = model.head; head.load_state_dict(parent['learned']['head']); model.train(False)
    teacher = deepcopy(head).eval().requires_grad_(False)
    initial = {n:p.detach().clone() for n,p in head.named_parameters()}
    optimizer = torch.optim.AdamW(head.parameters(),lr=experiment.get('lr',3e-5),weight_decay=.01)
    sampler = ReplaySampler(train.rows,plan.get('batch_size',512),seed,experiment.get('replay'))
    baseline = evaluate(head,selection)
    control = PlateauControl(baseline['task_mean_nll'])
    best_score = baseline['task_mean_nll']; best_step = 0; best = cpu_state(head)
    eligible_score = best_score; eligible_step = 0; eligible = deepcopy(best)
    history = []; begin = 0; previous = None
    exposures = np.zeros(len(train.rows),dtype=np.int64)
    recovery_path = directory/'last.pt'
    if recovery_path.exists():
        state = torch.load(recovery_path,map_location='cpu',weights_only=True)
        if state['binding']!=binding: raise ValueError('Fine-tuning recovery binding differs')
        head.load_state_dict(state['head']); optimizer.load_state_dict(state['optimizer'])
        control.load_state_dict(state['control']); begin=state['step']; previous=state['wandb']
        best_score,best_step,best=state['best_score'],state['best_step'],state['best']
        eligible_score,eligible_step,eligible=state['eligible_score'],state['eligible_step'],state['eligible']
        history=state['history'];exposures=state['exposures'].numpy()
    settings=deepcopy(config)
    settings.update(batch_size=plan.get('batch_size',512),head=head.config)
    settings['wandb'].update(name='paya-'+experiment['name'],project='Paya',log_models=False)
    settings['optim'].update(lr=experiment.get('lr',3e-5),label_smoothing=.05)
    settings['continuation'] = dict(parent_sha256=sha256(plan['parent']),experiment=experiment,
                                    plan=plan,binding=binding,source_sha256=code)
    started=perf_counter(); completed=False
    try:
        with open_run(settings,directory,model,{'train':train,'selection':selection},previous) as run:
            if recovery_path.exists():
                torch.set_rng_state(state['rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
            stop = control.bad>=control.stop_after
            for step in range(begin+1,plan['steps']+1):
                if stop: break
                indices=sampler.at(step-1);np.add.at(exposures,indices,1)
                values=update_step(head,teacher,train,indices,optimizer,initial,experiment)
                if step%20==0: run.log(dict(step=step,**{'train/'+k:v for k,v in values.items()}))
                if step%plan['evaluate_every']==0 or step==plan['steps']:
                    result=evaluate(head,selection);score=result['task_mean_nll']
                    failures=retention_failures(result,baseline)
                    history.append(dict(step=step,selection=result,retention_failures=failures,
                                        learning_rate=optimizer.param_groups[0]['lr']))
                    if score<best_score: best_score,best_step,best=score,step,cpu_state(head)
                    if not failures and score<eligible_score:
                        eligible_score,eligible_step,eligible=score,step,cpu_state(head)
                    stop=control.update(score,optimizer)
                    run.log(dict(step=step,epoch=step*plan.get('batch_size',512)/len(train.rows),
                        **{'selection/'+k:v for k,v in result['pooled'].items()},
                        **{'selection/task_macro_f1':result['task_macro_f1'],
                           'selection/task_macro_auroc':result['task_macro_auroc'],
                           'selection/task_mean_nll':score,'selection/retention_failures':len(failures)}))
                    write(root/'status.json',dict(phase='fine_tuning',experiment=experiment['name'],step=step,
                        total_steps=plan['steps'],wandb_url=run.url,latest_selection=result['pooled'],
                        task_macro_f1=result['task_macro_f1'],retention_failures=failures))
                    print(json.dumps(dict(experiment=experiment['name'],step=step,nll=score,
                        recovery_f1=recovery_f1(result),retention_failures=len(failures))),flush=True)
                if step%100==0 or step==plan['steps'] or stop:
                    state=dict(binding=binding,step=step,head=cpu_state(head),optimizer=optimizer.state_dict(),
                        control=control.state_dict(),best_score=best_score,best_step=best_step,best=best,
                        eligible_score=eligible_score,eligible_step=eligible_step,eligible=eligible,
                        rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                        exposures=torch.from_numpy(exposures.copy()),history=history,
                        wandb=run_identity(run,'online'))
                    with atomic_output(recovery_path) as out:torch.save(state,out)
            head.load_state_dict(best)
            full_baseline_path=root/'reference-full.json'
            if not full_baseline_path.exists():
                head.load_state_dict(parent['learned']['head']);write(full_baseline_path,evaluate(head,full_selection))
            reference=json.loads(full_baseline_path.read_text())
            head.load_state_dict(best); research=evaluate(head,full_selection)
            save_checkpoint(directory/'research.pt',model,{**parent['training'],'continuation_binding':binding,
                'continuation_step':best_step,'parent_sha256':sha256(plan['parent'])})
            head.load_state_dict(eligible); retained=evaluate(head,full_selection)
            save_checkpoint(directory/'eligible.pt',model,{**parent['training'],'continuation_binding':binding,
                'continuation_step':eligible_step,'parent_sha256':sha256(plan['parent'])})
            result=dict(binding=binding,experiment=experiment,parent_sha256=sha256(plan['parent']),
                steps_completed=state['step'] if recovery_path.exists() else 0,early_stopped=stop,
                best_step=best_step,eligible_step=eligible_step,monitor_baseline=baseline,history=history,
                full_selection=research,full_retention_failures=retention_failures(research,reference),
                full_recovery_f1_gain=recovery_f1(research)-recovery_f1(reference),
                retained_full_selection=retained,retained_full_failures=retention_failures(retained,reference),
                exposures=dict(draws=int(exposures.sum()),unique_rows=int((exposures>0).sum()),
                    maximum_row_repeats=int(exposures.max()),
                    per_task={t:int(sum(exposures[i] for i,r in enumerate(train.rows) if r['task']==t))
                              for t in sorted({r['task'] for r in train.rows})}),
                seconds=perf_counter()-started,wandb_url=run.url,source_sha256=code)
            write(result_path,result);completed=True
            run.summary.update({'local/result_complete':True,'best/step':best_step,
                'full_selection/recovery_f1_gain':result['full_recovery_f1_gain'],
                'full_selection/retention_failures':len(result['full_retention_failures'])})
    except Exception as error:
        if not completed:raise
        write(directory/'tracking-finalization-warning.json',dict(warning=type(error).__name__,local_results_valid=True))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True)
    args=parser.parse_args();plan=json.loads(Path(args.plan).read_text())
    torch.set_num_threads(8)
    root=Path(plan['directory']);root.mkdir(parents=True,exist_ok=True)
    config=yaml.safe_load(Path(plan.get('base_config','config/paya-compact-full.yaml')).read_text())
    model=PayaModel.from_assets(config['assets'],config['preparation'],config['head'],'cuda')
    parent=load_checkpoint(plan['parent'],model)
    if parent['temperature']!=1 or parent['calibration'] is not None:
        raise ValueError('Continuation requires the selected raw checkpoint')
    datasets={r:ChoiceDataset(config['data'][r]) for r in ['train','selection']}
    assert_disjoint(*datasets.values())
    train=Bank(plan.get('bank_train','cache/paya-full-bank/train'),limit=plan.get('train_limit',64000))
    full_selection=Bank(plan.get('bank_selection','cache/paya-development/selection'))
    for bank,role in [(train,'train'),(full_selection,'selection')]:verify_bank_identity(bank,model,datasets[role])
    order=np.random.default_rng(20261021).permutation(len(full_selection.rows))[:plan.get('selection_limit',16384)]
    selection=BankView(full_selection,positions=order)
    code={name:sha256(Path(__file__).parent/name) for name in ['finetune.py','finetune_controls.py',
        'selection_metrics.py','head.py','model.py','checkpoint.py','tracking.py','development.py','train.py','data.py']}
    if not (root/'gradient-probe.json').exists():write(root/'gradient-probe.json',gradient_probe(model.head,train))
    results=[]
    for experiment in plan['experiments']:
        results.append(run_experiment(model,parent,train,selection,full_selection,config,plan,experiment,root,code))
    write(root/'summary.json',dict(protocol='A: unchanged native roles',plan=plan,
        experiments=[dict(name=r['experiment']['name'],best_step=r['best_step'],
            pooled=r['full_selection']['pooled'],macro_f1=r['full_selection']['task_macro_f1'],
            macro_auroc=r['full_selection']['task_macro_auroc'],recovery_f1_gain=r['full_recovery_f1_gain'],
            retention_failures=r['full_retention_failures'],wandb_url=r['wandb_url']) for r in results]))
    write(root/'status.json',dict(phase='complete',summary=str(root/'summary.json')))


if __name__=='__main__': main()
