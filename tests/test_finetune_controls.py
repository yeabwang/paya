import copy

import numpy as np
import pytest
import torch

from scripts.finetune_controls import ReplaySampler, PlateauControl, anchor_loss, distillation_loss
from scripts.selection_metrics import score_logits
from paya.evaluate import metrics
from paya.classification import classification_metrics


def test_pcgrad_preserves_aligned_gradients_and_projects_conflicts():
    from scripts.finetune_controls import project_conflicting_gradients
    aligned=torch.tensor([[1.,0.],[1.,1.]])
    torch.testing.assert_close(project_conflicting_gradients(aligned),aligned.sum(0))
    conflicting=torch.tensor([[1.,0.],[-1.,1.]])
    torch.testing.assert_close(project_conflicting_gradients(conflicting),torch.tensor([.5,1.5]))


def test_vector_metrics_match_public_metrics_with_permuted_candidates_and_ties():
    rows = [dict(task='task',criteria=c,target=t) for c,t in
            [(['b','a'],0),(['a','b'],1),(['b','a'],1),(['a','b'],0)]]
    logits = np.array([[1.,1.],[0.,2.],[1.,-1.],[2.,-2.]])
    records = [dict(r,logits=z.tolist()) for r,z in zip(rows,logits)]
    result = score_logits(rows,logits)
    old = metrics(records); cls = classification_metrics(records)
    for key in ['accuracy','nll','brier','ece_15']:
        assert result['pooled'][key] == pytest.approx(old[key])
    for key in ['macro_f1','macro_auroc_ovr']:
        assert result['per_task']['task'][key] == pytest.approx(cls[key])
    assert result['per_task']['task']['per_class']['a']['average_precision'] is not None


def test_replay_is_step_recoverable_and_does_not_touch_dropout_rng():
    rows = [dict(task=t) for t in ['rr_category']*100+['sdb_label']*100+['strong']*100]
    initial = torch.get_rng_state().clone()
    sampler = ReplaySampler(rows,40,42,replay=.25)
    values = sampler.at(10)
    assert sum(rows[i]['task']=='strong' for i in values)==10
    np.testing.assert_equal(values,ReplaySampler(rows,40,42,replay=.25).at(10))
    assert torch.equal(initial,torch.get_rng_state())
    weak = ReplaySampler(rows,40,42,replay=0).at(10)
    assert not any(rows[i]['task']=='strong' for i in weak)


def test_penalties_ignore_padding_and_have_finite_gradient():
    z = torch.tensor([[1.,2.,-torch.inf]],requires_grad=True)
    teacher = torch.tensor([[2.,1.,-torch.inf]])
    value = distillation_loss(z,teacher,torch.tensor([[True,True,False]]))
    value.backward()
    assert torch.isfinite(value) and torch.isfinite(z.grad).all() and z.grad[0,2]==0
    head = torch.nn.Linear(3,2)
    initial = {n:p.detach().clone() for n,p in head.named_parameters()}
    assert anchor_loss(head,initial)==0
    with torch.no_grad(): head.weight.add_(.1)
    assert anchor_loss(head,initial)>0


def test_plateau_state_resumes_learning_rate_and_stopping_decisions():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter],lr=1e-3)
    first = PlateauControl(1.)
    assert not first.update(1.,optimizer)
    saved = copy.deepcopy(first.state_dict())
    second = PlateauControl(9.); second.load_state_dict(saved)
    other = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))],lr=1e-3)
    for _ in range(4):
        assert first.update(1.,optimizer)==second.update(1.,other)
        assert optimizer.param_groups[0]['lr']==other.param_groups[0]['lr']
    assert first.bad==5


@pytest.mark.parametrize('experiment', [{'anchor':.01,'distill':.1}, {'pcgrad':True}])
def test_finetuning_checkpoint_resume_reproduces_dropout_optimizer_and_replay(tmp_path, experiment):
    from types import SimpleNamespace
    from scripts.finetune import update_step
    from paya.head import InteractionHead

    torch.manual_seed(9)
    signal, question, criteria = torch.randn(24,1,512), torch.randn(24,768), torch.randn(24,2,768)
    rows=[dict(task='rr_category' if i<12 else 'protected') for i in range(24)]
    bank=SimpleNamespace(rows=rows,device='cpu',targets=torch.arange(24)%2)
    bank.batch=lambda i:(signal[i],torch.ones(len(i),1,dtype=torch.bool),question[i],criteria[i],
                         torch.ones(len(i),2,dtype=torch.bool))
    initial_head=InteractionHead(width=16,dropout=.1)
    initial={n:p.detach().clone() for n,p in initial_head.named_parameters()}
    def run(end,resume=None):
        head=copy.deepcopy(initial_head); teacher=copy.deepcopy(head).eval().requires_grad_(False)
        optimizer=torch.optim.AdamW(head.parameters(),lr=3e-5)
        control=PlateauControl(0.)
        begin=0;torch.manual_seed(42)
        if resume:
            saved=torch.load(resume,weights_only=True)
            head.load_state_dict(saved['head']);optimizer.load_state_dict(saved['optimizer'])
            control.load_state_dict(saved['control']);begin=saved['step'];torch.set_rng_state(saved['rng'])
        sampler=ReplaySampler(rows,8,42,replay=.25)
        for step in range(begin+1,end+1):
            values=update_step(head,teacher,bank,sampler.at(step-1),optimizer,initial,experiment)
            if step%3==0:control.update(values['loss'],optimizer)
        state=dict(head=head.state_dict(),optimizer=optimizer.state_dict(),control=control.state_dict(),
                   rng=torch.get_rng_state(),step=end)
        return state
    first=run(6);torch.save(first,tmp_path/'resume.pt')
    resumed=run(12,tmp_path/'resume.pt');continuous=run(12)
    assert resumed['control']==continuous['control']
    assert all(torch.equal(resumed['head'][key],value) for key,value in continuous['head'].items())
