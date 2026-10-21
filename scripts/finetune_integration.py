"""Native-input checks and a 16-example memorization diagnostic, not a benchmark."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import yaml
from paya.checkpoint import load_checkpoint,save_checkpoint
from paya.data import ChoiceDataset
from scripts.development import Bank
from scripts.finetune import BankView,evaluate,update_step,write
from paya.model import PayaModel
from paya.provenance import sha256
from paya.tracking import open_run

torch.set_num_threads(8);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
root=Path('runs/paya-finetune-integration');root.mkdir(exist_ok=True)
config=yaml.safe_load(Path('config/paya-compact-full.yaml').read_text())
config.update(batch_size=16)
config['optim']=dict(lr=.001,weight_decay=.01,label_smoothing=.05,optimizer_steps=400)
config['wandb'].update(name='paya-finetune-integration',project='Paya',log_models=False)
model=PayaModel.from_assets(config['assets'],config['preparation'],config['head'],'cuda')
parent=load_checkpoint('runs/paya-full-003/training/best.pt',model)
bank=Bank('cache/paya-full-bank/train',limit=64000)
tasks=sorted({r['task'] for r in bank.rows})
positions=[next(i for i,r in enumerate(bank.rows) if r['task']==task) for task in tasks]
for task,answer in [('af_label','af'),('rr_category','tachypnea'),('sdb_label','severe_ahi>=30')]:
    positions.append(next(i for i,r in enumerate(bank.rows) if r['task']==task and r['criteria'][r['target']]==answer and i not in positions))
assert len(positions)==16
tiny=BankView(bank,positions=positions)
source=ChoiceDataset(config['data']['train'])
native=[source.rows[int(bank.indices[i])] for i in positions]
assert [r['mapping_id'] for r in native]==[r['mapping_id'] for r in tiny.rows]
prepared=[model.preprocessor.prepare(r['state']) for r in native]
fresh=model.encode_batch(prepared)
cached=bank.signal[torch.tensor(positions,device='cuda'),0]
assert torch.allclose(fresh,cached,atol=2e-4,rtol=2e-4)
teacher=deepcopy(model.head).eval().requires_grad_(False)
initial={n:p.detach().clone() for n,p in model.head.named_parameters()}
optimizer=torch.optim.AdamW(model.head.parameters(),lr=.001,weight_decay=.01)
before=evaluate(model.head,tiny)
with open_run(config,root,model,{'train':tiny}) as run:
    for step in range(1,401):
        values=update_step(model.head,teacher,tiny,np.arange(16),optimizer,initial,{'smoothing':.05})
        if step%20==0:run.log({'step':step,**{'train/'+k:v for k,v in values.items()}})
    after=evaluate(model.head,tiny)
    assert after['pooled']['accuracy']==1.,'Small fitting diagnostic did not learn all 16 examples'
    assert all(not p.requires_grad for p in model.encoder.parameters())
    save_checkpoint(root/'diagnostic.pt',model,{'diagnostic_only':True,'parent_sha256':sha256('runs/paya-full-003/training/best.pt')})
    row=native[0];expected=model.predict(row['state'],row['question'],row['criteria'])
    restored=PayaModel.from_assets(config['assets'],config['preparation'],config['head'],'cuda')
    load_checkpoint(root/'diagnostic.pt',restored)
    actual=restored.predict(row['state'],row['question'],row['criteria'])
    assert actual['choice']==expected['choice']
    assert np.allclose(list(actual['probabilities'].values()),list(expected['probabilities'].values()),atol=1e-7,rtol=1e-7)
    result=dict(rows=16,tasks=len(tasks),before=before['pooled'],after=after['pooled'],
        fresh_frozen_features_match=True,checkpoint_reload_matches=True,encoders_frozen=True,
        diagnostic_only=True,wandb_url=run.url)
    write(root/'result.json',result)
    run.summary.update({'diagnostic/accuracy':1.,'diagnostic/reload_verified':True})
print(json.dumps(result),flush=True)
