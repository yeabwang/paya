"""Evaluate final pilot weights and three predefined interpolation candidates."""
import argparse
import json
from pathlib import Path
import torch
import yaml
from paya.checkpoint import load_checkpoint
from scripts.development import Bank
from scripts.finetune import evaluate,write
from scripts.finetune_controls import retention_failures,recovery_f1
from paya.model import PayaModel

torch.set_num_threads(8)
config=yaml.safe_load(Path('config/paya-compact-full.yaml').read_text())
model=PayaModel.from_assets(config['assets'],config['preparation'],config['head'],'cuda')
parent=load_checkpoint('runs/paya-full-003/training/best.pt',model)
bank=Bank('cache/paya-development/selection')
reference=evaluate(model.head,bank)
report={'reference':reference,'last_weights':{},'interpolation':{}}
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--plans',nargs='+',default=['finetune-pilot','finetune-preservation','finetune-conflict'])
args=parser.parse_args()
for plan_name in args.plans:
    plan=json.loads(Path('config/'+plan_name+'.json').read_text())
    root=Path(plan['directory'])
    for experiment in plan['experiments']:
        state=torch.load(root/experiment['name']/'last.pt',map_location='cpu',weights_only=True)
        model.head.load_state_dict(state['head'])
        result=evaluate(model.head,bank)
        report['last_weights'][experiment['name']]={'step':state['step'],'metrics':result,
            'recovery_f1_gain':recovery_f1(result)-recovery_f1(reference),
            'retention_failures':retention_failures(result,reference)}
        if experiment['name']=='replay':
            for alpha in [.25,.5,.75]:
                weights={k:(1-alpha)*parent['learned']['head'][k]+alpha*v for k,v in state['head'].items()}
                model.head.load_state_dict(weights);value=evaluate(model.head,bank)
                report['interpolation'][str(alpha)]={'metrics':value,
                    'recovery_f1_gain':recovery_f1(value)-recovery_f1(reference),
                    'retention_failures':retention_failures(value,reference)}
write('runs/paya-finetune-review.json',report)
for group in ['last_weights','interpolation']:
    print(json.dumps({'analysis':group,'results':{k:{'recovery_f1_gain':v['recovery_f1_gain'],
        'retention_failures':len(v['retention_failures']),'pooled':v['metrics']['pooled']} for k,v in report[group].items()}}),flush=True)
