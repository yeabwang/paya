"""Versioned group-stratified role manifests referencing existing waveform shards."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3

import numpy as np

from .atomic import atomic_output
from .provenance import fingerprint

ROLES = ('train','selection','calibration','confirmation')


def allocate_groups(counts, fractions=(.8,.1,.05,.05), seed=20261021, attempts=32):
    counts=np.asarray(counts,dtype=np.float64);fractions=np.asarray(fractions)
    support=(counts>0).sum(0)
    if (support<len(fractions)).any(): raise ValueError('Too few independent groups for every role')
    totals=counts.sum(0);target=fractions[:,None]*totals
    best=None;best_score=float('inf')
    for attempt in range(attempts):
        rng=np.random.default_rng(seed+attempt);assignment=np.full(len(counts),-1,dtype=int)
        mass=np.zeros_like(target);presence=np.zeros_like(target)
        valid=True
        for label in np.argsort(support,kind='stable'):
            for role in rng.permutation(len(fractions)):
                if presence[role,label]:continue
                available=np.flatnonzero((assignment<0)&(counts[:,label]>0))
                if not len(available):valid=False;break
                missing=presence[role]==0
                gain=((counts[available]>0)*missing/support).sum(1)
                gain+=rng.uniform(0,.01,len(available))
                group=int(available[gain.argmax()]);assignment[group]=role
                mass[role]+=counts[group];presence[role]+=(counts[group]>0)
            if not valid:break
        if not valid:continue
        remaining=np.flatnonzero(assignment<0)
        remaining=remaining[np.argsort(-counts[remaining].sum(1),kind='stable')]
        scale=totals**2
        for group in remaining:
            delta=((mass+counts[group]-target)**2-(mass-target)**2)/(scale[None,:]*fractions[:,None])
            role=int(delta.sum(1).argmin());assignment[group]=role;mass[role]+=counts[group]
        score=float((((mass-target)/totals)**2/fractions[:,None]).sum())
        if score<best_score:best,best_score=assignment.copy(),score
    if best is None:raise ValueError('No valid joint class/group assignment found; preserve existing roles')
    assert all(((counts[best==r]>0).sum(0)>0).all() for r in range(len(fractions)))
    return best,dict(objective=best_score,target_fractions=fractions.tolist(),
        achieved_row_fractions=[float(counts[best==r].sum()/counts.sum()) for r in range(len(fractions))])


def build_roles(audit_path='runs/paya-next-audit/metadata.sqlite3', output='data/experiment-roles-v2'):
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(audit_path)
    pairs=list(db.execute('SELECT DISTINCT task,answer FROM examples ORDER BY task,answer'))
    pair_index={tuple(v):i for i,v in enumerate(pairs)}
    groups=[r[0] for r in db.execute('SELECT DISTINCT component FROM components ORDER BY component')]
    group_index={g:i for i,g in enumerate(groups)}
    counts=np.zeros((len(groups),len(pairs)),dtype=np.int64)
    for group,task,answer,count in db.execute('SELECT component,task,answer,COUNT(*) FROM examples e '
            'JOIN components c USING(group_id) GROUP BY component,task,answer'):
        counts[group_index[group],pair_index[task,answer]]=count
    assignment,details=allocate_groups(counts)
    identity=fingerprint({'groups':groups,'counts':counts.tolist(),'assignment':assignment.tolist()})
    if (root/'report.json').exists():
        previous=json.loads((root/'report.json').read_text())
        if previous['assignment_sha256']!=identity:raise ValueError('Existing role assignment differs')
        return previous
    db.execute('CREATE TABLE IF NOT EXISTS roles_v2(component TEXT PRIMARY KEY,role TEXT)')
    if db.execute('SELECT COUNT(*) FROM roles_v2').fetchone()[0]:
        previous=dict(db.execute('SELECT component,role FROM roles_v2'))
        if previous!={g:ROLES[int(r)] for g,r in zip(groups,assignment)}:
            raise ValueError('Partial role assignment differs')
    else:
        db.executemany('INSERT INTO roles_v2 VALUES (?,?)',((g,ROLES[int(r)]) for g,r in zip(groups,assignment)))
        db.commit()
    joined=' FROM examples e JOIN components c USING(group_id) JOIN roles_v2 v USING(component) '
    report=dict(protocol='paya-roles-v2',assignment_sha256=identity,rows_unchanged=True,
        historical_test_unchanged=True,confirmation_locked=True,details=details,roles={})
    for role in ROLES:
        filters=defaultdict(lambda:defaultdict(list))
        for file,group,position in db.execute('SELECT filename,row_group,position'+joined+
                'WHERE v.role=? ORDER BY filename,row_group,position',(role,)):
            filters[file][str(group)].append(position)
        manifest=dict(files=sorted(filters),row_indices=filters,role_protocol='paya-roles-v2',
            assignment_sha256=identity,path_base=str(Path.cwd()),role=role)
        with atomic_output(root/(role+'.json'),'w') as out:json.dump(manifest,out)
        classes={}
        for task,answer,rows,components in db.execute('SELECT task,answer,COUNT(*),COUNT(DISTINCT component)'+
                joined+'WHERE v.role=? GROUP BY task,answer',(role,)):
            classes.setdefault(task,{})[answer]=dict(rows=rows,independent_groups=components)
        report['roles'][role]=dict(rows=sum(v['rows'] for c in classes.values() for v in c.values()),
            groups=int((assignment==ROLES.index(role)).sum()),class_coverage=classes,
            manifest=str(root/(role+'.json')))
    from .data import ChoiceDataset,assert_disjoint
    datasets={r:ChoiceDataset(root/(r+'.json')) for r in ROLES}
    assert_disjoint(*datasets.values())
    for role,data in datasets.items():
        report['roles'][role]['sha256']=data.sha256
        assert len(data.rows)==report['roles'][role]['rows']
    with atomic_output(root/'report.json','w') as out:json.dump(report,out,indent=2)
    db.close();return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',default='runs/paya-next-audit/metadata.sqlite3')
    parser.add_argument('--output',default='data/experiment-roles-v2')
    args=parser.parse_args();report=build_roles(args.audit,args.output)
    print(json.dumps({'phase':'roles_created','roles':{r:{k:v for k,v in d.items() if k!='class_coverage'}
                                                    for r,d in report['roles'].items()}}),flush=True)
