import json
import sqlite3

import pytest

from paya.evaluate import metrics
from paya.postprocess import reference_comparison


def test_historical_comparison_checks_identity_and_scores_both_models(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    reference=tmp_path/'runs/paya-full-002';reference.mkdir(parents=True)
    destination=tmp_path/'runs/candidate';destination.mkdir()
    old=[];new=[]
    db=sqlite3.connect(':memory:')
    db.executescript('CREATE TABLE predictions(id TEXT PRIMARY KEY,value TEXT); '
                     'CREATE TABLE origins(id TEXT PRIMARY KEY,task TEXT)')
    for i in range(4):
        target=i%2
        row={'mapping_id':str(i),'source_id':str(i),'group_id':'g'+str(i),'recording_id':'r'+str(i),
             'criteria':['a','b'] if i<2 else ['c','d'],'target':target}
        old.append({**row,'logits':[1.,0.]})
        new.append({**row,'logits':[3.,0.] if target==0 else [0.,3.]})
        db.execute('INSERT INTO predictions VALUES (?,?)',(str(i),json.dumps(new[-1])))
        db.execute('INSERT INTO origins VALUES (?,?)',(str(i),'task1' if i<2 else 'task2'))
    path=reference/'test.predictions.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in old))
    (reference/'test.json').write_text(json.dumps({'metrics':metrics(old),'uncalibrated_metrics':metrics(old),'temperature':1.}))
    tasks={task:{'uncalibrated':{'macro_f1':1.,'macro_auroc_ovr':1.},
                 'calibrated':{'macro_f1':1.,'macro_auroc_ovr':1.}} for task in ['task1','task2']}
    got=reference_comparison(db,destination,tasks,new,1.)
    assert got['reference']['uncalibrated']['accuracy']==.5
    assert got['candidate']['uncalibrated']['accuracy']==1.
    assert got['reference']['uncalibrated']['macro_auroc_ovr']==.5
    assert got['same_mapping_ids_answers_and_audit_ids']
    old[0]['target']=1
    path.write_text(''.join(json.dumps(r)+'\n' for r in old))
    with pytest.raises(AssertionError):
        reference_comparison(db,destination,tasks,new,1.)
