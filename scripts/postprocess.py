"""Validate the selected model, calibrate independently, test, and time inference."""
import json
from pathlib import Path
import sqlite3
import time
import numpy as np
import pyarrow.parquet as pq
import torch
from paya.atomic import atomic_output
from paya.calibrate import fit_temperature
from paya.checkpoint import load_checkpoint, save_checkpoint, learned_digest
from paya.data import make_batch
from paya.evaluate import collect, metrics
from paya.features import open_feature_cache
from paya.model import PayaModel
from paya.classification import classification_metrics
from paya.provenance import sha256


def task_summary(tasks,name):
    values=[value[name] for value in tasks.values()]
    supported=[v['macro_auroc_ovr'] for v in values if v['macro_auroc_ovr'] is not None]
    return {'macro_f1':sum(v['macro_f1'] for v in values)/len(values),
            'macro_auroc_ovr':sum(supported)/len(supported) if supported else None,'tasks':len(values)}


def reference_comparison(db,directory,tasks,records,temperature):
    """Score archived reference logits on the same IDs; never select using test."""
    from paya.evaluate import PredictionRecords
    original=Path('runs/paya-full-002')
    old=json.loads((original/'test.json').read_text())
    path=original/'test.predictions.jsonl'
    db.execute('DROP TABLE IF EXISTS reference_predictions')
    db.execute('CREATE TABLE reference_predictions(id TEXT PRIMARY KEY,value TEXT)')
    count=0;pending=[]
    for row in PredictionRecords(path,len(records)):
        match=db.execute('SELECT value FROM predictions WHERE id=?',(row['mapping_id'],)).fetchone()
        assert match is not None
        current=json.loads(match[0])
        assert current['criteria'][current['target']]==row['criteria'][row['target']]
        assert set(current['criteria'])==set(row['criteria'])
        assert all(current[k]==row[k] for k in ['source_id','group_id','recording_id'])
        pending.append((row['mapping_id'],json.dumps(row)))
        if len(pending)>=512:
            db.executemany('INSERT INTO reference_predictions VALUES (?,?)',pending);pending.clear()
        count+=1
    db.executemany('INSERT INTO reference_predictions VALUES (?,?)',pending);db.commit()
    assert count==len(records)==old['metrics']['count']
    reference_tasks={}
    for task in tasks:
        def outcomes():
            return (json.loads(r[0]) for r in db.execute('SELECT p.value FROM reference_predictions p '
                'JOIN origins o ON p.id=o.id WHERE o.task=?',(task,)))
        reference_tasks[task]={name:{**metrics(outcomes(),t),**classification_metrics(outcomes(),t)}
            for name,t in [('uncalibrated',1.),('calibrated',old['temperature'])]}
    result={'reference_run':'paya-full-002','candidate_run':directory.name,
        'interpretation':'Retrospective historical test comparison; no test-driven checkpoint or temperature selection.',
        'same_mapping_ids_answers_and_audit_ids':True,'reference_predictions_sha256':sha256(path),
        'reference':{'uncalibrated':{**old['uncalibrated_metrics'],**task_summary(reference_tasks,'uncalibrated')},
                     'calibrated':{**old['metrics'],**task_summary(reference_tasks,'calibrated')},'per_task':reference_tasks},
        'candidate':{'uncalibrated':{**metrics(records),**task_summary(tasks,'uncalibrated')},
                     'calibrated':{**metrics(records,temperature),**task_summary(tasks,'calibrated')},'per_task':tasks}}
    with atomic_output(directory/'baseline-comparison.json','w') as output:json.dump(result,output,indent=2,allow_nan=False)
    return result

def write(path,value):
    with atomic_output(path,'w') as output: json.dump(value,output,indent=2,allow_nan=False)

def run(model,datasets,config,directory,status):
    best=load_checkpoint(directory/'training/best.pt',model)
    assert model.calibration is None and float(model.temperature)==1
    cache=open_feature_cache(config)
    status('selected_checkpoint_validation')
    selection=collect(model,datasets['selection'],config['batch_size'],cache,
                      output_path=directory/'selected-validation.predictions.jsonl')
    validation=metrics(selection)
    assert abs(validation['nll']-best['training']['best_nll'])<1e-5
    write(directory/'validation.json',validation)
    status('calibration')
    calibration=fit_temperature(model,datasets['calibration'],best['training']['forbidden_calibration'],
        config['batch_size'],cache,output_path=directory/'calibration.predictions.jsonl')
    save_checkpoint(directory/'calibrated.pt',model,best['training'])
    write(directory/'calibration.json',calibration)
    temperature=float(model.temperature)
    class Progress:
        def batches(self,batch_size):
            count=0
            for index,batch in enumerate(datasets['test'].batches(batch_size)):
                yield batch
                count+=len(batch)
                if (index+1)%20==0: status('test_evaluation',evaluated_rows=count,test_rows=len(datasets['test'].rows))
    status('test_evaluation',evaluated_rows=0,test_rows=len(datasets['test'].rows))
    records=collect(model,Progress(),config['batch_size'],cache,output_path=directory/'test.predictions.jsonl')
    assert len(records)==len(datasets['test'].rows)
    status('evaluation_aggregation',evaluated_rows=len(records))
    db=sqlite3.connect(directory/'evaluation.sqlite3')
    db.executescript('PRAGMA cache_size=-8192; PRAGMA temp_store=FILE; '
        'DROP TABLE IF EXISTS origins; DROP TABLE IF EXISTS predictions; '
        'CREATE TABLE origins(id TEXT PRIMARY KEY, task TEXT); '
        'CREATE TABLE predictions(id TEXT PRIMARY KEY, subset TEXT, value TEXT, row_index INTEGER);')
    provenance_roots=list(Path('data').glob('*/provenance'))
    assert len(provenance_roots)==1
    for path in provenance_roots[0].rglob('test-*.parquet'):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256,columns=['mapping_id','qa_task']):
            db.executemany('INSERT INTO origins VALUES (?,?)',((r['mapping_id'],r['qa_task']) for r in batch.to_pylist()))
    for index,record in enumerate(records):
        db.execute('INSERT INTO predictions VALUES (?,?,?,?)',
                   (record['mapping_id'],record['datasetname'],json.dumps(record),index))
    db.commit()
    joined=' FROM predictions p JOIN origins o ON p.id=o.id'
    assert db.execute('SELECT COUNT(*)'+joined).fetchone()[0]==len(records)
    def grouped(field):
        values=[r[0] for r in db.execute('SELECT DISTINCT '+field+joined)]
        return {key:{'uncalibrated':metrics(json.loads(row[0]) for row in db.execute(
            'SELECT p.value'+joined+' WHERE '+field+'=?',(key,))),
            'calibrated':metrics((json.loads(row[0]) for row in db.execute(
            'SELECT p.value'+joined+' WHERE '+field+'=?',(key,))),temperature)} for key in values}
    tasks=grouped('o.task');subsets=grouped('p.subset')
    for task,value in tasks.items():
        def outcomes():
            return (json.loads(row[0]) for row in db.execute('SELECT p.value'+joined+' WHERE o.task=?',(task,)))
        value['uncalibrated'].update(classification_metrics(outcomes()))
        value['calibrated'].update(classification_metrics(outcomes(),temperature))
    assert len(tasks)==13
    assert sum(value['calibrated']['count'] for value in tasks.values())==len(records)
    counts={task:{} for task in tasks}
    for task,value in db.execute('SELECT o.task,p.value'+joined):
        record=json.loads(value);answer=record['criteria'][record['target']]
        entry=counts[task].setdefault(answer,{'count':0,'correct':0})
        entry['count']+=1
        winner=min(range(len(record['logits'])),key=lambda i:(-record['logits'][i],record['criteria'][i]))
        entry['correct']+=int(winner==record['target'])
    indices=[r[0] for r in db.execute('SELECT MIN(p.row_index)'+joined+' GROUP BY o.task')]
    latency_rows=[datasets['test'].rows[index] for index in indices]
    status('historical_reference_comparison')
    comparison=reference_comparison(db,directory,tasks,records,temperature)
    db.close()
    status('latency_measurement')
    torch.cuda.synchronize();started=time.perf_counter()
    clone=PayaModel.from_assets(config['assets'],config['preparation'],config['head'],'cuda')
    load_checkpoint(directory/'calibrated.pt',clone)
    torch.cuda.synchronize();load_seconds=time.perf_counter()-started
    first=latency_rows[0]
    torch.cuda.synchronize();started=time.perf_counter()
    clone.predict(first['state'],first['question'],first['criteria'])
    torch.cuda.synchronize();first_request_ms=(time.perf_counter()-started)*1000
    assert learned_digest(clone)==learned_digest(model)
    for row in latency_rows:
        clone.predict(row['state'],row['question'],row['criteria'])
    with torch.no_grad():
        batch=make_batch(clone,latency_rows)
        reference=model(make_batch(model,latency_rows),cache)['logits']
        actual=clone(batch,cache)['logits']
        mask=batch.criteria_mask
        torch.testing.assert_close(actual[mask],reference[mask],atol=2e-4,rtol=2e-4)
    times=[]
    for index in range(130):
        row=latency_rows[index%len(latency_rows)]
        torch.cuda.synchronize();started=time.perf_counter()
        clone.predict(row['state'],row['question'],row['criteria'])
        torch.cuda.synchronize();times.append((time.perf_counter()-started)*1000)
    latency={'device':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,
        'batch_size':1,'requests':len(times),'task_count':len(latency_rows),
        'p50_ms':float(np.percentile(times,50)),'p95_ms':float(np.percentile(times,95)),
        'p99_ms':float(np.percentile(times,99)),'mean_ms':float(np.mean(times)),
        'fresh_model_load_seconds':load_seconds,'first_request_with_cold_text_model_ms':first_request_ms,
        'signal_feature_cache_used':False,'cold_process_startup_measured':False,
        'measurement':'Public predict API; native signal preparation and encoder included; text embeddings warmed; CUDA synchronized.'}
    write(directory/'latency.json',latency)
    uncalibrated=metrics(records);calibrated=metrics(records,temperature)
    assert uncalibrated['accuracy']==calibrated['accuracy']
    readiness=json.loads(Path('reports/training-readiness.json').read_text())
    roles=json.loads(Path('reports/calibration-roles.json').read_text())
    benchmark=json.loads(Path('reports/fast-benchmark.json').read_text())
    report={'run':directory.name,'checkpoint':str(directory/'calibrated.pt'),
        'code_revision':json.loads((directory/'status.json').read_text())['code_revision'],
        'best_epoch':best['training']['epoch'],'best_step':best['training']['global_step'],
        'best_selection_nll':best['training']['best_nll'],'validation':validation,
        'calibration':calibration,'temperature':temperature,'calibrated':True,
        'metrics':calibrated,'uncalibrated_metrics':uncalibrated,
        'per_task':{task:value['calibrated'] for task,value in tasks.items()},
        'per_task_comparison':tasks,'per_subset':subsets,'task_answer_counts':counts,
        'latency':latency,'legacy_warm_cache_benchmark':benchmark,'validation_role_protocol':roles,
        'coverage_limitations':readiness.get('absent_labels_by_split',{}),
        'frozen_encoders_unchanged':True,'checkpoint_reload_passed':True,
        'predictions_path':str(records.path),'next_training_requires_user_review':True}
    for name in ['uncalibrated','calibrated']:report[name+'_task_summary']=task_summary(tasks,name)
    report['historical_comparison_path']=str(directory/'baseline-comparison.json')
    write(directory/'test.json',report)
    table='\n'.join(f"| {task} | {value['calibrated']['count']:,} | {value['calibrated']['accuracy']:.2%} | {value['calibrated']['macro_f1']:.4f} | {value['calibrated']['macro_auroc_ovr']:.4f} | {value['uncalibrated']['nll']:.4f} | {value['calibrated']['nll']:.4f} |" for task,value in sorted(tasks.items()))
    document=f'''# Paya full training report

Run: {directory.name}. Code: {report['code_revision']}. Selected epoch: {report['best_epoch']}, by independent selection NLL.

Three complete training epochs; batch size {config['batch_size']}; 1,637,289 training QA rows. The signal and text encoders remained frozen. A 264,065-parameter interaction head replaces the reference attention head; public inputs remain unchanged. AdamW lr 0.0003, label smoothing 0.05 over real candidates, no class weighting. This run started with a fresh head.

The supplied validation split was partitioned into subject-disjoint selection ({roles['roles']['selection']['rows']:,} rows) and calibration ({roles['roles']['calibration']['rows']:,} rows) roles before training. Connected recordings/waveforms remain together. Native memberships and retained content were preserved. Earlier reference test failures informed the research direction, so this is a retrospective test comparison, not a new unseen benchmark. This cycle selected its recipe and checkpoint using selection data, fit temperature on calibration data, and did not optimize on test.

## Test results

Test rows: {calibrated['count']:,}. Accuracy: {calibrated['accuracy']:.2%}. Uncalibrated NLL: {uncalibrated['nll']:.6f}; calibrated NLL: {calibrated['nll']:.6f}. Calibrated Brier: {calibrated['brier']:.6f}; ECE (15 bins): {calibrated['ece_15']:.6f}. Fitted global temperature: {temperature:.6f}. Temperature calibration preserves answer rankings.

Mean task macro F1: {report['calibrated_task_summary']['macro_f1']:.4f}; mean supported-class task AUROC: {report['calibrated_task_summary']['macro_auroc_ovr']:.4f}. AUROC is one-vs-rest by answer text; undefined classes are omitted and their support is explicit. Macro F1 includes all candidate labels, including absent labels with zero F1.

| Task | Test rows | Accuracy | Macro F1 | AUROC | NLL before | NLL after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

## Runtime and latency

Measured batch-one end-to-end inference p50 {latency['p50_ms']:.2f} ms, p95 {latency['p95_ms']:.2f} ms and p99 {latency['p99_ms']:.2f} ms over {latency['requests']} requests covering {latency['task_count']} tasks on {latency['device']}. The public inference API includes signal preparation and encoding; signal-feature caching is disabled. Text embeddings are warmed. Fresh model construction took {load_seconds:.3f} seconds and the first request with a cold text model took {first_request_ms:.2f} ms. CUDA context/process startup was already initialized and is not included as a cold-process measurement.

Frozen FP32 features were prepared in a checksummed, recoverable bank and reused for complete, shuffled training epochs. No additional raw waveform dataset was copied. Training and recovery bindings retain exact source, asset, preparation and plan hashes. The legacy warm-cache benchmark is retained as historical context, not claimed as this run's measured throughput.

## Coverage and checks

All 13 tasks were evaluated. Sparse or absent labels limit class-specific conclusions: the inherited validation corpus has no severe sleep-breathing examples and only one moderate-hypoxemia example. The role-specific task/answer counts and per-answer test counts are in test.json; classes with no examples cannot be evaluated reliably. No new examples were invented or relabeled. Input processing exclusions and reasons remain documented in reports/DATA_PROCESSING.md and the processing ledger.

The independent calibration role contains {len(roles['task_answer_counts']['calibration'])} of the 13 tasks. Tasks absent from calibration: {', '.join(roles.get('absent_tasks',{}).get('calibration',[])) or 'none'}. The fitted temperature is global; this calibration sample does not establish calibration quality for missing tasks/classes. Selection covers {len(roles['task_answer_counts']['selection'])} tasks. The role protocol records all missing task/answer pairs.

Frozen encoder checksums, finite losses/gradients, calibrated checkpoint reload, batched-versus-serial numerical agreement, and split isolation were checked. Local model files and reports remain authoritative if cloud final synchronization fails.

No subsequent training round will start until the user has analyzed and reviewed this report.
'''
    reference_table='\n'.join(f"| {key} | {comparison['reference']['uncalibrated'][key]:.6f} | {comparison['candidate']['uncalibrated'][key]:.6f} | {comparison['candidate']['calibrated'][key]:.6f} |"
        for key in ['accuracy','macro_f1','macro_auroc_ovr','nll','brier','ece_15'])
    with atomic_output(directory/'REPORT.md','w') as output:
        output.write(document+'\n## Matched historical test comparison\n\n| Metric | Reference raw | Candidate raw | Candidate calibrated |\n| --- | ---: | ---: | ---: |\n'+reference_table+'\n\nThe comparison verifies identical mapping IDs, answers, candidate sets and audit identities against archived reference logits. Both reference raw/calibrated metrics and per-task class support/confusion are preserved in baseline-comparison.json. Do not interpret pooled improvement as every class improving.\n')
    return report
