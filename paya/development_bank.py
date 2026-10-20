"""Build a bounded, reproducible frozen-feature development bank on the GPU."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from paya.atomic import atomic_output
from paya.data import ChoiceDataset, make_batch, assert_disjoint
from paya.features import open_feature_cache
from paya.model import PayaModel
from paya.provenance import fingerprint,sha256


def main():
    torch.set_num_threads(8)
    root=Path('cache/paya-development');root.mkdir(exist_ok=True)
    config=yaml.safe_load(Path('config/paya-fast.yaml').read_text())
    datasets={k:ChoiceDataset(config['data'][k]) for k in ['train','selection']}
    assert_disjoint(*datasets.values())
    model=PayaModel.from_assets(config['assets'],config['preparation'],config['head'],'cuda')
    cache=open_feature_cache(config)
    for role,dataset in datasets.items():
        destination=root/role
        if (destination/'manifest.json').exists():
            print(json.dumps({'role':role,'status':'already_complete'}),flush=True);continue
        destination.mkdir(exist_ok=True)
        started=perf_counter();metadata=[];locations={};tasks={};questions={};candidates={};census=Counter()
        # Only metadata is read during sampling; the test/calibration files are never opened.
        for file in dataset.files:
            origins={row['mapping_id']:row['qa_task'] for row in pq.ParquetFile(
                Path('data/pulselm-paya/provenance')/file.parent.name/file.name).read(columns=['mapping_id','qa_task']).to_pylist()}
            pf=pq.ParquetFile(file)
            for group in range(pf.metadata.num_row_groups):
                rows=pf.read_row_group(group,columns=['mapping_id','source_id','group_id','question','criteria','target']).to_pylist()
                selected=[]
                for position,row in enumerate(rows):
                    # Uniform deterministic QA sample, independent of labels and signal values.
                    if role=='train' and int(hashlib.sha256(('paya-dev-v1:'+row['mapping_id']).encode()).hexdigest()[:8],16)%16:
                        continue
                    row['task']=origins[row['mapping_id']]
                    row['task_index']=tasks.setdefault(row['task'],len(tasks))
                    row['question_index']=questions.setdefault(row['question'],len(questions))
                    row['criteria_indices']=[candidates.setdefault(c,len(candidates)) for c in row['criteria']]
                    census[row['task']+'::'+row['criteria'][row['target']]]+=1
                    selected.append((len(metadata),position));metadata.append(row)
                if selected:locations[str(file),group]=selected
        count=len(metadata);width=max(len(r['criteria']) for r in metadata)
        signals=np.lib.format.open_memmap(destination/'signals.npy',mode='w+',dtype=np.float32,shape=(count,512))
        q=np.array([r['question_index'] for r in metadata],dtype=np.int32)
        ci=np.full((count,width),-1,dtype=np.int32)
        for i,r in enumerate(metadata):ci[i,:len(r['criteria'])]=r['criteria_indices']
        np.save(destination/'question_indices.npy',q);np.save(destination/'criteria_indices.npy',ci)
        np.save(destination/'targets.npy',np.array([r['target'] for r in metadata],dtype=np.int64))
        np.save(destination/'task_indices.npy',np.array([r['task_index'] for r in metadata],dtype=np.int32))
        np.save(destination/'questions.npy',model.text_encoder.encode(list(questions)).cpu().numpy())
        np.save(destination/'criteria.npy',model.text_encoder.encode(list(candidates)).cpu().numpy())
        complete=0;current_file=None;reader=None
        for (file,group),selected in locations.items():
            if file!=current_file:
                reader=pq.ParquetFile(file);current_file=file
            rows=reader.read_row_group(group).take([position for _,position in selected]).to_pylist()
            assert all(row['mapping_id']==metadata[index]['mapping_id'] for row,(index,_) in zip(rows,selected))
            batch=make_batch(model,rows)
            features=cache.get_many_or_encode(model,batch.states).cpu().numpy()
            for (index,_),value in zip(selected,features):signals[index]=value
            complete+=len(selected)
            if complete//10000!=(complete-len(selected))//10000:
                print(json.dumps({'role':role,'encoded':complete,'rows':count,'seconds':perf_counter()-started}),flush=True)
        signals.flush();assert complete==count and np.isfinite(signals).all()
        with atomic_output(destination/'rows.jsonl','w') as out:
            for row in metadata:out.write(json.dumps(row)+'\n')
        files={p.name:sha256(p) for p in destination.iterdir() if p.is_file() and p.suffix in ['.npy','.jsonl']}
        manifest={'role':role,'rows':count,'dataset_sha256':dataset.sha256,'model_identity':model.identity(),
            'sampling':'SHA256(paya-dev-v1:mapping_id) first32 modulo16==0' if role=='train' else 'all selection QA rows',
            'tasks':tasks,'questions':questions,'criteria':candidates,'class_counts':dict(census),
            'seconds':perf_counter()-started,'files':files}
        manifest['binding']=fingerprint(manifest)
        with atomic_output(destination/'manifest.json','w') as out:json.dump(manifest,out,indent=2)
        print(json.dumps({'role':role,'rows':count,'status':'complete','seconds':manifest['seconds']}),flush=True)

if __name__=='__main__':main()
