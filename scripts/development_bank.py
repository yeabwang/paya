"""Build a bounded, reproducible frozen-feature development bank on the GPU."""
from collections import Counter
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from time import perf_counter
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import yaml
from paya.atomic import atomic_output
from paya.data import ChoiceDataset, make_batch, assert_disjoint
from paya.features import open_feature_cache
from paya.model import PayaModel
from paya.provenance import fingerprint,sha256


def waveform_rows(table):
    """Keep uniform single-channel Arrow samples as NumPy views, not Python floats."""
    states=table['state'].combine_chunks()
    if states.null_count or states.type.names!=['signals']:
        return table.to_pylist()
    signals=states.field('signals')
    if signals.null_count or not (pc.list_value_length(signals).to_numpy()==1).all():
        return table.to_pylist()
    channels=signals.flatten()
    if set(channels.type.names)!={'samples','metadata'}:return table.to_pylist()
    samples=channels.field('samples')
    lengths=pc.list_value_length(samples).to_numpy()
    if samples.null_count or not len(lengths) or not (lengths==lengths[0]).all():
        return table.to_pylist()
    values=samples.flatten()
    if values.null_count:return table.to_pylist()
    matrix=values.to_numpy(zero_copy_only=False).reshape(len(table),int(lengths[0]))
    metadata=channels.field('metadata').to_pylist()
    rows=table.drop(['state']).to_pylist()
    for row,values,meta in zip(rows,matrix,metadata):
        row['state']={'signals':[{'samples':values,'metadata':meta}]}
    return rows


def resume_signals(destination,count,binding):
    progress=destination/'progress.json'
    state=json.loads(progress.read_text()) if progress.exists() else {'binding':binding,'chunks':[]}
    if state['binding']!=binding:
        raise ValueError('Partial feature bank bindings differ')
    values=np.lib.format.open_memmap(destination/'signals.npy',mode='r+' if progress.exists() else 'w+',
        dtype=np.float32,shape=(count,512))
    if values.shape!=(count,512) or values.dtype!=np.float32:
        raise ValueError('Partial feature bank shape differs')
    end=0
    for chunk in state['chunks']:
        if chunk['start']!=end or not end<chunk['end']<=count:
            raise ValueError('Partial feature bank journal is not contiguous')
        raw=values[end:chunk['end']]
        if not np.isfinite(raw).all() or hashlib.sha256(raw.tobytes()).hexdigest()!=chunk['sha256']:
            raise ValueError('Committed feature bank chunk changed')
        end=chunk['end']
    return values,state


def commit_signals(destination,values,state,end,location_end):
    begin=state['chunks'][-1]['end'] if state['chunks'] else 0
    if end==begin:return
    raw=values[begin:end]
    if not np.isfinite(raw).all():raise ValueError('Nonfinite feature bank chunk')
    values.flush()
    with (destination/'signals.npy').open('r+b') as source:os.fsync(source.fileno())
    state['chunks'].append({'start':begin,'end':end,'location_end':location_end,
        'sha256':hashlib.sha256(raw.tobytes()).hexdigest()})
    with atomic_output(destination/'progress.json','w') as output:json.dump(state,output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-train',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(8)
    root=Path('cache/paya-full-bank' if args.full_train else 'cache/paya-development');root.mkdir(exist_ok=True)
    config=yaml.safe_load(Path('config/paya-fast.yaml').read_text())
    provenance_roots=list(Path('data').glob('*/provenance'))
    if len(provenance_roots)!=1:
        raise ValueError('Expected exactly one local provenance root')
    provenance=provenance_roots[0]
    datasets={k:ChoiceDataset(config['data'][k]) for k in ['train','selection']}
    assert_disjoint(*datasets.values())
    model=PayaModel.from_assets(config['assets'],config['preparation'],config['head'],'cuda')
    cache=open_feature_cache(config)
    for role,dataset in datasets.items():
        if args.full_train and role!='train':continue
        destination=root/role
        if (destination/'manifest.json').exists():
            print(json.dumps({'role':role,'status':'already_complete'}),flush=True);continue
        destination.mkdir(exist_ok=True)
        started=perf_counter();metadata=[];locations={};tasks={};questions={};candidates={};census=Counter()
        # Only metadata is read during sampling; the test/calibration files are never opened.
        for file in dataset.files:
            origins={row['mapping_id']:row['qa_task'] for row in pq.ParquetFile(
                provenance/file.parent.name/file.name).read(columns=['mapping_id','qa_task']).to_pylist()}
            pf=pq.ParquetFile(file)
            for group in range(pf.metadata.num_row_groups):
                rows=pf.read_row_group(group,columns=['mapping_id','source_id','group_id','question','criteria','target']).to_pylist()
                selected=[]
                for position,row in enumerate(rows):
                    # Uniform deterministic QA sample, independent of labels and signal values.
                    if role=='train' and not args.full_train and int(hashlib.sha256(('paya-dev-v1:'+row['mapping_id']).encode()).hexdigest()[:8],16)%16:
                        continue
                    row['task']=origins[row['mapping_id']]
                    row['task_index']=tasks.setdefault(row['task'],len(tasks))
                    row['question_index']=questions.setdefault(row['question'],len(questions))
                    row['criteria_indices']=[candidates.setdefault(c,len(candidates)) for c in row['criteria']]
                    census[row['task']+'::'+row['criteria'][row['target']]]+=1
                    selected.append((len(metadata),position));metadata.append(row)
                if selected:locations[str(file),group]=selected
        count=len(metadata);width=max(len(r['criteria']) for r in metadata)
        needed=count*(512*4+width*4+2048)
        if not (destination/'signals.npy').exists() and shutil.disk_usage(root).free<needed:
            raise ValueError('Insufficient data-disk space for the bounded feature bank')
        binding=fingerprint({'dataset':dataset.sha256,'model':model.identity(),'rows':count,
            'full_train':args.full_train,'builder_sha256':sha256(Path(__file__))})
        signals,state=resume_signals(destination,count,binding)
        q=np.array([r['question_index'] for r in metadata],dtype=np.int32)
        ci=np.full((count,width),-1,dtype=np.int32)
        for i,r in enumerate(metadata):ci[i,:len(r['criteria'])]=r['criteria_indices']
        np.save(destination/'question_indices.npy',q);np.save(destination/'criteria_indices.npy',ci)
        np.save(destination/'targets.npy',np.array([r['target'] for r in metadata],dtype=np.int64))
        np.save(destination/'task_indices.npy',np.array([r['task_index'] for r in metadata],dtype=np.int32))
        np.save(destination/'questions.npy',model.text_encoder.encode(list(questions)).cpu().numpy())
        np.save(destination/'criteria.npy',model.text_encoder.encode(list(candidates)).cpu().numpy())
        complete=state['chunks'][-1]['end'] if state['chunks'] else 0
        cursor=state['chunks'][-1]['location_end'] if state['chunks'] else 0
        assert sum(len(v) for i,v in enumerate(locations.values()) if i<cursor)==complete
        print(json.dumps({'role':role,'resumed_rows':complete,'rows':count}),flush=True)
        current_file=None;reader=None
        for location,((file,group),selected) in enumerate(locations.items()):
            if location<cursor:continue
            if file!=current_file:
                reader=pq.ParquetFile(file);current_file=file
            rows=waveform_rows(reader.read_row_group(group).take([position for _,position in selected]))
            assert all(row['mapping_id']==metadata[index]['mapping_id'] for row,(index,_) in zip(rows,selected))
            prepared=[model.preprocessor.prepare(row['state']) for row in rows]
            features=cache.get_many_or_encode(model,prepared).cpu().numpy()
            for (index,_),value in zip(selected,features):signals[index]=value
            complete+=len(selected)
            if complete//10000!=(complete-len(selected))//10000:
                commit_signals(destination,signals,state,complete,location+1)
                print(json.dumps({'role':role,'encoded':complete,'rows':count,'seconds':perf_counter()-started}),flush=True)
        commit_signals(destination,signals,state,complete,len(locations))
        signals.flush();assert complete==count and all(np.isfinite(signals[i:i+10000]).all() for i in range(0,count,10000))
        with atomic_output(destination/'rows.jsonl','w') as out:
            for row in metadata:out.write(json.dumps(row)+'\n')
        files={p.name:sha256(p) for p in destination.iterdir() if p.is_file() and p.suffix in ['.npy','.jsonl']}
        manifest={'role':role,'rows':count,'dataset_sha256':dataset.sha256,'model_identity':model.identity(),
            'sampling':'all native training QA rows' if args.full_train else
                ('SHA256(paya-dev-v1:mapping_id) first32 modulo16==0' if role=='train' else 'all selection QA rows'),
            'tasks':tasks,'questions':questions,'criteria':candidates,'class_counts':dict(census),
            'seconds':perf_counter()-started,'files':files}
        manifest['binding']=fingerprint(manifest)
        with atomic_output(destination/'manifest.json','w') as out:json.dump(manifest,out,indent=2)
        print(json.dumps({'role':role,'rows':count,'status':'complete','seconds':manifest['seconds']}),flush=True)

if __name__=='__main__':main()
