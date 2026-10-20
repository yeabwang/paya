import numpy as np
import pytest
import torch

from paya.development import InteractionHead, class_weights, smoothed_choice_loss


def test_smoothing_never_uses_padded_candidates():
    z = torch.tensor([[1., 2., -torch.inf], [2., 1., 0.]], requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    y = torch.tensor([0, 2])
    loss = smoothed_choice_loss(z, y, mask, .1)
    expected = torch.nn.functional.cross_entropy(z[:1, :2], y[:1], label_smoothing=.1)
    assert float(loss[0].detach()) == pytest.approx(float(expected.detach()))
    loss.sum().backward()
    assert torch.isfinite(z.grad).all() and z.grad[0, 2] == 0


def test_class_weights_are_training_only_and_preserve_task_mass():
    rows = [{'task': 'a', 'criteria': ['low','high'], 'target': i} for i in [0]*9+[1]]
    weights = class_weights(rows)
    assert np.mean(weights) == pytest.approx(1.)
    assert weights[-1] > weights[0] and np.isfinite(weights).all()


def test_interaction_head_candidate_permutation_padding_and_signal_gradient():
    torch.manual_seed(7)
    head = InteractionHead(width=16, dropout=0).eval()
    s = torch.randn(2, 1, 512, requires_grad=True)
    q, c = torch.randn(2, 768), torch.randn(2, 3, 768)
    sm = torch.ones(2, 1, dtype=torch.bool)
    cm = torch.tensor([[True, True, False], [True, True, True]])
    z = head(s, sm, q, c, cm)
    order = [2, 0, 1]
    torch.testing.assert_close(head(s, sm, q, c[:, order], cm[:, order]), z[:, order])
    z[cm].sum().backward()
    assert s.grad.abs().sum() > 0 and z[0, 2].isneginf()


def test_bank_indices_align_rows_and_features_and_detect_corruption(tmp_path):
    import json
    from paya.development import Bank
    from paya.provenance import sha256
    arrays = {'signals':np.arange(4*512,dtype=np.float32).reshape(4,512),
        'questions':np.ones((2,768),dtype=np.float32), 'criteria':np.ones((3,768),dtype=np.float32),
        'question_indices':np.array([0,1,0,1]), 'criteria_indices':np.array([[0,1,-1]]*4),
        'targets':np.array([0,1,0,1])}
    for name,values in arrays.items():np.save(tmp_path/(name+'.npy'),values)
    (tmp_path/'rows.jsonl').write_text(''.join(json.dumps({'index':i})+'\n' for i in range(4)))
    manifest={'rows':4,'binding':'fixture','files':{p.name:sha256(p) for p in tmp_path.iterdir()}}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    bank=Bank(tmp_path,limit=3,device='cpu')
    for i,row in enumerate(bank.rows):
        assert row['index']==bank.indices[i]
        assert bank.signal[i,0,0]==row['index']*512
    assert bank.batch(torch.tensor([0]))[-1].tolist()==[[True,True,False]]
    (tmp_path/'targets.npy').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):
        Bank(tmp_path,device='cpu')


def test_public_interaction_checkpoint_roundtrip(tmp_path):
    from conftest import EncoderDouble, TextDouble, state
    from paya.model import PayaModel
    from paya.checkpoint import save_checkpoint, load_checkpoint
    settings={'kind':'interaction','width':16,'dropout':0.}
    first=PayaModel(EncoderDouble(512),TextDouble(),head=settings).eval()
    answer=first.predict(state(condition='processed'),'Which category?',['a','b'])
    save_checkpoint(tmp_path/'model.pt',first)
    second=PayaModel(EncoderDouble(512),TextDouble(),head=settings).eval()
    # The fixture encoder assets are injected; share the same frozen weights.
    second.encoder.load_state_dict(first.encoder.state_dict())
    load_checkpoint(tmp_path/'model.pt',second)
    assert second.predict(state(condition='processed'),'Which category?',['a','b'])==answer


def test_sqlite_features_reuse_across_heads_but_reject_changed_assets(model,tmp_path):
    import json
    from conftest import state
    from paya.features import SQLiteFeatureCache
    from paya.head import DecisionHead
    cache=SQLiteFeatureCache(tmp_path/'cache.sqlite3')
    prepared=model.preprocessor.prepare(state(condition='processed'))
    expected=cache.get_or_encode(model,prepared);calls=model.encoder.calls
    binding=json.loads(cache.db.execute('SELECT binding FROM metadata').fetchone()[0])
    binding['model']['head']=model.head.config
    with cache.db:cache.db.execute('UPDATE metadata SET binding=?',(json.dumps(binding),))
    model.head=DecisionHead(width=32,heads=4,layers=1)
    torch.testing.assert_close(cache.get_or_encode(model,prepared),expected)
    assert model.encoder.calls==calls
    model.asset_identity={'different':True}
    with pytest.raises(ValueError,match='contract mismatch'):
        cache.get_or_encode(model,prepared)


def test_partial_bank_commits_recover_and_reject_corrupted_chunks(tmp_path):
    from paya.development_bank import resume_signals,commit_signals
    values,state=resume_signals(tmp_path,5,'binding')
    values[:3]=7
    commit_signals(tmp_path,values,state,3,2)
    recovered,state=resume_signals(tmp_path,5,'binding')
    assert state['chunks'][-1]['end']==3 and (recovered[:3]==7).all()
    with pytest.raises(ValueError,match='bindings'):
        resume_signals(tmp_path,5,'changed')
    recovered[1,0]=9;recovered.flush()
    with pytest.raises(ValueError,match='chunk changed'):
        resume_signals(tmp_path,5,'binding')


def test_epoch_batches_cover_every_row_and_resume_without_changing_global_rng():
    from paya.development import EpochBatches
    initial=torch.get_rng_state().clone()
    sampler=EpochBatches(103,16,42)
    first=[sampler.at(i).clone() for i in range(7)]
    assert sorted(torch.cat(first).tolist())==list(range(103))
    assert len(first[-1])==7
    resumed=EpochBatches(103,16,42)
    assert torch.equal(resumed.at(5),first[5])
    assert not torch.equal(sampler.at(7),first[0])
    assert torch.equal(initial,torch.get_rng_state())


def test_arrow_views_preserve_exact_preparation_identity_and_heterogeneous_fallback():
    from conftest import state
    from paya.development_bank import waveform_rows
    from paya.preprocessing import SignalPreprocessor
    import pyarrow as pa
    rows=[{'mapping_id':str(i),'state':state(rate=125,duration=10,condition='processed')} for i in range(3)]
    table=pa.Table.from_pylist(rows).take([2,0])
    fast=waveform_rows(table);slow=table.to_pylist();processor=SignalPreprocessor()
    for first,second in zip(fast,slow):
        assert first['mapping_id']==second['mapping_id']
        assert isinstance(first['state']['signals'][0]['samples'],np.ndarray)
        assert processor.prepare(first['state']).identity==processor.prepare(second['state']).identity
    rows[1]['state']['signals'].append(rows[1]['state']['signals'][0])
    heterogeneous=pa.Table.from_pylist(rows)
    assert waveform_rows(heterogeneous)==heterogeneous.to_pylist()
