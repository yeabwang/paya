"""Batch acceleration preserves native signals, candidate mapping and frozen features."""
import copy
import pytest
import torch
from conftest import state
from paya.features import SQLiteFeatureCache, open_feature_cache
from paya.vendor.spotr import encode_batch, WIDTH


def test_sqlite_cache_batch_dedup_reload_and_corruption(tmp_path,model,monkeypatch):
    first=model.preprocessor.prepare(state(condition='processed'))
    other=copy.deepcopy(state(condition='processed'))
    other['signals'][0]['samples'][0]+=0.4
    second=model.preprocessor.prepare(other)
    expected=torch.cat([model.encode(first),model.encode(second),model.encode(first)])
    cache=SQLiteFeatureCache(tmp_path/'features.sqlite3')
    actual=cache.get_many_or_encode(model,[first,second,first])
    torch.testing.assert_close(actual,expected)
    assert cache.db.execute('SELECT COUNT(*) FROM features').fetchone()[0]==2
    monkeypatch.setattr(model,'encode_batch',lambda _: pytest.fail('Cache missed verified embedding'))
    restored=SQLiteFeatureCache(tmp_path/'features.sqlite3')
    torch.testing.assert_close(restored.get_many_or_encode(model,[second,first]),expected[[1,0]])
    with restored.db: restored.db.execute("UPDATE features SET digest='corrupt'")
    with pytest.raises(ValueError,match='content changed'):
        restored.get_or_encode(model,first)


def test_batch_cache_preserves_model_logits_and_identity(tmp_path,model):
    states=[state(condition='processed'),state(rate=125,duration=2,condition='processed')]
    batch=model.make_batch(states,['Question?']*2,[['slow','normal','fast']]*2)
    with torch.no_grad():
        expected=model(batch)['logits']
        config={'feature_cache':str(tmp_path/'features.sqlite3'),'feature_cache_format':'sqlite'}
        cache=open_feature_cache(config)
        torch.testing.assert_close(model(batch,cache)['logits'],expected)
        assert all(p.grad is None for p in model.encoder.parameters())
        model.asset_identity={'changed':True}
        with pytest.raises(ValueError,match='contract mismatch'):
            model(batch,cache)


def test_vendor_batch_shape_and_per_recording_independence():
    class Core(torch.nn.Module):
        def forward(self,x):
            b,c,l=x.shape
            return x.mean((1,2))[:,None,None].expand(b,1+c+2*((l+199)//200),WIDTH)
    samples=torch.stack([torch.ones(1,1250),torch.ones(1,1250)*2])
    actual=encode_batch(Core(),samples)
    assert actual.shape==(2,512)
    assert torch.equal(actual[:,0],torch.tensor([1.,2.]))
    with pytest.raises(ValueError): encode_batch(Core(),samples[0])
