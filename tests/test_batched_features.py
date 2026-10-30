"""Variable-length signal-token caching preserves the public numerical path."""

import copy

import pytest
import torch

from conftest import state
from paya.features import SQLiteFeatureCache, open_feature_cache


def test_sqlite_cache_dedup_reload_and_corruption(tmp_path, model, monkeypatch):
    first = model.preprocessor.prepare(state(condition="processed"))
    other = copy.deepcopy(state(condition="processed"))
    other["signals"][0]["samples"][0] += 0.4
    second = model.preprocessor.prepare(other)
    cache = SQLiteFeatureCache(tmp_path / "tokens.sqlite3")
    try:
        a, mask = cache.get_or_encode(model, first)
        cache.get_or_encode(model, second)
        assert cache.db.execute("SELECT COUNT(*) FROM tokens").fetchone()[0] == 2
        monkeypatch.setattr(
            model, "encode", lambda _: pytest.fail("Cache missed verified tokens")
        )
        b, other_mask = cache.get_or_encode(model, first)
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        assert torch.equal(mask, other_mask)
        with cache.db:
            cache.db.execute("UPDATE tokens SET digest='corrupt'")
        with pytest.raises(ValueError, match="changed"):
            cache.get_or_encode(model, first)
    finally:
        cache.close()


def test_cache_preserves_logits_and_separates_encoder_identity(tmp_path, model):
    states = [
        state(condition="processed"),
        state(rate=125, duration=2, condition="processed"),
    ]
    batch = model.make_batch(
        states, ["Question?"] * 2, [["slow", "normal", "fast"]] * 2
    )
    cache = open_feature_cache({"feature_cache": str(tmp_path / "tokens.sqlite3")})
    try:
        with torch.no_grad():
            expected = model(batch)["logits"]
            torch.testing.assert_close(
                model(batch, cache)["logits"], expected, atol=0, rtol=0
            )
            before = model.encoder.calls
            model.head.config["dropout"] = 0.2
            model(batch, cache)
            assert model.encoder.calls == before
            model.asset_identity = {"changed": True}
            model(batch, cache)
            assert model.encoder.calls == before + 2
    finally:
        cache.close()


def test_repeated_waveform_is_encoded_once_per_batch(model):
    request = state(condition="processed")
    batch = model.make_batch(
        [request, request], ["q", "q2"], [["a", "b"], ["a", "b", "c"]]
    )
    model(batch)
    assert model.encoder.calls == 1
