import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from paya.inference import load_bundle, load_checkpoint
from paya.provenance import PUBLIC_API, identity
from paya.text import IDENTITY, TextEncoder


def test_checkpoint_requires_current_identity(tmp_path):
    (tmp_path / "source.json").write_text("{}")
    current = identity(tmp_path)
    assert "model_code" not in current and "text_code" not in current
    for key in current:
        checkpoint = tmp_path / "invalid.pt"
        torch.save({"identity": current | {key: "changed"}}, checkpoint)
        with pytest.raises(ValueError, match="provenance"):
            load_checkpoint(checkpoint, tmp_path)


def test_bundle_requires_current_schema_and_api(tmp_path):
    for manifest in (
        {"schema_version": 1, "public_api": PUBLIC_API},
        {"schema_version": 2, "public_api": {"channel_order": "canonical-physical-v1"}},
    ):
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError):
            load_bundle(tmp_path)


def test_text_uses_third_party_weights_and_cache_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HF_HOME", raising=False)
    folder = tmp_path / "third-party" / "biolord"
    folder.mkdir(parents=True)
    (folder / "source.json").write_text(
        json.dumps(
            {
                "model_id": IDENTITY["model_id"],
                "revision": IDENTITY["revision"],
                "files": {},
            }
        )
    )
    calls = {}

    class Encoder:
        max_seq_length = 100
        tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(text))

        def eval(self):
            return self

        def requires_grad_(self, enabled):
            return self

        def encode(self, strings, **kwargs):
            return torch.ones(len(strings), 768)

    def load(path, **kwargs):
        calls.update(path=path, **kwargs)
        return Encoder()

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=load)
    )
    assert TextEncoder().encode(["new text"]).shape == (1, 768)
    assert Path(calls["path"]) == Path("third-party/biolord")
    assert calls["cache_folder"] == "cache/huggingface"
    assert calls["revision"] is None
    assert calls["trust_remote_code"] is False
    assert not (tmp_path / "assets").exists()
