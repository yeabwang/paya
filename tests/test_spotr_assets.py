import json

import pytest
import torch
from safetensors.torch import load_file

from paya.vendor import signal as assets
from paya.provenance import sha256, verify_source


def test_checkpoint_hash_is_checked_before_tensor_loading(tmp_path, monkeypatch):
    checkpoint = tmp_path/"untrusted.pt"
    checkpoint.write_bytes(b"not the pinned checkpoint")
    def unexpected(*args, **kwargs):
        pytest.fail("Unverified checkpoint must not be loaded")
    monkeypatch.setattr(torch, "load", unexpected)
    with pytest.raises(ValueError, match="SHA-256"):
        assets.download_encoder(tmp_path/"assets", checkpoint)
    assert not (tmp_path/"assets"/"source.json").exists()


def test_conversion_is_strict_and_records_checked_assets(tmp_path, monkeypatch):
    core = torch.nn.Linear(2, 2)
    checkpoint = tmp_path/"upstream.pt"
    torch.save(core.state_dict(), checkpoint)
    card = tmp_path/"README.md"
    card.write_text("SPOTR test fixture", encoding="utf-8")
    monkeypatch.setattr(assets, "CHECKPOINT_SHA256", sha256(checkpoint))
    monkeypatch.setattr(assets, "SPOTREncoder", lambda **config: torch.nn.Linear(2, 2))
    def download(filename, **options):
        assert filename == "README.md"
        assert options["repo_id"] == assets.MODEL_ID and options["revision"] == assets.REVISION
        assert options["token"] is False
        return str(card)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    directory = tmp_path/"assets"
    source = assets.download_encoder(directory, checkpoint)
    verified = verify_source(directory, assets.MODEL_ID, assets.REVISION)
    assert verified["files"] == source["files"]
    assert json.loads((directory/"config.json").read_text()) == assets.CONFIG
    assert (directory/"LICENSE").is_file()
    for name, value in core.state_dict().items():
        torch.testing.assert_close(load_file(str(directory/"model.safetensors"))[name], value)
    assert assets.download_encoder(directory)["files"] == source["files"]
    torch.save({"unexpected": torch.zeros(2)}, checkpoint)
    monkeypatch.setattr(assets, "CHECKPOINT_SHA256", sha256(checkpoint))
    with pytest.raises(RuntimeError, match="state_dict"):
        assets.download_encoder(tmp_path/"bad-assets", checkpoint)
    assert not (tmp_path/"bad-assets"/"source.json").exists()
