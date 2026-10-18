import json

import numpy as np
import torch
from torch import nn

import paya.features as features
from paya.provenance import sha256


def test_fresh_cache_preserves_physical_channel_layout_and_detects_changes(
    tmp_path, monkeypatch
):
    root = tmp_path / "data"
    root.mkdir()
    np.save(root / "x.npy", np.arange(40, dtype=np.float32).reshape(2, 1, 20))
    row = {
        "source_id": "window",
        "signal_file": "x.npy",
        "index": 0,
        "valid_channels": [0],
        "sampling_rate_hz": 65,
        "channels": [{"sensor_type": "EEG"}],
    }
    for split in ("train", "selection", "calibration"):
        (root / f"{split}.jsonl").write_text(
            json.dumps(row) + "\n" if split == "train" else ""
        )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "signals": {"x.npy": sha256(root / "x.npy")},
                "splits": {
                    s: sha256(root / f"{s}.jsonl")
                    for s in ("train", "selection", "calibration")
                },
            }
        )
    )

    class Core(nn.Module):
        def forward(self, x):
            return x.mean(-1)[..., None, None].expand(-1, -1, 105, 768)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.core = Core()

    monkeypatch.setattr(features, "build_model", lambda *a: Model())
    monkeypatch.setattr(features, "evidence_identity", lambda *a: {"test": "pinned"})
    path = tmp_path / "evidence.json"
    features.extract(root, "fake", path, "cpu")
    cache = features.load_cache(path, root, "fake")
    assert cache["window"].shape == (1, 105, 768)
    torch.testing.assert_close(cache["window"].float(), torch.full((1, 105, 768), 9.5))
    descriptor = json.loads(path.read_text())
    descriptor["data_manifest"] = "wrong"
    path.write_text(json.dumps(descriptor))
    import pytest

    with pytest.raises(ValueError, match="dataset"):
        features.load_cache(path, root, "fake")


def test_interrupted_extraction_resumes_without_duplicate_windows(
    tmp_path, monkeypatch
):
    root = tmp_path / "data"
    root.mkdir()
    np.save(root / "x.npy", np.arange(60, dtype=np.float32).reshape(3, 1, 20))
    rows = [
        {
            "source_id": f"w{i}",
            "signal_file": "x.npy",
            "index": i,
            "valid_channels": [0],
            "sampling_rate_hz": 65,
            "channels": [{"sensor_type": "EEG"}],
        }
        for i in range(3)
    ]
    for split in ("train", "selection", "calibration"):
        (root / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows) if split == "train" else ""
        )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "signals": {"x.npy": sha256(root / "x.npy")},
                "splits": {
                    s: sha256(root / f"{s}.jsonl")
                    for s in ("train", "selection", "calibration")
                },
            }
        )
    )

    class Core(nn.Module):
        def __init__(self, interrupt=False):
            super().__init__()
            self.calls, self.interrupt = 0, interrupt

        def forward(self, values):
            self.calls += 1
            if self.interrupt and self.calls == 2:
                raise RuntimeError("interrupted")
            return values.mean(-1)[..., None, None].expand(-1, -1, 105, 768)

    class Model(nn.Module):
        def __init__(self, core):
            super().__init__()
            self.core = core

    monkeypatch.setattr(features, "evidence_identity", lambda *args: {"test": "pinned"})
    monkeypatch.setattr(features, "build_model", lambda *args: Model(Core()))
    complete = tmp_path / "complete.json"
    features.extract(root, "fake", complete, "cpu", batch_size=1)
    monkeypatch.setattr(
        features, "build_model", lambda *args: Model(Core(interrupt=True))
    )
    resumed = tmp_path / "resumed.json"
    import pytest

    with pytest.raises(RuntimeError, match="interrupted"):
        features.extract(root, "fake", resumed, "cpu", batch_size=1)
    # Represent a successfully flushed marker for the first completed window.
    marker = resumed.with_suffix(".progress.json")
    progress = json.loads(marker.read_text())
    progress.update(windows={"w0": [0, 1]}, offset=1)
    marker.write_text(json.dumps(progress))
    core = Core()
    monkeypatch.setattr(features, "build_model", lambda *args: Model(core))
    features.extract(root, "fake", resumed, "cpu", batch_size=1)
    assert core.calls == 2
    assert sha256(complete.with_suffix(".npy")) == sha256(resumed.with_suffix(".npy"))
    assert (
        json.loads(resumed.read_text())["windows"]
        == json.loads(complete.read_text())["windows"]
    )
