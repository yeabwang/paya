import json

import numpy as np
import pytest
import torch
import yaml
from torch import nn

import paya.train as training
from paya.model import SignalDecision


class Core(nn.Module):
    embed_dim = 8

    def __init__(self):
        super().__init__()
        self.base = nn.Linear(1, 8)

    def forward(self, x):
        return self.base(x.mean(-1)[..., None, None].expand(-1, -1, 27, 1))


class Text:
    def encode(self, strings):
        return torch.stack(
            [torch.arange(768).float() / 768 + len(s) / 100 for s in strings]
        )


def test_training_checkpoint_and_exact_epoch_resume(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    data = tmp_path / "data"
    data.mkdir()
    np.save(data / "signal.npy", np.arange(4 * 20, dtype=np.float32).reshape(4, 1, 20))
    for split, group in [("train", "person1"), ("selection", "person2")]:
        rows = []
        for i in range(4):
            rows.append(
                {
                    "source_id": f"{split}/{i}",
                    "dataset_id": "source",
                    "group_id": group,
                    "signal_file": "signal.npy",
                    "index": i,
                    "valid_channels": [0],
                    "sampling_rate_hz": 65,
                    "channels": [{"sensor_type": "EEG"}],
                    "questions": [
                        {
                            "family": "native",
                            "question": "Which?",
                            "choices": ["one", "two"],
                            "label": i % 2,
                            "templates": ["Which?"],
                            "heldout_question": "Select.",
                        }
                    ],
                }
            )
        (data / f"{split}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
    (data / "manifest.json").write_text("{}")
    (data / "texts.json").write_text('["Which?","one","two"]')
    bank = tmp_path / "text.pt"
    bank.write_bytes(b"verified fake cache")
    config = {
        "seed": 42,
        "threads": 2,
        "assets": "fake",
        "head": {},
        "data": {
            "root": str(data),
            "text_cache": str(bank),
            "feature_cache": "fake",
            "batch_size": 2,
        },
        "optim": {"epochs": 2, "patience": 5, "lr": 0.001, "weight_decay": 0.01},
        "wandb": {"mode": "disabled"},
    }
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(config))

    class Evidence:
        def __getitem__(self, key):
            return torch.arange(27 * 8).float().reshape(1, 27, 8) / 100

    monkeypatch.setattr(training, "load_cache", lambda *a: Evidence())
    monkeypatch.setattr(training, "TextEncoder", lambda *a: Text())
    monkeypatch.setattr(training, "identity", lambda *a: {"test": "pinned"})
    monkeypatch.setattr(
        training,
        "build_model",
        lambda *a: SignalDecision(Core(), width=16, heads=2, layers=1),
    )
    training.train(cfg, tmp_path / "complete", "cpu")
    save = training.save_checkpoint

    def interrupted(path, *args):
        save(path, *args)
        if path.name == "latest.pt":
            raise RuntimeError("interrupt")

    monkeypatch.setattr(training, "save_checkpoint", interrupted)
    with pytest.raises(RuntimeError, match="interrupt"):
        training.train(cfg, tmp_path / "resumed", "cpu")
    monkeypatch.setattr(training, "save_checkpoint", save)
    training.train(cfg, tmp_path / "resumed", "cpu", tmp_path / "resumed/latest.pt")
    expected = torch.load(tmp_path / "complete/latest.pt", weights_only=True)
    actual = torch.load(tmp_path / "resumed/latest.pt", weights_only=True)
    assert actual["step"] == expected["step"] == 4
    for key in expected["head"]:
        torch.testing.assert_close(
            actual["head"][key], expected["head"][key], rtol=0, atol=0
        )
    assert expected["run_identity"] == actual["run_identity"]
