"""Promotion and fine-tuning contracts for the supported attention head."""

import copy

import pytest
import torch
import yaml

from conftest import new_model, state
from paya.checkpoint import learned_digest, load_checkpoint, save_checkpoint
from paya.head import AttentionHead
from paya.reporting import task_report
from test_training_contracts import write_dataset


def test_attention_order_parameter_count_and_padded_candidates():
    head = AttentionHead(dropout=0).eval()
    assert sum(parameter.numel() for parameter in head.parameters()) == 480257
    order = []
    head.self_attention.register_forward_hook(lambda *_: order.append("self"))
    head.cross_attention.register_forward_hook(lambda *_: order.append("cross"))
    signal = torch.randn(2, 4, 512)
    valid = torch.tensor([[True, True, True, False], [True, True, True, True]])
    question, candidates = torch.randn(2, 768), torch.randn(2, 3, 768)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    expected = head(signal, valid, question, candidates, mask)
    assert order == ["self", "cross"] and head.roles.shape == (2, 128)
    candidates[~mask] = 1000
    torch.testing.assert_close(
        head(signal, valid, question, candidates, mask), expected
    )


def test_checkpoint_rejects_modified_head_and_retired_configuration(tmp_path):
    model = new_model()
    path = tmp_path / "head.pt"
    save_checkpoint(path, model)
    payload = torch.load(path, weights_only=True)
    payload["learned"]["head"]["roles"].add_(1)
    torch.save(payload, path)
    with pytest.raises(ValueError, match="checksum"):
        load_checkpoint(path, model)
    from paya.model import PayaModel

    with pytest.raises(ValueError, match="head kind"):
        PayaModel(model.encoder, model.text_encoder, head={"kind": "interaction"})


def test_fine_tuning_starts_with_parent_weights_and_fresh_state(tmp_path, monkeypatch):
    import paya.train as module

    parent = new_model()
    with torch.no_grad():
        parent.head.roles.add_(0.3)
    parent.temperature.fill_(2)
    parent.calibration = {"learned_sha256": learned_digest(parent), "temperature": 2.0}
    parent_path = tmp_path / "parent.pt"
    save_checkpoint(parent_path, parent)
    paths = {}
    for role in ("train", "selection"):
        path = tmp_path / f"{role}.jsonl"
        write_dataset(path, role, f"{role}:w", role)
        paths[role] = str(path)
    config = {
        "assets": {},
        "data": paths,
        "batch_size": 1,
        "threads": 2,
        "optim": {"epochs": 2, "lr": 0.001, "eval_every_steps": 1},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(module.PayaModel, "from_assets", lambda *a, **k: new_model())
    initial = []
    original = module.save_checkpoint

    def capture(path, model, training=None):
        if training and training["global_step"] == 0:
            initial.append(
                (
                    learned_digest(model),
                    float(model.temperature),
                    model.calibration,
                    copy.deepcopy(training["optimizer"]),
                )
            )
        return original(path, model, training)

    monkeypatch.setattr(module, "save_checkpoint", capture)
    model, history = module.train(
        config_path, tmp_path / "fine-tune", initialize_from=parent_path
    )
    assert initial[0][:3] == (learned_digest(parent), 1.0, None)
    assert initial[0][3]["state"] == {}
    assert [row["step"] for row in history] == [1, 2]
    assert learned_digest(model) != learned_digest(parent)
    saved = torch.load(tmp_path / "fine-tune/last.pt", weights_only=True)
    assert saved["training"]["initialization"]["learned_sha256"] == learned_digest(
        parent
    )
    assert saved["temperature"] == 1 and saved["calibration"] is None
    with pytest.raises(ValueError, match="not both"):
        module.train(
            config_path,
            tmp_path / "bad",
            resume=parent_path,
            initialize_from=parent_path,
        )


def test_task_report_preserves_absent_support_and_missing_task():
    rows = [
        {"task": "rhythm", "criteria": ["a", "b"], "target": 0, "logits": [1.0, 0.0]},
        {"criteria": ["a", "b"], "target": 1, "logits": [0.0, 1.0]},
    ]
    report = task_report(rows)
    assert report["rows_without_task"] == 1
    assert report["tasks"]["rhythm"]["per_class"]["b"]["recall"] is None
    assert report["task_mean"]["macro_auroc_ovr"] is None


def test_processed_signal_metadata_does_not_enter_learned_head(model):
    request = state("ppg", 125, 10, condition="processed")
    expected = model.predict(request, "q", ["a", "b"])
    request["signals"][0]["metadata"].update(
        sampling_rate_hz=250, duration_s=5, placement="wrist", subject_id="audit"
    )
    assert model.predict(request, "q", ["a", "b"]) == expected
