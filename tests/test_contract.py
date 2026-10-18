import hashlib

import numpy as np
import pytest
import torch
from torch import nn

from paya.inference import answer
from paya.inputs import METADATA_DIM, prepare_signal
from paya.model import SignalDecision
from paya.text import IDENTITY, TextEncoder
from paya.train import choice_loss


class Core(nn.Module):
    embed_dim = 8

    def __init__(self):
        super().__init__()
        self.base = nn.Linear(1, 8)
        self.calls = 0

    def forward(self, signals):
        self.calls += 1
        x = signals.mean(-1)[..., None, None].expand(-1, -1, 27, 1)
        return self.base(x)


class Text:
    def __init__(self):
        self.calls = 0

    def encode(self, strings):
        self.calls += 1
        return torch.stack(
            [
                torch.rand(
                    768,
                    generator=torch.Generator().manual_seed(
                        int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)
                    ),
                )
                for s in strings
            ]
        )


def request():
    return {
        "signal": {
            "sampling_rate_hz": 178,
            "channels": [
                {"sensor_type": "EEG", "samples": list(np.linspace(0, 1, 20))},
                {"sensor_type": "ECG", "samples": list(np.linspace(1, 3, 20))},
            ],
        },
        "question": "Which condition is present?",
        "choices": ["First condition.", "Second condition.", "Third condition."],
    }


def test_public_input_choice_channel_and_audit_id_invariance():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    model = SignalDecision(Core(), width=16, heads=2, layers=1, dropout=0).eval()
    text = Text()
    r = request()
    first = answer(model, text, r)
    assert model.core.calls == text.calls == 1
    assert abs(sum(first["probabilities"]) - 1) < 1e-6
    r["dataset_id"] = "an entirely new dataset"
    r["signal"]["channels"][0]["dataset_id"] = "ignored"
    np.testing.assert_allclose(
        answer(model, text, r)["probabilities"], first["probabilities"], atol=1e-6
    )
    r["signal"]["channels"].reverse()
    np.testing.assert_allclose(
        answer(model, text, r)["probabilities"], first["probabilities"], atol=1e-6
    )
    order = [2, 0, 1]
    r["choices"] = [r["choices"][i] for i in order]
    np.testing.assert_allclose(
        answer(model, text, r)["probabilities"],
        np.array(first["probabilities"])[order],
        atol=1e-6,
    )
    assert all("slot" not in name for name, _ in model.named_parameters())


def test_physical_units_and_input_validation():
    a = {
        "sampling_rate_hz": 65,
        "channels": [{"sensor_type": "ECG", "unit": "mV", "samples": [1.0] * 20}],
    }
    b = {
        "sampling_rate_hz": 65,
        "channels": [{"sensor_type": "ECG", "unit": "uV", "samples": [1000.0] * 20}],
    }
    x, m = prepare_signal(a)
    y, n = prepare_signal(b)
    torch.testing.assert_close(x, y)
    torch.testing.assert_close(m, n)
    assert m.shape == (1, METADATA_DIM)
    with pytest.raises(ValueError):
        prepare_signal({**a, "sampling_rate_hz": float("nan")})
    b["channels"][0]["samples"][0] = float("inf")
    with pytest.raises(ValueError):
        prepare_signal(b)
    with pytest.raises(ValueError):
        prepare_signal({**a, "channels": [{"samples": [0.0] * 9}]})


def test_code_identity_survives_git_line_endings(tmp_path):
    from paya.provenance import source_sha256

    a, b = tmp_path / "windows.py", tmp_path / "linux.py"
    a.write_bytes(b"x=1\r\ny=2\r\n")
    b.write_bytes(b"x=1\ny=2\n")
    assert source_sha256(a) == source_sha256(b)


def test_text_cache_miss_batching_and_fresh_equivalence(tmp_path):
    class Encoder:
        max_seq_length = 10
        tokenizer = type("Tokenizer", (), {"encode": lambda self, s, **kw: list(s)})()

        def encode(self, strings, **kwargs):
            return Text().encode(strings)

    text = TextEncoder()
    text.model = Encoder()
    expected = text.encode(["new", "words", "new"])
    path = tmp_path / "text.pt"
    text.save(path)
    cached = TextEncoder(path)
    torch.testing.assert_close(expected, cached.encode(["new", "words", "new"]))
    with pytest.raises(ValueError, match="context"):
        text.encode(["far too long for this encoder"])
    torch.save({"identity": {**IDENTITY, "revision": "wrong"}, "embeddings": {}}, path)
    with pytest.raises(ValueError, match="pinned"):
        TextEncoder(path)


def test_shared_head_learns_with_frozen_encoder():
    torch.set_num_threads(2)
    model = SignalDecision(Core(), width=16, heads=2, layers=1)
    model.train()
    inputs = dict(
        question=torch.randn(2, 768),
        choices=torch.randn(2, 3, 768),
        ending=torch.randn(2, 768),
        choice_mask=torch.tensor([[True, True, False], [True, True, True]]),
        sensor_metadata=torch.randn(2, 2, METADATA_DIM),
        sensor_mask=torch.ones(2, 2, dtype=torch.bool),
        signals=torch.randn(2, 2, 20),
    )
    logits = model(**inputs)
    assert logits[0, 2] == -torch.inf
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert all(p.grad is None for p in model.core.parameters())
    assert model.head.sensor_proj[0].weight.grad.abs().sum() > 0
    assert model.head.signal_proj[1].weight.grad.abs().sum() > 0
    assert model.core.calls == 1


def test_smoothing_excludes_padded_alternatives():
    logits = torch.tensor([[1.0, 2.0, -torch.inf], [2.0, 3.0, 1.0]], requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    labels = torch.tensor([1, 0])
    torch.testing.assert_close(
        choice_loss(logits, labels, mask),
        torch.nn.functional.cross_entropy(logits, labels),
    )
    value = choice_loss(logits, labels, mask, 0.1)
    assert torch.isfinite(value)
    value.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad[0, 2] == 0


def test_exact_ties_do_not_select_by_candidate_position():
    model = SignalDecision(Core(), width=16, heads=2, layers=1, dropout=0)
    with torch.no_grad():
        for parameter in model.head.parameters():
            parameter.zero_()
    r = request()
    first = answer(model, Text(), r)
    r["choices"].reverse()
    assert answer(model, Text(), r)["choice"] == first["choice"]


def test_public_channel_permutations_have_identical_numerical_encoder_inputs():
    class ObservedCore(Core):
        def forward(self, x):
            self.observed = x.detach().clone()
            return super().forward(x)

    model = SignalDecision(
        ObservedCore(), width=16, heads=2, layers=1, dropout=0
    ).eval()
    r = request()
    answer(model, Text(), r)
    first = model.core.observed
    r["signal"]["channels"].reverse()
    r["dataset_id"] = "ignored"
    answer(model, Text(), r)
    torch.testing.assert_close(first, model.core.observed, atol=0, rtol=0)


def test_raw_assessment_matches_public_requests_and_reuses_exact_fp32_memory(tmp_path):
    from paya.assess import probabilities
    from paya.evaluate import score_window

    r = request()
    values = np.array([c["samples"] for c in r["signal"]["channels"]], dtype=np.float32)
    np.save(tmp_path / "signals.npy", values[None])
    q = {
        "question": r["question"],
        "choices": r["choices"],
        "label": 1,
        "family": "native",
        "heldout_question": "Another wording?",
    }
    row = {
        "source_id": "one",
        "group_id": "group",
        "dataset_id": "task",
        "signal_file": "signals.npy",
        "index": 0,
        "valid_channels": [0, 1],
        "sampling_rate_hz": 178,
        "channels": [
            {k: v for k, v in c.items() if k != "samples"}
            for c in r["signal"]["channels"]
        ],
        "questions": [q, q],
    }
    model = SignalDecision(Core(), width=16, heads=2, layers=1, dropout=0).eval()
    text = Text()
    expected = answer(model, text, r)
    before = model.core.calls
    records, memory, metadata = score_window(model, text, row, tmp_path, "cpu", {})
    assert model.core.calls == before + 1 and memory.dtype == torch.float32
    for record in records:
        np.testing.assert_allclose(
            probabilities(record["logits"]), expected["probabilities"], atol=1e-6
        )
    before = model.core.calls
    score_window(model, text, row, tmp_path, "cpu", {}, True, memory, metadata)
    assert model.core.calls == before
