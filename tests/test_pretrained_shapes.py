"""Real frozen weights and synthetic waveforms verify shapes, not decision accuracy."""

import json
from pathlib import Path

import pytest
import torch

from paya.model import PayaModel
from conftest import state


@pytest.mark.integration
def test_real_spotr_and_biolord_shapes_and_gradients(monkeypatch):
    assets = {"spotr": "third-party/spotr", "text": "third-party/biolord"}
    if not all((Path(path)/"source.json").exists() for path in assets.values()):
        pytest.skip("Pinned encoder assets have not been prepared")
    model = PayaModel.from_assets(assets, head={"width": 256, "layers": 2, "heads": 4, "dropout": 0})
    examples = {}
    shapes = []
    hook = model.encoder.core.register_forward_hook(lambda m, x, y: shapes.append((list(x[0].shape), list(y.shape))))
    for kind, rate, duration, expected in [("ppg", 125, 10, [1, 1, 1250]),
                                          ("ppg", 65, 6, [1, 1, 390]),
                                          ("temperature", 1, 2, [1, 1, 2]),
                                          ("ppg", 200, 64, [1, 1, 12800])]:
        prepared = model.preprocessor.prepare(state(kind, rate, duration, condition="processed"))
        frames = model.preprocessor.format(prepared)
        assert list(torch.stack(frames.frames).shape) == expected
        encoded = model.encode(prepared)
        assert encoded.values.shape == (expected[0], 512) and encoded.intervals_s.shape == (expected[0], 2)
        assert torch.isfinite(encoded.values).all()
        assert shapes[-1] == (expected, [1, 2+2*((expected[-1]+199)//200), 512])
        examples[f"{kind}-{rate}Hz"] = {"native_samples": round(rate*duration),
            "spotr_input": expected, "spotr_tokens": shapes[-1][1], "spotr_global": list(encoded.values.shape),
            "intervals_s": encoded.intervals_s.tolist()}
    joint = state(condition="processed")
    joint["signals"].append(state("eeg", condition="processed")["signals"][0])
    encoded_joint = model.encode(model.preprocessor.prepare(joint))
    assert encoded_joint.values.shape == (1, 512)
    assert shapes[-1] == ([1, 2, 3000], [1, 33, 512])
    mixed_rate = state("ppg", 125, 10, condition="processed")
    mixed_rate["signals"].append(state("ecg", 200, 10, condition="processed")["signals"][0])
    for channel in mixed_rate["signals"]:
        channel["metadata"]["start_time_s"] = 0
    encoded_mixed = model.encode(model.preprocessor.prepare(mixed_rate))
    assert encoded_mixed.values.shape == (2, 512)
    assert shapes[-2:] == [([1, 1, 1250], [1, 16, 512]), ([1, 1, 2000], [1, 22, 512])]
    hook.remove()
    batch = model.make_batch([state("ppg", 125, 10, condition="processed")],
                             ["Which category describes this signal?"], [["Low", "Normal", "High"]])
    with torch.no_grad():
        result = model(batch)
    assert result["logits"].shape == (1, 3) and torch.isfinite(result["logits"]).all()
    predicted = model.predict(state(), "Which category?", ["Slow", "Normal", "Fast"])
    assert list(predicted) == ["choice", "probabilities", "confidence"]
    assert len(predicted["probabilities"]) == 3 and abs(sum(predicted["probabilities"])-1) < 1e-6
    model.train()
    torch.nn.functional.cross_entropy(model(batch)["logits"], torch.tensor([1])).backward()
    assert all(p.grad is None and not p.requires_grad for p in model.encoder.parameters())
    assert all(p.grad is None and not p.requires_grad for p in model.text_encoder.model.parameters())
    assert model.head.scorer.weight.grad.isfinite().all()
    model.zero_grad(set_to_none=True)
    with pytest.raises(ValueError, match="context"):
        model.text_encoder.encode(["question "*200])
    multichannel = state()
    multichannel["signals"].append(state("eeg")["signals"][0])
    mixed = model.make_batch([multichannel, state("temperature", 1, 64, condition="processed")],
                             ["Which category?", "Which category?"],
                             [["Slow", "Fast"], ["Low", "Normal", "High", "Uncertain"]])
    model.eval()
    with torch.no_grad():
        mixed_result = model(mixed)
    assert mixed_result["logits"].shape == (2, 4)
    assert mixed_result["logits"][0, 2:].isneginf().all()
    assert torch.isfinite(mixed_result["logits"][mixed.criteria_mask]).all()
    import paya.preprocessing as preprocessing
    def unexpected(*args, **kwargs):
        pytest.fail("Processed inference must not enter NeuroKit")
    monkeypatch.setattr(preprocessing, "clean_channel", unexpected)
    monkeypatch.setattr(preprocessing, "check_eeg_channels", unexpected)
    model.predict(state("ppg", 125, 10, condition="processed"), "Question?", ["Yes", "No"])
    report = {"purpose": "synthetic shape checks; untrained head; no performance evaluation",
              "preparation": model.preprocessor.identity(), "examples": examples,
              "question": list(batch.question.shape), "criteria": list(batch.criteria.shape),
              "text_sequence_length": batch.criteria.shape[1]+1,
              "role_embeddings": list(model.head.roles.shape), "decision_logits": list(result["logits"].shape),
              "mixed_batch_logits": [2, 4],
              "joint_channel_global": list(encoded_joint.values.shape),
              "mixed_rate_globals": list(encoded_mixed.values.shape),
              "trainable_parameters": sum(p.numel() for p in model.trainable_parameters()),
              "gradient_boundary": "decision head only; no optimizer step", "output_keys": list(predicted)}
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/architecture-shapes.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
