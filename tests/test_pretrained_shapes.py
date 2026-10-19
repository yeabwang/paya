"""Real frozen weights, synthetic waveforms: shape smoke checks only."""

import json
from pathlib import Path

import pytest
import torch

from paya.model import PayaModel
from conftest import state


@pytest.mark.integration
def test_all_three_real_frozen_encoders_and_both_signal_routes():
    assets = {"normwear": "third-party/normwear", "mantis": "third-party/mantis", "text": "third-party/biolord"}
    if not all((Path(path)/"source.json").exists() for path in assets.values()):
        pytest.skip("Pinned encoder assets have not been prepared")
    model = PayaModel.from_assets(assets, head={"width": 256, "layers": 2, "heads": 4, "dropout": 0})
    batch = model.make_batch([state()], ["Which heart-rate category describes this signal?"],
                             [["Slow heart rate", "Normal heart rate", "Fast heart rate"]])
    report = {"purpose": "synthetic shape checks; untrained router/head; no performance evaluation",
              "preparation": model.preprocessor.identity(),
              "question": list(batch.question.shape), "criteria": list(batch.criteria.shape),
              "text_sequence_length": batch.criteria.shape[1]+1,
              "role_embeddings": list(model.head.roles.shape), "routes": {}}
    signals, owners, compatible = model.router_inputs(batch)
    assert [list(signal.shape) for signal in signals] == [[3000]]
    report["router_native_channels"] = [list(signal.shape) for signal in signals]
    report["router_gate_input_width"] = model.router.gate[0].in_features
    with torch.no_grad():
        for expert, name in enumerate(("normwear", "mantis")):
            encoder_input = model.preprocessor.format_for_encoder(batch.states[0], expert)
            native = model.encoders[expert](encoder_input)
            adapted = model.adapters[expert](native.values)
            result = model(batch, routes=torch.tensor([expert]))
            assert native.values.shape[-1] == (768, 512)[expert]
            assert adapted.shape[-1] == 768
            assert native.times.shape == (len(native.values),)
            assert result["logits"].shape == (1, 3) and torch.isfinite(result["logits"]).all()
            report["routes"][name] = {"encoder_frames": list(encoder_input.frames.shape),
                "native_vectors": list(native.values.shape), "head_signal": list(adapted.shape),
                "relative_times": list(native.times.shape),
                "router_logits": list(result["router_logits"].shape), "decision_logits": list(result["logits"].shape)}
    result = model.predict(state(), "Which category?", ["Slow", "Normal", "Fast"])
    assert list(result) == ["choice", "probabilities", "confidence"]
    assert len(result["probabilities"]) == 3 and abs(sum(result["probabilities"])-1) < 1e-6
    assert all(not p.requires_grad for p in model.encoders.parameters())
    # Verify gradients stop at frozen encoders without an optimizer step.
    model.train()
    gradients = model(batch)
    (torch.nn.functional.cross_entropy(gradients["logits"], torch.tensor([1]))
     - gradients["router_logits"].log_softmax(-1)[0, 1]).backward()
    assert all(p.grad is None for p in model.encoders.parameters())
    assert all(p.grad is None and not p.requires_grad for p in model.text_encoder.model.parameters())
    assert model.head.scorer.weight.grad.isfinite().all()
    assert model.router.gate[-1].weight.grad.isfinite().all()
    model.zero_grad(set_to_none=True)
    report["gradient_boundary"] = "router/head only; no optimizer step"
    with pytest.raises(ValueError, match="context"):
        model.text_encoder.encode(["question "*200])
    multichannel = state()
    multichannel["signals"].append(state("eeg")["signals"][0])
    mixed = model.make_batch([multichannel, state("temperature", 1, 16)], ["Which category?", "Which category?"],
                             [["Slow", "Fast"], ["Low", "Normal", "High", "Uncertain"]])
    model.eval()
    with torch.no_grad():
        mixed_result = model(mixed, routes=torch.tensor([0, 1]))
    assert mixed_result["logits"].shape == (2, 4)
    assert mixed_result["logits"][0, 2:].isneginf().all()
    assert torch.isfinite(mixed_result["logits"][mixed.criteria_mask]).all()
    report["mixed_batch_logits"] = [2, 4]
    report["mixed_batch"] = "two-channel NormWear request and temperature Mantis request; 2/4 choices"
    report["output_keys"] = list(result)
    report["output_probabilities_shape"] = [len(result["probabilities"])]
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/architecture-shapes.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
