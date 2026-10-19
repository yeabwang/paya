import copy
import inspect

import pytest
import torch

from paya.encoders import FixedWidthAdapter, NormWearEncoder, MantisEncoder
from paya.head import DecisionHead
from paya.inputs import ENDING
from paya.train import choice_loss, expert_losses, routing_targets
from conftest import state


def test_fixed_adapters_are_lossless_and_have_no_parameters():
    for width in (512, 768):
        adapter = FixedWidthAdapter(width)
        x = torch.arange(2*width, dtype=torch.float64).reshape(2, width)/8
        result = adapter(x)
        assert result.shape == (2, 768) and not list(adapter.parameters()) and not adapter.state_dict()
        torch.testing.assert_close(result[:, :width], x, rtol=0, atol=0)
        assert result[:, width:].count_nonzero() == 0
        torch.testing.assert_close(result @ result.T, x @ x.T, rtol=0, atol=0)


def test_layer_shapes_and_separate_text_encoding(model):
    seen = {}
    handles = []
    for name, module in (("conv1", model.router.waveform[0]), ("conv2", model.router.waveform[2]),
                         ("question", model.router.question[0]), ("gate1", model.router.gate[0]),
                         ("gate2", model.router.gate[2]), ("text_input", model.head.text_input[0]),
                         ("key", model.head.blocks[0].cross_attention.key),
                         ("ff", model.head.blocks[0].feed_forward[0]), ("scorer", model.head.scorer)):
        handles.append(module.register_forward_hook(lambda m, x, y, name=name: seen.setdefault(name, []).append((tuple(x[0].shape), tuple(y.shape)))))
    batch = model.make_batch([state()], ["Which category?"], [["slow", "normal", "fast"]])
    result = model(batch)
    for handle in handles:
        handle.remove()
    assert model.text_encoder.calls[-1] == ["Which category?", "slow", "normal", "fast", ENDING]
    assert result["logits"].shape == (1, 3) and result["router_logits"].shape == (1, 2)
    assert seen["conv1"][0] == ((1, 1, 390), (1, 16, 195))
    assert seen["conv2"][0] == ((1, 16, 195), (1, 32, 98))
    assert seen["question"][0] == ((1, 768), (1, 128))
    assert seen["gate1"][0] == ((1, 192), (1, 64))
    assert seen["gate2"][0] == ((1, 64), (1, 2))
    assert seen["text_input"][0] == ((1, 5, 768), (1, 5, 256))
    assert seen["key"][0][0][-1] == 768 and seen["key"][0][1][-1] == 256
    assert seen["ff"][0] == ((1, 5, 256), (1, 5, 1024))
    assert seen["scorer"][0] == ((1, 3, 256), (1, 3, 1))


def test_public_three_fields_and_single_selected_expert(model):
    with torch.no_grad():
        model.router.gate[-1].weight.zero_()
        model.router.gate[-1].bias.copy_(torch.tensor([2., 0.]))
    result = model.predict(state=state(), question="Which category?", criteria=["slow", "normal", "fast"])
    assert list(result) == ["choice", "probabilities", "confidence"]
    assert abs(sum(result["probabilities"])-1) < 1e-6
    assert result["confidence"] == result["probabilities"][["slow", "normal", "fast"].index(result["choice"])]
    assert model.encoders[0].calls == 1 and model.encoders[1].calls == 0
    with torch.no_grad():
        model.router.gate[-1].bias.copy_(torch.tensor([0., 2.]))
    model.predict(state(), "Which category?", ["slow", "normal", "fast"])
    assert model.encoders[0].calls == 1 and model.encoders[1].calls == 1


def test_stage_order_preparation_then_router_then_encoder(model, monkeypatch):
    events = []
    original = model.preprocessor.prepare
    def prepare(value):
        prepared = original(value); events.append("prepared"); return prepared
    monkeypatch.setattr(model.preprocessor, "prepare", prepare)
    hooks = [model.router.register_forward_pre_hook(lambda *_: events.append("router"))]
    hooks += [encoder.register_forward_pre_hook(lambda *_: events.append("encoder")) for encoder in model.encoders]
    model.predict(state(), "Question?", ["a", "b"])
    for hook in hooks:
        hook.remove()
    assert events == ["prepared", "router", "encoder"]


def test_candidate_permutation_channel_permutation_and_audit_invariance(model):
    criteria = ["slow", "normal", "fast"]
    request = state()
    request["signals"].append(state("ppg", 500)["signals"][0])
    first = model.predict(request, "Which category?", criteria)
    request["signals"].reverse()
    request["dataset_id"] = "audit only"
    request["signals"][0]["metadata"]["subject_id"] = "audit only"
    second = model.predict(request, "Which category?", criteria)
    torch.testing.assert_close(torch.tensor(first["probabilities"]), torch.tensor(second["probabilities"]), atol=1e-6, rtol=1e-6)
    order = [2, 0, 1]
    third = model.predict(request, "Which category?", [criteria[i] for i in order])
    torch.testing.assert_close(torch.tensor(third["probabilities"]), torch.tensor(first["probabilities"])[order], atol=1e-6, rtol=1e-6)


def test_heterogeneous_batch_masks_gradients_and_frozen_boundaries(model):
    model.train()
    batch = model.make_batch([state(), state("temperature", 1, 16)], ["q", "q2"], [["a", "b"], ["a", "b", "c", "d"]])
    result = model(batch)
    assert result["logits"].shape == (2, 4) and result["logits"][0, 2:].isneginf().all()
    assert result["router_logits"][1, 0].isneginf() and result["routes"][1] == 1
    loss = choice_loss(result["logits"], torch.tensor([0, 1]), batch.criteria_mask)
    loss = loss - result["router_logits"][0].log_softmax(-1)[1]
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.head.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.router.parameters())
    assert all(p.grad is None and not p.requires_grad for p in model.encoders.parameters())
    assert not model.encoders.training and all(not e.training for e in model.encoders)
    assert not list(model.adapters.parameters())
    assert "metadata" not in inspect.signature(model.head.forward).parameters
    assert all(name.startswith(("router.", "head.")) for name, p in model.named_parameters() if p.requires_grad)


def test_training_teacher_uses_only_compatible_experts_and_backpropagates(model):
    batch = model.make_batch([state(), state("temperature", 1, 16)], ["q", "q2"], [["a", "b"], ["c", "d"]])
    model.train()
    losses, compatible = expert_losses(model, batch, torch.tensor([0, 1]), gradients=True)
    assert losses.shape == compatible.shape == (2, 2) and losses[1, 0].isinf()
    targets = routing_targets(losses.detach(), compatible)
    assert targets[1, 0] == 0 and targets[1, 1] == 1
    losses[compatible].mean().backward()
    assert model.head.scorer.weight.grad.abs().sum() > 0


def test_no_compatible_expert_or_signal_attention_mask_is_rejected(model):
    batch = model.make_batch([state()], ["q"], [["a", "b"]])
    frames, owners, compatible = model.router_inputs(batch)
    with pytest.raises(ValueError, match="compatible"):
        model.router(frames, owners, batch.question, torch.zeros_like(compatible))
    with pytest.raises(ValueError, match="real signal"):
        model.head(torch.zeros(1, 2, 768), torch.zeros(1, 2, dtype=torch.bool), torch.zeros(1, 2, 3),
                   batch.question, batch.criteria, batch.ending, batch.criteria_mask)


def test_encoder_wrappers_keep_native_shape_contracts():
    from torch import nn
    from paya.preprocessing import SignalPreprocessor
    class NormCore(nn.Module):
        def forward(self, x):
            return torch.arange(560, dtype=torch.float32)[None, None, :, None].expand(len(x), x.shape[1], 560, 768)
    class MantisCore(nn.Module):
        def forward(self, x):
            return x.mean(-1).expand(-1, 512)
    prepared = SignalPreprocessor().prepare(state())
    norm = NormWearEncoder(NormCore())(prepared.views[0])
    mantis = MantisEncoder(MantisCore())(prepared.views[1])
    assert norm.values.shape == (105, 768) and norm.positions.shape == (105, 3)
    assert mantis.values.shape[1] == 512 and mantis.positions.shape[1] == 3
