import copy
import inspect

import pytest
import torch

from paya.encoders import FixedWidthAdapter, NormWearEncoder, MantisEncoder
from paya.head import DecisionHead
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
    assert model.text_encoder.calls[-1] == ["Which category?", "slow", "normal", "fast"]
    assert model.head.roles.shape == (2, 256)
    assert "ending" not in vars(batch)
    assert result["logits"].shape == (1, 3) and result["router_logits"].shape == (1, 2)
    assert seen["conv1"] == [((1, 1, 3000), (1, 16, 1500))]
    assert seen["conv2"] == [((1, 16, 1500), (1, 32, 750))]
    assert seen["question"][0] == ((1, 768), (1, 128))
    assert seen["gate1"][0] == ((1, 160), (1, 64))
    assert seen["gate2"][0] == ((1, 64), (1, 2))
    assert seen["text_input"][0] == ((1, 4, 768), (1, 4, 256))
    assert seen["key"][0][0][-1] == 768 and seen["key"][0][1][-1] == 256
    assert seen["ff"][0] == ((1, 4, 256), (1, 4, 1024))
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


@pytest.mark.parametrize("expert", [0, 1])
def test_stage_order_clean_then_router_then_selected_format_and_encoder(model, monkeypatch, expert):
    events = []
    original_clean = model.preprocessor.clean
    original_format = model.preprocessor.format_for_encoder
    original_convert = model.preprocessor._convert
    targets = []
    def clean(value):
        cleaned = original_clean(value); events.append("cleaned"); return cleaned
    def format_signal(value, selected):
        events.append(f"format:{selected}")
        return original_format(value, selected)
    def convert(values, rate, target):
        targets.append(target)
        return original_convert(values, rate, target)
    monkeypatch.setattr(model.preprocessor, "clean", clean)
    monkeypatch.setattr(model.preprocessor, "format_for_encoder", format_signal)
    monkeypatch.setattr(model.preprocessor, "_convert", convert)
    with torch.no_grad():
        model.router.gate[-1].weight.zero_()
        model.router.gate[-1].bias.fill_(-1)
        model.router.gate[-1].bias[expert] = 1
    hooks = [model.router.register_forward_pre_hook(lambda *_: events.append("router"))]
    hooks += [encoder.register_forward_pre_hook(lambda *_, index=index: events.append(f"encoder:{index}"))
              for index, encoder in enumerate(model.encoders)]
    model.predict(state(), "Question?", ["a", "b"])
    for hook in hooks:
        hook.remove()
    assert events == ["cleaned", "router", f"format:{expert}", f"encoder:{expert}"]
    assert targets == [65 if expert == 0 else 500]


def test_cleaning_keeps_one_native_array_per_channel_without_encoder_formatting(model, monkeypatch):
    import paya.preprocessing as module
    def unexpected(*args, **kwargs):
        pytest.fail("Encoder formatting ran before routing")
    monkeypatch.setattr(model.preprocessor, "_convert", unexpected)
    monkeypatch.setattr(module, "frame_starts", unexpected)
    request = state("ecg", 500, 16)
    request["signals"].append(state("temperature", 1, 16)["signals"][0])
    for channel in request["signals"]:
        channel["metadata"]["start_time_s"] = 0
    batch = model.make_batch([request], ["Which category?"], [["a", "b"]])
    signals, owners, compatible = model.router_inputs(batch)
    assert sorted(len(signal) for signal in signals) == [16, 8000]
    assert sorted(c.rate for c in batch.states[0].channels) == [1, 500]
    assert owners.tolist() == [0, 0] and compatible.tolist() == [[False, True]]
    assert model.router(signals, owners, batch.question, compatible).shape == (1, 2)


def test_router_pools_native_channels_without_padding(model):
    question = torch.randn(2, 768)
    signals = [torch.randn(17), torch.randn(17), torch.randn(64)]
    owners = torch.tensor([0, 0, 1])
    expected = [model.router.waveform(signal[None, None]).mean(-1)[0] for signal in signals]
    seen = []
    hook = model.router.gate[0].register_forward_pre_hook(lambda _, args: seen.append(args[0].detach().clone()))
    logits = model.router(signals, owners, question, torch.ones(2, 2, dtype=torch.bool))
    hook.remove()
    assert logits.shape == (2, 2) and seen[0].shape == (2, 160)
    torch.testing.assert_close(seen[0][0, :32], (expected[0]+expected[1])/2)
    torch.testing.assert_close(seen[0][1, :32], expected[2])
    logits.sum().backward()
    assert model.router.waveform[0].weight.grad.abs().sum() > 0


def test_router_result_is_independent_of_other_native_channel_lengths(model):
    first = model.make_batch([state()], ["q"], [["a", "b"]])
    mixed = model.make_batch([state(), state("temperature", 1, 16)], ["q", "q2"], [["a", "b"], ["a", "b"]])
    signal, owners, eligible = model.router_inputs(first)
    alone = model.router(signal, owners, first.question, eligible)
    signal, owners, eligible = model.router_inputs(mixed)
    together = model.router(signal, owners, mixed.question, eligible)
    torch.testing.assert_close(alone[0], together[0], atol=1e-6, rtol=1e-6)


def test_one_cleaned_signal_can_be_reused_for_questions_and_either_encoder(model, monkeypatch):
    cleaned = model.preprocessor.clean(state())
    original = [channel.samples.copy() for channel in cleaned.channels]
    def unexpected(*args, **kwargs):
        pytest.fail("An already cleaned signal was cleaned again")
    monkeypatch.setattr(model.preprocessor, "clean", unexpected)
    for expert, question in enumerate(("Is the rhythm regular?", "Is the waveform morphology normal?")):
        with torch.no_grad():
            model.router.gate[-1].weight.zero_()
            model.router.gate[-1].bias.fill_(-1)
            model.router.gate[-1].bias[expert] = 1
        model.predict(cleaned, question, ["yes", "no"])
        for channel, before in zip(cleaned.channels, original):
            torch.testing.assert_close(torch.from_numpy(channel.samples), torch.from_numpy(before), atol=0, rtol=0)
    assert [encoder.calls for encoder in model.encoders] == [1, 1]


@pytest.mark.parametrize("invalid", ["owner", "missing", "shape", "nonfinite"])
def test_router_rejects_invalid_native_channels(model, invalid):
    signals, owners = [torch.zeros(16)], torch.tensor([0])
    if invalid == "owner":
        owners[0] = 1
    elif invalid == "missing":
        signals = []
    elif invalid == "shape":
        signals[0] = torch.zeros(1, 16)
    else:
        signals[0][0] = torch.nan
    with pytest.raises(ValueError):
        model.router(signals, owners, torch.zeros(1, 768), torch.ones(1, 2, dtype=torch.bool))


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
    signals, owners, compatible = model.router_inputs(batch)
    with pytest.raises(ValueError, match="compatible"):
        model.router(signals, owners, batch.question, torch.zeros_like(compatible))
    with pytest.raises(ValueError, match="real signal"):
        model.head(torch.zeros(1, 2, 768), torch.zeros(1, 2, dtype=torch.bool), torch.zeros(1, 2),
                   batch.question, batch.criteria, batch.criteria_mask)


def test_encoder_wrappers_keep_native_shape_contracts():
    from torch import nn
    from paya.preprocessing import SignalPreprocessor
    class NormCore(nn.Module):
        def forward(self, x):
            return torch.arange(560, dtype=torch.float32)[None, None, :, None].expand(len(x), x.shape[1], 560, 768)
    class MantisCore(nn.Module):
        def forward(self, x):
            return x.mean(-1).expand(-1, 512)
    processor = SignalPreprocessor()
    cleaned = processor.clean(state())
    norm = NormWearEncoder(NormCore())(processor.format_for_encoder(cleaned, 0))
    mantis = MantisEncoder(MantisCore())(processor.format_for_encoder(cleaned, 1))
    assert norm.values.shape == (105, 768) and norm.times.shape == (105,)
    assert mantis.values.shape[1] == 512 and mantis.times.shape == (len(mantis.values),)
    torch.testing.assert_close(norm.times[0], torch.tensor(0.5))
    torch.testing.assert_close(norm.times[1:].reshape(8, 13)[:, 0], (torch.arange(8)+0.5)/8)
    assert torch.equal(norm.times[1:].reshape(8, 13), norm.times[1:].reshape(8, 13)[:, :1].expand(-1, 13))


def test_time_encoding_changes_cross_attention_keys_without_changing_values():
    from paya.head import CrossAttention, temporal_encoding
    layer = CrossAttention(width=4, heads=1, dropout=0).eval()
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.zero_()
        layer.query.bias[0] = 1
        layer.value.weight[0, 0] = 1
        layer.output.weight[0, 0] = 1
    signal = torch.zeros(1, 2, 768)
    signal[0, 1, 0] = 1
    text, mask = torch.zeros(1, 1, 4), torch.ones(1, 2, dtype=torch.bool)
    times = torch.tensor([[0.0, 0.25]])
    first = layer(text, signal, temporal_encoding(times, 4), mask)
    second = layer(text, signal, temporal_encoding(times.flip(1), 4), mask)
    torch.testing.assert_close(first[0, 0, 0], torch.sigmoid(torch.tensor(0.5)))
    torch.testing.assert_close(second[0, 0, 0], torch.sigmoid(torch.tensor(-0.5)))


@pytest.mark.parametrize("width", [1, 7, 256])
def test_temporal_encoding_contains_only_relative_time(width):
    from paya.head import temporal_encoding
    times = torch.tensor([[0.0, 0.25, 0.5]])
    encoded = temporal_encoding(times, width)
    assert encoded.shape == (1, 3, width) and torch.isfinite(encoded).all()
    torch.testing.assert_close(encoded[0, :, 0], torch.sin(2*torch.pi*times[0]))


@pytest.mark.parametrize("times", [torch.zeros(1, 2, 3), torch.tensor([[0.0, float("nan")]]),
                                  torch.tensor([[-0.1, 0.5]]), torch.tensor([[0.5, 1.1]])])
def test_head_rejects_extra_coordinates_or_invalid_relative_times(times):
    head = DecisionHead(dropout=0)
    with pytest.raises(ValueError, match="(?i)(relative times|finite)"):
        head(torch.zeros(1, 2, 768), torch.ones(1, 2, dtype=torch.bool), times,
             torch.zeros(1, 768), torch.zeros(1, 2, 768), torch.ones(1, 2, dtype=torch.bool))


def test_head_ignores_padding_and_preserves_time_embedding_pairs():
    head = DecisionHead(dropout=0).eval()
    generator = torch.Generator().manual_seed(7)
    signal = torch.randn(1, 3, 768, generator=generator)
    times = torch.tensor([[0.0, 0.25, 0.75]])
    question = torch.randn(1, 768, generator=generator)
    criteria = torch.randn(1, 2, 768, generator=generator)
    mask = torch.ones(1, 3, dtype=torch.bool)
    choices = torch.ones(1, 2, dtype=torch.bool)
    expected = head(signal, mask, times, question, criteria, choices)
    order = [2, 0, 1]
    actual = head(signal[:, order], mask, times[:, order], question, criteria, choices)
    torch.testing.assert_close(actual, expected)
    padded = head(torch.cat((signal, torch.full((1, 1, 768), 100.0)), 1),
                  torch.tensor([[True, True, True, False]]), torch.cat((times, torch.ones(1, 1)), 1),
                  question, torch.cat((criteria, torch.full((1, 1, 768), 100.0)), 1),
                  torch.tensor([[True, True, False]]))
    torch.testing.assert_close(padded[:, :2], expected)
    assert padded[0, 2].isneginf()
