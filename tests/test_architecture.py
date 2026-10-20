import inspect

import numpy as np
import pytest
import torch

from paya.encoders import SPOTREncoder
from paya.head import DecisionHead
from paya.train import choice_loss
from conftest import state


def test_layer_shapes_and_separate_text_encoding(model):
    seen, handles = {}, []
    for name, module in (("text_input", model.head.text_input[0]),
                         ("key", model.head.blocks[0].cross_attention.key),
                         ("ff", model.head.blocks[0].feed_forward[0]), ("scorer", model.head.scorer)):
        handles.append(module.register_forward_hook(lambda m, x, y, name=name:
                       seen.setdefault(name, []).append((tuple(x[0].shape), tuple(y.shape)))))
    batch = model.make_batch([state()], ["Which category?"], [["slow", "normal", "fast"]])
    result = model(batch)
    for handle in handles:
        handle.remove()
    assert model.text_encoder.calls[-1] == ["Which category?", "slow", "normal", "fast"]
    assert model.head.roles.shape == (2, 256) and "ending" not in vars(batch)
    assert set(result) == {"logits", "criteria_mask"} and result["logits"].shape == (1, 3)
    assert seen["text_input"][0] == ((1, 4, 768), (1, 4, 256))
    assert seen["key"][0] == ((1, 1, 512), (1, 1, 256))
    assert seen["ff"][0] == ((1, 4, 256), (1, 4, 1024))
    assert seen["scorer"][0] == ((1, 3, 256), (1, 3, 1))


def test_public_output_and_one_frozen_signal_encoder(model):
    choices = ["slow", "normal", "fast"]
    result = model.predict(state(), "Which category?", choices)
    assert list(result) == ["choice", "probabilities", "confidence"]
    assert abs(sum(result["probabilities"])-1) < 1e-6
    assert result["confidence"] == result["probabilities"][choices.index(result["choice"])]
    assert model.encoder.calls == 1
    assert not hasattr(model, "router") and not hasattr(model, "adapters")


@pytest.mark.parametrize("condition", ["raw", "processed"])
def test_stage_order_native_preparation_then_framing_and_encoding(model, monkeypatch, condition):
    import paya.preprocessing as module
    events = []
    original_clean, original_format = module.clean_channel, model.preprocessor.format
    def clean(channel):
        events.append("neurokit")
        return original_clean(channel)
    def format_signal(prepared):
        events.append("format")
        return original_format(prepared)
    monkeypatch.setattr(module, "clean_channel", clean)
    monkeypatch.setattr(model.preprocessor, "format", format_signal)
    hook = model.encoder.register_forward_pre_hook(lambda *_: events.append("spotr"))
    model.predict(state(condition=condition), "Question?", ["a", "b"])
    hook.remove()
    assert events == (["neurokit"] if condition == "raw" else []) + ["format", "spotr"]


def test_prepared_signal_reuse(model, monkeypatch):
    request = state("ecg", 10, 60, condition="processed")
    request["signals"].append(state("ppg", 10, 60, condition="processed")["signals"][0])
    for channel in request["signals"]:
        channel["metadata"]["start_time_s"] = 0
    prepared = model.preprocessor.prepare(request)
    assert [c.rate for c in prepared.channels] == [10, 10]
    originals = [c.samples.copy() for c in prepared.channels]
    def unexpected(*args, **kwargs):
        pytest.fail("Prepared signals must not be prepared again")
    monkeypatch.setattr(model.preprocessor, "prepare", unexpected)
    for question in ("First question?", "Second question?"):
        model.predict(prepared, question, ["yes", "no"])
    for channel, original in zip(prepared.channels, originals):
        np.testing.assert_array_equal(channel.samples, original)


def test_candidate_permutation_and_audit_invariance(model):
    criteria = ["slow", "normal", "fast"]
    request = state()
    request["signals"].append(state("ppg", 500)["signals"][0])
    first = model.predict(request, "Which category?", criteria)
    request["dataset_id"] = "audit only"
    request["signals"][0]["metadata"]["subject_id"] = "audit only"
    second = model.predict(request, "Which category?", criteria)
    torch.testing.assert_close(torch.tensor(first["probabilities"]), torch.tensor(second["probabilities"]), atol=1e-6, rtol=1e-6)
    order = [2, 0, 1]
    third = model.predict(request, "Which category?", [criteria[i] for i in order])
    torch.testing.assert_close(torch.tensor(third["probabilities"]), torch.tensor(first["probabilities"])[order], atol=1e-6, rtol=1e-6)


def test_heterogeneous_batch_masks_gradients_and_frozen_boundaries(model):
    model.train()
    batch = model.make_batch([state(), state("ieeg", 1000, 2)], ["q", "q2"], [["a", "b"], ["a", "b", "c", "d"]])
    result = model(batch)
    assert result["logits"].shape == (2, 4) and result["logits"][0, 2:].isneginf().all()
    choice_loss(result["logits"], torch.tensor([0, 1]), batch.criteria_mask).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.head.parameters())
    assert all(p.grad is None and not p.requires_grad for p in model.encoder.parameters())
    assert not model.encoder.training
    assert "metadata" not in inspect.signature(model.head.forward).parameters
    assert all(name.startswith("head.") for name, p in model.named_parameters() if p.requires_grad)


def test_empty_signal_attention_mask_is_rejected(model):
    batch = model.make_batch([state()], ["q"], [["a", "b"]])
    with pytest.raises(ValueError, match="real signal"):
        model.head(torch.zeros(1, 2, 512), torch.zeros(1, 2, dtype=torch.bool), torch.zeros(1, 2, 2),
                   batch.question, batch.criteria, batch.criteria_mask)


def test_encoder_wrapper_keeps_native_recording_and_duration():
    from torch import nn
    from paya.preprocessing import SignalPreprocessor
    class Core(nn.Module):
        def forward(self, x):
            batch, channels, length = x.shape
            tokens = torch.zeros(batch, 1+channels+2*((length+199)//200), 512)
            tokens[:, 0] = x.mean((1, 2))[:, None]
            tokens[:, 1:] = 100
            return tokens
    processor = SignalPreprocessor()
    prepared = processor.prepare(state())
    frames = processor.format(prepared)
    encoded = SPOTREncoder(Core())(frames)
    assert encoded.values.shape == (1, 512) and encoded.intervals_s.shape == (1, 2)
    torch.testing.assert_close(encoded.values[0], frames.samples.mean().expand(512))
    torch.testing.assert_close(encoded.intervals_s, frames.intervals_s)


def test_single_signal_token_keeps_physical_time_in_attention_values():
    from paya.head import CrossAttention, interval_encoding
    layer = CrossAttention(width=8, heads=1, dropout=0).eval()
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.zero_()
        layer.output.weight.copy_(torch.eye(8))
    signal, text = torch.zeros(1, 1, 512), torch.zeros(1, 1, 8)
    mask = torch.ones(1, 1, dtype=torch.bool)
    first_time = interval_encoding(torch.tensor([[[0., 10.]]]), 8)
    second_time = interval_encoding(torch.tensor([[[0., 5.]]]), 8)
    first = layer(text, signal, first_time, mask)
    second = layer(text, signal, second_time, mask)
    torch.testing.assert_close(first, first_time)
    torch.testing.assert_close(second, second_time)
    assert not torch.allclose(first, second)


@pytest.mark.parametrize("width", [2, 7, 256])
def test_interval_encoding_keeps_both_elapsed_endpoints(width):
    from paya.head import interval_encoding
    intervals = torch.tensor([[[0., 10.], [0.25, 20.], [5., 100.]]])
    encoded = interval_encoding(intervals, width)
    assert encoded.shape == (1, 3, width) and torch.isfinite(encoded).all()
    torch.testing.assert_close(encoded[..., 0], torch.sin(2*torch.pi*intervals[..., 0]))
    torch.testing.assert_close(encoded[..., width//2], torch.sin(2*torch.pi*intervals[..., 1]))


@pytest.mark.parametrize("intervals", [torch.zeros(1, 2), torch.zeros(1, 2, 3),
    torch.tensor([[[0., float("nan")], [0., 1.]]]),
    torch.tensor([[[-0.1, 1.], [0., 1.]]]), torch.tensor([[[2., 1.], [0., 1.]]]),
    torch.tensor([[[0., 0.], [0., 1.]]])])
def test_head_rejects_invalid_physical_intervals(intervals):
    head = DecisionHead(dropout=0)
    with pytest.raises(ValueError, match="(?i)(intervals|finite)"):
        head(torch.zeros(1, 2, 512), torch.ones(1, 2, dtype=torch.bool), intervals,
             torch.zeros(1, 768), torch.zeros(1, 2, 768), torch.ones(1, 2, dtype=torch.bool))


def test_head_ignores_padding_and_preserves_signal_interval_pairs():
    head = DecisionHead(dropout=0).eval()
    generator = torch.Generator().manual_seed(7)
    signal = torch.randn(1, 3, 512, generator=generator)
    intervals = torch.tensor([[[0., 9.984], [0.016, 10.], [0., 6.]]])
    question = torch.randn(1, 768, generator=generator)
    criteria = torch.randn(1, 2, 768, generator=generator)
    mask = torch.ones(1, 3, dtype=torch.bool)
    choices = torch.ones(1, 2, dtype=torch.bool)
    expected = head(signal, mask, intervals, question, criteria, choices)
    order = [2, 0, 1]
    actual = head(signal[:, order], mask, intervals[:, order], question, criteria, choices)
    torch.testing.assert_close(actual, expected)
    padded = head(torch.cat((signal, torch.full((1, 1, 512), 100.0)), 1),
                  torch.tensor([[True, True, True, False]]), torch.cat((intervals, torch.zeros(1, 1, 2)), 1),
                  question, torch.cat((criteria, torch.full((1, 1, 768), 100.0)), 1),
                  torch.tensor([[True, True, False]]))
    torch.testing.assert_close(padded[:, :2], expected)
    assert padded[0, 2].isneginf()


@pytest.mark.parametrize("shape", [(1, 2), (1, 1250), (4, 201), (128, 12800)])
def test_encoder_runs_once_for_the_complete_recording(shape):
    from torch import nn
    from paya.preprocessing import EncoderInput
    seen = []
    class Core(nn.Module):
        def forward(self, x):
            seen.append(tuple(x.shape))
            batch, channels, length = x.shape
            return x.mean((1, 2))[:, None, None].expand(batch, 1+channels+2*((length+199)//200), 512)
    samples = torch.full(shape, 2.)
    intervals = torch.tensor([[0., 10.]])
    result = SPOTREncoder(Core())(EncoderInput(samples, intervals))
    assert seen == [(1, *shape)]
    torch.testing.assert_close(result.values, torch.full((1, 512), 2.))
    torch.testing.assert_close(result.intervals_s, intervals)


def test_same_samples_at_different_rates_keep_different_physical_times(model):
    import copy
    first = state("ppg", 125, 10, condition="processed")
    second = copy.deepcopy(first)
    second["signals"][0]["metadata"].update(sampling_rate_hz=250, duration_s=5)
    a = model.encode(model.preprocessor.prepare(first))
    b = model.encode(model.preprocessor.prepare(second))
    torch.testing.assert_close(a.values, b.values, atol=0, rtol=0)
    torch.testing.assert_close(a.intervals_s, b.intervals_s*2, atol=0, rtol=0)


def test_encoder_rejects_invalid_interval_before_running_frozen_weights():
    from torch import nn
    from paya.preprocessing import EncoderInput
    class Core(nn.Module):
        def forward(self, x):
            pytest.fail("Invalid intervals must fail before frozen encoding")
    with pytest.raises(ValueError, match="interval"):
        SPOTREncoder(Core())(EncoderInput(torch.zeros(1, 2), torch.tensor([[1., 0.]])))


@pytest.mark.parametrize("shape", [(0, 200), (129, 200), (1, 1), (1, 12801), (200,)])
def test_encoder_rejects_inputs_outside_pretrained_capacity(shape):
    from torch import nn
    from paya.preprocessing import EncoderInput
    class Core(nn.Module):
        def forward(self, x):
            pytest.fail("Invalid waveform shape must fail before frozen encoding")
    with pytest.raises(ValueError, match="SPOTR expects"):
        SPOTREncoder(Core())(EncoderInput(torch.zeros(shape), torch.tensor([[0., 1.]])))


@pytest.mark.parametrize("channels,length", [(1, 2), (2, 199), (3, 200), (2, 201), (1, 12800)])
def test_spotr_core_native_normalization_padding_and_shapes(channels, length):
    from paya.vendor.spotr import SPOTREncoder as Core
    core = Core(encoder_dim=16, num_heads=8, n_layers=1).eval()
    x = torch.randn(1, channels, length)
    with torch.no_grad():
        tokens, normalized = core.encode(x)
        shifted = core(x+100)
        constant = core(torch.full_like(x, 37))
    padded = 200*((length+199)//200)
    assert tokens.shape == (1, 1+channels+padded//100, 16)
    assert normalized.shape == (1, channels, padded)
    expected = (x-x.mean(-1, keepdim=True))/(x.std(-1, keepdim=True)+1e-6)
    torch.testing.assert_close(normalized[..., :length], expected)
    assert (normalized[..., length:] == 0).all()
    torch.testing.assert_close(shifted, tokens, atol=2e-5, rtol=2e-5)
    assert torch.isfinite(constant).all()
