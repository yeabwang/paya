import inspect

import numpy as np
import pytest
import torch

from paya.encoders import SignalEncoder
from paya.head import AttentionHead
from paya.train import choice_loss
from conftest import state


def test_layer_shapes_and_separate_text_encoding(model):
    batch = model.make_batch(
        [state()], ["Which category?"], [["slow", "normal", "fast"]]
    )
    result = model(batch)
    assert model.text_encoder.calls[-1] == ["Which category?", "slow", "normal", "fast"]
    assert batch.question.shape == (1, 768) and batch.criteria.shape == (1, 3, 768)
    assert result["logits"].shape == (1, 3)
    assert model.head.signal[1].in_features == 512
    assert model.head.text[0].in_features == 768
    assert model.head.config["width"] == 128


def test_public_output_and_one_frozen_signal_encoder(model):
    choices = ["slow", "normal", "fast"]
    result = model.predict(state(), "Which category?", choices)
    assert list(result) == ["choice", "confidence", "probabilities"]
    assert list(result["probabilities"]) == choices
    assert abs(sum(result["probabilities"].values()) - 1) < 1e-6
    assert result["confidence"] == result["probabilities"][result["choice"]]
    assert model.encoder.calls == 1
    assert not hasattr(model, "router") and not hasattr(model, "adapters")


def test_choice_keyed_output_matches_requested_format_and_json_roundtrip(
    model, monkeypatch
):
    import json

    choices = ["choice 1", "choice 2", "choice 3", "choice 4"]
    values = torch.tensor([[0.82, 0.09, 0.06, 0.03]])
    monkeypatch.setattr(model, "forward", lambda batch: {"logits": values.log()})
    result = model.predict(state(condition="processed"), "Question?", choices)
    assert result["choice"] == "choice 1"
    assert result["confidence"] == pytest.approx(0.82)
    assert result["probabilities"] == pytest.approx(
        dict(zip(choices, values[0].tolist()))
    )
    assert json.loads(json.dumps(result)) == result


def test_no_time_features_in_encoder_input_or_head_signature(model):
    encoder_input = model.preprocessor.format(
        model.preprocessor.prepare(state(condition="processed"))
    )
    assert set(vars(encoder_input)) == {"samples"}
    assert list(inspect.signature(model.head.forward).parameters) == [
        "signal",
        "signal_mask",
        "question",
        "criteria",
        "criteria_mask",
    ]


@pytest.mark.parametrize("condition", ["raw", "processed"])
def test_stage_order_native_preparation_then_framing_and_encoding(
    model, monkeypatch, condition
):
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
    hook = model.encoder.register_forward_pre_hook(lambda *_: events.append("encoder"))
    model.predict(state(condition=condition), "Question?", ["a", "b"])
    hook.remove()
    assert events == (["neurokit"] if condition == "raw" else []) + [
        "format",
        "encoder",
    ]


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
    torch.testing.assert_close(
        torch.tensor(list(first["probabilities"].values())),
        torch.tensor(list(second["probabilities"].values())),
        atol=1e-6,
        rtol=1e-6,
    )
    order = [2, 0, 1]
    third = model.predict(request, "Which category?", [criteria[i] for i in order])
    torch.testing.assert_close(
        torch.tensor(list(third["probabilities"].values())),
        torch.tensor(list(first["probabilities"].values()))[order],
        atol=1e-6,
        rtol=1e-6,
    )


def test_heterogeneous_batch_masks_gradients_and_frozen_boundaries(model):
    model.train()
    batch = model.make_batch(
        [state(), state("ieeg", 1000, 2)],
        ["q", "q2"],
        [["a", "b"], ["a", "b", "c", "d"]],
    )
    result = model(batch)
    assert result["logits"].shape == (2, 4) and result["logits"][0, 2:].isneginf().all()
    choice_loss(result["logits"], torch.tensor([0, 1]), batch.criteria_mask).backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.head.parameters()
    )
    assert all(
        p.grad is None and not p.requires_grad for p in model.encoder.parameters()
    )
    assert not model.encoder.training
    assert "metadata" not in inspect.signature(model.head.forward).parameters
    assert all(
        name.startswith("head.")
        for name, p in model.named_parameters()
        if p.requires_grad
    )


def test_empty_signal_attention_mask_is_rejected(model):
    batch = model.make_batch([state()], ["q"], [["a", "b"]])
    with pytest.raises(ValueError, match="nonempty boolean mask"):
        model.head(
            torch.zeros(1, 2, 512),
            torch.zeros(1, 2, dtype=torch.bool),
            batch.question,
            batch.criteria,
            batch.criteria_mask,
        )


def test_encoder_wrapper_keeps_complete_native_recording():
    from torch import nn
    from paya.preprocessing import SignalPreprocessor

    class Core(nn.Module):
        def forward(self, x):
            batch, channels, length = x.shape
            tokens = torch.zeros(batch, 1 + channels + 2 * ((length + 199) // 200), 512)
            tokens[:, 0] = x.mean((1, 2))[:, None]
            tokens[:, 1:] = 100
            return tokens

    processor = SignalPreprocessor()
    prepared = processor.prepare(state())
    frames = processor.format(prepared)
    encoded, mask = SignalEncoder(Core())(frames)
    assert encoded.shape == (1, 1 + 1 + 2 * ((3000 + 199) // 200), 512)
    assert mask.all()
    torch.testing.assert_close(encoded[0, 0], frames.samples.mean().expand(512))
    assert (encoded[0, 1:] == 100).all()


def test_head_candidate_padding_permutation_and_signal_gradient():
    torch.manual_seed(7)
    head = AttentionHead(width=16, dropout=0).eval()
    signal = torch.randn(2, 4, 512, requires_grad=True)
    question, criteria = torch.randn(2, 768), torch.randn(2, 3, 768)
    mask = torch.tensor([[True, True, True, False], [True, True, True, True]])
    choices = torch.tensor([[True, True, False], [True, True, True]])
    expected = head(signal, mask, question, criteria, choices)
    order = [2, 0, 1]
    torch.testing.assert_close(
        head(signal, mask, question, criteria[:, order], choices[:, order]),
        expected[:, order],
        atol=2e-6,
        rtol=2e-6,
    )
    changed = signal.detach().clone()
    changed[~mask] = 1000
    torch.testing.assert_close(
        head(changed, mask, question, criteria, choices), expected
    )
    expected[choices].sum().backward()
    assert signal.grad.abs().sum() > 0 and expected[0, 2].isneginf()
    assert (signal.grad[~mask] == 0).all()


@pytest.mark.parametrize("shape", [(1, 2), (1, 1250), (4, 201), (128, 12800)])
def test_encoder_runs_once_for_the_complete_recording(shape):
    from torch import nn
    from paya.preprocessing import EncoderInput

    seen = []

    class Core(nn.Module):
        def forward(self, x):
            seen.append(tuple(x.shape))
            batch, channels, length = x.shape
            return x.mean((1, 2))[:, None, None].expand(
                batch, 1 + channels + 2 * ((length + 199) // 200), 512
            )

    samples = torch.full(shape, 2.0)
    result, mask = SignalEncoder(Core())(EncoderInput(samples))
    assert seen == [(1, *shape)]
    torch.testing.assert_close(
        result, torch.full((1, 1 + shape[0] + 2 * ((shape[1] + 199) // 200), 512), 2.0)
    )
    assert mask.sum() == 1 + shape[0] + (shape[1] + 99) // 100


def test_same_processed_samples_at_different_rates_have_no_time_features(model):
    import copy

    first = state("ppg", 125, 10, condition="processed")
    second = copy.deepcopy(first)
    second["signals"][0]["metadata"].update(sampling_rate_hz=250, duration_s=5)
    a, a_mask = model.encode(model.preprocessor.prepare(first))
    b, b_mask = model.encode(model.preprocessor.prepare(second))
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert torch.equal(a_mask, b_mask)
    assert model.predict(first, "Question?", ["a", "b"]) == model.predict(
        second, "Question?", ["a", "b"]
    )


@pytest.mark.parametrize("shape", [(0, 200), (129, 200), (1, 1), (1, 12801), (200,)])
def test_encoder_rejects_inputs_outside_pretrained_capacity(shape):
    from torch import nn
    from paya.preprocessing import EncoderInput

    class Core(nn.Module):
        def forward(self, x):
            pytest.fail("Invalid waveform shape must fail before frozen encoding")

    with pytest.raises(ValueError, match="expects finite"):
        SignalEncoder(Core())(EncoderInput(torch.zeros(shape)))


@pytest.mark.parametrize(
    "channels,length", [(1, 2), (2, 199), (3, 200), (2, 201), (1, 12800)]
)
def test_vendor_core_native_normalization_padding_and_shapes(channels, length):
    torch.manual_seed(0)
    from paya.vendor import signal

    Core = signal.SPOTREncoder
    core = Core(encoder_dim=16, num_heads=8, n_layers=1).eval()
    x = torch.randn(1, channels, length)
    with torch.no_grad():
        tokens, normalized = core.encode(x)
        shifted = core(x + 100)
        constant = core(torch.full_like(x, 37))
    padded = 200 * ((length + 199) // 200)
    assert tokens.shape == (1, 1 + channels + padded // 100, 16)
    assert normalized.shape == (1, channels, padded)
    expected = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(normalized[..., :length], expected)
    assert (normalized[..., length:] == 0).all()
    torch.testing.assert_close(shifted, tokens, atol=2e-5, rtol=2e-5)
    assert torch.isfinite(constant).all()
