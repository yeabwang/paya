"""Load checkpoints and answer requests through one canonical waveform computation."""

import json
from pathlib import Path

import torch

from .inputs import ENDING, prepare_signal, validate_question
from .model import build_model
from .provenance import (
    PUBLIC_API,
    identity,
    sha256,
)
from .text import TextEncoder


def load_checkpoint(checkpoint, assets, text=None, device="cpu"):
    """Verify checkpoint compatibility and load a frozen model for inference.

    assets names a verified NormWear encoder directory. An optional text
    encoder can supply an existing embedding cache or a bundled tokenizer.
    Return (model, text_encoder, checkpoint_payload). Head tensors are loaded
    strictly; unsupported runtime, schema, or encoder identities are rejected.
    """
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("identity") != identity(assets):
        raise ValueError("Model/schema/text/encoder provenance differs from checkpoint")
    model = build_model(assets, payload["config"])
    model.head.load_state_dict(payload["head"], strict=True)
    model.temperature.fill_(payload.get("temperature", 1.0))
    model.to(device).eval().requires_grad_(False)
    return model, text or TextEncoder(payload.get("text_cache"), device=device), payload


def canonical_inputs(signal):
    """Return validated channel tensors in deterministic physical order.

    Sort metadata and sample bytes together so input channel permutations
    produce the same floating-point reduction order. Return [channels, samples]
    and [channels, METADATA_DIM] tensors.
    """
    signals, metadata = prepare_signal(signal)
    order = sorted(
        range(len(signals)),
        key=lambda i: (metadata[i].numpy().tobytes(), signals[i].numpy().tobytes()),
    )
    return signals[order], metadata[order]


def logits(
    model,
    text,
    question,
    choices,
    metadata,
    device="cpu",
    signals=None,
    signal_memory=None,
):
    """Encode a question and candidates and return [1, choices] calibrated logits.

    Supply signals as [channels, samples] or signal_memory as
    [1, channels, tokens, 768], with matching unbatched sensor metadata.
    This helper also supports exact FP32 evidence reuse during evaluation.
    """
    validate_question(question, choices)
    vectors = text.encode([question, *choices, ENDING]).to(device)
    model.eval()
    evidence = (
        {"signals": signals[None].to(device)}
        if signals is not None
        else {"signal_memory": signal_memory}
    )
    return model(
        **evidence,
        sensor_metadata=metadata[None].to(device),
        sensor_mask=torch.ones(1, len(metadata), dtype=torch.bool, device=device),
        question=vectors[:1],
        choices=vectors[None, 1:-1],
        ending=vectors[-1:],
        choice_mask=torch.ones(1, len(choices), dtype=torch.bool, device=device),
    )


@torch.inference_mode()
def answer(model, text, request, device="cpu"):
    """Choose a supplied answer from a signal, question, and choice descriptions.

    request contains signal, question and choices. Run canonical input
    preparation, one encoder/head computation, and softmax. Return choice,
    probability, and probabilities in the supplied choice order. Exact ties
    are resolved by description text rather than candidate position.
    """
    signals, metadata = canonical_inputs(request["signal"])
    scores = logits(
        model,
        text,
        request["question"],
        request["choices"],
        metadata,
        device,
        signals=signals,
    )
    probabilities = scores.softmax(-1)[0].cpu().tolist()
    best = max(probabilities)
    index = min(
        (i for i, p in enumerate(probabilities) if p == best),
        key=lambda i: request["choices"][i],
    )
    return {
        "choice": request["choices"][index],
        "probability": probabilities[index],
        "probabilities": probabilities,
    }


def load_bundle(root, device="cpu"):
    """Verify and load a complete export, returning ``(model, text_encoder)``.

    Files must remain within the bundle and match their recorded checksums.
    The manifest and checkpoint must match the runtime contract. Encoder weights
    and tokenizer files are loaded from the bundle, allowing offline use.
    """
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != 2:
        raise ValueError("Unsupported export manifest schema")
    if manifest.get("public_api") != PUBLIC_API:
        raise ValueError(
            "Export public input adapter differs from the runtime contract"
        )
    for filename, expected in manifest["files"].items():
        path = (root / filename).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != expected:
            raise ValueError("Exported artifact file is missing or changed")
    text = TextEncoder(
        root / "text.pt", device=device, model_path=root / "text_encoder"
    )
    model, text, _ = load_checkpoint(root / "model.pt", root / "normwear", text, device)
    if abs(float(model.temperature) - float(manifest["temperature"])) > 1e-6:
        raise ValueError("Exported temperature differs from its manifest")
    return model, text
