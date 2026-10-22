"""Paya versions, frozen-model identities and artifact checksums."""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

from .vendor import signal

PACKAGE_VERSION = "1.0.0"
VERSION = f"v{PACKAGE_VERSION.split('.')[0]}"
INPUT_SCHEMA_VERSION = int(VERSION[1:])
RUNTIME_CONTRACT = f"paya-decision-{VERSION}"
PREPARATION_CONTRACT = f"native-raw-or-processed-{VERSION}"
NATIVE_CONTRACT = f"wearable-native-no-unit-conversion-{VERSION}"
CACHE_CONTRACT = f"signal-global-channel-temporal-tokens-fp32-{VERSION}"
TEXT_CONTRACT = f"biolord-separate-text-{VERSION}"

TEXT_ID = "FremyCompany/BioLORD-2023"
TEXT_REVISION = "167aab527b238a50ca65224e6319215d2ff4fc9f"
TEXT_IDENTITY = {
    "contract": TEXT_CONTRACT,
    "model_id": TEXT_ID,
    "revision": TEXT_REVISION,
    "pooling": "sentence-transformers-default",
    "normalize": True,
    "dimension": 768,
}

ARCHITECTURE = {
    "signal_encoder": signal.IDENTITY,
    "pipeline": [
        "validate",
        "clean_raw_only",
        "stack_channels",
        "frozen_encoding",
        "decision_head",
    ],
    "signal_width": signal.WIDTH,
    "sampling": "declared-native-rate",
    "format": "one-aligned-recording",
    "minimum_samples": signal.MIN_SAMPLES,
    "signal_types": ["eeg", "ieeg", "ecg", "ppg"],
    "signal_tokens": "1+C+2*ceil(L/200)",
    "channel_order": "as-supplied",
    "signal_output": "global-channel-temporal-tokens",
    "signal_mask": "exclude-fully-padded-temporal-tokens",
    "attention_order": ["question-candidate-self-attention", "signal-cross-attention"],
    "required_metadata": ["sensor_type", "sampling_rate_hz", "duration_s"],
    "signal_condition": {"values": ["raw", "processed"], "default": "raw"},
    "text_processing": f"{TEXT_CONTRACT}-normalized-mean-128",
    "text_roles": ["question", "candidate"],
    "head_inputs": ["signal", "question", "criteria"],
    "input_schema": INPUT_SCHEMA_VERSION,
    "output": ["choice", "confidence", "probabilities"],
    "probabilities": "choice-to-probability",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def source_identity():
    """Identify supported Python sources without changing the package version."""
    root = Path(__file__).resolve().parent
    return fingerprint(
        {
            path.relative_to(root).as_posix(): hashlib.sha256(
                path.read_text(encoding="utf-8").replace("\r\n", "\n").encode()
            ).hexdigest()
            for path in sorted(root.rglob("*.py"))
        }
    )


def feature_source_identity():
    """Identify only code affecting frozen signal tokens and their preparation."""
    root = Path(__file__).resolve().parent
    return fingerprint(
        {
            name: hashlib.sha256(
                (root / name).read_text(encoding="utf-8").replace("\r\n", "\n").encode()
            ).hexdigest()
            for name in (
                "encoders.py",
                "preprocessing.py",
                "inputs.py",
                "physiology.py",
                "vendor/spotr.py",
            )
        }
    )


def runtime_identity():
    """Record the numerical environment required for exact training recovery."""
    import torch

    return {
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in (
                "torch",
                "numpy",
                "scipy",
                "neurokit2",
                "transformers",
                "sentence-transformers",
            )
        },
        "cuda": torch.version.cuda,
    }


def contained_file(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Missing asset or path outside its declared directory")
    return path


def verify_source(directory, model_id, revision):
    root = Path(directory)
    source = json.loads((root / "source.json").read_text(encoding="utf-8"))
    if source.get("model_id") != model_id or source.get("revision") != revision:
        raise ValueError(f"Expected pinned {model_id}@{revision}")
    files = source.get("files")
    if isinstance(files, list):
        files = {entry["file"]: entry["sha256"] for entry in files}
    if not isinstance(files, dict) or not files:
        raise ValueError("Asset source needs nonempty file checksums")
    for name, digest in files.items():
        if sha256(contained_file(root, name)) != digest:
            raise ValueError(f"Asset checksum changed: {name}")
    if "model.safetensors" not in files:
        raise ValueError("Pinned safetensors weights are required")
    return {"model_id": model_id, "revision": revision, "files": files}


def verify_assets(assets):
    expected = {
        signal.ASSET_KEY: (signal.MODEL_ID, signal.REVISION),
        "text": (TEXT_ID, TEXT_REVISION),
    }
    if set(assets) != set(expected):
        raise ValueError(
            f"Specify exactly {signal.ASSET_KEY} and text asset directories"
        )
    return {name: verify_source(assets[name], *spec) for name, spec in expected.items()}
