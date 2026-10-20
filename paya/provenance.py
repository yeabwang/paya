"""Paya versions, frozen-model identities and artifact checksums."""

import hashlib
import json
from pathlib import Path

PACKAGE_VERSION = "1.0.0"
VERSION = f"v{PACKAGE_VERSION.split('.')[0]}"
INPUT_SCHEMA_VERSION = int(VERSION[1:])
RUNTIME_CONTRACT = f"paya-spotr-decision-{VERSION}"
PREPARATION_CONTRACT = f"spotr-native-raw-or-processed-{VERSION}"
NATIVE_CONTRACT = f"wearable-native-no-unit-conversion-{VERSION}"
CACHE_CONTRACT = f"spotr-global-fp32-intervals-{VERSION}"
TEXT_CONTRACT = f"biolord-separate-text-{VERSION}"

SPOTR_ID = "5gyyyyy/SPOTR"
SPOTR_REVISION = "5a963ffaca341c6d875c69713133de526f6b905b"
SPOTR_CODE_REVISION = "57b3c9f47cc7d20bc9852b84df9545f1dbb887f6"
SPOTR_CHECKPOINT_SHA256 = "94ac49b21c9e22391fe8eb2869a057b1c3ddf776497b16e5e91491e4e1dbdbea"
SPOTR_CONFIG = {"encoder_dim": 512, "patch_size": 100, "num_heads": 8, "n_layers": 12}
SPOTR_MAX_CHANNELS = 128
SPOTR_MAX_SAMPLES = 12800
TEXT_ID = "FremyCompany/BioLORD-2023"
TEXT_REVISION = "167aab527b238a50ca65224e6319215d2ff4fc9f"
TEXT_IDENTITY = {"contract": TEXT_CONTRACT, "model_id": TEXT_ID, "revision": TEXT_REVISION,
                 "pooling": "sentence-transformers-default", "normalize": True, "dimension": 768}

ARCHITECTURE = {"signal_encoder": SPOTR_ID, "encoder_code_revision": SPOTR_CODE_REVISION,
                "pipeline": ["validate", "clean_raw_only", "stack_channels", "frozen_encoding", "decision_head"],
                "signal_width": 512, "sampling": "declared-native-rate",
                "format": "one-aligned-recording", "minimum_samples": 2,
                "signal_types": ["eeg", "ieeg", "ecg", "ppg"], "signal_tokens": 1,
                "channel_order": "as-supplied", "signal_output": "global-token",
                "encoder_config": SPOTR_CONFIG,
                "required_metadata": ["sensor_type", "sampling_rate_hz", "duration_s"],
                "signal_condition": {"values": ["raw", "processed"], "default": "raw"},
                "text_processing": f"{TEXT_CONTRACT}-normalized-mean-128",
                "text_roles": ["question", "candidate"], "signal_position": "elapsed-start-end-sinusoidal-keys-values",
                "input_schema": INPUT_SCHEMA_VERSION,
                "output": ["choice", "probabilities", "confidence"]}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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
    expected = {"spotr": (SPOTR_ID, SPOTR_REVISION), "text": (TEXT_ID, TEXT_REVISION)}
    if set(assets) != set(expected):
        raise ValueError("Specify exactly spotr and text asset directories")
    return {name: verify_source(assets[name], *spec) for name, spec in expected.items()}
