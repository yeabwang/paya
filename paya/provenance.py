"""Paya versions, frozen-model identities and artifact checksums."""

import hashlib
import json
from pathlib import Path

PACKAGE_VERSION = "1.0.0"
VERSION = f"v{PACKAGE_VERSION.split('.')[0]}"
INPUT_SCHEMA_VERSION = int(VERSION[1:])
RUNTIME_CONTRACT = f"paya-body-decision-{VERSION}"
PREPARATION_CONTRACT = f"neurokit-native-physiology-{VERSION}"
NATIVE_CONTRACT = f"wearable-native-cleaning-{VERSION}"
CACHE_CONTRACT = f"two-encoder-fp32-fixed-adapter-{VERSION}"
TEXT_CONTRACT = f"biolord-separate-text-{VERSION}"

NORMWEAR_ID = "mosaic-laboratory/normwear"
NORMWEAR_REVISION = "c35a3d1ecda1cb971f1f89caaa3ee4b4c6ad8e84"
MANTIS_ID = "paris-noah/MantisV2"
MANTIS_REVISION = "8f6ca35cb54ab14b120618943c6fca5ddf5a76a6"
MANTIS_IMPLEMENTATION = "mantis-tsfm==1.1.0"
TEXT_ID = "FremyCompany/BioLORD-2023"
TEXT_REVISION = "167aab527b238a50ca65224e6319215d2ff4fc9f"
TEXT_IDENTITY = {"contract": TEXT_CONTRACT, "model_id": TEXT_ID, "revision": TEXT_REVISION,
                 "pooling": "sentence-transformers-default", "normalize": True, "dimension": 768}

ARCHITECTURE = {"router": f"waveform-conv16-conv32-question128-gate64-top1-{VERSION}",
                "signal_width": 768, "adapters": ["identity768", "right-zero-pad512-to768"],
                "mantis_layer": 2, "mantis_output": "combined",
                "text_processing": f"{TEXT_CONTRACT}-normalized-mean-128",
                "ending": "The answer is:", "input_schema": INPUT_SCHEMA_VERSION,
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
    expected = {"normwear": (NORMWEAR_ID, NORMWEAR_REVISION),
                "mantis": (MANTIS_ID, MANTIS_REVISION), "text": (TEXT_ID, TEXT_REVISION)}
    if set(assets) != set(expected):
        raise ValueError("Specify exactly normwear, mantis, and text asset directories")
    return {name: verify_source(assets[name], *spec) for name, spec in expected.items()}
