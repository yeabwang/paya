"""Versioned runtime contracts, pinned encoder origins, and integrity fingerprints.

Loading checks use explicit computation versions and encoder identities. Exact
source hashes are retained for auditing and strict training resume, so ordinary
formatting and docstring edits do not invalidate inference artifacts.
"""

import hashlib
import math
from pathlib import Path

NORMWEAR_ID = "mosaic-laboratory/normwear"
NORMWEAR_REVISION = "c35a3d1ecda1cb971f1f89caaa3ee4b4c6ad8e84"
TEXT_ID = "FremyCompany/BioLORD-2023"
TEXT_REVISION = "167aab527b238a50ca65224e6319215d2ff4fc9f"
RUNTIME_CONTRACT = "paya-choice-fp32-v1"
EVIDENCE_CONTRACT = "sensor-time-scale-fp16-v1"
PUBLIC_API = {"contract": RUNTIME_CONTRACT, "channel_order": "canonical-physical-v1"}


def sha256(path):
    """Return a file's SHA-256 fingerprint without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_sha256(path):
    """Hash UTF-8 source after normalizing Windows and Unix line endings."""
    return hashlib.sha256(
        Path(path).read_text(encoding="utf-8").encode("utf-8")
    ).hexdigest()


def identity(asset_dir):
    """Describe the runtime, physical input schema, and pinned encoder snapshot.

    The head configuration is stored separately in each checkpoint and its
    tensors are loaded strictly. Bump the runtime contract for a computation
    change; formatting changes belong only in source provenance.
    """
    from .inputs import SCHEMA_VERSION
    from .text import IDENTITY

    return {
        "schema": 2,
        "runtime_contract": RUNTIME_CONTRACT,
        "input_schema": SCHEMA_VERSION,
        "text": IDENTITY,
        "signal_source": sha256(Path(asset_dir) / "source.json"),
    }


def validate_calibration(binding, checkpoint_sha256):
    """Require a calibrator bound to these weights and the supported waveform API."""
    if binding.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("Calibration belongs to different weights")
    if binding.get("public_api") != PUBLIC_API:
        raise ValueError("Calibration belongs to a different public input computation")
    temperature = float(binding["temperature"])
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Calibration temperature must be finite and positive")
    return temperature


def source_provenance():
    """Record current package source fingerprints for audit, without gating loads."""
    root = Path(__file__).parent
    return {
        p.relative_to(root).as_posix(): source_sha256(p)
        for p in sorted(root.rglob("*.py"))
    }
