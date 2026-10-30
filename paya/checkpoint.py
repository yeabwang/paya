"""Save decision-head weights and validate model and calibration identities."""

import hashlib
from pathlib import Path

import torch

from .provenance import RUNTIME_CONTRACT, source_identity, runtime_identity, fingerprint
from .atomic import atomic_output


def learned_state(model):
    return {
        "head": {
            key: tensor.detach().cpu().clone()
            for key, tensor in model.head.state_dict().items()
        }
    }


def learned_digest(model):
    digest = hashlib.sha256()
    for module, state in learned_state(model).items():
        for name, tensor in sorted(state.items()):
            digest.update(
                f"{module}.{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode()
            )
            digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(path, model, training=None):
    """Atomically save learned weights and optional resumable training state."""
    temperature = float(model.temperature)
    if not 0 < temperature < float("inf"):
        raise ValueError("Invalid checkpoint temperature")
    if model.calibration is None and temperature != 1.0:
        raise ValueError(
            "A non-unit temperature needs outcome-fitted calibration provenance"
        )
    if model.calibration and model.calibration["learned_sha256"] != learned_digest(
        model
    ):
        raise ValueError("Calibration does not match current decision-head weights")
    if model.calibration and abs(model.calibration["temperature"] - temperature) > 1e-7:
        raise ValueError("Calibration temperature changed")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "contract": RUNTIME_CONTRACT,
        "identity": model.identity(),
        "identity_sha256": fingerprint(model.identity()),
        "learned": learned_state(model),
        "learned_sha256": learned_digest(model),
        "source_sha256": source_identity(),
        "runtime": runtime_identity(),
        "temperature": temperature,
        "calibration": model.calibration,
        "training": training,
    }
    with atomic_output(path) as output:
        torch.save(payload, output)


def load_checkpoint(path, model):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("contract") != RUNTIME_CONTRACT
        or payload.get("identity") != model.identity()
        or payload.get("identity_sha256") != fingerprint(model.identity())
    ):
        raise ValueError(
            "Checkpoint is not the exact current architecture/assets/preparation"
        )
    if set(payload.get("learned", {})) != {"head"}:
        raise ValueError("Checkpoint must contain only decision-head weights")
    model.head.load_state_dict(payload["learned"]["head"], strict=True)
    if learned_digest(model) != payload.get("learned_sha256"):
        raise ValueError("Decision-head checksum changed")
    temperature = float(payload["temperature"])
    if not 0 < temperature < float("inf"):
        raise ValueError("Invalid checkpoint temperature")
    calibration = payload.get("calibration")
    if calibration is None and temperature != 1.0:
        raise ValueError(
            "A non-unit temperature needs outcome-fitted calibration provenance"
        )
    if calibration and (
        calibration["learned_sha256"] != learned_digest(model)
        or abs(calibration["temperature"] - temperature) > 1e-7
    ):
        raise ValueError(
            "Calibration binding differs from the saved weights/temperature"
        )
    model.temperature.fill_(temperature)
    model.calibration = calibration
    model.eval()
    return payload
