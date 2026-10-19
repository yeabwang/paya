"""Save router/head weights and validate model and calibration identities."""

import hashlib
from pathlib import Path

import torch

from .provenance import RUNTIME_CONTRACT


def learned_state(model):
    return {name: {key: tensor.detach().cpu().clone() for key, tensor in getattr(model, name).state_dict().items()}
            for name in ("router", "head")}


def learned_digest(model):
    digest = hashlib.sha256()
    for module, state in learned_state(model).items():
        for name, tensor in sorted(state.items()):
            digest.update(f"{module}.{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
            digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(path, model, training=None):
    """Atomically save learned weights and optional resumable training state."""
    temperature = float(model.temperature)
    if not 0 < temperature < float("inf"):
        raise ValueError("Invalid checkpoint temperature")
    if model.calibration is None and temperature != 1.0:
        raise ValueError("A non-unit temperature needs outcome-fitted calibration provenance")
    if model.calibration and model.calibration["learned_sha256"] != learned_digest(model):
        raise ValueError("Calibration does not match current router/head weights")
    if model.calibration and abs(model.calibration["temperature"]-temperature) > 1e-7:
        raise ValueError("Calibration temperature changed")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"contract": RUNTIME_CONTRACT, "identity": model.identity(),
               "learned": learned_state(model), "temperature": temperature,
               "calibration": model.calibration, "training": training}
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, model):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("contract") != RUNTIME_CONTRACT or payload.get("identity") != model.identity():
        raise ValueError("Checkpoint is not the exact current two-encoder architecture/assets/preparation")
    for module in ("router", "head"):
        getattr(model, module).load_state_dict(payload["learned"][module], strict=True)
    temperature = float(payload["temperature"])
    if not 0 < temperature < float("inf"):
        raise ValueError("Invalid checkpoint temperature")
    calibration = payload.get("calibration")
    if calibration is None and temperature != 1.0:
        raise ValueError("A non-unit temperature needs outcome-fitted calibration provenance")
    if calibration and (calibration["learned_sha256"] != learned_digest(model)
                        or abs(calibration["temperature"]-temperature) > 1e-7):
        raise ValueError("Calibration binding differs from the saved weights/temperature")
    model.temperature.fill_(temperature)
    model.calibration = calibration
    model.eval()
    return payload
