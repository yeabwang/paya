"""FP32 frozen features bound to the full prepared-view and encoder contract."""

from pathlib import Path

import torch

from .encoders import EncodedSignal
from .provenance import fingerprint

CACHE_CONTRACT = "two-encoder-fp32-fixed-adapter-v2"


def tensor_digest(values, positions):
    import hashlib
    digest = hashlib.sha256()
    for tensor in (values, positions):
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


class FeatureCache:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def get_or_encode(self, model, prepared, expert):
        binding = {"contract": CACHE_CONTRACT, "model": model.identity(),
                   "prepared": prepared.identity, "expert": expert}
        path = self.directory / (fingerprint(binding) + ".pt")
        if path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload.get("binding") != binding:
                raise ValueError("Frozen feature cache contract mismatch")
            values, positions = payload["values"], payload["positions"]
            if values.ndim != 2 or values.shape[1] != 768 or positions.shape != (len(values), 3):
                raise ValueError("Frozen feature cache shape mismatch")
            if not 1 <= len(values) <= model.preprocessor.config.max_signal_tokens:
                raise ValueError("Frozen feature cache exceeds the token budget")
            if values.dtype != torch.float32 or positions.dtype != torch.float32:
                raise ValueError("Frozen feature cache must preserve FP32 computation")
            if not torch.isfinite(values).all() or not torch.isfinite(positions).all():
                raise ValueError("Nonfinite cached features")
            if tensor_digest(values, positions) != payload.get("digest"):
                raise ValueError("Cached feature content changed")
            return EncodedSignal(values.to(model.device), positions.to(model.device))
        encoded = model.encode(prepared, expert)
        values, positions = encoded.values.cpu().float(), encoded.positions.cpu().float()
        temporary = path.with_suffix(".partial")
        torch.save({"binding": binding, "values": values, "positions": positions,
                    "digest": tensor_digest(values, positions)}, temporary)
        temporary.replace(path)
        return encoded
