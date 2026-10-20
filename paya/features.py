"""Cache frozen FP32 features by signal, preprocessing and encoder identity."""

from pathlib import Path

import torch

from .provenance import CACHE_CONTRACT, fingerprint
from .atomic import atomic_output


def tensor_digest(values):
    import hashlib
    digest = hashlib.sha256()
    digest.update(values.contiguous().numpy().tobytes())
    return digest.hexdigest()


class FeatureCache:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def get_or_encode(self, model, prepared):
        binding = {"contract": CACHE_CONTRACT, "model": model.identity(),
                   "prepared": prepared.identity}
        path = self.directory / (fingerprint(binding) + ".pt")
        if path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload.get("binding") != binding:
                raise ValueError("Frozen feature cache contract mismatch")
            values = payload["values"]
            if values.shape != (1, 512):
                raise ValueError("Frozen feature cache shape mismatch")
            if values.dtype != torch.float32:
                raise ValueError("Frozen feature cache must preserve FP32 computation")
            if not torch.isfinite(values).all():
                raise ValueError("Nonfinite cached features")
            if tensor_digest(values) != payload.get("digest"):
                raise ValueError("Cached feature content changed")
            return values.to(model.device)
        encoded = model.encode(prepared)
        values = encoded.cpu().float()
        with atomic_output(path) as output:
            torch.save({"binding": binding, "values": values,
                        "digest": tensor_digest(values)}, output)
        return encoded
