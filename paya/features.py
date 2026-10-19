"""Cache frozen FP32 features by signal, preprocessing and encoder identity."""

from pathlib import Path

import torch

from .encoders import EncodedSignal
from .provenance import CACHE_CONTRACT, fingerprint


def tensor_digest(values, intervals_s):
    import hashlib
    digest = hashlib.sha256()
    for tensor in (values, intervals_s):
        digest.update(tensor.contiguous().numpy().tobytes())
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
            values, intervals = payload["values"], payload["intervals_s"]
            if values.ndim != 2 or values.shape[1] != 512 or intervals.shape != (len(values), 2):
                raise ValueError("Frozen feature cache shape mismatch")
            if not 1 <= len(values) <= model.preprocessor.config.max_signal_tokens:
                raise ValueError("Frozen feature cache exceeds the token budget")
            if values.dtype != torch.float32 or intervals.dtype != torch.float32:
                raise ValueError("Frozen feature cache must preserve FP32 computation")
            if not torch.isfinite(values).all() or not torch.isfinite(intervals).all():
                raise ValueError("Nonfinite cached features")
            if (intervals[:, 0] < 0).any() or (intervals[:, 1] <= intervals[:, 0]).any():
                raise ValueError("Cached interval times are invalid")
            if tensor_digest(values, intervals) != payload.get("digest"):
                raise ValueError("Cached feature content changed")
            return EncodedSignal(values.to(model.device), intervals.to(model.device))
        encoded = model.encode(prepared)
        values, intervals = encoded.values.cpu().float(), encoded.intervals_s.cpu().float()
        temporary = path.with_suffix(".partial")
        torch.save({"binding": binding, "values": values, "intervals_s": intervals,
                    "digest": tensor_digest(values, intervals)}, temporary)
        temporary.replace(path)
        return encoded
