"""Cache frozen FP32 features by signal, preprocessing and encoder identity."""

from pathlib import Path

import torch

from .encoders import EncodedSignal
from .provenance import CACHE_CONTRACT, fingerprint


def tensor_digest(values, times):
    import hashlib
    digest = hashlib.sha256()
    for tensor in (values, times):
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


class FeatureCache:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def get_or_encode(self, model, cleaned, expert):
        binding = {"contract": CACHE_CONTRACT, "model": model.identity(),
                   "cleaned": cleaned.identity, "expert": expert}
        path = self.directory / (fingerprint(binding) + ".pt")
        if path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload.get("binding") != binding:
                raise ValueError("Frozen feature cache contract mismatch")
            values, times = payload["values"], payload["times"]
            if values.ndim != 2 or values.shape[1] != 768 or times.shape != (len(values),):
                raise ValueError("Frozen feature cache shape mismatch")
            if not 1 <= len(values) <= model.preprocessor.config.max_signal_tokens:
                raise ValueError("Frozen feature cache exceeds the token budget")
            if values.dtype != torch.float32 or times.dtype != torch.float32:
                raise ValueError("Frozen feature cache must preserve FP32 computation")
            if not torch.isfinite(values).all() or not torch.isfinite(times).all():
                raise ValueError("Nonfinite cached features")
            if ((times < 0) | (times > 1)).any():
                raise ValueError("Cached relative times must lie within [0,1]")
            if tensor_digest(values, times) != payload.get("digest"):
                raise ValueError("Cached feature content changed")
            return EncodedSignal(values.to(model.device), times.to(model.device))
        encoded = model.encode(cleaned, expert)
        values, times = encoded.values.cpu().float(), encoded.times.cpu().float()
        temporary = path.with_suffix(".partial")
        torch.save({"binding": binding, "values": values, "times": times,
                    "digest": tensor_digest(values, times)}, temporary)
        temporary.replace(path)
        return encoded
