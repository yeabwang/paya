"""Checksummed FP32 signal-token caches using the public batch-one encoding path."""

import hashlib
import json
from pathlib import Path
import sqlite3

import torch

from .atomic import atomic_output
from .provenance import fingerprint


def tensor_digest(values, mask):
    content = values.detach().cpu().contiguous().numpy().tobytes()
    validity = mask.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(content + validity).hexdigest()


def validate_tokens(values, mask):
    if (
        values.ndim != 2
        or values.shape[1] != 512
        or not len(values)
        or values.dtype != torch.float32
        or mask.dtype != torch.bool
        or mask.shape != values.shape[:1]
        or not mask.any()
        or not mask[0]
        or not torch.isfinite(values).all()
    ):
        raise ValueError("Signal token cache shape, precision or mask mismatch")


class FeatureCache:
    """Keep each recording's tokens and validity mask in one atomic file."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def close(self):
        pass

    def get_or_encode(self, model, prepared):
        binding = {"model": model.feature_identity(), "prepared": prepared.identity}
        path = self.directory / (fingerprint(binding) + ".pt")
        if path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload.get("binding") != binding:
                raise ValueError("Signal token cache contract mismatch")
            values, mask = payload["values"], payload["mask"]
            validate_tokens(values, mask)
            if tensor_digest(values, mask) != payload.get("digest"):
                raise ValueError("Signal token cache content changed")
            return values.to(model.device), mask.to(model.device)
        values, mask = model.encode(prepared)
        validate_tokens(values, mask)
        with atomic_output(path) as output:
            torch.save(
                {
                    "binding": binding,
                    "values": values.cpu(),
                    "mask": mask.cpu(),
                    "digest": tensor_digest(values, mask),
                },
                output,
            )
        return values, mask


class SQLiteFeatureCache:
    """Store variable-length tokens and masks, independently of head settings."""

    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript("""
            PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA cache_size=-16384;
            CREATE TABLE IF NOT EXISTS bindings(id TEXT PRIMARY KEY, identity TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS tokens(binding TEXT, waveform TEXT, count INTEGER,
                values_blob BLOB, mask_blob BLOB, digest TEXT, PRIMARY KEY(binding,waveform));
        """)

    def close(self):
        self.db.close()

    def get_or_encode(self, model, prepared):
        identity = model.feature_identity()
        binding = fingerprint(identity)
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO bindings VALUES (?,?)",
                (binding, json.dumps(identity, sort_keys=True)),
            )
        row = self.db.execute(
            "SELECT count,values_blob,mask_blob,digest FROM tokens WHERE binding=? AND waveform=?",
            (binding, prepared.identity),
        ).fetchone()
        if row is not None:
            count, content, mask_bytes, digest = row
            if (
                count < 1
                or len(content) != count * 512 * 4
                or len(mask_bytes) != count
                or hashlib.sha256(content + mask_bytes).hexdigest() != digest
            ):
                raise ValueError("Signal token cache content or shape changed")
            values = (
                torch.frombuffer(bytearray(content), dtype=torch.float32)
                .clone()
                .reshape(count, 512)
            )
            mask = torch.frombuffer(bytearray(mask_bytes), dtype=torch.bool).clone()
            validate_tokens(values, mask)
            return values.to(model.device), mask.to(model.device)
        values, mask = model.encode(prepared)
        validate_tokens(values, mask)
        content = values.detach().cpu().contiguous().numpy().tobytes()
        mask_bytes = mask.detach().cpu().contiguous().numpy().tobytes()
        with self.db:
            self.db.execute(
                "INSERT INTO tokens VALUES (?,?,?,?,?,?)",
                (
                    binding,
                    prepared.identity,
                    len(values),
                    content,
                    mask_bytes,
                    hashlib.sha256(content + mask_bytes).hexdigest(),
                ),
            )
        return values, mask


def open_feature_cache(config):
    if not config.get("feature_cache"):
        return None
    kind = config.get("feature_cache_format", "sqlite")
    if kind == "sqlite":
        return SQLiteFeatureCache(config["feature_cache"])
    if kind == "files":
        return FeatureCache(config["feature_cache"])
    raise ValueError("feature_cache_format must be files or sqlite")
