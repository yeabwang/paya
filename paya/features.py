"""Cache frozen FP32 features by signal, preprocessing and encoder identity."""

from pathlib import Path
import hashlib
import json
import sqlite3

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


class SQLiteFeatureCache:
    """Batch native encodes and keep verified FP32 features in one durable store."""

    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript('''PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;
            PRAGMA cache_size=-16384; PRAGMA temp_store=FILE;
            CREATE TABLE IF NOT EXISTS metadata(binding TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS features(id TEXT PRIMARY KEY, values_blob BLOB, digest TEXT);''')

    @torch.no_grad()
    def get_many_or_encode(self, model, prepared):
        binding = json.dumps({"contract":CACHE_CONTRACT,"model":model.identity()},sort_keys=True)
        existing = self.db.execute('SELECT binding FROM metadata').fetchone()
        if existing and existing[0] != binding:
            raise ValueError("Frozen feature cache contract mismatch")
        if not existing:
            with self.db: self.db.execute('INSERT INTO metadata VALUES (?)',(binding,))
        unique = {value.identity:value for value in prepared}
        found = {}
        keys = list(unique)
        for start in range(0,len(keys),500):
            chunk=keys[start:start+500]
            for key,content,digest in self.db.execute(
                    'SELECT id,values_blob,digest FROM features WHERE id IN ('+','.join('?' for _ in chunk)+')',chunk):
                if len(content)!=512*4 or hashlib.sha256(content).hexdigest()!=digest:
                    raise ValueError("Cached feature content changed")
                values=torch.frombuffer(bytearray(content),dtype=torch.float32).clone()
                if not torch.isfinite(values).all():
                    raise ValueError("Nonfinite cached features")
                found[key]=values
        shapes = {}
        for key,value in unique.items():
            if key not in found:
                shape=tuple(model.preprocessor.format(value).samples.shape)
                shapes.setdefault(shape,[]).append((key,value))
        for entries in shapes.values():
            encoded=model.encode_batch([value for _,value in entries]).cpu().float()
            updates=[]
            for (key,_),values in zip(entries,encoded):
                content=values.contiguous().numpy().tobytes()
                updates.append((key,content,hashlib.sha256(content).hexdigest()))
                found[key]=values
            with self.db:
                self.db.executemany('INSERT INTO features VALUES (?,?,?)',updates)
        return torch.stack([found[value.identity] for value in prepared]).to(model.device)

    def get_or_encode(self, model, prepared):
        return self.get_many_or_encode(model,[prepared])


def open_feature_cache(config):
    if not config.get("feature_cache"):
        return None
    kind=config.get("feature_cache_format","files")
    if kind=="sqlite": return SQLiteFeatureCache(config["feature_cache"])
    if kind=="files": return FeatureCache(config["feature_cache"])
    raise ValueError("feature_cache_format must be files or sqlite")
