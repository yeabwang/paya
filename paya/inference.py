"""Load an exported model for inference."""

import json
from pathlib import Path

from .checkpoint import load_checkpoint
from .model import PayaModel
from .provenance import RUNTIME_CONTRACT, contained_file, sha256
from .vendor import signal

__all__ = ["load_model"]


def load_model(directory, device="cpu"):
    root = Path(directory).resolve()
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("contract") != RUNTIME_CONTRACT:
        raise ValueError("Unsupported model architecture")
    for name, expected in manifest["files"].items():
        if sha256(contained_file(root, name)) != expected:
            raise ValueError(f"Model file checksum changed: {name}")
    config = json.loads((root/"config.json").read_text(encoding="utf-8"))
    assets = {name: root/name for name in (signal.ASSET_KEY, "text")}
    model = PayaModel.from_assets(assets, config["preparation"], config["head"], device)
    if manifest["identity"] != model.identity():
        raise ValueError("Model identities differ")
    load_checkpoint(root/"decision.pt", model)
    return model
