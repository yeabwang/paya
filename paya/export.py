"""Write self-contained Paya models with verified encoder assets."""

import argparse
import json
import shutil
from pathlib import Path

import torch

from .inputs import ENDING
from .provenance import (
    PUBLIC_API,
    TEXT_ID,
    TEXT_REVISION,
    identity,
    sha256,
    source_provenance,
    validate_calibration,
)
from .text import TextEncoder


def export(checkpoint, assets, output, calibration=None):
    """Export head weights, encoders, text cache, notices, and an integrity manifest.

    ``assets`` is an encoder-only NormWear directory. A supplied calibrator must
    bind the exact checkpoint and public inference contract. Existing calibrated
    state is retained when no new calibrator is supplied. Nonempty output
    directories are never overwritten; optimizer and RNG state are omitted.
    """
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(root)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    current_identity = identity(assets)
    if state.get("identity") != current_identity:
        raise ValueError("Checkpoint encoder or runtime contract differs from export")
    checkpoint_sha256 = sha256(checkpoint)
    temperature = float(state.get("temperature", 1.0))
    if calibration:
        binding = json.loads(Path(calibration).read_text())
        temperature = validate_calibration(binding, checkpoint_sha256)
    root.mkdir(parents=True, exist_ok=True)
    exported_state = {
        "head": state["head"],
        "config": {"head": state["config"].get("head", {})},
    }
    exported_state.update(
        identity=current_identity,
        temperature=temperature,
    )
    torch.save(exported_state, root / "model.pt")
    shutil.copytree(assets, root / "normwear")
    text = TextEncoder()
    text.encode([ENDING])
    text.model.save(str(root / "text_encoder"))
    if text.model_path is not None:
        for name in ("README.md", "LICENSE", "LICENSE.metadata.json", "upstream.json"):
            notice = Path(text.model_path) / name
            if notice.is_file():
                shutil.copy2(notice, root / "text_encoder" / name)
    text_source = {
        "model_id": TEXT_ID,
        "revision": TEXT_REVISION,
        "files": {
            p.relative_to(root / "text_encoder").as_posix(): sha256(p)
            for p in (root / "text_encoder").rglob("*")
            if p.is_file()
        },
    }
    (root / "text_encoder/source.json").write_text(json.dumps(text_source, indent=2))
    text.save(root / "text.pt")
    manifest = {
        "schema_version": 2,
        "source_checkpoint_sha256": checkpoint_sha256,
        "public_api": dict(PUBLIC_API),
        "source_provenance": source_provenance(),
        "temperature": temperature,
        "files": {
            p.relative_to(root).as_posix(): sha256(p)
            for p in root.rglob("*")
            if p.is_file()
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    """Export a selected checkpoint and optional bound calibration file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(export(args.checkpoint, args.assets, args.output, args.calibration))[
            :300
        ]
    )


if __name__ == "__main__":
    main()
