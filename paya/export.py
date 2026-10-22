"""Export models with verified weights and encoder assets."""

import argparse
import json
from pathlib import Path
import shutil

import yaml

from .checkpoint import load_checkpoint, save_checkpoint
from .model import PayaModel
from .provenance import RUNTIME_CONTRACT, sha256, verify_assets


def export_model(model, assets, output):
    root = Path(output).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Export destination must be empty")
    if any(root.is_relative_to(Path(path).resolve()) for path in assets.values()):
        raise ValueError("Export cannot be nested inside encoder assets")
    if verify_assets(assets) != model.asset_identity:
        raise ValueError("Export assets do not match the trained model")
    root.mkdir(parents=True, exist_ok=True)
    for name, path in assets.items():
        source = Path(path).resolve()
        for entry in source.rglob("*"):
            if not entry.resolve().is_relative_to(source):
                raise ValueError("Asset tree contains an escaping link")
        shutil.copytree(
            source, root / name, ignore=shutil.ignore_patterns(".cache", "__pycache__")
        )
    project = Path(__file__).resolve().parent.parent
    shutil.copytree(
        project / "paya",
        root / "code" / "paya",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copyfile(project / "requirements-model.txt", root / "requirements-model.txt")
    save_checkpoint(root / "decision.pt", model)
    config = {
        "preparation": model.identity()["preparation"]["config"],
        "head": model.head.config,
    }
    (root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    manifest = {
        "contract": RUNTIME_CONTRACT,
        "identity": model.identity(),
        "calibrated": model.calibration is not None,
        "files": {
            path.relative_to(root).as_posix(): sha256(path)
            for path in root.rglob("*")
            if path.is_file()
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/paya.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model = PayaModel.from_assets(
        config["assets"], config.get("preparation"), config.get("head")
    )
    load_checkpoint(args.checkpoint, model)
    print(
        json.dumps(export_model(model, config["assets"], args.output), indent=2)[:1000]
    )
