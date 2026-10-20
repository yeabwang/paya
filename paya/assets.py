"""Download, verify and convert the pinned SPOTR-Base encoder checkpoint."""

import argparse
import json
from pathlib import Path
import shutil

import torch
from safetensors.torch import save_file

from .provenance import (SPOTR_ID, SPOTR_REVISION, SPOTR_CODE_REVISION,
                         SPOTR_CHECKPOINT_SHA256, SPOTR_CONFIG, sha256, verify_source)
from .vendor.spotr import SPOTREncoder


def download_spotr(directory, checkpoint=None):
    """Verify published tensor weights before converting them to safetensors."""
    from huggingface_hub import hf_hub_download
    root = Path(directory)
    if (root/"source.json").exists():
        return verify_source(root, SPOTR_ID, SPOTR_REVISION)
    root.mkdir(parents=True, exist_ok=True)
    options = dict(repo_id=SPOTR_ID, revision=SPOTR_REVISION, token=False, cache_dir="cache/huggingface")
    checkpoint = Path(checkpoint) if checkpoint else Path(hf_hub_download(filename="SPOTREncoder.pt", **options))
    if sha256(checkpoint) != SPOTR_CHECKPOINT_SHA256:
        raise ValueError("SPOTR checkpoint differs from the published SHA-256")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    core = SPOTREncoder(**SPOTR_CONFIG)
    core.load_state_dict(state, strict=True)
    temporary = root/"model.safetensors.partial"
    save_file({name: value.contiguous() for name, value in state.items()}, str(temporary))
    temporary.replace(root/"model.safetensors")
    (root/"config.json").write_text(json.dumps(SPOTR_CONFIG, indent=2), encoding="utf-8")
    card = hf_hub_download(filename="README.md", **options)
    shutil.copyfile(card, root/"README.md")
    shutil.copyfile(Path(__file__).parent/"vendor"/"LICENSE", root/"LICENSE")
    source = {"model_id": SPOTR_ID, "revision": SPOTR_REVISION,
              "implementation_revision": SPOTR_CODE_REVISION,
              "upstream_checkpoint": "SPOTREncoder.pt", "upstream_sha256": SPOTR_CHECKPOINT_SHA256,
              "files": {name: sha256(root/name) for name in ("model.safetensors", "config.json", "README.md", "LICENSE")}}
    (root/"source.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    return source


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spotr", type=Path, default=Path("third-party/spotr"))
    parser.add_argument("--checkpoint", type=Path, help="Optional local copy of the published SPOTREncoder.pt")
    args = parser.parse_args()
    print(json.dumps(download_spotr(args.spotr, args.checkpoint), indent=2))
