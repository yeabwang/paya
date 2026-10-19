"""Download and verify the pinned MantisV2 checkpoint."""

import argparse
import json
from pathlib import Path

from .provenance import MANTIS_ID, MANTIS_REVISION, MANTIS_IMPLEMENTATION, sha256, verify_source


def download_mantis(directory):
    from huggingface_hub import snapshot_download
    root = Path(directory)
    if (root / "source.json").exists():
        return verify_source(root, MANTIS_ID, MANTIS_REVISION)
    snapshot_download(MANTIS_ID, revision=MANTIS_REVISION, local_dir=str(root),
                      allow_patterns=["config.json", "model.safetensors", "README.md"])
    source = {"model_id": MANTIS_ID, "revision": MANTIS_REVISION,
              "implementation": MANTIS_IMPLEMENTATION,
              "files": {name: sha256(root/name) for name in ("config.json", "model.safetensors", "README.md")}}
    (root/"source.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    return source


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mantis", type=Path, default=Path("third-party/mantis"))
    print(json.dumps(download_mantis(parser.parse_args().mantis), indent=2))
