"""Package only active source/configuration; credentials, retired code and weights are excluded."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--with-data-assets", action="store_true")
    p.add_argument("--with-data-metadata", action="store_true")
    a = p.parse_args()
    root = Path(__file__).resolve().parent.parent
    files = [root / name for name in ("pyproject.toml", "README.md")]
    for name in ("paya", "config", "tests", "deploy", "benchmarks"):
        files.extend(
            x
            for x in (root / name).rglob("*")
            if x.is_file() and "__pycache__" not in x.parts
        )
    for name in ("normwear", "biolord"):
        files.extend(
            x
            for x in (root / "third-party" / name).rglob("*")
            if x.is_file()
            and (x.suffix in (".json", ".md", ".txt") or x.name == "LICENSE")
        )
    if a.with_data_assets:
        for name in (
            "third-party/normwear",
            "third-party/biolord",
            "data/body-signals-v1",
        ):
            files.extend(
                x
                for x in (root / name).rglob("*")
                if x.is_file() and "__pycache__" not in x.parts
            )
    elif a.with_data_metadata:
        files.extend((root / "data/body-signals-v1").glob("*.json*"))
    files = sorted(set(files))
    manifest = {
        str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in files
    }
    with tarfile.open(a.output, "w:gz") as archive:
        for f in files:
            archive.add(f, arcname="paya/" + str(f.relative_to(root)), recursive=False)
    a.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            {
                "files": len(files),
                "bytes": a.output.stat().st_size,
                "sha256": hashlib.sha256(a.output.read_bytes()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
