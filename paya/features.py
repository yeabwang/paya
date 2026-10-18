"""Extract and resume memory-mapped frozen signal evidence for head training."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .inputs import prepare_signal
from .model import GroundedChoiceHead, build_model
from .provenance import EVIDENCE_CONTRACT, identity, sha256


def evidence_identity(assets):
    """Describe the frozen encoder and versioned pooling/storage computation."""
    model = identity(assets)
    return {
        "contract": EVIDENCE_CONTRACT,
        "signal_source": model["signal_source"],
        "time_bins": 8,
        "encoder_precision": "fp32",
        "storage_precision": "float16",
    }


def load_cache(path, data_root, assets):
    """Verify a cache and return source-ID lookup over its memory-mapped array.

    The descriptor binds the encoder, dataset manifest, array shape, dtype, and
    checksum. Each lookup returns [channels, 105, 768] float16 evidence.
    """
    path = Path(path)
    descriptor = json.loads(path.read_text())
    if descriptor["identity"] != evidence_identity(assets) or descriptor[
        "data_manifest"
    ] != sha256(Path(data_root) / "manifest.json"):
        raise ValueError(
            "Frozen evidence cache does not match encoder, physical inputs or dataset"
        )
    source = (path.parent / descriptor["file"]).resolve()
    if not source.is_relative_to(path.parent.resolve()):
        raise ValueError("Evidence file escapes cache directory")
    if sha256(source) != descriptor["sha256"]:
        raise ValueError("Frozen evidence file checksum changed")
    memory = np.load(source, mmap_mode="r")
    if memory.dtype != np.float16 or memory.shape != tuple(descriptor["shape"]):
        raise ValueError("Invalid evidence array")

    class Evidence:
        def __getitem__(self, key):
            start, count = descriptor["windows"][key]
            return torch.from_numpy(np.array(memory[start : start + count], copy=True))

    return Evidence()


@torch.inference_mode()
def extract(data_root, assets, output, device="cuda", batch_size=8):
    """Extract frozen evidence with bounded batch memory and verifiable resume.

    Write an FP16 NumPy array, progress marker, and final JSON descriptor.
    The current cache has 8 time bins, 13 scale bins, one CLS token and 768
    features. Existing completed descriptors are immutable; partial arrays
    require matching progress provenance before continuing.
    """
    root, output = Path(data_root), Path(output)
    if output.exists():
        raise FileExistsError("Evidence outputs are immutable; choose a new descriptor")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_text())
    for filename, expected in manifest["signals"].items():
        if sha256(root / filename) != expected:
            raise ValueError("Prepared measurements changed")
    groups, arrays, windows = defaultdict(list), {}, {}
    for split in ("train", "selection", "calibration"):
        if sha256(root / f"{split}.jsonl") != manifest["splits"][split]:
            raise ValueError("Prepared split changed")
        for line in (root / f"{split}.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["source_id"] in windows:
                raise ValueError("Duplicate waveform crosses cache boundaries")
            windows[row["source_id"]] = None
            groups[row["signal_file"], tuple(row["valid_channels"])].append(row)
    sensors = sum(
        len(row["valid_channels"]) for rows in groups.values() for row in rows
    )
    filename = output.with_suffix(".npy")
    progress_path = output.with_suffix(".progress.json")
    expected = {
        "identity": evidence_identity(assets),
        "data_manifest": sha256(root / "manifest.json"),
        "shape": [sensors, 105, 768],
    }
    offset, done = 0, 0
    if filename.exists():
        if not progress_path.exists():
            raise ValueError("Partial evidence file lacks a verified resume marker")
        progress = json.loads(progress_path.read_text())
        if progress["provenance"] != expected:
            raise ValueError("Evidence resume inputs changed")
        windows.update(progress["windows"])
        offset, done = progress["offset"], len(progress["windows"])
        memory = np.lib.format.open_memmap(filename, mode="r+")
        if memory.shape != tuple(expected["shape"]):
            raise ValueError("Partial evidence shape changed")
    else:
        memory = np.lib.format.open_memmap(
            filename, mode="w+", dtype=np.float16, shape=(sensors, 105, 768)
        )

    def save_progress(array):
        """Flush evidence before atomically recording its completed window offsets."""
        array.flush()
        temporary = progress_path.with_suffix(".partial")
        temporary.write_text(
            json.dumps(
                {
                    "provenance": expected,
                    "offset": offset,
                    "windows": {k: v for k, v in windows.items() if v is not None},
                }
            )
        )
        temporary.replace(progress_path)

    save_progress(memory)
    model = build_model(assets, {}).to(device).eval()
    for (signal_file, valid), rows in groups.items():
        rows = [r for r in rows if windows[r["source_id"]] is None]
        arrays.setdefault(signal_file, np.load(root / signal_file, mmap_mode="r"))
        for begin in range(0, len(rows), batch_size):
            current = rows[begin : begin + batch_size]
            signals = []
            for row in current:
                raw = arrays[signal_file][row["index"]][list(valid)]
                x, _ = prepare_signal(
                    {
                        "sampling_rate_hz": row["sampling_rate_hz"],
                        "channels": [
                            {**m, "samples": v} for m, v in zip(row["channels"], raw)
                        ],
                    }
                )
                signals.append(x)
            patches = model.core(torch.stack(signals).to(device))
            features = GroundedChoiceHead.signal_memory(patches).cpu().half().numpy()
            if features.shape[2:] != (105, 768) or not np.isfinite(features).all():
                raise ValueError("Unsupported or nonfinite evidence grid")
            for row, values in zip(current, features):
                memory[offset : offset + len(values)] = values
                windows[row["source_id"]] = [offset, len(values)]
                offset += len(values)
            done += len(current)
            if done % 400 < len(current):
                save_progress(memory)
                print(
                    json.dumps(
                        {"encoded_windows": done, "total_windows": len(windows)}
                    ),
                    flush=True,
                )
    memory.flush()
    del memory
    descriptor = {
        "identity": evidence_identity(assets),
        "data_manifest": sha256(root / "manifest.json"),
        "file": filename.name,
        "sha256": sha256(filename),
        "shape": [sensors, 105, 768],
        "windows": windows,
    }
    output.write_text(json.dumps(descriptor))
    print(
        json.dumps(
            {
                "encoded_windows": done,
                "sensors": sensors,
                "bytes": filename.stat().st_size,
            }
        ),
        flush=True,
    )
    return descriptor


def main():
    """Extract or resume frozen signal evidence for an explicit dataset and encoder."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("cache/paya-evidence.json"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--threads", type=int, default=6)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    extract(a.data, a.assets, a.output, a.device, a.batch_size)


if __name__ == "__main__":
    main()
