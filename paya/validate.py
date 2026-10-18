"""Validate prepared split integrity, group separation, and spectral supervision."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.signal import welch

from .inputs import prepare_signal, validate_question
from .provenance import sha256

TASKS = ("wesad", "uci_har", "drive_fatigue", "epilepsy", "ecg_abnormal", "mitbih")
BANDS = (("delta", 1, 4), ("theta", 4, 8), ("alpha", 8, 13), ("beta", 13, 30))


def spectral_label(signal, rate):
    """Return the strongest defined EEG band and summed Welch powers.

    Use the physical sample rate and half-open band intervals across all
    channels. The totals are supervision/audit evidence, not model inputs.
    """
    frequency, power = welch(
        signal, fs=rate, nperseg=min(signal.shape[-1], int(2 * rate)), axis=-1
    )
    totals = [
        float(power[..., (frequency >= low) & (frequency < high)].sum())
        for _, low, high in BANDS
    ]
    return int(np.argmax(totals)), totals


def validate(root, output):
    """Check hashes, unique windows, independent groups, questions, and spectral labels.

    Require exactly the prepared train/selection/calibration splits and all six
    source tasks. Recompute spectral supervision from waveform arrays. Write
    and return an audit record; failures raise before a successful audit is saved.
    """
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported dataset manifest schema")
    if (
        set(manifest["splits"]) != {"train", "selection", "calibration"}
        or (root / "test.jsonl").exists()
    ):
        raise ValueError("Only active training/selection/calibration data belong here")
    arrays = {}
    for filename, digest in manifest["signals"].items():
        path = (root / filename).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != digest:
            raise ValueError("Measurement provenance mismatch")
        arrays[filename] = np.load(path, mmap_mode="r")
    groups, ids, texts, counts, window_counts = {}, {}, set(), Counter(), Counter()
    spectral_checked = 0
    for split, digest in manifest["splits"].items():
        if sha256(root / f"{split}.jsonl") != digest:
            raise ValueError("Question manifest changed")
        for line in (root / f"{split}.jsonl").read_text().splitlines():
            r = json.loads(line)
            window_counts[split] += 1
            if r["source_id"] in ids:
                raise ValueError("Repeated window in active dataset")
            ids[r["source_id"]] = split
            if r["group_id"] in groups and groups[r["group_id"]] != split:
                raise ValueError("Group leakage")
            groups[r["group_id"]] = split
            raw = arrays[r["signal_file"]][r["index"]][r["valid_channels"]]
            prepare_signal(
                {
                    "sampling_rate_hz": r["sampling_rate_hz"],
                    "channels": [
                        {**m, "samples": x} for m, x in zip(r["channels"], raw)
                    ],
                }
            )
            for q in r["questions"]:
                validate_question(q["question"], q["choices"])
                if not 0 <= q["label"] < len(q["choices"]):
                    raise ValueError("Target outside answer set")
                if q["heldout_question"] in q["templates"]:
                    raise ValueError("Held-out wording in training templates")
                counts[r["dataset_id"], split, q["family"], q["label"]] += 1
                texts.update(q["choices"])
                texts.update(q["templates"])
                texts.add(q["heldout_question"])
                if q["family"] in ("eeg_band", "eeg_delta_theta"):
                    band, totals = spectral_label(raw, r["sampling_rate_hz"])
                    expected = (
                        band
                        if q["family"] == "eeg_band"
                        else int(totals[1] > totals[0])
                    )
                    if q["label"] != expected:
                        raise ValueError("Spectral label disagrees with waveform")
                    np.testing.assert_allclose(
                        totals, r["evidence_audit_only"]["band_power"], rtol=1e-6
                    )
                    spectral_checked += 1
    if {k[0] for k in counts} != set(TASKS):
        raise ValueError("Original six tasks are not all present")
    if not texts <= set(json.loads((root / "texts.json").read_text())):
        raise ValueError("Text list is incomplete")
    result = {
        "status": "integrity_passed",
        "windows": dict(window_counts),
        "unique_windows": len(ids),
        "questions": sum(counts.values()),
        "spectral_targets_recomputed": spectral_checked,
        "label_counts": {"/".join(map(str, k)): v for k, v in counts.items()},
        "manifest_sha256": sha256(root / "manifest.json"),
    }
    Path(output).write_text(json.dumps(result, indent=2))
    return result


def main():
    """Validate a prepared dataset and write its integrity audit."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = validate(a.data, a.output)
    print(json.dumps({k: v for k, v in r.items() if k != "label_counts"}))


if __name__ == "__main__":
    main()
