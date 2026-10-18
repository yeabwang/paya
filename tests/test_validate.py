import json

import numpy as np
import pytest

from paya.provenance import sha256
from paya.validate import TASKS, spectral_label, validate


@pytest.fixture
def released_data(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    waveform = np.sin(2 * np.pi * 2 * np.arange(390) / 65).astype(np.float32)
    np.save(root / "signals.npy", np.tile(waveform, (18, 1, 1)))
    texts = set()
    index = 0
    for split in ("train", "selection", "calibration"):
        rows = []
        for task in TASKS:
            questions = [
                {
                    "family": "native",
                    "question": "Which category describes this signal?",
                    "choices": ["Category one.", "Category two."],
                    "label": 0,
                    "templates": ["Which category describes this signal?"],
                    "heldout_question": "Choose the matching category.",
                }
            ]
            evidence = {}
            if task in ("epilepsy", "drive_fatigue"):
                _, powers = spectral_label(waveform[None], 65)
                questions.append(
                    {
                        "family": "eeg_delta_theta",
                        "question": "Which band has greater power?",
                        "choices": ["Delta.", "Theta."],
                        "label": 0,
                        "templates": ["Which band has greater power?"],
                        "heldout_question": "Choose the stronger band.",
                    }
                )
                evidence = {"band_power": powers}
            for question in questions:
                texts.update(question["choices"])
                texts.update(question["templates"])
                texts.add(question["heldout_question"])
            rows.append(
                {
                    "source_id": f"{task}/{split}/window",
                    "dataset_id": task,
                    "group_id": f"{task}/{split}/group",
                    "signal_file": "signals.npy",
                    "index": index,
                    "valid_channels": [0],
                    "sampling_rate_hz": 65,
                    "channels": [{"sensor_type": "unknown"}],
                    "questions": questions,
                    "evidence_audit_only": evidence,
                }
            )
            index += 1
        (root / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    (root / "texts.json").write_text(json.dumps(sorted(texts)))
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "signals": {"signals.npy": sha256(root / "signals.npy")},
                "splits": {
                    split: sha256(root / f"{split}.jsonl")
                    for split in ("train", "selection", "calibration")
                },
            }
        )
    )
    return root


def test_validates_release_data_without_conversion_inputs(released_data, tmp_path):
    audit = tmp_path / "audit.json"
    result = validate(released_data, audit)
    assert result["status"] == "integrity_passed"
    assert result["windows"] == {"train": 6, "selection": 6, "calibration": 6}
    assert result["unique_windows"] == 18
    assert result["questions"] == 24
    assert result["spectral_targets_recomputed"] == 6
    assert json.loads(audit.read_text()) == result


@pytest.mark.parametrize("change", ["array", "group"])
def test_rejects_changed_arrays_and_group_overlap(released_data, tmp_path, change):
    if change == "array":
        np.save(released_data / "signals.npy", np.zeros((18, 1, 390), np.float32))
        error = "provenance"
    else:
        path = released_data / "selection.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["group_id"] = f"{TASKS[0]}/train/group"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        manifest_path = released_data / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["splits"]["selection"] = sha256(path)
        manifest_path.write_text(json.dumps(manifest))
        error = "Group leakage"
    audit = tmp_path / "audit.json"
    with pytest.raises(ValueError, match=error):
        validate(released_data, audit)
    assert not audit.exists()
