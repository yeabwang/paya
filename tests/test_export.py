import json
import sys
from pathlib import Path

import pytest
import torch

import paya.export as bundle_export
from paya.assess import calibration_record
from paya.assess import main as assess_main
from paya.inference import load_bundle
from paya.provenance import PUBLIC_API, identity, sha256, validate_calibration


def test_calibration_command_produces_an_exportable_binding(tmp_path, monkeypatch):
    assets = tmp_path / "normwear"
    assets.mkdir()
    (assets / "source.json").write_text("{}")
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {"head": {"weight": torch.ones(1)}, "config": {}, "identity": identity(assets)},
        checkpoint,
    )
    header = {
        "type": "prediction_manifest",
        "split": "calibration",
        "checkpoint_sha256": sha256(checkpoint),
        "public_api": dict(PUBLIC_API),
        "evidence_precision": "fp32",
        "encoder_batch_size": 1,
        "temperature_in_graph": 1.0,
        "forbidden_calibration_groups": ["training"],
    }
    records = [
        {"logits": [4.0, -4.0], "target": i % 2, "group_id": f"g{i % 2}"}
        for i in range(8)
    ]
    predictions = tmp_path / "calibration.jsonl"
    predictions.write_text("\n".join(json.dumps(r) for r in [header, *records]) + "\n")
    calibration = tmp_path / "temperature.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "assess",
            "--predictions",
            str(predictions),
            "--output",
            str(calibration),
            "--fit-global",
        ],
    )
    assess_main()
    binding = json.loads(calibration.read_text())
    assert binding["public_api"] == PUBLIC_API
    assert binding["predictions_sha256"] == sha256(predictions)
    assert binding["calibration_nll_fitted"] <= binding["calibration_nll_raw"]

    class SavedEncoder:
        def save(self, directory):
            directory = Path(directory)
            directory.mkdir()
            (directory / "config.json").write_text("{}")

    class Text:
        model_path = None
        model = SavedEncoder()

        def encode(self, strings):
            return torch.ones(len(strings), 768)

        def save(self, path):
            torch.save({}, path)

    monkeypatch.setattr(bundle_export, "TextEncoder", Text)
    output = tmp_path / "export"
    manifest = bundle_export.export(checkpoint, assets, output, calibration)
    payload = torch.load(output / "model.pt", weights_only=True)
    assert payload["temperature"] == binding["temperature"]
    assert payload["identity"]["schema"] == manifest["schema_version"] == 2
    assert set(payload) == {"head", "config", "identity", "temperature"}
    assert payload["config"] == {"head": {}}
    assert manifest["public_api"] == PUBLIC_API
    assert torch.equal(payload["head"]["weight"], torch.ones(1))

    # Tampering must fail before any encoder or checkpoint is loaded.
    (output / "text_encoder/config.json").write_text("changed")
    with pytest.raises(ValueError, match="missing or changed"):
        load_bundle(output)


def test_calibration_rejects_wrong_path_groups_weights_and_temperature():
    header = {
        "split": "calibration",
        "checkpoint_sha256": "selected",
        "public_api": dict(PUBLIC_API),
        "evidence_precision": "fp32",
        "encoder_batch_size": 1,
        "forbidden_calibration_groups": ["training"],
    }
    records = [
        {"logits": [1.0, -1.0], "target": 0, "group_id": "a"},
        {"logits": [-1.0, 1.0], "target": 1, "group_id": "b"},
    ]
    for change in (
        {"split": "selection"},
        {"temperature_in_graph": 2.0},
        {"evidence_precision": "float16"},
        {"public_api": {}},
        {"forbidden_calibration_groups": ["a"]},
    ):
        with pytest.raises(ValueError):
            calibration_record(header | change, records, "predictions")
    binding = calibration_record(header, records, "predictions")
    with pytest.raises(ValueError, match="different weights"):
        validate_calibration(binding, "another checkpoint")
    for temperature in (float("nan"), float("inf"), 0.0, -1.0):
        with pytest.raises(ValueError, match="finite and positive"):
            validate_calibration(binding | {"temperature": temperature}, "selected")
