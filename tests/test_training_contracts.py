import json

import pytest
import torch

from paya.checkpoint import load_checkpoint, save_checkpoint, learned_digest
from paya.data import ChoiceDataset, assert_disjoint
from paya.features import FeatureCache
from paya.calibrate import fit_temperature
from paya.export import export_bundle, load_bundle
from paya.text import TextEncoder
from conftest import new_model, state


def write_dataset(path, group="subject-a", source="recording-a:window-1", recording="recording-a"):
    row = {"state": state(), "question": "Which category?", "criteria": ["slow", "normal", "fast"],
           "target": 1, "source_id": source, "group_id": group, "recording_id": recording}
    path.write_text(json.dumps(row)+"\n", encoding="utf-8")
    return ChoiceDataset(path)


def test_new_data_contract_and_split_isolation(tmp_path):
    training = write_dataset(tmp_path/"train.jsonl")
    selection = write_dataset(tmp_path/"selection.jsonl", "subject-b", "recording-b:w", "recording-b")
    assert_disjoint(training, selection)
    overlapping = write_dataset(tmp_path/"bad.jsonl", "subject-b", "recording-a:w2", "recording-a")
    with pytest.raises(ValueError, match="recording_id"):
        assert_disjoint(training, overlapping)
    batch = next(training.batches(1, seed=7))
    assert batch[0]["criteria"][batch[0]["target"]] == "normal"


def test_fp32_cache_reuses_only_matching_preparation_and_detects_corruption(model, tmp_path):
    cache = FeatureCache(tmp_path)
    prepared = model.preprocessor.clean(state())
    first = cache.get_or_encode(model, prepared, 1)
    count = model.encoders[1].calls
    second = cache.get_or_encode(model, prepared, 1)
    assert model.encoders[1].calls == count
    torch.testing.assert_close(first.values, second.values, atol=0, rtol=0)
    torch.testing.assert_close(first.times, second.times, atol=0, rtol=0)
    changed = state(); changed["signals"][0]["samples"][0] += 0.1
    cache.get_or_encode(model, model.preprocessor.clean(changed), 1)
    assert model.encoders[1].calls == count+1
    path = next(tmp_path.glob("*.pt"))
    payload = torch.load(path, weights_only=True)
    payload["values"][0, 0] += 1
    torch.save(payload, path)
    key = payload["binding"]["cleaned"]
    prepared_to_load = prepared if prepared.identity == key else model.preprocessor.clean(changed)
    with pytest.raises(ValueError, match="content changed"):
        cache.get_or_encode(model, prepared_to_load, 1)


def test_feature_cache_hit_skips_formatting_and_remains_encoder_specific(model, tmp_path, monkeypatch):
    cache = FeatureCache(tmp_path)
    cleaned = model.preprocessor.clean(state())
    calls = []
    original = model.preprocessor.format_for_encoder
    def format_signal(signal, expert):
        calls.append(expert)
        return original(signal, expert)
    monkeypatch.setattr(model.preprocessor, "format_for_encoder", format_signal)
    for expert in (0, 1, 0, 1):
        cache.get_or_encode(model, cleaned, expert)
    assert calls == [0, 1]
    assert len(list(tmp_path.glob("*.pt"))) == 2
    for path in tmp_path.glob("*.pt"):
        payload = torch.load(path, weights_only=True)
        assert payload["times"].shape == (len(payload["values"]),)
        assert "positions" not in payload


def test_checkpoint_rejects_a_different_text_and_position_contract(model, tmp_path):
    path = tmp_path/"incompatible.pt"
    save_checkpoint(path, model)
    payload = torch.load(path, weights_only=True)
    payload["identity"]["architecture"]["text_roles"].append("ending")
    payload["identity"]["architecture"]["signal_position"] = "time-scale-cls"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="two-encoder"):
        load_checkpoint(path, model)


def test_checkpoint_roundtrip_rejects_incompatible_head_and_preserves_output(model, tmp_path):
    before = model.predict(state(), "q", ["a", "b", "c"])
    path = tmp_path/"new.pt"
    save_checkpoint(path, model)
    other = new_model()
    load_checkpoint(path, other)
    after = other.predict(state(), "q", ["a", "b", "c"])
    assert before == after
    payload = torch.load(path, weights_only=True)
    assert set(payload["learned"]) == {"router", "head"}
    assert "encoders" not in payload
    torch.save({"contract": "unrecognized", "head": {}}, tmp_path/"invalid.pt")
    with pytest.raises(ValueError, match="two-encoder"):
        load_checkpoint(tmp_path/"invalid.pt", other)


def test_old_text_banks_and_unfitted_temperatures_are_rejected(model, tmp_path):
    torch.save({"identity": {"dimension": 768}, "embeddings": {}}, tmp_path/"old-text.pt")
    with pytest.raises(ValueError, match="Text bank"):
        TextEncoder(tmp_path/"old-text.pt")
    for strings in ([], [""], "question", [None]):
        with pytest.raises(ValueError, match="separate"):
            TextEncoder().encode(strings)
    model.temperature.fill_(2)
    with pytest.raises(ValueError, match="provenance"):
        save_checkpoint(tmp_path/"unfitted.pt", model)


def test_outcome_temperature_is_bound_to_frozen_router_and_head(model, tmp_path):
    calibration = write_dataset(tmp_path/"calibration.jsonl", "heldout", "heldout:w", "heldout")
    fitted = fit_temperature(model, calibration, {"group_id": ["train"]})
    assert fitted["learned_sha256"] == learned_digest(model) and model.temperature > 0
    assert fitted["after"]["nll"] <= fitted["before"]["nll"]+1e-6
    save_checkpoint(tmp_path/"calibrated.pt", model)
    loaded = new_model(); load_checkpoint(tmp_path/"calibrated.pt", loaded)
    assert loaded.calibration == model.calibration
    with pytest.raises(ValueError, match="overlaps"):
        fit_temperature(model, calibration, {"recording_id": ["heldout"]})
    with torch.no_grad():
        model.head.scorer.weight.add_(0.01)
    with pytest.raises(ValueError, match="Calibration"):
        save_checkpoint(tmp_path/"stale.pt", model)
    model.train()
    assert model.calibration is None and model.temperature == 1


def test_bundle_roundtrip_and_integrity_with_tiny_asset_doubles(model, tmp_path, monkeypatch):
    import paya.export as module
    assets = {}
    for name in ("normwear", "mantis", "text"):
        root = tmp_path/name; root.mkdir()
        (root/"weights.txt").write_text(name, encoding="utf-8")
        assets[name] = root
    monkeypatch.setattr(module, "verify_assets", lambda _: model.asset_identity)
    monkeypatch.setattr(module.PayaModel, "from_assets", lambda *args, **kwargs: new_model())
    root = tmp_path/"bundle"
    manifest = export_bundle(model, assets, root)
    assert not manifest["calibrated"]
    loaded = load_bundle(root)
    assert loaded.predict(state(), "q", ["a", "b"]) == model.predict(state(), "q", ["a", "b"])
    with pytest.raises(FileExistsError):
        export_bundle(model, assets, root)
    (root/"mantis"/"weights.txt").write_text("modified", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(root)


def test_training_entrypoint_warmup_routing_and_resume_with_tiny_encoders(tmp_path, monkeypatch):
    import yaml
    import paya.train as module
    model = new_model()
    frozen = {name: value.clone() for name, value in model.encoders.state_dict().items()}
    original_gate = model.router.gate[-1].weight.detach().clone()
    monkeypatch.setattr(module.PayaModel, "from_assets", lambda *args, **kwargs: model)
    paths = {}
    for split in ("train", "selection", "calibration", "test"):
        path = tmp_path/f"{split}.jsonl"
        write_dataset(path, split, f"{split}:w", split)
        paths[split] = str(path)
    config = {"assets": {}, "data": paths, "batch_size": 1, "threads": 2,
              "feature_cache": str(tmp_path/"features"),
              "optim": {"epochs": 2, "warmup_epochs": 1, "lr": 0.001}}
    config_path = tmp_path/"config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    _, history = module.train(config_path, tmp_path/"run")
    assert [row["phase"] for row in history] == ["head_warmup", "routed"]
    best = torch.load(tmp_path/"run"/"best.pt", weights_only=True)
    assert best["training"]["epoch"] == 2  # never select an untrained warmup router
    assert not torch.equal(model.router.gate[-1].weight, original_gate)
    for name, value in model.encoders.state_dict().items():
        torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)
    _, resumed = module.train(config_path, tmp_path/"run", resume=tmp_path/"run"/"last.pt")
    assert resumed == []
