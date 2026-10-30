"""Exercise bounded loading and recovery from a committed mid-epoch checkpoint."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml

from conftest import new_model, state
from paya.atomic import atomic_output
from paya.checkpoint import learned_digest
from paya.data import ChoiceDataset, assert_disjoint
from paya.evaluate import collect, metrics, PredictionRecords


def rows(count, split="train"):
    return [
        {
            "state": state(rate=125, duration=1, condition="processed"),
            "question": "Which category?",
            "criteria": ["slow", "normal", "fast"],
            "target": i % 3,
            "mapping_id": f"{split}-{i}",
            "source_id": f"{split}:w{i}",
            "group_id": split,
            "recording_id": split,
        }
        for i in range(count)
    ]


def test_jsonl_and_parquet_shuffle_resume_and_lazy_validation(tmp_path, monkeypatch):
    original = rows(521)
    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text(
        "\n".join(json.dumps(r) for r in original) + "\n", encoding="utf-8"
    )
    parquet = tmp_path / "data.parquet"
    pq.write_table(pa.Table.from_pylist(original), parquet, row_group_size=128)
    manifest = tmp_path / "data.json"
    manifest.write_text(json.dumps({"files": [parquet.name]}))
    for path in (jsonl, manifest):
        dataset = ChoiceDataset(path)
        assert not isinstance(dataset.rows, list)
        assert len(dataset.rows) == 521
        assert dataset.rows[-1] == original[-1]
        batches = list(dataset.batches(17, seed=9))
        assert len({r["mapping_id"] for batch in batches for r in batch}) == 521
        for batch in batches:
            for row in batch:
                index = int(row["mapping_id"].split("-")[-1])
                assert (
                    row["criteria"][row["target"]]
                    == original[index]["criteria"][original[index]["target"]]
                )
        assert list(dataset.batches(17, seed=9, start_batch=20)) == batches[20:]
        assert list(dataset.batches(17, seed=10)) != batches
        monkeypatch.setattr(
            dataset, "_read_block", lambda _: pytest.fail("Audit read signal samples")
        )
        other = tmp_path / (path.stem + "-other.jsonl")
        other.write_text(json.dumps(rows(1, "selection")[0]) + "\n")
        assert_disjoint(dataset, ChoiceDataset(other))


def test_dataset_changes_and_oversize_blocks_fail_before_use(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(rows(1)[0]) + "\n")
    dataset = ChoiceDataset(path)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        next(dataset.batches(1))
    parquet = tmp_path / "large.parquet"
    pq.write_table(pa.Table.from_pylist(rows(2050)), parquet, row_group_size=2050)
    with pytest.raises(ValueError, match="memory budget"):
        ChoiceDataset(parquet)


def test_atomic_failure_preserves_previous_file_and_streaming_metrics(tmp_path, model):
    path = tmp_path / "checkpoint"
    path.write_bytes(b"committed")
    with pytest.raises(RuntimeError):
        with atomic_output(path) as output:
            output.write(b"incomplete")
            raise RuntimeError("simulated failed write")
    assert path.read_bytes() == b"committed"
    assert not list(tmp_path.glob("*.partial"))
    source = tmp_path / "data.jsonl"
    source.write_text("\n".join(json.dumps(r) for r in rows(7)) + "\n")
    predictions = collect(model, ChoiceDataset(source), batch_size=2)
    assert isinstance(predictions, PredictionRecords)
    assert len(predictions) == 7
    assert next(iter(predictions))["mapping_id"] == "train-0"
    assert metrics(iter(predictions)) == metrics(list(predictions))


def test_dictionary_encoded_signals_do_not_bypass_numeric_memory_budget(
    tmp_path, monkeypatch
):
    import paya.data as module

    record = rows(1)[0]
    record["state"]["signals"][0]["samples"] = [1.0] * 4096
    path = tmp_path / "dictionary.parquet"
    pq.write_table(pa.Table.from_pylist([record]), path, use_dictionary=True)
    assert pq.ParquetFile(path).metadata.row_group(0).total_byte_size < 16384
    monkeypatch.setattr(module, "MAX_BLOCK_BYTES", 16384)
    with pytest.raises(ValueError, match="memory budget"):
        ChoiceDataset(path)


@pytest.mark.parametrize("cache_format", ["files", "sqlite"])
@pytest.mark.parametrize("step_selection", [False, True])
def test_mid_epoch_resume_matches_uninterrupted_weights_with_dropout(
    tmp_path, monkeypatch, cache_format, step_selection
):
    import paya.train as module
    from paya.head import AttentionHead

    def factory(*args, **kwargs):
        model = new_model()
        model.head = AttentionHead(width=128, dropout=0.2)
        return model

    monkeypatch.setattr(module.PayaModel, "from_assets", factory)
    paths = {}
    for split, count in [("train", 11), ("selection", 4)]:
        path = tmp_path / (split + ".jsonl")
        path.write_text("\n".join(json.dumps(r) for r in rows(count, split)) + "\n")
        paths[split] = str(path)
    config = {
        "assets": {},
        "data": paths,
        "batch_size": 3,
        "threads": 2,
        "seed": 42,
        "feature_cache": str(
            tmp_path / ("features.sqlite3" if cache_format == "sqlite" else "features")
        ),
        "feature_cache_format": cache_format,
        "optim": {"epochs": 2, "lr": 0.001, "checkpoint_every_steps": 1},
    }
    if step_selection:
        config["optim"].update(eval_every_steps=2)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    uninterrupted, _ = module.train(path, tmp_path / "baseline")
    expected = learned_digest(uninterrupted)
    original = module.save_checkpoint

    def crash(destination, model, training=None):
        original(destination, model, training)
        if Path(destination).name == "last.pt" and training["global_step"] == 2:
            raise RuntimeError("simulated crash after committed step")

    monkeypatch.setattr(module, "save_checkpoint", crash)
    with pytest.raises(RuntimeError, match="simulated crash"):
        module.train(path, tmp_path / "interrupted")
    checkpoint = tmp_path / "interrupted/last.pt"
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["training"]["epoch"] == 0 and saved["training"]["next_batch"] == 2
    assert (
        saved["training"]["metrics_bytes"]
        == (tmp_path / "interrupted/steps.jsonl").stat().st_size
    )
    with (tmp_path / "interrupted/steps.jsonl").open("a") as output:
        output.write('{"step":999,"uncommitted":true}\n')
    monkeypatch.setattr(module, "save_checkpoint", original)
    resumed, _ = module.train(path, tmp_path / "interrupted", resume=checkpoint)
    assert learned_digest(resumed) == expected
    final = torch.load(checkpoint, weights_only=True)
    assert final["training"]["global_step"] == 8
    committed = [
        json.loads(line)["step"]
        for line in (tmp_path / "interrupted/steps.jsonl").read_text().splitlines()
    ]
    assert committed == list(range(1, 9))
    assert final["training"]["forbidden_calibration"]["datasets"]
    changed = rows(11, "train")
    changed[0]["target"] = 1
    Path(paths["train"]).write_text("\n".join(json.dumps(r) for r in changed) + "\n")
    with pytest.raises(ValueError, match="dataset content differs"):
        module.train(path, tmp_path / "interrupted", resume=checkpoint)
