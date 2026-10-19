"""Check offline telemetry, run lifecycle and checkpoint continuation."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import yaml

from conftest import new_model
from test_training_contracts import write_dataset
from paya.provenance import VERSION, PACKAGE_VERSION
from paya.tracking import open_run


@pytest.fixture
def wandb_capture(tmp_path, monkeypatch):
    for variable, folder in (("WANDB_CACHE_DIR", "cache"), ("WANDB_DATA_DIR", "staging"),
                             ("WANDB_CONFIG_DIR", "config")):
        monkeypatch.setenv(variable, str(tmp_path/folder))
    monkeypatch.setenv("WANDB_SILENT", "true")
    import wandb
    captured = SimpleNamespace(runs=[], logs=[], artifacts=[], finished=[])
    original_init = wandb.init
    original_log = wandb.sdk.wandb_run.Run.log
    original_artifact = wandb.sdk.wandb_run.Run.log_artifact
    original_finish = wandb.sdk.wandb_run.Run._finish

    def initialize(**kwargs):
        run = original_init(**kwargs)
        captured.runs.append(run)
        return run

    def log(run, data, *args, **kwargs):
        captured.logs.append((run.id, dict(data)))
        return original_log(run, data, *args, **kwargs)

    def artifact(run, value, *args, **kwargs):
        captured.artifacts.append((run.id, value, kwargs))
        return original_artifact(run, value, *args, **kwargs)

    def finish(run, exit_code=None):
        captured.finished.append((run.id, exit_code, dict(run.summary)))
        return original_finish(run, exit_code=exit_code)

    monkeypatch.setattr(wandb, "init", initialize)
    monkeypatch.setattr(wandb.sdk.wandb_run.Run, "log", log)
    monkeypatch.setattr(wandb.sdk.wandb_run.Run, "log_artifact", artifact)
    monkeypatch.setattr(wandb.sdk.wandb_run.Run, "_finish", finish)
    return captured


@pytest.mark.parametrize("interrupted", [False, True])
def test_offline_training_metrics_artifact_and_resume(tmp_path, monkeypatch, wandb_capture, interrupted):
    import paya.train as module
    import wandb
    monkeypatch.setattr(module.PayaModel, "from_assets", lambda *args, **kwargs: new_model())
    paths = {}
    for split in ("train", "selection"):
        path = tmp_path/f"{split}.jsonl"
        write_dataset(path, split, f"{split}:w", split)
        paths[split] = str(path)
    config = {"assets": {}, "data": paths, "batch_size": 1, "threads": 2,
              "optim": {"epochs": 2, "warmup_epochs": 1, "lr": 0.001},
              "wandb": {"mode": "offline", "project": "paya-test", "log_every": 1}}
    config_path = tmp_path/"training.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    directory = tmp_path/"run"
    collect = module.collect
    calls = 0

    def fail_second_selection(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted selection")
        return collect(*args, **kwargs)

    if interrupted:
        monkeypatch.setattr(module, "collect", fail_second_selection)
        with pytest.raises(RuntimeError, match="interrupted selection"):
            module.train(config_path, directory)
        saved = torch.load(directory/"last.pt", weights_only=True)
        assert saved["training"]["global_step"] == 1
        assert wandb_capture.finished[0][1] == 1
        assert wandb.run is None
        monkeypatch.setattr(module, "collect", collect)
        module.train(config_path, directory, resume=directory/"last.pt")
    else:
        module.train(config_path, directory)

    latest = torch.load(directory/"last.pt", weights_only=True)
    exported = torch.load(directory/"best-decision.pt", weights_only=True)
    assert latest["training"]["global_step"] == 2
    assert latest["training"]["wandb"]["id"] == wandb_capture.runs[-1].id
    assert exported["training"] is None
    assert set(exported["learned"]) == {"router", "head"}
    assert exported["contract"].endswith(VERSION)
    logs = [row for _, row in wandb_capture.logs]
    assert any("train/answer_loss" in row and "train/router_loss" not in row for row in logs)
    assert any("train/router_loss" in row for row in logs)
    selected = [row for row in logs if "selection/nll" in row]
    assert selected[-1]["step"] == 2
    assert selected[-1]["selection/route_fraction/normwear"] + selected[-1]["selection/route_fraction/mantis"] == 1
    assert all(name in selected[-1] for name in ("selection/accuracy", "selection/brier", "selection/ece_15"))
    assert len(wandb_capture.artifacts) == 1
    _, artifact, kwargs = wandb_capture.artifacts[0]
    assert artifact.type == "model" and kwargs["aliases"] == ["best"]
    assert artifact.metadata["version"] == VERSION
    assert "decision.pt" in artifact.manifest.entries
    run_config = dict(wandb_capture.runs[-1].config)
    assert run_config["version"] == VERSION
    assert run_config["datasets"]["train"]["records"] == 1
    assert len(run_config["datasets"]["train"]["sha256"]) == 64
    assert "data" not in run_config["training"] and "assets" not in run_config["training"]
    if interrupted:
        assert wandb_capture.runs[0].id != wandb_capture.runs[1].id
        assert run_config["resumed_from_run_id"] == wandb_capture.runs[0].id
    assert wandb_capture.finished[-1][1] == 0
    assert "best/selection_nll" in wandb_capture.finished[-1][2]
    assert list(directory.glob("wandb/offline-run-*/*.wandb"))
    assert wandb.run is None


def test_disabled_and_invalid_tracking_never_initialize_wandb(tmp_path, monkeypatch):
    import wandb
    def unexpected(**kwargs):
        pytest.fail("Disabled/invalid tracking must not start a run")
    monkeypatch.setattr(wandb, "init", unexpected)
    with open_run({}, tmp_path, None, {}) as run:
        assert run is None
    for settings in ({"mode": "invalid"}, {"log_every": 0}, {"log_every": True}):
        with pytest.raises(ValueError):
            with open_run({"wandb": settings}, tmp_path, None, {}):
                pass


def test_online_resume_uses_saved_identity_without_contacting_server(tmp_path, model, monkeypatch):
    import wandb
    calls, definitions = [], []
    @contextmanager
    def initialize(**kwargs):
        calls.append(kwargs)
        yield SimpleNamespace(define_metric=lambda *args, **kw: definitions.append((args, kw)))
    monkeypatch.setattr(wandb, "init", initialize)
    previous = {"id": "existing", "project": "paya", "entity": "team", "mode": "online"}
    with open_run({"wandb": {"mode": "online"}}, tmp_path, model, {}, previous):
        pass
    assert calls[0]["id"] == "existing" and calls[0]["resume"] == "must"
    assert calls[0]["entity"] == "team"
    assert calls[0]["config"] is None  # Keep the existing online run configuration.
    assert (("selection/*",), {"step_metric": "epoch"}) in definitions
    for change in ({"project": "different"}, {"entity": "different"}):
        with pytest.raises(ValueError, match="saved W&B"):
            with open_run({"wandb": {"mode": "online", **change}}, tmp_path, model, {}, previous):
                pass
    assert len(calls) == 1


def test_package_and_artifact_versions_have_one_source():
    import importlib.metadata
    from paya import __version__, provenance
    assert __version__ == PACKAGE_VERSION == importlib.metadata.version("paya") == "1.0.0"
    assert VERSION == "v1" and provenance.INPUT_SCHEMA_VERSION == 1
    for name in ("RUNTIME_CONTRACT", "PREPARATION_CONTRACT", "NATIVE_CONTRACT", "CACHE_CONTRACT", "TEXT_CONTRACT"):
        assert getattr(provenance, name).endswith(f"-{VERSION}")
