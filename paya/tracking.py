"""W&B metrics, checkpoint artifacts and run continuation."""

from contextlib import contextmanager

from .checkpoint import save_checkpoint
from .provenance import VERSION, sha256


@contextmanager
def open_run(config, directory, model, datasets, previous=None):
    """Create a tracking context; disabled mode does not initialize W&B."""
    settings = config.get("wandb", {})
    mode = settings.get("mode", "disabled")
    interval = settings.get("log_every", 10)
    if mode not in ("online", "offline", "disabled"):
        raise ValueError("wandb.mode must be online, offline, or disabled")
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
        raise ValueError("wandb.log_every must be a positive integer")
    if mode == "disabled":
        yield None
        return

    import wandb
    project = settings.get("project", "paya")
    entity = settings.get("entity")
    continuation = {}
    if previous and mode == previous["mode"] == "online":
        if project != previous["project"] or entity not in (None, previous["entity"]):
            raise ValueError("Online resume must use the saved W&B project and entity")
        entity = previous["entity"]
        continuation = {"id": previous["id"], "resume": "must"}
    with wandb.init(project=project, entity=entity, name=settings.get("name") or directory.name,
                     mode=mode, dir=str(directory), save_code=False, **continuation,
                     config=None if continuation else {"version": VERSION, "model": model.identity(),
                             "training": {k: v for k, v in config.items()
                                          if k not in ("wandb", "data", "assets", "feature_cache")},
                             "datasets": {name: {"records": len(data.rows), "sha256": sha256(data.path)}
                                          for name, data in datasets.items()},
                             "resumed_from_run_id": previous["id"] if previous else None}) as run:
        run.define_metric("step")
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="step")
        run.define_metric("selection/*", step_metric="epoch")
        yield run


def run_identity(run, mode):
    return {"id": run.id, "project": run.project, "entity": run.entity, "mode": mode} if run else None


def log_best(run, model, directory, epoch, nll):
    """Log inference weights and their selection result, excluding optimizer/data state."""
    import wandb
    path = directory/"best-decision.pt"
    save_checkpoint(path, model)
    artifact = wandb.Artifact(f"paya-{VERSION}-{run.id}", type="model",
                              metadata={"epoch": epoch, "selection_nll": nll, "version": VERSION})
    artifact.add_file(str(path), name="decision.pt")
    run.log_artifact(artifact, aliases=["best"])
    run.summary.update({"best/epoch": epoch, "best/selection_nll": nll})
