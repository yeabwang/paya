"""Fit a global probability temperature on held-out outcomes."""

import argparse
import json
import math
from pathlib import Path

from scipy.optimize import minimize_scalar
import yaml

from .checkpoint import learned_digest, load_checkpoint, save_checkpoint
from .data import ChoiceDataset, assert_disjoint
from .evaluate import collect, metrics
from .features import open_feature_cache
from .model import PayaModel
from .reporting import task_report


def fit_temperature(
    model, dataset, forbidden, batch_size=8, cache=None, output_path=None
):
    if "datasets" in forbidden:
        for reference in forbidden["datasets"]:
            previous = ChoiceDataset(reference["path"])
            try:
                if previous.sha256 != reference["sha256"]:
                    raise ValueError("Training/selection dataset changed since fitting")
                assert_disjoint(previous, dataset)
            except ValueError as error:
                raise ValueError(
                    f"Calibration overlaps training/selection: {error}"
                ) from error
            finally:
                previous.close()
    else:
        for field in ("source_id", "group_id", "recording_id"):
            for identifier in forbidden.get(field, []):
                if dataset.db.execute(
                    "SELECT 1 FROM audit WHERE field=? AND value=?", (field, identifier)
                ).fetchone():
                    raise ValueError(
                        f"Calibration overlaps training/selection on {field}"
                    )
    records = collect(model, dataset, batch_size, cache, output_path=output_path)
    before = metrics(records)
    objective = lambda log_t: metrics(records, math.exp(log_t))["nll"]
    result = minimize_scalar(objective, bounds=(-4, 4), method="bounded")
    value = math.exp(result.x) if result.success and result.fun < before["nll"] else 1.0
    model.temperature.fill_(value)
    model.calibration = {
        "learned_sha256": learned_digest(model),
        "temperature": float(model.temperature),
        "dataset_sha256": dataset.sha256,
        "count": len(records),
        "before": before,
        "after": metrics(records, float(model.temperature)),
        "tasks_before": task_report(records),
        "tasks_after": task_report(records, float(model.temperature)),
    }
    return model.calibration


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/paya.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.resolve() == args.checkpoint.resolve() or args.output.exists():
        raise ValueError(
            "Use a new calibrated checkpoint path; preserve the raw checkpoint"
        )
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model = PayaModel.from_assets(
        config["assets"], config.get("preparation"), config.get("head"), args.device
    )
    payload = load_checkpoint(args.checkpoint, model)
    if not payload["training"]:
        raise ValueError("Calibration requires a retrained decision checkpoint")
    dataset = ChoiceDataset(config["data"]["calibration"])
    cache = open_feature_cache(config)
    fitted = fit_temperature(
        model,
        dataset,
        payload["training"]["forbidden_calibration"],
        config.get("batch_size", 8),
        cache,
        args.output.with_suffix(".predictions.jsonl"),
    )
    save_checkpoint(args.output, model, payload["training"])
    print(json.dumps(fitted, indent=2))
