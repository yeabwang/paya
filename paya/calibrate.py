"""Fit a global probability temperature on held-out outcomes."""

import argparse
import json
import math
from pathlib import Path

from scipy.optimize import minimize_scalar
import yaml

from .checkpoint import learned_digest, load_checkpoint, save_checkpoint
from .data import ChoiceDataset
from .evaluate import collect, metrics
from .features import FeatureCache
from .model import PayaModel
from .provenance import sha256


def fit_temperature(model, dataset, forbidden, batch_size=8, cache=None):
    for field in ("source_id", "group_id", "recording_id"):
        if {row[field] for row in dataset.rows} & set(forbidden.get(field, [])):
            raise ValueError(f"Calibration overlaps training/selection on {field}")
    records = collect(model, dataset, batch_size, cache)
    before = metrics(records)
    objective = lambda log_t: metrics(records, math.exp(log_t))["nll"]
    result = minimize_scalar(objective, bounds=(-4, 4), method="bounded")
    value = math.exp(result.x) if result.success and result.fun < before["nll"] else 1.0
    model.temperature.fill_(value)
    model.calibration = {"learned_sha256": learned_digest(model), "temperature": float(model.temperature),
                         "dataset_sha256": sha256(dataset.path), "count": len(records),
                         "before": before, "after": metrics(records, float(model.temperature))}
    return model.calibration


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/paya.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model = PayaModel.from_assets(config["assets"], config.get("preparation"), config.get("head"), args.device)
    payload = load_checkpoint(args.checkpoint, model)
    if not payload["training"]:
        raise ValueError("Calibration requires a retrained decision checkpoint")
    dataset = ChoiceDataset(config["data"]["calibration"])
    cache = FeatureCache(config["feature_cache"]) if config.get("feature_cache") else None
    fitted = fit_temperature(model, dataset, payload["training"]["forbidden_calibration"],
                             config.get("batch_size", 8), cache)
    save_checkpoint(args.output, model, payload["training"])
    print(json.dumps(fitted, indent=2))
