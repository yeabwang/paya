"""Outcome evaluation uses the same preprocessing, top-1 gate, and shared head."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .data import ChoiceDataset, make_batch


@torch.no_grad()
def collect(model, dataset, batch_size=8, cache=None):
    model.eval()
    records = []
    for rows in dataset.batches(batch_size):
        result = model(make_batch(model, rows), cache=cache)
        for row, logits, route in zip(rows, result["logits"], result["routes"].tolist()):
            records.append({"logits": logits[:len(row["criteria"])].cpu().tolist(),
                            "target": row["target"], "criteria": row["criteria"], "route": route,
                            **{k: row[k] for k in ("source_id", "group_id", "recording_id")}})
    return records


def metrics(records, temperature=1.0):
    if not records or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Need predictions and a finite positive temperature")
    losses, briers, correct, confidence = [], [], [], []
    for row in records:
        z = np.asarray(row["logits"], dtype=np.float64)/temperature
        z -= z.max()
        p = np.exp(z); p /= p.sum()
        target = row["target"]
        losses.append(float(np.log(np.exp(z).sum())-z[target]))
        briers.append(float(((p - np.eye(len(p))[target])**2).sum()))
        winner = min(range(len(p)), key=lambda i: (-p[i], row["criteria"][i]))
        correct.append(float(winner == target)); confidence.append(float(p[winner]))
    correct, confidence = np.asarray(correct), np.asarray(confidence)
    bins = np.minimum((confidence*15).astype(int), 14)
    reliability = [{"bin": i, "count": int((bins == i).sum()),
                    "accuracy": float(correct[bins == i].mean()),
                    "confidence": float(confidence[bins == i].mean())}
                   for i in range(15) if (bins == i).any()]
    return {"count": len(records), "accuracy": float(correct.mean()), "nll": float(np.mean(losses)),
            "brier": float(np.mean(briers)), "ece_15": float(sum(r["count"]/len(records)*abs(r["accuracy"]-r["confidence"])
                                                                       for r in reliability)),
            "reliability": reliability}


if __name__ == "__main__":
    from .checkpoint import load_checkpoint
    from .features import FeatureCache
    from .model import PayaModel
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/paya.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("selection", "calibration", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model = PayaModel.from_assets(config["assets"], config.get("preparation"), config.get("head"), args.device)
    load_checkpoint(args.checkpoint, model)
    dataset = ChoiceDataset(config["data"][args.split])
    cache = FeatureCache(config["feature_cache"]) if config.get("feature_cache") else None
    records = collect(model, dataset, config.get("batch_size", 8), cache)
    report = {"split": args.split, "calibrated": model.calibration is not None,
              "metrics": metrics(records, float(model.temperature)), "predictions": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))
