"""Measure decision accuracy, NLL, Brier score and calibration error."""

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import torch
import yaml

from .data import ChoiceDataset, make_batch
from .atomic import atomic_output


class PredictionRecords:
    def __init__(self, path, count):
        self.path, self.count = Path(path), count

    def __iter__(self):
        with self.path.open(encoding="utf-8") as source:
            for line in source:
                yield json.loads(line)

    def __len__(self):
        return self.count


@torch.no_grad()
def collect(model, dataset, batch_size=8, cache=None, output_path=None):
    model.eval()
    if output_path is None:
        directory=tempfile.TemporaryDirectory(prefix='paya-predictions-')
        records=collect(model,dataset,batch_size,cache,output_path=Path(directory.name)/'predictions.jsonl')
        records._temporary=directory
        return records
    if output_path is not None:
        count = 0
        with atomic_output(output_path, "w") as output:
            for rows in dataset.batches(batch_size):
                result = model(make_batch(model, rows), cache=cache)
                for row, logits in zip(rows, result["logits"]):
                    output.write(json.dumps({"logits": logits[:len(row["criteria"])].cpu().tolist(),
                        "target": row["target"], "criteria": row["criteria"],
                        **{k: row[k] for k in ("source_id", "group_id", "recording_id")},
                        **{k: row[k] for k in ("mapping_id", "datasetname") if k in row}}, allow_nan=False)+"\n")
                    count += 1
        return PredictionRecords(output_path, count)


def metrics(records, temperature=1.0):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Need predictions and a finite positive temperature")
    count = 0
    losses = briers = correct = 0.0
    bin_count = np.zeros(15, dtype=np.int64)
    bin_correct = np.zeros(15)
    bin_confidence = np.zeros(15)
    for row in records:
        z = np.asarray(row["logits"], dtype=np.float64)/temperature
        if not np.isfinite(z).all():
            raise ValueError("Nonfinite prediction logits")
        z -= z.max()
        p = np.exp(z); p /= p.sum()
        target = row["target"]
        losses += float(np.log(np.exp(z).sum())-z[target])
        briers += float(((p - np.eye(len(p))[target])**2).sum())
        winner = min(range(len(p)), key=lambda i: (-p[i], row["criteria"][i]))
        hit = float(winner == target)
        confidence = float(p[winner])
        correct += hit
        count += 1
        index = min(int(confidence*15),14)
        bin_count[index] += 1
        bin_correct[index] += hit
        bin_confidence[index] += confidence
    if not count:
        raise ValueError("Need predictions and a finite positive temperature")
    reliability = [{"bin": i, "count": int(bin_count[i]),
                    "accuracy": float(bin_correct[i]/bin_count[i]),
                    "confidence": float(bin_confidence[i]/bin_count[i])}
                   for i in range(15) if bin_count[i]]
    return {"count": count, "accuracy": correct/count, "nll": losses/count,
            "brier": briers/count, "ece_15": float(sum(r["count"]/count*abs(r["accuracy"]-r["confidence"])
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
    records = collect(model, dataset, config.get("batch_size", 8), cache,
                      output_path=args.output.with_suffix(".predictions.jsonl"))
    report = {"split": args.split, "calibrated": model.calibration is not None,
              "metrics": metrics(records, float(model.temperature)), "predictions_path": str(records.path)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output(args.output,"w") as output:
        json.dump(report,output,indent=2)
    print(json.dumps(report["metrics"], indent=2))
