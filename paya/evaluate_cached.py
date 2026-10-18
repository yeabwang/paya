"""Evaluate saved checkpoints using cached training evidence for diagnostics."""

import argparse
import json
from pathlib import Path

import torch
import yaml

from .assess import write_predictions
from .data import ChoiceDataset, loader, to_device
from .features import load_cache
from .inference import load_checkpoint
from .provenance import sha256
from .text import TextEncoder


@torch.inference_mode()
def predict(model, dataset, batch_size, device):
    """Return labelled prediction records from shape-compatible cached batches.

    Preserve audit IDs and supplied candidate order, and omit padded logits.
    These records reflect the training-evidence path rather than raw public
    waveform inference.
    """
    model.eval()
    records = []
    for batch in loader(dataset, batch_size):
        logits = model(**to_device(batch["inputs"], device)).cpu()
        for i, item in enumerate(batch["records"]):
            n = len(item["descriptions"])
            records.append(
                {
                    key: item[key]
                    for key in ("source_id", "group_id", "dataset_id", "family")
                }
                | {
                    "choices": item["descriptions"],
                    "logits": logits[i, :n].tolist(),
                    "target": item["labels"],
                }
            )
    return records


def evaluate(
    checkpoint, config, split, output, only=None, families=None, heldout_wording=False
):
    """Write predictions and metrics for a cached-evidence evaluation split.

    Check target exclusion/group boundaries for transfer scoring. This path
    is diagnostic and does not fit public waveform calibration.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = TextEncoder(config["data"]["text_cache"])
    cache = load_cache(
        config["data"]["feature_cache"], config["data"]["root"], config["assets"]
    )
    ds = ChoiceDataset(
        config["data"]["root"],
        "selection" if split == "heldout_domain" else split,
        text,
        cache=cache,
        families=families,
        heldout_wording=heldout_wording,
    )
    if split == "heldout_domain":
        ds.examples += ChoiceDataset(
            config["data"]["root"],
            "calibration",
            text,
            cache=cache,
            families=families,
            heldout_wording=heldout_wording,
        ).examples
    if only:
        ds.examples = [(r, q) for r, q in ds.examples if r["dataset_id"] in only]
    model, _, payload = load_checkpoint(
        checkpoint,
        config["assets"],
        text,
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    if split == "heldout_domain" and (
        not only or not set(only) <= set(payload["config"]["data"].get("exclude", []))
    ):
        raise ValueError(
            "Held-out domain scoring requires datasets excluded from all decision training/selection"
        )
    if split == "calibration":
        excluded = set(payload["config"]["data"].get("exclude", []))
        ds.examples = [
            (r, q) for r, q in ds.examples if r["dataset_id"] not in excluded
        ]
    groups_path = Path(checkpoint).parent / "groups.json"
    saved_groups = json.loads(groups_path.read_text())
    if split == "heldout_domain" and {r["group_id"] for r, _ in ds.examples} & set(
        saved_groups["train"] + saved_groups["selection"]
    ):
        raise ValueError("Target domain intersects decision development groups")
    records = predict(
        model, ds, config["data"]["batch_size"], next(model.parameters()).device
    )
    header = {
        "type": "prediction_manifest",
        "split": split,
        "checkpoint_sha256": sha256(checkpoint),
        "data_manifest_sha256": sha256(Path(config["data"]["root"]) / "manifest.json"),
        "identity": payload["identity"],
        "heldout_wording": heldout_wording,
        "forbidden_calibration_groups": saved_groups["train"]
        + saved_groups["selection"],
        "temperature_in_graph": float(model.temperature),
        "excluded_training_datasets": payload["config"]["data"].get("exclude", []),
        "interpretation": "held out from decision training/selection; pretrained encoder exposure not audited"
        if split == "heldout_domain"
        else "development evidence",
    }
    result = write_predictions(output, header, records)
    print(json.dumps(result["native_mean"]))
    return result


def main():
    """Evaluate a checkpoint using its prepared data and frozen-evidence cache."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--split", choices=["selection", "calibration", "heldout_domain"], required=True
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--only", nargs="+")
    p.add_argument("--families", nargs="+")
    p.add_argument("--heldout-wording", action="store_true")
    a = p.parse_args()
    config = yaml.safe_load(a.config.read_text())
    torch.set_num_threads(config.get("threads", 4))
    evaluate(
        a.checkpoint, config, a.split, a.output, a.only, a.families, a.heldout_wording
    )


if __name__ == "__main__":
    main()
