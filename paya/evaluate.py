"""Evaluate waveforms through the canonical FP32, batch-one inference path."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .assess import write_predictions
from .inference import canonical_inputs, load_checkpoint, logits
from .provenance import PUBLIC_API, sha256, source_provenance
from .text import TextEncoder


@torch.inference_mode()
def score_window(
    model, text, row, root, device, arrays, wording=False, memory=None, metadata=None
):
    """Score every question about a window using one FP32 waveform encoding.

    Return (records, memory, metadata). Reuse supplied exact evidence for
    reserved wording or a compatible frozen encoder; do not substitute the
    FP16 training cache for this computation.
    """
    if memory is None:
        filename = row["signal_file"]
        path = (Path(root) / filename).resolve()
        if not path.is_relative_to(Path(root).resolve()):
            raise ValueError("Signal path escapes prepared data")
        if filename not in arrays:
            arrays[filename] = np.load(path, mmap_mode="r")
        values = arrays[filename][row["index"]][row["valid_channels"]]
        signals, metadata = canonical_inputs(
            {
                "sampling_rate_hz": row["sampling_rate_hz"],
                "channels": [
                    {**m, "samples": v} for m, v in zip(row["channels"], values)
                ],
            }
        )
        model.eval()
        memory = model.head.signal_memory(
            model.core(signals[None].to(device)), model.memory_time_bins
        )
    records = []
    for q in row["questions"]:
        z = (
            logits(
                model,
                text,
                q["heldout_question"] if wording else q["question"],
                q["choices"],
                metadata,
                device,
                signal_memory=memory,
            )[0]
            .cpu()
            .tolist()
        )
        records.append(
            {k: row[k] for k in ("source_id", "group_id", "dataset_id")}
            | {
                "family": q["family"],
                "choices": q["choices"],
                "target": q["label"],
                "logits": z,
            }
        )
    return records, memory, metadata


def run(config, directory, transfers=None, transfer_families=None):
    """Score selection, calibration, wording, and explicitly supplied transfer runs.

    ``transfers`` maps excluded dataset IDs to run directories. All supplied
    transfer checkpoints must exclude their target from training and selection.
    FP32 encoder evidence is reused only between questions for the same window
    and between checkpoints sharing the frozen encoder and pooling contract.
    """
    root = Path(config["data"]["root"])
    directory = Path(directory)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = TextEncoder(config["data"]["text_cache"], device=device)
    model, _, payload = load_checkpoint(
        directory / "best.pt", config["assets"], text, device
    )
    if float(model.temperature) != 1.0:
        raise ValueError("Raw public assessment requires uncalibrated weights")
    manifest = json.loads((root / "manifest.json").read_text())
    for name, digest in manifest["signals"].items():
        if sha256(root / name) != digest:
            raise ValueError("Prepared signal measurements changed")
    groups = json.loads((directory / "groups.json").read_text())
    forbidden = groups["train"] + groups["selection"]
    predictions = {key: [] for key in ("selection", "calibration", "heldout-wording")}
    arrays, targets = {}, {}
    for task, folder in (transfers or {}).items():
        folder = Path(folder)
        other, _, state = load_checkpoint(
            folder / "best.pt", config["assets"], text, device
        )
        if task not in state["config"]["data"].get("exclude", []):
            raise ValueError("Transfer target was trained")
        if other.memory_time_bins != model.memory_time_bins:
            raise ValueError(
                "Transfer checkpoints must share the evidence pooling layout"
            )
        if float(other.temperature) != 1.0:
            raise ValueError("Transfer assessment requires uncalibrated weights")
        target_groups = json.loads((folder / "groups.json").read_text())
        targets[task] = {
            "model": other,
            "state": state,
            "folder": folder,
            "forbidden": target_groups["train"] + target_groups["selection"],
            "records": [],
        }
    done = 0
    for split in ("selection", "calibration"):
        if sha256(root / f"{split}.jsonl") != manifest["splits"][split]:
            raise ValueError("Prepared split changed")
        for line in (root / f"{split}.jsonl").read_text().splitlines():
            row = json.loads(line)
            if split == "calibration" and row["group_id"] in forbidden:
                raise ValueError("Calibration group overlap")
            records, memory, metadata = score_window(
                model, text, row, root, device, arrays
            )
            predictions[split].extend(records)
            if split == "selection":
                wording, _, _ = score_window(
                    model, text, row, root, device, arrays, True, memory, metadata
                )
                predictions["heldout-wording"].extend(wording)
            if row["dataset_id"] in targets:
                target = targets[row["dataset_id"]]
                if row["group_id"] in target["forbidden"]:
                    raise ValueError("Transfer target group overlap")
                questions = [
                    q
                    for q in row["questions"]
                    if transfer_families is None or q["family"] in transfer_families
                ]
                if questions:
                    target_records, _, _ = score_window(
                        target["model"],
                        text,
                        {**row, "questions": questions},
                        root,
                        device,
                        arrays,
                        False,
                        memory,
                        metadata,
                    )
                    target["records"].extend(target_records)
            done += 1
            if done % 200 == 0:
                print(json.dumps({"public_fp32_windows_scored": done}), flush=True)

    def save(path, records, split, checkpoint, state, excluded_groups, wording=False):
        header = {
            "type": "prediction_manifest",
            "split": split,
            "checkpoint_sha256": sha256(checkpoint),
            "data_manifest_sha256": sha256(root / "manifest.json"),
            "identity": state["identity"],
            "public_api": dict(PUBLIC_API),
            "source_provenance": source_provenance(),
            "evidence_precision": "fp32",
            "encoder_batch_size": 1,
            "channel_order": PUBLIC_API["channel_order"],
            "heldout_wording": wording,
            "temperature_in_graph": 1.0,
            "forbidden_calibration_groups": excluded_groups,
            "excluded_training_datasets": state["config"]["data"].get("exclude", []),
            "interpretation": "development evidence; not independent confirmation",
        }
        return write_predictions(path, header, records)

    for split, records in predictions.items():
        save(
            directory / f"public-{split}.jsonl",
            records,
            "selection" if split == "heldout-wording" else split,
            directory / "best.pt",
            payload,
            forbidden,
            split == "heldout-wording",
        )
    for target in targets.values():
        if not target["records"]:
            raise ValueError("No questions matched the requested transfer families")
        save(
            target["folder"] / "public-heldout-domain.jsonl",
            target["records"],
            "heldout_domain",
            target["folder"] / "best.pt",
            target["state"],
            target["forbidden"],
        )
    print(json.dumps({"public_fp32_assessment_windows": done}), flush=True)


def main():
    """Run public-path evaluation with optional explicit transfer checkpoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--transfer",
        nargs=2,
        action="append",
        default=[],
        metavar=("DATASET", "RUN_DIR"),
    )
    parser.add_argument("--transfer-family", action="append")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    torch.set_num_threads(config.get("threads", 6))
    run(config, args.run_dir, dict(args.transfer), args.transfer_family)


if __name__ == "__main__":
    main()
