"""Verify export integrity and representative public requests; cache comparison is optional."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from paya.features import load_cache
from paya.inference import answer, load_bundle, load_checkpoint, logits
from paya.inputs import prepare_signal
from paya.text import TextEncoder


def verify(
    bundle, data, output, checkpoint=None, assets=None, feature_cache=None, device=None
):
    """Check one selection window per dataset using the actual exported model.

    Verify one encoder/head call, semantic permutation invariance, and artifact
    reload agreement. An optional source checkpoint is compared using the same
    local text encoder. FP16 training-cache differences are reported separately
    and are never treated as public-path equivalence or accuracy evidence.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    bundle, root, output = Path(bundle), Path(data).resolve(), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model, text = load_bundle(bundle, device)
    reloaded, reloaded_text = load_bundle(bundle, device)
    source, source_text = None, None
    if checkpoint:
        source_text = TextEncoder(device=device, model_path=bundle / "text_encoder")
        source, _, _ = load_checkpoint(
            checkpoint, assets or bundle / "normwear", source_text, device
        )
        source.temperature.copy_(model.temperature)
    evidence = (
        load_cache(feature_cache, root, assets or bundle / "normwear")
        if feature_cache
        else None
    )
    rows, results = {}, {}
    for line in (root / "selection.jsonl").read_text().splitlines():
        row = json.loads(line)
        rows.setdefault(row["dataset_id"], row)
    for name, row in rows.items():
        signal_path = (root / row["signal_file"]).resolve()
        if not signal_path.is_relative_to(root):
            raise ValueError("Signal file escapes prepared data")
        values = np.load(signal_path, mmap_mode="r")[row["index"]][
            row["valid_channels"]
        ]
        question = next(q for q in row["questions"] if q["family"] == "native")
        request = {
            "signal": {
                "sampling_rate_hz": row["sampling_rate_hz"],
                "channels": [
                    {**m, "samples": v.tolist()}
                    for m, v in zip(row["channels"], values)
                ],
            },
            "question": question["question"],
            "choices": question["choices"],
        }
        counts = {"encoder": 0, "head": 0}

        def core_hook(*args):
            counts["encoder"] += 1

        def head_hook(*args):
            counts["head"] += 1

        handles = [
            model.core.register_forward_hook(core_hook),
            model.head.register_forward_hook(head_hook),
        ]
        try:
            reference = answer(model, text, request, device)
        finally:
            for handle in handles:
                handle.remove()
        if counts != {"encoder": 1, "head": 1}:
            raise AssertionError("Repeated decision computation")
        loaded = answer(reloaded, reloaded_text, request, device)
        np.testing.assert_allclose(
            reference["probabilities"], loaded["probabilities"], atol=1e-4, rtol=1e-4
        )
        record = {
            "calls": counts,
            "export_reload_max_probability_delta": float(
                np.max(
                    np.abs(
                        np.array(reference["probabilities"]) - loaded["probabilities"]
                    )
                )
            ),
            "choice": reference["choice"],
            "probability": reference["probability"],
        }
        if source is not None:
            original = answer(source, source_text, request, device)
            np.testing.assert_allclose(
                reference["probabilities"],
                original["probabilities"],
                atol=1e-4,
                rtol=1e-4,
            )
            record["source_checkpoint_max_probability_delta"] = float(
                np.max(
                    np.abs(
                        np.array(reference["probabilities"]) - original["probabilities"]
                    )
                )
            )
        permuted = deepcopy(request)
        permuted["dataset_id"] = "ignored audit identifier"
        permuted["signal"]["channels"].reverse()
        permuted["choices"].reverse()
        changed = answer(model, text, permuted, device)
        np.testing.assert_allclose(
            reference["probabilities"],
            changed["probabilities"][::-1],
            atol=1e-5,
            rtol=1e-5,
        )
        if reference["choice"] != changed["choice"]:
            raise AssertionError("Semantic answer changed under permutation")
        record["permutation_passed"] = True
        if evidence is not None:
            _, metadata = prepare_signal(request["signal"])
            with torch.inference_mode():
                cached = (
                    logits(
                        model,
                        text,
                        request["question"],
                        request["choices"],
                        metadata,
                        device,
                        signal_memory=evidence[row["source_id"]][None].to(device),
                    )
                    .softmax(-1)[0]
                    .cpu()
                    .tolist()
                )
            record["fp16_evidence_max_probability_delta"] = float(
                np.max(np.abs(np.array(reference["probabilities"]) - cached))
            )
            record["fp16_evidence_same_choice"] = (
                request["choices"][int(np.argmax(cached))] == reference["choice"]
            )
        results[name] = record
        (output.parent / f"{name}-request.json").write_text(json.dumps(request))
    value = {
        "status": "passed",
        "device": device,
        "temperature": float(model.temperature),
        "tasks": results,
        "scope": "one selection window per dataset; integrity and API checks only",
    }
    output.write_text(json.dumps(value, indent=2))
    return value


def main():
    """Verify a bundle, with optional source-checkpoint and training-cache diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("checkpoint", "assets", "feature-cache"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--device")
    args = parser.parse_args()
    torch.set_num_threads(6)
    print(
        json.dumps(
            verify(
                args.bundle,
                args.data,
                args.output,
                args.checkpoint,
                args.assets,
                args.feature_cache,
                args.device,
            )
        )
    )


if __name__ == "__main__":
    main()
