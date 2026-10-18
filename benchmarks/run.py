"""Measure cold loading and warm public-request latency with cached and fresh text."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from paya.inference import answer, load_bundle, load_checkpoint
from paya.text import TextEncoder


def benchmark(checkpoint, assets, request, device="cuda", iterations=50, bundle=None):
    """Measure complete public inference after separately recording cold startup.

    Report warm p50/p95 for cached text and freshly encoded text with the
    models resident. CUDA timings synchronize the selected device. A bundle
    or a checkpoint plus encoder directory supplies the model.
    """
    started = time.perf_counter()
    if bundle:
        model, text = load_bundle(bundle, device)
    else:
        text = TextEncoder(device=device)
        model, _, _ = load_checkpoint(checkpoint, assets, text, device)
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    cold = time.perf_counter() - started
    first_start = time.perf_counter()
    answer(model, text, request, device)
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    first_request_ms = 1000 * (time.perf_counter() - first_start)
    results = {}
    for mode in ("cached_text", "new_text"):
        timings = []
        for _ in range(iterations):
            if mode == "new_text":
                text.embeddings.clear()
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            output = answer(model, text, request, device)
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            timings.append(1000 * (time.perf_counter() - start))
        results[mode] = {
            "p50_ms": float(np.percentile(timings, 50)),
            "p95_ms": float(np.percentile(timings, 95)),
            "iterations": iterations,
        }
    return {
        "device": torch.cuda.get_device_name(torch.device(device))
        if device.startswith("cuda")
        else "cpu",
        "precision": "FP32",
        "batch_size": 1,
        "model_load_seconds": cold,
        "first_request_ms": first_request_ms,
        "signal_shape": [
            len(request["signal"]["channels"]),
            len(request["signal"]["channels"][0]["samples"]),
        ],
        "choices": len(request["choices"]),
        "temperature": float(model.temperature),
        "artifact": str(bundle or checkpoint),
        "timings": results,
        "last_prediction": output,
    }


def main():
    """Benchmark a complete JSON request and save timing measurements."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--assets", type=Path)
    p.add_argument("--bundle", type=Path)
    p.add_argument("--request", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--iterations", type=int, default=50)
    a = p.parse_args()
    if not a.bundle and not (a.checkpoint and a.assets):
        p.error("Supply --bundle or both --checkpoint and --assets")
    value = benchmark(
        a.checkpoint,
        a.assets,
        json.loads(a.request.read_text()),
        a.device,
        a.iterations,
        a.bundle,
    )
    a.output.write_text(json.dumps(value, indent=2))
    print(json.dumps(value))


if __name__ == "__main__":
    main()
