"""Run public evaluation, calibration, export, verification, and latency measurement."""

import argparse
import json
from pathlib import Path

import torch
import yaml

from benchmarks.run import benchmark
from benchmarks.verify import verify
from paya.assess import calibration_record, read_predictions, report
from paya.evaluate import run as evaluate_public
from paya.export import export
from paya.provenance import sha256


def finalize(config_path, directory, output, transfers=None, transfer_families=None):
    """Finalize a completed run at an explicit, previously empty export location.

    Transfer jobs are supplied by the caller. Calibration uses separate groups,
    and timings run only after assessment and artifact verification complete.
    """
    config = yaml.safe_load(Path(config_path).read_text())
    directory, output = Path(directory), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    checkpoint = directory / "best.pt"
    torch.set_num_threads(config.get("threads", 6))
    evaluate_public(config, directory, transfers, transfer_families)
    predictions = directory / "public-calibration.jsonl"
    header, calibration = read_predictions(predictions)
    binding = calibration_record(header, calibration, sha256(predictions))
    calibration_path = directory / "global-temperature.json"
    calibration_path.write_text(json.dumps(binding, indent=2))
    _, selection = read_predictions(directory / "public-selection.jsonl")
    temperature = binding["temperature"]
    result = {
        "raw_selection": report(selection),
        "scaled_selection": report(selection, temperature),
        "raw_calibration": report(calibration),
        "scaled_calibration": report(calibration, temperature),
    }
    (directory / "calibration-comparison.json").write_text(json.dumps(result, indent=2))
    export(checkpoint, config["assets"], output, calibration_path)
    verification = verify(
        output,
        config["data"]["root"],
        directory / "public-verification.json",
        checkpoint=checkpoint,
        assets=config["assets"],
    )
    measurements = {
        request.stem: benchmark(
            checkpoint,
            config["assets"],
            json.loads(request.read_text()),
            device=verification["device"],
            iterations=30,
            bundle=output,
        )
        for request in directory.glob("*-request.json")
    }
    (directory / "latency.json").write_text(json.dumps(measurements, indent=2))
    return {
        "finished": str(directory),
        "global_temperature": temperature,
        "artifact": str(output),
    }


def main():
    """Finalize an explicitly selected training run and optional transfer experiments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--transfer",
        nargs=2,
        action="append",
        default=[],
        metavar=("DATASET", "RUN_DIR"),
    )
    parser.add_argument("--transfer-family", action="append")
    args = parser.parse_args()
    print(
        json.dumps(
            finalize(
                args.config,
                args.run_dir,
                args.output,
                dict(args.transfer),
                args.transfer_family,
            )
        )
    )


if __name__ == "__main__":
    main()
