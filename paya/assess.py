"""Compute grouped metrics, read prediction records, and fit bound global calibration."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)

from .provenance import PUBLIC_API, sha256


def probabilities(logits, temperature=1.0):
    """Compute stable float64 softmax after dividing logits by a scalar temperature."""
    z = np.asarray(logits, dtype=np.float64) / temperature
    z -= z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=-1, keepdims=True)


def ece(confidence, correct):
    """Compute top-label expected calibration error using 15 equal-width bins."""
    bins = np.minimum((confidence * 15).astype(int), 14)
    return float(
        sum(
            (bins == i).mean()
            * abs(confidence[bins == i].mean() - correct[bins == i].mean())
            for i in range(15)
            if (bins == i).any()
        )
    )


def metrics(records, temperature=1.0, bootstrap=200):
    """Measure one dataset/family with a fixed candidate count.

    Return accuracy, balanced accuracy, macro F1, AUROC when defined, NLL,
    Brier score, calibration bins and confidence gates. Optional bootstrap
    intervals resample whole groups and retain within-group window dependence.
    """
    z = np.array([r["logits"] for r in records])
    y = np.array([r["target"] for r in records])
    p = probabilities(z, temperature)
    pred = p.argmax(1)
    correct = (pred == y).astype(float)
    confidence = p.max(1)
    classes = np.arange(p.shape[1])
    auc = None
    if len(np.unique(y)) == len(classes):
        auc = float(
            roc_auc_score(y, p[:, 1])
            if len(classes) == 2
            else roc_auc_score(y, p, multi_class="ovo", average="macro")
        )
    groups = np.array([r["group_id"] for r in records])
    unique = np.unique(groups)
    result = {
        "count": len(y),
        "groups": len(unique),
        "accuracy": float(correct.mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(
            f1_score(y, pred, labels=classes, average="macro", zero_division=0)
        ),
        "recall_by_class": recall_score(
            y, pred, labels=classes, average=None, zero_division=0
        ).tolist(),
        "auroc_macro_ovo": auc,
        "ece_15": ece(confidence, correct),
        "nll": float(-np.log(p[np.arange(len(y)), y].clip(1e-12)).mean()),
        "brier": float(((p - np.eye(len(classes))[y]) ** 2).sum(1).mean()),
        "reliability": [],
        "confidence_gates": {},
        "group_evidence": "limited"
        if len(unique) < 5
        else "recorded; independence depends on source provenance",
    }
    for lo in np.arange(0, 1, 0.1):
        selected = (confidence >= lo) & (
            confidence < lo + 0.1 if lo < 0.9 else confidence <= 1
        )
        if selected.any():
            result["reliability"].append(
                {
                    "lower": float(lo),
                    "count": int(selected.sum()),
                    "confidence": float(confidence[selected].mean()),
                    "accuracy": float(correct[selected].mean()),
                }
            )
    for threshold in (0.8, 0.9, 0.95):
        keep = confidence >= threshold
        result["confidence_gates"][str(threshold)] = {
            "coverage": float(keep.mean()),
            "accuracy": float(correct[keep].mean()) if keep.any() else None,
        }
    if len(unique) >= 2 and bootstrap:
        rng = np.random.default_rng(42)
        # Resample independent groups, preserving correlated windows within each selected group.
        indexes = {g: np.flatnonzero(groups == g) for g in unique}
        estimates, auc_estimates = [], []
        for _ in range(bootstrap):
            idx = np.concatenate(
                [indexes[g] for g in rng.choice(unique, len(unique), replace=True)]
            )
            estimates.append((correct[idx].mean(), ece(confidence[idx], correct[idx])))
            if auc is not None and len(np.unique(y[idx])) == len(classes):
                auc_estimates.append(
                    float(
                        roc_auc_score(y[idx], p[idx, 1])
                        if len(classes) == 2
                        else roc_auc_score(
                            y[idx], p[idx], multi_class="ovo", average="macro"
                        )
                    )
                )
        result["group_bootstrap_ci95"] = dict(
            zip(
                ("accuracy", "ece_15"),
                np.percentile(estimates, [2.5, 97.5], axis=0).T.tolist(),
            )
        )
        if auc_estimates:
            result["group_bootstrap_ci95"]["auroc_macro_ovo"] = np.percentile(
                auc_estimates, [2.5, 97.5]
            ).tolist()
        result["group_bootstrap_valid_auroc_draws"] = len(auc_estimates)
    return result


def report(records, temperature=1.0, bootstrap=200):
    """Group records by dataset/family and return unweighted native/family means.

    Undefined metrics are omitted from their respective means. These means
    are distinct from pooled question accuracy.
    """
    groups = defaultdict(list)
    for r in records:
        groups[f"{r['dataset_id']}/{r['family']}"].append(r)
    by_family = {
        key: metrics(rows, temperature, bootstrap) for key, rows in groups.items()
    }
    native = [m for k, m in by_family.items() if k.endswith("/native")]

    def mean(key, values):
        defined = [m[key] for m in values if m[key] is not None]
        return float(np.mean(defined)) if defined else None

    return {
        "temperature": temperature,
        "by_family": by_family,
        "native_mean": {
            k: mean(k, native)
            for k in (
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "auroc_macro_ovo",
                "ece_15",
                "nll",
                "brier",
            )
        },
        "family_mean": {
            k: mean(k, list(by_family.values()))
            for k in (
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "auroc_macro_ovo",
                "ece_15",
                "nll",
                "brier",
            )
        },
    }


def fit_global(records, forbidden_groups=()):
    """Fit a single positive temperature on independent calibration groups.

    Reject empty data, overlap with forbidden groups, and fewer than two
    calibration groups. Minimize target NLL within the fixed 0.05 to 20 bounds.
    """
    if not records or {r["group_id"] for r in records} & set(forbidden_groups):
        raise ValueError("Calibration must use independent held-out groups")
    if len({r["group_id"] for r in records}) < 2:
        raise ValueError("Global calibration needs at least two independent groups")

    def loss(log_t):
        return negative_log_likelihood(records, np.exp(log_t))

    return float(
        np.exp(
            minimize_scalar(loss, bounds=(np.log(0.05), np.log(20)), method="bounded").x
        )
    )


def negative_log_likelihood(records, temperature=1.0):
    """Return mean target NLL across records with potentially different choice counts."""
    return float(
        np.mean(
            [
                -np.log(
                    probabilities(r["logits"], temperature)[r["target"]].clip(1e-12)
                )
                for r in records
            ]
        )
    )


def calibration_record(header, records, predictions_sha256):
    """Fit one temperature to independent, uncalibrated public-path predictions.

    Return the binding consumed by export, including its checkpoint, prediction
    checksum, API contract, calibration groups, and fitting-set NLL comparison.
    Cached-evidence predictions cannot calibrate the waveform inference path.
    """
    if header["split"] != "calibration":
        raise ValueError("Only the declared calibration split can fit a temperature")
    if header.get("temperature_in_graph", 1.0) != 1.0:
        raise ValueError("Do not refit an already calibrated graph")
    if header.get("public_api") != PUBLIC_API:
        raise ValueError("Calibration requires canonical public-path predictions")
    if (
        header.get("evidence_precision") != "fp32"
        or header.get("encoder_batch_size") != 1
    ):
        raise ValueError("Calibration requires FP32 batch-one waveform predictions")
    temperature = fit_global(records, header["forbidden_calibration_groups"])
    before = negative_log_likelihood(records)
    after = negative_log_likelihood(records, temperature)
    if after > before:
        temperature, after = 1.0, before
    return {
        "temperature": temperature,
        "checkpoint_sha256": header["checkpoint_sha256"],
        "predictions_sha256": predictions_sha256,
        "public_api": dict(PUBLIC_API),
        "evaluation_path": "public_fp32_encoder_batch_one",
        "calibration_groups": sorted({r["group_id"] for r in records}),
        "calibration_nll_raw": before,
        "calibration_nll_fitted": after,
        "interpretation": "fitting-set improvement; not independent confirmation",
    }


def write_predictions(path, header, records):
    """Write a provenance header, JSONL predictions, and adjacent metric summary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        stream.write(json.dumps(header) + "\n")
        for record in records:
            stream.write(json.dumps(record) + "\n")
    result = {"provenance": header, **report(records)}
    path.with_suffix(".metrics.json").write_text(json.dumps(result, indent=2))
    return result


def read_predictions(path):
    """Read a prediction_manifest header and its subsequent JSONL prediction records."""
    with Path(path).open() as stream:
        header = json.loads(next(stream))
        if header.get("type") != "prediction_manifest":
            raise ValueError("Missing prediction provenance")
        return header, [json.loads(line) for line in stream]


def main():
    """Write metrics or fit a checkpoint-bound temperature from public calibration records."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fit-global", action="store_true")
    a = p.parse_args()
    header, records = read_predictions(a.predictions)
    if a.fit_global:
        value = calibration_record(header, records, sha256(a.predictions))
    else:
        value = {"provenance": header, **report(records)}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
