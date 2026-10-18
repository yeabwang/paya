import numpy as np
import pytest

from paya.assess import fit_global, metrics, report
from paya.validate import spectral_label


def test_spectral_questions_have_known_physical_ground_truth():
    for frequency, expected in [(2, 0), (6, 1), (10, 2), (20, 3)]:
        t = np.arange(390) / 65
        label, totals = spectral_label(np.sin(2 * np.pi * frequency * t)[None], 65)
        assert label == expected and totals[expected] > 0.9 * sum(totals)


def test_group_intervals_and_calibration_boundaries():
    records = [
        {
            "logits": [4.0, -2.0] if i % 2 else [-2.0, 4.0],
            "target": i % 2,
            "group_id": f"g{i % 4}",
            "dataset_id": "one",
            "family": "native",
        }
        for i in range(40)
    ]
    r = report(records)
    assert r["by_family"]["one/native"]["groups"] == 4
    assert "group_bootstrap_ci95" in r["by_family"]["one/native"]
    assert r["by_family"]["one/native"]["group_bootstrap_ci95"]["auroc_macro_ovo"] == [
        0.0,
        0.0,
    ]
    assert r["by_family"]["one/native"]["group_bootstrap_valid_auroc_draws"] > 0
    with pytest.raises(ValueError, match="held-out"):
        fit_global(records, ["g0"])
    assert 0.05 <= fit_global(records) <= 20
    assert metrics(records, bootstrap=0)["count"] == 40
