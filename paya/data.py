"""Fresh JSONL decisions using separate state, question and ordered criteria."""

import json
from pathlib import Path
import random

from .inputs import validate_text


class ChoiceDataset:
    def __init__(self, path):
        self.path = Path(path)
        self.rows = []
        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            validate_text(row.get("question"), row.get("criteria"))
            if not isinstance(row.get("state"), dict):
                raise ValueError(f"Row {number} needs the new state object")
            target = row.get("target")
            if isinstance(target, bool) or not isinstance(target, int) or not 0 <= target < len(row["criteria"]):
                raise ValueError(f"Row {number} has an invalid target index")
            for name in ("source_id", "group_id", "recording_id"):
                if not isinstance(row.get(name), str) or not row[name].strip():
                    raise ValueError(f"Row {number} requires audit field {name}")
            self.rows.append(row)
        if not self.rows:
            raise ValueError("Dataset is empty")

    def batches(self, batch_size, seed=None):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        order = list(range(len(self.rows)))
        rng = random.Random(seed)
        if seed is not None:
            rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            rows = []
            for index in order[start:start+batch_size]:
                row = self.rows[index].copy()
                if seed is not None:
                    permutation = list(range(len(row["criteria"])))
                    rng.shuffle(permutation)
                    row["criteria"] = [row["criteria"][i] for i in permutation]
                    row["target"] = permutation.index(row["target"])
                rows.append(row)
            yield rows


def assert_disjoint(*datasets):
    for field in ("source_id", "group_id", "recording_id"):
        seen = set()
        for dataset in datasets:
            current = {row[field] for row in dataset.rows}
            if seen & current:
                raise ValueError(f"Dataset splits overlap on {field}")
            seen |= current


def make_batch(model, rows):
    return model.make_batch([r["state"] for r in rows], [r["question"] for r in rows],
                            [r["criteria"] for r in rows])
