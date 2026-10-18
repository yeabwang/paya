"""Read prepared question families and build deterministic, shape-compatible batches."""

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .inputs import ENDING, prepare_signal, validate_question


class ChoiceDataset(Dataset):
    """Expand prepared waveform rows into individual question/choice examples.

    Audit IDs remain in returned records and never enter scoring inputs.
    The epoch and seed control wording and candidate-order augmentation.
    """

    def __init__(
        self,
        root,
        split,
        text,
        augment=False,
        seed=42,
        cache=None,
        exclude=(),
        families=None,
        heldout_wording=False,
    ):
        """Read one split and validate targets, with optional dataset/family filtering."""
        self.root, self.text, self.augment, self.seed, self.epoch = (
            Path(root),
            text,
            augment,
            seed,
            0,
        )
        self.examples, self.arrays, self.cache = [], {}, cache
        self.heldout_wording = heldout_wording
        for line in (self.root / f"{split}.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["dataset_id"] in exclude:
                continue
            for q in row["questions"]:
                if families is not None and q["family"] not in families:
                    continue
                validate_question(q["question"], q["choices"])
                if not 0 <= q["label"] < len(q["choices"]):
                    raise ValueError("Question target is outside supplied alternatives")
                self.examples.append((row, q))

    def __len__(self):
        """Return the number of expanded question examples."""
        return len(self.examples)

    def __getitem__(self, index):
        """Return one question with physical metadata, text vectors, and its target.

        Candidate shuffling remaps the label. A supplied cache replaces waveform
        input with frozen evidence after validating the physical channel count.
        """
        row, q = self.examples[index]
        rng = random.Random(self.seed + self.epoch * len(self) + index)
        order = list(range(len(q["choices"])))
        if self.augment:
            rng.shuffle(order)
        prompt = rng.choice(q["templates"]) if self.augment else q["question"]
        if self.heldout_wording:
            prompt = q["heldout_question"]
        choices = [q["choices"][i] for i in order]
        vectors = self.text.encode([prompt, *choices, ENDING])
        filename = row["signal_file"]
        if filename not in self.arrays:
            path = (self.root / filename).resolve()
            if not path.is_relative_to(self.root.resolve()):
                raise ValueError("Signal file must remain in prepared root")
            self.arrays[filename] = np.load(path, mmap_mode="r")
        values = self.arrays[filename][row["index"]][row["valid_channels"]]
        signal, metadata = prepare_signal(
            {
                "sampling_rate_hz": row["sampling_rate_hz"],
                "channels": [
                    {**m, "samples": x} for m, x in zip(row["channels"], values)
                ],
            }
        )
        item = {
            "question": vectors[0],
            "choices": vectors[1:-1],
            "ending": vectors[-1],
            "sensor_metadata": metadata,
            "labels": order.index(q["label"]),
            "order": order,
            "descriptions": choices,
            "family": q["family"],
            "dataset_id": row["dataset_id"],
            "group_id": row["group_id"],
            "source_id": row["source_id"],
        }
        if self.cache is None:
            item["signals"] = signal
        else:
            memory = self.cache[row["source_id"]]
            if memory.ndim != 3 or memory.shape[0] != len(metadata):
                raise ValueError("Cache does not match physical channel layout")
            item["signal_memory"] = memory
        return item


class ShapeSampler(Sampler):
    """Group examples by waveform shape and physical channel count."""

    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset, self.batch_size, self.shuffle = dataset, batch_size, shuffle

    def __iter__(self):
        """Yield deterministic shape-compatible index batches for the current epoch."""
        groups = defaultdict(list)
        for i, (r, _) in enumerate(self.dataset.examples):
            groups[(r["signal_file"], len(r["valid_channels"]))].append(i)
        rng = random.Random(self.dataset.seed + self.dataset.epoch)
        batches = []
        for indices in groups.values():
            if self.shuffle:
                rng.shuffle(indices)
            batches.extend(
                indices[i : i + self.batch_size]
                for i in range(0, len(indices), self.batch_size)
            )
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self):
        """Return the number of batches, including each incomplete shape bucket."""
        groups = defaultdict(int)
        for r, _ in self.dataset.examples:
            groups[(r["signal_file"], len(r["valid_channels"]))] += 1
        return sum(
            (n + self.batch_size - 1) // self.batch_size for n in groups.values()
        )


def collate(items):
    """Stack one shape-compatible batch and pad only the candidate dimension.

    Return model inputs, integer labels, and original audit records.
    choice_mask identifies valid alternatives; padded candidates have no target.
    """
    b, k = len(items), max(len(x["choices"]) for x in items)
    choices = torch.zeros(b, k, items[0]["choices"].shape[-1])
    mask = torch.zeros(b, k, dtype=torch.bool)
    for i, x in enumerate(items):
        choices[i, : len(x["choices"])] = x["choices"]
        mask[i, : len(x["choices"])] = True
    field = "signals" if "signals" in items[0] else "signal_memory"
    metadata = torch.stack([x["sensor_metadata"] for x in items])
    return {
        "inputs": {
            field: torch.stack([x[field] for x in items]),
            "sensor_metadata": metadata,
            "sensor_mask": torch.ones(metadata.shape[:2], dtype=torch.bool),
            "question": torch.stack([x["question"] for x in items]),
            "choices": choices,
            "ending": torch.stack([x["ending"] for x in items]),
            "choice_mask": mask,
        },
        "labels": torch.tensor([x["labels"] for x in items]),
        "records": items,
    }


def to_device(inputs, device):
    """Move every model-input tensor to a device without altering audit records."""
    return {k: v.to(device) for k, v in inputs.items()}


def loader(dataset, batch_size, train=False):
    """Create a single-process DataLoader using shape buckets and candidate padding."""
    return DataLoader(
        dataset,
        batch_sampler=ShapeSampler(dataset, batch_size, train),
        collate_fn=collate,
        num_workers=0,
    )
