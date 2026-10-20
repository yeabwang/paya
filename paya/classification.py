"""Class-sensitive metrics aligned by answer text, with disk-backed exact AUROC."""

from collections import Counter
import math
import sqlite3
import tempfile

import numpy as np


def classification_metrics(records, temperature=1.0):
    """Score one task; report undefined class AUCs without inventing values.

    Candidate order may vary, but every row must contain the same answer set.
    Scores are sorted on disk, keeping memory bounded for large evaluations.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    confusion = Counter()
    labels, count = None, 0
    with tempfile.TemporaryDirectory(prefix="paya-auc-") as directory:
        db = sqlite3.connect(directory + "/scores.sqlite3")
        try:
            db.executescript("PRAGMA cache_size=-8192; PRAGMA temp_store=FILE; "
                             "CREATE TABLE scores(label TEXT, score REAL, positive INTEGER)")
            pending = []
            for row in records:
                criteria = row["criteria"]
                current = frozenset(criteria)
                if len(current) != len(criteria) or len(current) < 2:
                    raise ValueError("Need distinct candidate answers")
                if labels is None:
                    labels = current
                if current != labels:
                    raise ValueError("Classification metrics require one consistent task vocabulary")
                target = row["target"]
                if isinstance(target, bool) or not isinstance(target, int) or not 0 <= target < len(criteria):
                    raise ValueError("Invalid candidate target")
                logits = np.asarray(row["logits"], dtype=np.float64) / temperature
                if logits.shape != (len(criteria),) or not np.isfinite(logits).all():
                    raise ValueError("Need one finite logit per candidate")
                logits -= logits.max()
                probabilities = np.exp(logits)
                probabilities /= probabilities.sum()
                truth = criteria[target]
                winner = min(range(len(criteria)), key=lambda i: (-probabilities[i], criteria[i]))
                confusion[truth, criteria[winner]] += 1
                pending.extend((answer, float(score), int(answer == truth))
                               for answer, score in zip(criteria, probabilities))
                count += 1
                if len(pending) >= 4096:
                    db.executemany("INSERT INTO scores VALUES (?,?,?)", pending)
                    pending.clear()
            if not count:
                raise ValueError("Need at least one prediction")
            db.executemany("INSERT INTO scores VALUES (?,?,?)", pending)
            db.execute("CREATE INDEX ranked ON scores(label,score)")
            per_class = {}
            for label in sorted(labels):
                support = sum(v for (truth, _), v in confusion.items() if truth == label)
                predicted = sum(v for (_, winner), v in confusion.items() if winner == label)
                tp = confusion[label, label]
                negative = count - support
                auc = None
                if support and negative:
                    below, concordant = 0, 0.0
                    for _, positives, size in db.execute(
                            "SELECT score,SUM(positive),COUNT(*) FROM scores WHERE label=? "
                            "GROUP BY score ORDER BY score", (label,)):
                        negatives = size - positives
                        concordant += positives * (below + 0.5 * negatives)
                        below += negatives
                    auc = concordant / (support * negative)
                per_class[label] = {"support": support, "predicted": predicted,
                    "correct": tp, "precision": tp / predicted if predicted else 0.0,
                    "recall": tp / support if support else None,
                    "f1": 2 * tp / (support + predicted) if support + predicted else 0.0,
                    "auroc_ovr": auc}
            aucs = [v["auroc_ovr"] for v in per_class.values() if v["auroc_ovr"] is not None]
            observed = [v for v in per_class.values() if v["support"]]
            return {"count": count, "macro_f1": sum(v["f1"] for v in per_class.values()) / len(labels),
                    "observed_class_macro_recall": sum(v["recall"] for v in observed) / len(observed),
                    "macro_auroc_ovr": sum(aucs) / len(aucs) if aucs else None,
                    "auroc_supported_classes": len(aucs), "candidate_classes": len(labels),
                    "per_class": per_class,
                    "confusion": {truth: {winner: confusion[truth, winner] for winner in sorted(labels)}
                                  for truth in sorted(labels)}}
        finally:
            db.close()
