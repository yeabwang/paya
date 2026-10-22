"""Task and class metrics without treating dataset names as task identities."""

import json
from pathlib import Path
import sqlite3
import tempfile

from .classification import classification_metrics
from .evaluate import metrics


def task_report(records, temperature=1.0):
    """Report supplied tasks with bounded memory; disclose rows lacking task labels."""
    with tempfile.TemporaryDirectory(prefix="paya-task-metrics-") as directory:
        db = sqlite3.connect(str(Path(directory) / "predictions.sqlite3"))
        try:
            db.executescript(
                "PRAGMA cache_size=-8192; PRAGMA temp_store=FILE; "
                "CREATE TABLE records(task TEXT, payload TEXT);"
            )
            missing, pending = 0, []
            for row in records:
                task = row.get("task")
                if task is None:
                    missing += 1
                    continue
                if not isinstance(task, str) or not task.strip():
                    raise ValueError("Prediction task must be a nonempty string")
                pending.append((task, json.dumps(row)))
                if len(pending) >= 2048:
                    db.executemany("INSERT INTO records VALUES (?,?)", pending)
                    pending.clear()
            db.executemany("INSERT INTO records VALUES (?,?)", pending)
            db.execute("CREATE INDEX by_task ON records(task)")
            tasks = {}
            for (task,) in db.execute(
                "SELECT DISTINCT task FROM records ORDER BY task"
            ).fetchall():

                def rows():
                    return (
                        json.loads(row[0])
                        for row in db.execute(
                            "SELECT payload FROM records WHERE task=?", (task,)
                        )
                    )

                tasks[task] = {
                    **metrics(rows(), temperature),
                    **classification_metrics(rows(), temperature),
                }
            mean = {}
            for key in (
                "accuracy",
                "nll",
                "brier",
                "ece_15",
                "macro_f1",
                "macro_auroc_ovr",
            ):
                values = [row[key] for row in tasks.values() if row[key] is not None]
                mean[key] = sum(values) / len(values) if values else None
            return {"tasks": tasks, "task_mean": mean, "rows_without_task": missing}
        finally:
            db.close()
