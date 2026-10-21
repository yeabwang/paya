"""Audit fixed pilot sample support without loading signal features or identities."""
from collections import Counter
import json
from pathlib import Path
import numpy as np

from paya.atomic import atomic_output
from paya.provenance import sha256


def support(path, limit):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    wanted = set(map(int, np.random.default_rng(20261021).permutation(manifest['rows'])[:limit]))
    counts = Counter()
    with (path / 'rows.jsonl').open() as source:
        for index, line in enumerate(source):
            if index in wanted:
                row = json.loads(line)
                counts[row['task'], row['criteria'][row['target']]] += 1
    return {'rows': sum(counts.values()), 'task_answer_rows': {
        task: {answer: count for (t, answer), count in sorted(counts.items()) if t == task}
        for task in sorted({t for t, _ in counts})}}


def main():
    report = {'train': support('cache/paya-full-bank/train', 64000),
              'monitor': support('cache/paya-development/selection', 16384),
              'source_sha256': sha256(Path(__file__))}
    coverage = json.loads(Path('runs/paya-next-audit/coverage.json').read_text())['task_answer_support']
    for role in ['train', 'monitor']:
        observed = report[role]['task_answer_rows']
        report[role]['missing_task_answers'] = [
            {'task': task, 'answer': answer} for task, answers in coverage.items()
            for answer in answers if answer not in observed.get(task, {})]
    with atomic_output('runs/paya-finetune-sample-support.json', 'w') as out:
        json.dump(report, out, indent=2, allow_nan=False)
    print(json.dumps({role: {'rows': report[role]['rows'],
                             'missing_task_answers': report[role]['missing_task_answers']}
                      for role in ['train', 'monitor']}), flush=True)


if __name__ == '__main__':
    main()
