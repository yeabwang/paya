"""Retrieve only aggregate pilot reports; never download weights or datasets."""
import json
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    destination = root / '.temp' / 'reports' / 'next'
    destination.mkdir(parents=True, exist_ok=True)
    reports = {
        'pilot-summary.json': 'runs/paya-finetune-pilot-001/summary.json',
        'preservation-summary.json': 'runs/paya-finetune-preservation-001/summary.json',
        'conflict-summary.json': 'runs/paya-finetune-conflict-001/summary.json',
        'details.json': 'runs/paya-finetune-details.json',
        'sample-support.json': 'runs/paya-finetune-sample-support.json',
        'online.json': 'runs/paya-finetune-online.json',
        'review.json': 'runs/paya-finetune-review.json',
        'role-report.json': 'data/experiment-roles-v2/report.json',
        'integration.json': 'runs/paya-finetune-integration/result.json',
        'gradient-probe.json': 'runs/paya-finetune-pilot-001/gradient-probe.json',
        'coverage.json': 'runs/paya-next-audit/coverage.json',
    }
    for name, remote in reports.items():
        result = subprocess.run([sys.executable, str(root / '.temp' / 'remote.py'),
                                 '--target', 'new', 'fetch', remote],
                                cwd=root, capture_output=True, check=True)
        value = json.loads(result.stdout.decode('utf-8'))
        (destination / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps({'report': name, 'bytes': len(result.stdout)}), flush=True)


if __name__ == '__main__':
    main()
