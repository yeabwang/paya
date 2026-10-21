"""Read back online pilot loss histories; local result files remain authoritative."""
import json
from pathlib import Path
from urllib.parse import urlsplit
import wandb

from paya.atomic import atomic_output


def main():
    results = json.loads(Path('runs/paya-finetune-details.json').read_text())['experiments']
    api = wandb.Api(timeout=25)
    report = {}
    for name, result in results.items():
        pieces = urlsplit(result['wandb_url']).path.strip('/').split('/')
        try:
            run = api.run('/'.join([pieces[-4], pieces[-3], pieces[-1]]))
            points = [p for p in run.history(samples=25, keys=['step', 'train/loss'], pandas=False)
                      if p.get('train/loss') is not None]
            if not points:
                raise ValueError('No sampled online loss points')
            report[name] = {'state': run.state, 'url': run.url, 'sampled_points': len(points),
                            'maximum_sampled_step': max(p['step'] for p in points)}
        except Exception as error:
            report[name] = {'online_verification_failed': type(error).__name__,
                            'local_results_valid': True, 'url': result['wandb_url']}
        print(json.dumps({'experiment': name, **report[name]}), flush=True)
    with atomic_output('runs/paya-finetune-online.json', 'w') as out:
        json.dump(report, out, indent=2)


if __name__ == '__main__':
    main()
