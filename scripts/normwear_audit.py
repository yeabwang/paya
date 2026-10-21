"""Fetch pinned public configuration/source for review, without weights or execution."""
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


def read(url):
    with urlopen(Request(url, headers={'User-Agent': 'Paya-source-audit'}), timeout=30) as response:
        data = response.read(1_000_001)
    if len(data) > 1_000_000:
        raise ValueError('Source-audit response exceeded the one-megabyte bound')
    return data


def main():
    root = Path(__file__).resolve().parents[1]
    target = root / '.temp/research/normwear'
    target.mkdir(parents=True, exist_ok=True)
    repository = 'mosaic-laboratory/normwear'
    info = json.loads(read('https://huggingface.co/api/models/' + repository))
    revision = info['sha']
    report = {'repository': repository, 'revision': revision,
              'license': info.get('cardData', {}).get('license'),
              'weights_downloaded': False, 'custom_code_executed': False, 'files': {}}
    for name in ['config.json', 'configuration_normwear.py', 'modeling_normwear.py',
                 'normwear.py', 'layers.py', 'README.md']:
        url = f'https://huggingface.co/{repository}/resolve/{revision}/{name}'
        data = read(url)
        (target / name).write_bytes(data)
        report['files'][name] = {'url': url, 'bytes': len(data),
                                'sha256': hashlib.sha256(data).hexdigest()}
    upstream = 'Mobile-Sensing-and-UbiComp-Laboratory/NormWear'
    github = json.loads(read('https://api.github.com/repos/' + upstream + '/commits/main'))
    report['github_revision'] = github['sha']
    for name in ['main_model.py', 'LICENSE']:
        url = f'https://raw.githubusercontent.com/{upstream}/{github["sha"]}/{name}'
        data = read(url)
        (target / ('github-' + name)).write_bytes(data)
        report['files']['github-' + name] = {'url': url, 'bytes': len(data),
                                            'sha256': hashlib.sha256(data).hexdigest()}
    output = root / '.temp/reports/next/normwear-source.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'revision': revision, 'license': report['license'],
                      'files': len(report['files']), 'weights_downloaded': False}))


if __name__ == '__main__':
    main()
