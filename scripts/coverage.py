"""Bounded metadata audit of QA agreement, independent support and split roles."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3

import pyarrow.parquet as pq
import yaml

from paya.atomic import atomic_output
from paya.data import ChoiceDataset, assert_disjoint
from paya.provenance import fingerprint, sha256


class Components:
    def __init__(self, groups):
        self.parent = {g: g for g in groups}

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, values):
        roots = sorted({self.find(v) for v in values})
        for root in roots[1:]:
            self.parent[root] = roots[0]


def summarize(db):
    """Return aggregate counts only; patient and waveform identifiers stay local."""
    groups = Components(r[0] for r in db.execute('SELECT DISTINCT group_id FROM examples'))
    for field in ['waveform', 'recording']:
        for values, in db.execute(f'SELECT GROUP_CONCAT(DISTINCT group_id) FROM examples '
                                  f'GROUP BY {field} HAVING COUNT(DISTINCT group_id)>1'):
            groups.union(values.split(','))
    db.execute('CREATE TABLE IF NOT EXISTS components(group_id TEXT PRIMARY KEY,component TEXT)')
    db.executemany('INSERT OR REPLACE INTO components VALUES (?,?)',
                   ((g, groups.find(g)) for g in groups.parent))
    db.commit()
    joined = ' FROM examples e JOIN components c USING(group_id) '
    if db.execute('SELECT 1'+joined+'GROUP BY component HAVING COUNT(DISTINCT role)>1 LIMIT 1').fetchone():
        raise ValueError('Connected subject components overlap split roles')
    coverage = defaultdict(lambda: defaultdict(dict))
    for role, task, answer, count, waves, recordings, components in db.execute(
            'SELECT role,task,answer,COUNT(*),COUNT(DISTINCT waveform),COUNT(DISTINCT recording),'
            'COUNT(DISTINCT component)'+joined+'GROUP BY role,task,answer'):
        largest = db.execute('SELECT MAX(n) FROM (SELECT COUNT(*) n'+joined+
            'WHERE role=? AND task=? AND answer=? GROUP BY component)', (role, task, answer)).fetchone()[0]
        coverage[role][task][answer] = dict(rows=count, waveforms=waves, recordings=recordings,
            independent_groups=components, largest_group_fraction=largest/count)
    contradictions = []
    for role, task, waves, rows in db.execute(
            'SELECT role,task,COUNT(*),SUM(n) FROM (SELECT role,task,waveform,COUNT(*) n FROM examples '
            'GROUP BY role,task,waveform HAVING COUNT(DISTINCT answer)>1) GROUP BY role,task'):
        contradictions.append(dict(role=role, task=task, waveforms=waves, affected_qa_rows=rows))
    combined = {}
    for task, answer, count, waves, components in db.execute(
            'SELECT task,answer,COUNT(*),COUNT(DISTINCT waveform),COUNT(DISTINCT component)'+joined+
            'GROUP BY task,answer'):
        combined.setdefault(task, {})[answer] = dict(rows=count, waveforms=waves,
            independent_groups=components, can_cover_four_roles_by_group_count=components >= 4)
    return dict(coverage=coverage, task_answer_support=combined,
        contradictory_waveform_labels=contradictions,
        independent_components=len({groups.find(g) for g in groups.parent}),
        split_component_overlap=False,
        insufficient_for_four_roles=[dict(task=t, answer=a, **v) for t, labels in combined.items()
                                   for a, v in labels.items() if v['independent_groups'] < 4],
        limitations=['Connected components reflect known identities, not proof against undisclosed aliases.',
                    'Near-duplicate temporal overlap cannot be ruled out without original timing metadata.',
                    'QA disagreements are flagged; no labels or waveform rows are modified.'])


def audit(config_path, destination):
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(config_path).read_text())
    datasets = {role: ChoiceDataset(config['data'][role]) for role in ['train', 'selection', 'calibration']}
    # Test labels are not used by this audit. Existing test identity index is checked only for isolation.
    test = ChoiceDataset(config['data']['test'])
    assert_disjoint(*datasets.values(), test)
    roots = list(Path('data').glob('*/provenance'))
    if len(roots) != 1:
        raise ValueError('Expected one local QA provenance root')
    binding = fingerprint({'datasets': {r: d.sha256 for r, d in datasets.items()},
                           'code': sha256(Path(__file__))})
    db = sqlite3.connect(destination/'metadata.sqlite3')
    db.executescript('PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA cache_size=-16384; '
        'PRAGMA temp_store=FILE; CREATE TABLE IF NOT EXISTS settings(binding TEXT); '
        'CREATE TABLE IF NOT EXISTS files(role TEXT,path TEXT,digest TEXT,PRIMARY KEY(role,path)); '
        'CREATE TABLE IF NOT EXISTS examples(id TEXT PRIMARY KEY,role TEXT,subset TEXT,task TEXT,answer TEXT,'
        'waveform TEXT,group_id TEXT,recording TEXT,question_hash TEXT,vocabulary TEXT,'
        'filename TEXT,row_group INTEGER,position INTEGER);')
    previous = db.execute('SELECT binding FROM settings').fetchone()
    if previous and previous[0] != binding:
        raise ValueError('Audit bindings changed; use a new output directory')
    if not previous:
        db.execute('INSERT INTO settings VALUES (?)', (binding,)); db.commit()
    processed = 0
    for role, dataset in datasets.items():
        for file in dataset.files:
            digest = sha256(file)
            saved = db.execute('SELECT digest FROM files WHERE role=? AND path=?', (role, str(file))).fetchone()
            if saved:
                if saved[0] != digest:
                    raise ValueError('Completed audit shard changed')
                continue
            origin = roots[0]/file.parent.name/file.name
            origins = {r['mapping_id']: r for b in pq.ParquetFile(origin).iter_batches(
                batch_size=512, columns=['mapping_id', 'qa_task', 'upstream_answer']) for r in b.to_pylist()}
            reader = pq.ParquetFile(file)
            with db:
                for group in range(reader.num_row_groups):
                    rows = reader.read_row_group(group, columns=['mapping_id', 'group_id', 'recording_id',
                        'source_id', 'question', 'criteria', 'target']).to_pylist()
                    values = []
                    for position, row in enumerate(rows):
                        ref = origins[row['mapping_id']]
                        answer = row['criteria'][row['target']]
                        if answer != ref['upstream_answer']:
                            raise ValueError('Mapped QA answer differs from supplied QA provenance')
                        values.append((row['mapping_id'], role, file.parent.name, ref['qa_task'], answer,
                            row['source_id'], row['group_id'], row['recording_id'],
                            hashlib.sha256(row['question'].encode()).hexdigest(),
                            json.dumps(sorted(row['criteria'])), str(file), group, position))
                    db.executemany('INSERT INTO examples VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)', values)
                db.execute('INSERT INTO files VALUES (?,?,?)', (role, str(file), digest))
            processed += 1
            if processed % 25 == 0:
                print(json.dumps({'phase': 'metadata_audit', 'completed_shards': processed}), flush=True)
    db.executescript('CREATE INDEX IF NOT EXISTS by_waveform ON examples(waveform); '
        'CREATE INDEX IF NOT EXISTS by_recording ON examples(recording); '
        'CREATE INDEX IF NOT EXISTS by_group ON examples(group_id); '
        'CREATE INDEX IF NOT EXISTS by_class ON examples(role,task,answer);')
    report = summarize(db)
    report.update(binding=binding, datasets={r: {'rows':len(d.rows),'sha256':d.sha256}
                    for r,d in datasets.items()}, qa_answers_match_provenance=True,
                    test_used_for_optimization=False, waveform_data_changed=False)
    for role, data in datasets.items():
        assert db.execute('SELECT COUNT(*) FROM examples WHERE role=?', (role,)).fetchone()[0] == len(data.rows)
    with atomic_output(destination/'coverage.json', 'w') as out:
        json.dump(report, out, indent=2, allow_nan=False)
    db.close()
    print(json.dumps({'phase':'audit_complete','independent_components':report['independent_components'],
        'contradictory_waveform_labels':report['contradictory_waveform_labels'],
        'insufficient_for_four_roles':report['insufficient_for_four_roles']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/paya-compact-full.yaml')
    parser.add_argument('--output', default='runs/paya-next-audit')
    args = parser.parse_args(); audit(args.config, args.output)
