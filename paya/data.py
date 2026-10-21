"""Read bounded decision blocks and enforce split isolation with a disk index."""

from collections.abc import Sequence
import json
import math
from pathlib import Path
import random
import sqlite3

from .atomic import atomic_output
from .inputs import validate_text
from .provenance import fingerprint, sha256

AUDIT_FIELDS = ("source_id", "group_id", "recording_id")
MAX_BLOCK_BYTES = 64 * 1024 * 1024
MAX_BLOCK_ROWS = 2048
INDEX_VERSION = 2


def validate_row(row, location):
    validate_text(row.get("question"), row.get("criteria"))
    if not isinstance(row.get("state"), dict):
        raise ValueError(f"{location} needs a state object")
    target = row.get("target")
    if isinstance(target, bool) or not isinstance(target, int) or not 0 <= target < len(row["criteria"]):
        raise ValueError(f"{location} has an invalid target index")
    for name in AUDIT_FIELDS:
        if not isinstance(row.get(name), str) or not row[name].strip():
            raise ValueError(f"{location} requires audit field {name}")


class _Rows(Sequence):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return self.dataset.count

    def __iter__(self):
        for block in self.dataset._blocks():
            yield from self.dataset._read_block(block)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        block = self.dataset.db.execute(
            'SELECT * FROM blocks WHERE row_start<=? ORDER BY row_start DESC LIMIT 1', (index,)).fetchone()
        return self.dataset._read_block(block)[index-block["row_start"]]


class ChoiceDataset:
    """Keep waveform rows on disk; hold at most one bounded block and one batch."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        self.row_indices = None
        self.manifest_snapshot = None
        if self.path.is_dir():
            self.files = sorted(self.path.rglob("*.parquet"))
            index = self.path / ".paya-index.sqlite3"
        elif self.path.suffix == ".json":
            manifest = json.loads(self.path.read_text(encoding="utf-8"))
            base = Path(manifest.get("path_base", self.path.parent))
            self.files = [(base / p).resolve() for p in manifest["files"]]
            info = self.path.stat()
            self.manifest_snapshot = (info.st_size, info.st_mtime_ns)
            if "row_indices" in manifest:
                self.row_indices = {}
                for name, groups in manifest['row_indices'].items():
                    source = str((base/name).resolve())
                    if Path(source) not in self.files or not isinstance(groups,dict):
                        raise ValueError('Row selection must refer to a declared source file')
                    self.row_indices[source] = {}
                    for group, values in groups.items():
                        if str(int(group)) != group or int(group)<0 or not isinstance(values,list):
                            raise ValueError('Invalid row-group selection')
                        if any(isinstance(v,bool) or not isinstance(v,int) or v<0 for v in values) or len(set(values))!=len(values):
                            raise ValueError('Row positions must be distinct nonnegative integers')
                        self.row_indices[source][int(group)] = values
            index = self.path.with_suffix(".paya-index.sqlite3")
        else:
            self.files = [self.path]
            index = self.path.with_name(self.path.name + ".paya-index.sqlite3")
        if not self.files or len(set(self.files)) != len(self.files):
            raise ValueError("Dataset must name nonempty, distinct input files")
        if any(p.suffix not in (".jsonl", ".parquet") for p in self.files):
            raise ValueError("Dataset files must be JSONL or Parquet")
        if self.row_indices is not None and any(p.suffix != '.parquet' for p in self.files):
            raise ValueError('Row selections require bounded Parquet sources')
        self.snapshots = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in self.files]
        self.file_stats = {name: (size, modified) for name, size, modified in self.snapshots}
        identities = [(str(p), sha256(p)) for p in self.files]
        self.sha256 = identities[0][1] if len(identities) == 1 else fingerprint(identities)
        if self.row_indices is not None:
            self.sha256 = fingerprint({'files':identities,'row_indices':self.row_indices})
        index = index.with_name(index.stem + "." + self.sha256[:16] + index.suffix)
        binding = json.dumps({"version": INDEX_VERSION, "files": identities,
                              **({'row_indices':self.row_indices} if self.row_indices is not None else {})}, sort_keys=True)
        valid = False
        if index.exists():
            try:
                with sqlite3.connect(index) as check:
                    valid = check.execute("SELECT value FROM metadata WHERE key='binding'").fetchone() == (binding,)
            except sqlite3.DatabaseError:
                valid = False
        if not valid:
            self._build_index(index, binding)
        self.index_path = index
        self.db = sqlite3.connect(index)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA cache_size=-8192")
        self.db.execute("PRAGMA temp_store=FILE")
        self.count, self.block_count = self.db.execute('SELECT SUM(nrows),COUNT(*) FROM blocks').fetchone()
        if not self.count:
            raise ValueError("Dataset is empty")
        self.rows = _Rows(self)

    def assert_unchanged(self):
        if self.manifest_snapshot is not None:
            info = self.path.stat()
            if (info.st_size,info.st_mtime_ns) != self.manifest_snapshot:
                raise ValueError('Dataset manifest changed during training')
        current = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in self.files]
        if current != self.snapshots:
            raise ValueError("Dataset changed during training")

    def _build_index(self, index, binding):
        with atomic_output(index) as temporary:
            db = sqlite3.connect(temporary.name)
            try:
                db.executescript('''
                    PRAGMA cache_size=-8192;
                    PRAGMA temp_store=FILE;
                    CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);
                    CREATE TABLE blocks(id INTEGER PRIMARY KEY,filename TEXT,kind TEXT,position INTEGER,
                        nrows INTEGER,row_start INTEGER,byte_length INTEGER);
                    CREATE INDEX block_start ON blocks(row_start);
                    CREATE TABLE audit(field TEXT,value TEXT,PRIMARY KEY(field,value)) WITHOUT ROWID;
                ''')
                offset = 0
                def add(path, kind, position, rows, byte_length=0):
                    nonlocal offset
                    for i, row in enumerate(rows):
                        validate_row(row, f"{path.name} row {offset+i+1}")
                    db.execute('INSERT INTO blocks(filename,kind,position,nrows,row_start,byte_length) VALUES(?,?,?,?,?,?)',
                               (str(path),kind,position,len(rows),offset,byte_length))
                    db.executemany('INSERT OR IGNORE INTO audit VALUES(?,?)',
                        ((field,row[field]) for row in rows for field in AUDIT_FIELDS))
                    offset += len(rows)
                for path in self.files:
                    if path.suffix == ".parquet":
                        import pyarrow.parquet as pq
                        pf = pq.ParquetFile(path)
                        if "state" not in pf.schema_arrow.names:
                            raise ValueError("Parquet dataset requires state")
                        columns = ["question","criteria","target",*AUDIT_FIELDS]
                        selected = self.row_indices.get(str(path),{}) if self.row_indices is not None else None
                        if selected is not None and any(g>=pf.metadata.num_row_groups for g in selected):
                            raise ValueError('Selected row group does not exist')
                        for group in range(pf.metadata.num_row_groups):
                            if selected is not None and not selected.get(group):
                                continue
                            info = pf.metadata.row_group(group)
                            widths = {"BOOLEAN": 1, "INT32": 4, "INT64": 8, "INT96": 12, "FLOAT": 4, "DOUBLE": 8}
                            numeric_bytes = sum(info.column(i).num_values * widths.get(info.column(i).physical_type,0)
                                                for i in range(info.num_columns))
                            if info.num_rows > MAX_BLOCK_ROWS or max(info.total_byte_size,numeric_bytes) > MAX_BLOCK_BYTES:
                                raise ValueError("Parquet row group exceeds memory budget; repack with 256 rows per group")
                            table = pf.read_row_group(group,columns=columns)
                            if selected is not None:
                                if max(selected[group]) >= len(table):
                                    raise ValueError('Selected row position is out of bounds')
                                table = table.take(selected[group])
                            rows = table.to_pylist()
                            for row in rows:
                                row["state"] = {}
                            if rows:
                                add(path,"parquet",group,rows)
                    else:
                        with path.open("rb") as source:
                            rows=[]; start=source.tell()
                            while True:
                                position=source.tell()
                                line=source.readline(MAX_BLOCK_BYTES+1)
                                if len(line)>MAX_BLOCK_BYTES:
                                    raise ValueError("JSONL row exceeds memory budget")
                                if not line:
                                    if rows: add(path,"jsonl",start,rows,source.tell()-start)
                                    break
                                if rows and (len(rows)>=256 or source.tell()-start>MAX_BLOCK_BYTES):
                                    add(path,"jsonl",start,rows,position-start)
                                    rows=[];start=position
                                if line.strip():
                                    rows.append(json.loads(line))
                if not offset:
                    raise ValueError("Dataset is empty")
                db.execute("INSERT INTO metadata VALUES('binding',?)",(binding,))
                db.commit()
            finally:
                db.close()

    def _blocks(self, seed=None):
        if seed is None:
            yield from self.db.execute('SELECT * FROM blocks ORDER BY id')
            return
        rng = random.Random(seed)
        a = rng.randrange(1,self.block_count+1)
        while math.gcd(a,self.block_count) != 1:
            a = rng.randrange(1,self.block_count+1)
        b = rng.randrange(self.block_count)
        for i in range(self.block_count):
            identifier = (a*i+b)%self.block_count+1
            yield self.db.execute('SELECT * FROM blocks WHERE id=?',(identifier,)).fetchone()

    def _read_block(self, block):
        info=Path(block['filename']).stat()
        if (info.st_size,info.st_mtime_ns) != self.file_stats[block['filename']]:
            raise ValueError("Dataset changed during training")
        if block["kind"] == "parquet":
            import pyarrow.parquet as pq
            table = pq.ParquetFile(block["filename"]).read_row_group(block["position"])
            if self.row_indices is not None:
                table = table.take(self.row_indices[block['filename']][block['position']])
            rows = table.to_pylist()
        else:
            with open(block["filename"],"rb") as source:
                source.seek(block["position"])
                rows = [json.loads(line) for line in source.read(block["byte_length"]).splitlines() if line.strip()]
        if len(rows) != block["nrows"]:
            raise ValueError("Dataset block changed")
        return rows

    def batches(self, batch_size, seed=None, start_batch=0):
        if isinstance(batch_size,bool) or not isinstance(batch_size,int) or batch_size<1:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(start_batch,bool) or not isinstance(start_batch,int) or start_batch<0:
            raise ValueError("start_batch must be a nonnegative integer")
        self.assert_unchanged()
        skip=start_batch*batch_size
        pending=[]
        for block in self._blocks(seed):
            if skip>=block['nrows']:
                skip-=block['nrows']
                continue
            values=self._read_block(block)
            order=list(range(len(values)))
            if seed is not None:
                random.Random(f'{seed}:block:{block["id"]}').shuffle(order)
            for position in order[skip:]:
                row=values[position].copy()
                if seed is not None:
                    permutation=list(range(len(row['criteria'])))
                    random.Random(f'{seed}:block:{block["id"]}:row:{position}').shuffle(permutation)
                    row['criteria']=[row['criteria'][i] for i in permutation]
                    row['target']=permutation.index(row['target'])
                pending.append(row)
                if len(pending)==batch_size:
                    yield pending
                    pending=[]
            skip=0
        if pending:
            yield pending


def assert_disjoint(*datasets):
    for i, left in enumerate(datasets):
        for right in datasets[i+1:]:
            left.db.execute('ATTACH DATABASE ? AS other',(str(right.index_path),))
            try:
                overlap=left.db.execute('SELECT a.field FROM audit a JOIN other.audit b '
                    'ON a.field=b.field AND a.value=b.value LIMIT 1').fetchone()
                if overlap:
                    raise ValueError(f'Dataset splits overlap on {overlap[0]}')
            finally:
                left.db.execute('DETACH DATABASE other')


def make_batch(model, rows):
    return model.make_batch([r["state"] for r in rows], [r["question"] for r in rows],
                            [r["criteria"] for r in rows])
