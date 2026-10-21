import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from paya.data import ChoiceDataset,assert_disjoint
from paya.group_split import allocate_groups
from paya.provenance import sha256
from conftest import state


def test_group_assignment_is_deterministic_and_covers_rare_labels():
    counts=np.array([[10,1] if i<6 else [20,0] for i in range(24)])
    first,report=allocate_groups(counts,attempts=3)
    second,_=allocate_groups(counts,attempts=3)
    np.testing.assert_equal(first,second)
    assert all((counts[first==role].sum(0)>0).all() for role in range(4))
    assert sum(report['achieved_row_fractions'])==pytest.approx(1.)
    with pytest.raises(ValueError,match='Too few'):
        allocate_groups(np.array([[1,1],[1,1],[1,1]]))


def test_row_filtered_manifests_preserve_rows_and_isolate_shared_shards(tmp_path):
    rows=[dict(state=state(rate=125,duration=1,condition='processed'),question='Which?',criteria=['a','b'],target=i%2,
               source_id=f'w{i}',group_id=f'g{i}',recording_id=f'r{i}') for i in range(4)]
    source=tmp_path/'data.parquet';pq.write_table(pa.Table.from_pylist(rows),source,row_group_size=2)
    paths=[]
    for role,selection in [('train',{'0':[0],'1':[1]}),('calibration',{'0':[1],'1':[0]})]:
        path=tmp_path/(role+'.json')
        path.write_text(json.dumps(dict(files=[str(source)],row_indices={str(source):selection})))
        paths.append(path)
    first,second=[ChoiceDataset(p) for p in paths]
    assert len(first.rows)==2 and list(first.rows)==[rows[0],rows[3]]
    assert list(second.rows)==[rows[1],rows[2]]
    assert first.sha256!=second.sha256
    assert ChoiceDataset(source).sha256==sha256(source)
    assert_disjoint(first,second)
    paths[0].write_text(paths[0].read_text()+' ')
    with pytest.raises(ValueError,match='manifest changed'):first.assert_unchanged()


@pytest.mark.parametrize('positions',[[0,0],[10],[True]])
def test_row_filtered_manifest_rejects_duplicate_invalid_or_outside_positions(tmp_path,positions):
    row=dict(state=state(rate=125,duration=1,condition='processed'),question='Which?',criteria=['a','b'],target=0,source_id='w',group_id='g',recording_id='r')
    source=tmp_path/'data.parquet';pq.write_table(pa.Table.from_pylist([row]),source)
    manifest=tmp_path/'view.json'
    manifest.write_text(json.dumps(dict(files=[str(source)],row_indices={str(source):{'0':positions}})))
    with pytest.raises(ValueError):ChoiceDataset(manifest)
