import sqlite3

import pytest

from scripts.coverage import summarize


def fixture_db():
    db = sqlite3.connect(':memory:')
    db.execute('CREATE TABLE examples(role TEXT,task TEXT,answer TEXT,waveform TEXT,recording TEXT,group_id TEXT)')
    return db


def test_coverage_counts_independent_groups_and_flags_conflicting_qa():
    db = fixture_db()
    db.executemany('INSERT INTO examples VALUES (?,?,?,?,?,?)', [
        ('train','af','af','w1','r1','g1'), ('train','af','af','w1','r1','g1'),
        ('train','af','normal','w1','r2','g2'), ('train','af','normal','w2','r3','g3'),
        ('selection','af','normal','w3','r4','g4')])
    report = summarize(db)
    assert report['independent_components'] == 3
    value = report['coverage']['train']['af']['af']
    assert value['rows'] == 2 and value['waveforms'] == 1 and value['independent_groups'] == 1
    assert report['contradictory_waveform_labels'][0]['waveforms'] == 1
    assert report['contradictory_waveform_labels'][0]['affected_qa_rows'] == 3
    assert not report['task_answer_support']['af']['normal']['can_cover_four_roles_by_group_count']


def test_coverage_rejects_recording_link_across_roles():
    db = fixture_db()
    db.executemany('INSERT INTO examples VALUES (?,?,?,?,?,?)', [
        ('train','af','af','w1','record','g1'), ('calibration','af','af','w2','record','g2')])
    with pytest.raises(ValueError, match='overlap'):
        summarize(db)
