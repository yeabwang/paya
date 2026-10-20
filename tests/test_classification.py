import numpy as np
import pytest
from sklearn.metrics import f1_score, roc_auc_score

from paya.classification import classification_metrics


def test_answer_alignment_tied_auc_and_macro_f1():
    labels = ['a', 'b', 'c']
    logits = np.array([[1, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 2], [0, 1, 2]])
    targets = np.array([0, 1, 1, 2, 0])
    rows = []
    for i, (z, y) in enumerate(zip(logits, targets)):
        order = np.roll(np.arange(3), i % 3).tolist()
        rows.append({'criteria': [labels[j] for j in order], 'target': order.index(int(y)),
                     'logits': z[order].tolist()})
    got = classification_metrics(iter(rows))
    p = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    assert got['macro_f1'] == pytest.approx(f1_score(targets, logits.argmax(-1), average='macro'))
    assert got['macro_auroc_ovr'] == pytest.approx(roc_auc_score(targets, p, multi_class='ovr'))
    assert sum(v['support'] for v in got['per_class'].values()) == 5
    assert sum(sum(v.values()) for v in got['confusion'].values()) == 5


def test_absent_classes_are_explicit_not_fake_auc():
    got = classification_metrics([{'criteria': ['a', 'b'], 'target': 0, 'logits': [1., 0.]}])
    assert got['macro_auroc_ovr'] is None
    assert got['auroc_supported_classes'] == 0
    assert got['per_class']['b']['recall'] is None
    assert got['macro_f1'] == 0.5


@pytest.mark.parametrize('rows', [[], [dict(criteria=['a','b'],target=0,logits=[float('nan'),0])],
    [dict(criteria=['a','b'],target=0,logits=[0,0]), dict(criteria=['a','c'],target=0,logits=[0,0])]])
def test_invalid_records_fail(rows):
    with pytest.raises(ValueError):
        classification_metrics(rows)
