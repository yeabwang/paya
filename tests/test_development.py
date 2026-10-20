import numpy as np
import pytest
import torch

from paya.development import InteractionHead, class_weights, smoothed_choice_loss


def test_smoothing_never_uses_padded_candidates():
    z = torch.tensor([[1., 2., -torch.inf], [2., 1., 0.]], requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    y = torch.tensor([0, 2])
    loss = smoothed_choice_loss(z, y, mask, .1)
    expected = torch.nn.functional.cross_entropy(z[:1, :2], y[:1], label_smoothing=.1)
    assert float(loss[0].detach()) == pytest.approx(float(expected.detach()))
    loss.sum().backward()
    assert torch.isfinite(z.grad).all() and z.grad[0, 2] == 0


def test_class_weights_are_training_only_and_preserve_task_mass():
    rows = [{'task': 'a', 'criteria': ['low','high'], 'target': i} for i in [0]*9+[1]]
    weights = class_weights(rows)
    assert np.mean(weights) == pytest.approx(1.)
    assert weights[-1] > weights[0] and np.isfinite(weights).all()


def test_interaction_head_candidate_permutation_padding_and_signal_gradient():
    torch.manual_seed(7)
    head = InteractionHead(width=16, dropout=0).eval()
    s = torch.randn(2, 1, 512, requires_grad=True)
    q, c = torch.randn(2, 768), torch.randn(2, 3, 768)
    sm = torch.ones(2, 1, dtype=torch.bool)
    cm = torch.tensor([[True, True, False], [True, True, True]])
    z = head(s, sm, q, c, cm)
    order = [2, 0, 1]
    torch.testing.assert_close(head(s, sm, q, c[:, order], cm[:, order]), z[:, order])
    z[cm].sum().backward()
    assert s.grad.abs().sum() > 0 and z[0, 2].isneginf()


def test_bank_indices_align_rows_and_features_and_detect_corruption(tmp_path):
    import json
    from paya.development import Bank
    from paya.provenance import sha256
    arrays = {'signals':np.arange(4*512,dtype=np.float32).reshape(4,512),
        'questions':np.ones((2,768),dtype=np.float32), 'criteria':np.ones((3,768),dtype=np.float32),
        'question_indices':np.array([0,1,0,1]), 'criteria_indices':np.array([[0,1,-1]]*4),
        'targets':np.array([0,1,0,1])}
    for name,values in arrays.items():np.save(tmp_path/(name+'.npy'),values)
    (tmp_path/'rows.jsonl').write_text(''.join(json.dumps({'index':i})+'\n' for i in range(4)))
    manifest={'rows':4,'binding':'fixture','files':{p.name:sha256(p) for p in tmp_path.iterdir()}}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    bank=Bank(tmp_path,limit=3,device='cpu')
    for i,row in enumerate(bank.rows):
        assert row['index']==bank.indices[i]
        assert bank.signal[i,0,0]==row['index']*512
    assert bank.batch(torch.tensor([0]))[-1].tolist()==[[True,True,False]]
    (tmp_path/'targets.npy').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):
        Bank(tmp_path,device='cpu')
