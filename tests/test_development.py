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
