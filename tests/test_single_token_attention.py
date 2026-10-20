import torch
from torch.nn import functional as F

from paya.head import CrossAttention


def test_single_signal_inference_matches_general_attention_with_masking():
    torch.manual_seed(19)
    layer = CrossAttention(32, 4, .1).eval()
    text, signal = torch.randn(3, 5, 32), torch.randn(3, 1, 512)
    mask = torch.tensor([[True], [False], [True]])
    def split(x):
        return x.reshape(len(x), x.shape[1], 4, -1).transpose(1, 2)
    general = F.scaled_dot_product_attention(split(layer.query(text)), split(layer.key(signal)),
        split(layer.value(signal)), attn_mask=mask[:, None, None, :], dropout_p=0.)
    expected = layer.output(general.transpose(1, 2).reshape(text.shape))
    torch.testing.assert_close(layer(text, signal, mask), expected, atol=1e-6, rtol=1e-6)


def test_training_keeps_general_dropout_path_and_inference_skips_unused_projections():
    layer = CrossAttention(16, 4, .1)
    calls=[]
    hook = layer.query.register_forward_hook(lambda *args:calls.append(True))
    text, signal = torch.randn(2, 3, 16), torch.randn(2, 1, 512)
    mask=torch.ones(2, 1, dtype=torch.bool)
    layer(text, signal, mask)
    assert len(calls)==1
    layer.eval();layer(text, signal, mask)
    assert len(calls)==1
    hook.remove()
