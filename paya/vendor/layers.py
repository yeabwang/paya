# Copyright (c) School of Computing, Information, and Data Science, University of California San Diego.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

"""Modified upstream patch embedding and transformer layers used by NormWear."""

from torch import nn
from torch.nn import functional as F


class Mlp(nn.Module):
    """Two-layer GELU feed-forward block preserving the last tensor dimension."""

    def __init__(self, width, hidden):
        super().__init__()
        self.fc1 = nn.Linear(width, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, width)

    def forward(self, x):
        """Transform [..., width] inputs through the expanded hidden representation."""
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    """Multi-head self-attention with the upstream projection parameter layout."""

    def __init__(self, width, heads):
        super().__init__()
        self.num_heads, self.head_dim = heads, width // heads
        self.qkv = nn.Linear(width, width * 3, bias=True)
        self.proj = nn.Linear(width, width)

    def forward(self, x):
        """Apply dropout-free self-attention to [batch, tokens, width] inputs."""
        batch, count, width = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch, count, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        return self.proj(x.transpose(1, 2).reshape(batch, count, width))


class Block(nn.Module):
    """Pre-normalized attention and feed-forward residual block."""

    def __init__(self, width, heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, eps=1e-6)
        self.attn = Attention(width, heads)
        self.norm2 = nn.LayerNorm(width, eps=1e-6)
        self.mlp = Mlp(width, int(width * mlp_ratio))

    def forward(self, x):
        """Return [batch, tokens, width] after both residual updates."""
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    """Strided 2D patch projection with a directly calculated output grid."""

    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.patch_hw = self.output_grid(img_size)
        self.num_patches = self.patch_hw[0] * self.patch_hw[1]

    def output_grid(self, img_size):
        """Return the integer convolution grid for an image height/width pair."""
        return tuple(
            int(size) // patch for size, patch in zip(img_size, self.patch_size)
        )

    def forward(self, x):
        """Convert [batch, channels, height, width] images to flattened patch tokens."""
        return self.proj(x).flatten(2).transpose(1, 2)
