# Copyright (c) School of Computing, Information, and Data Science, University of California San Diego.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

"""Modified NormWear signal encoder and its retained wavelet preprocessing."""

import torch
from torch import nn
from torch.nn import functional as F

from .layers import Block, PatchEmbed


class EncoderLayer(nn.Module):
    """Apply a shared patch transformer and optional cross-channel CLS fusion."""

    def __init__(self, embed_dim, num_heads, mlp_ratio, fuse):
        super().__init__()
        self.variate_encoder = Block(embed_dim, num_heads, mlp_ratio)
        if fuse:
            self.cls_fusion = Block(embed_dim, num_heads, mlp_ratio)

    def forward(self, x, nvar):
        """Transform flattened sensor patches, then fuse CLS tokens across real sensors."""
        x = self.variate_encoder(x)
        if hasattr(self, "cls_fusion"):
            _, count, hidden = x.shape
            x = x.reshape(-1, nvar, count, hidden)
            cls = self.cls_fusion(x[:, :, 0]).unsqueeze(2)
            x = torch.cat((cls, x[:, :, 1:]), dim=2).reshape(-1, count, hidden)
        return x


class NormWear(nn.Module):
    """Retained pretrained waveform encoder producing sensor CLS and time/scale patches."""

    def __init__(
        self,
        img_size=(387, 65),
        patch_size=(9, 5),
        in_chans=3,
        nvar=4,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        fuse_freq=2,
    ):
        """Construct encoder layers from the pinned upstream configuration.

        nvar is retained for configuration compatibility; forward derives the
        actual channel count from the supplied waveform tensor.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim),
            requires_grad=False,
        )
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderLayer(embed_dim, num_heads, mlp_ratio, i % fuse_freq == 0)
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def position_embedding(self, image_shape):
        """Interpolate frozen patch positions to the current convolution grid."""
        original = self.patch_embed.patch_hw
        new = self.patch_embed.output_grid(image_shape)
        if original == new:
            return self.pos_embed
        positions = (
            self.pos_embed[:, 1:]
            .reshape(1, *original, self.embed_dim)
            .permute(0, 3, 1, 2)
        )
        positions = F.interpolate(
            positions, size=new, mode="bicubic", align_corners=False
        )
        positions = positions.permute(0, 2, 3, 1).flatten(1, 2)
        return torch.cat((self.pos_embed[:, :1], positions), dim=1)

    def forward(self, signals):
        """Encode [batch, sensors, samples] into normalized sensor/patch embeddings.

        Apply the retained signal/first-difference/second-difference wavelet
        frontend, patch embedding, transformer blocks, and final layer normalization.
        """
        batch, sensors, length = signals.shape
        spec = cwt_wrap(signals.reshape(batch * sensors, length), 0.1, 64).float()
        positions = self.position_embedding(spec.shape[-2:])
        patches = self.patch_embed(spec) + positions[:, 1:]
        cls = (self.cls_token + self.pos_embed[:, :1]).expand(batch * sensors, -1, -1)
        patches = torch.cat((cls, patches), dim=1)
        for block in self.encoder_blocks:
            patches = block(patches, nvar=sensors)
        patches = self.norm(patches)
        return patches.reshape(batch, sensors, patches.shape[1], self.embed_dim)


def ricker_wavelet(points, scale):
    """Return the normalized Ricker wavelet for a scalar tensor scale."""

    A = 2 / (torch.sqrt(3 * scale) * torch.pi**0.25)  # Normalization factor
    wsq = scale**2
    vec = torch.arange(0, points, device=scale.device) - (points - 1.0) / 2
    xsq = vec**2
    mod = 1 - xsq / wsq
    gauss = torch.exp(-xsq / (2 * wsq))
    wavelet = A * mod * gauss
    return wavelet


def cwt_ricker(x, lowest_scale, largest_scale, step=1):
    """Convolve [batch, samples] waveforms with the retained Ricker scale bank.

    Return [batch, scales, output_samples]. Kernel length is derived from
    the largest scale and input length. With the preserved padding rule,
    an even-length kernel adds one output sample.
    """
    _, sequence_length = x.shape
    x = x.unsqueeze(1)  # Add channel dimension, now (batch_size, 1, sequence_length)

    # Generate scales
    scales = torch.arange(lowest_scale, largest_scale + step, step, device=x.device)
    num_scales = scales.shape[0]

    # Prepare the wavelet basis for each scale
    wavelet_len = min(10 * largest_scale, sequence_length)
    wavelets = torch.stack([ricker_wavelet(wavelet_len, scale) for scale in scales])
    wavelets = wavelets.view(num_scales, 1, -1)  # (num_scales, 1, wavelet_len)

    # Perform convolution for each scale
    cwt_output = F.conv1d(x, wavelets, padding=wavelet_len // 2)

    return cwt_output


def cwt_wrap(x, lowest_scale, largest_scale, step=1):
    # x: bn, L
    # return: bn, 3, L, n_mels
    """Stack signal and two finite differences, then return [batch, 3, time, scales]."""
    d1 = x[:, 1:] - x[:, :-1]  # bn, L-1
    d2 = d1[:, 1:] - d1[:, :-1]  # bn, L-2
    x = (
        torch.stack([x[:, 2:], d1[:, 1:], d2]).float().permute(1, 0, 2)
    )  # batch, 3, length-2
    bn, n_, new_L = x.shape
    cwt_res = cwt_ricker(
        x.reshape(bn * n_, new_L), lowest_scale, largest_scale, step=step
    )  # bn*3, 65, new_L
    _, n_scale, new_L = cwt_res.shape
    return cwt_res.reshape(bn, n_, n_scale, new_L).permute(0, 1, 3, 2)
