"""Frozen signal encoding and the shared trainable candidate-scoring head."""

import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

from .inputs import METADATA_DIM
from .provenance import NORMWEAR_ID, NORMWEAR_REVISION, sha256
from .vendor.normwear import NormWear


class EvidenceAttention(nn.Module):
    """Cosine cross-attention with a bounded learned logit scale."""

    def __init__(self, width, heads=4):
        super().__init__()
        if width % heads:
            raise ValueError("Fusion width must be divisible by attention heads")
        self.heads = heads
        self.q, self.k, self.v, self.out = [nn.Linear(width, width) for _ in range(4)]
        self.log_scale = nn.Parameter(torch.tensor(math.log(5.0)))

    def forward(self, query, memory, valid=None):
        """Attend from [batch, queries, width] to [batch, memory, width].

        valid optionally masks memory positions. Return the original query shape;
        reject examples with no valid memory token.
        """

        def split(x):
            return x.reshape(x.shape[0], x.shape[1], self.heads, -1).transpose(1, 2)

        q, k = (
            F.normalize(split(self.q(query)), dim=-1),
            F.normalize(split(self.k(memory)), dim=-1),
        )
        scores = (q @ k.transpose(-2, -1)) * self.log_scale.clamp(
            max=math.log(20)
        ).exp()
        if valid is not None:
            if not valid.any(-1).all():
                raise ValueError("Attention needs at least one valid evidence token")
            scores = scores.masked_fill(~valid[:, None, None, :], -torch.inf)
        out = scores.softmax(-1) @ split(self.v(memory))
        return self.out(out.transpose(1, 2).reshape(query.shape))


class GroundedChoiceHead(nn.Module):
    """Score variable answer sets against shared signal, sensor, and text evidence."""

    def __init__(
        self,
        signal_hidden=768,
        text_hidden=768,
        width=256,
        layers=2,
        heads=4,
        dropout=0.1,
    ):
        super().__init__()
        self.signal_proj = nn.Sequential(
            nn.LayerNorm(signal_hidden), nn.Linear(signal_hidden, width)
        )
        self.sensor_proj = nn.Sequential(
            nn.Linear(METADATA_DIM, width), nn.GELU(), nn.Linear(width, width)
        )
        # Relative time and scale describe evidence positions, never channel rank.
        self.position_proj = nn.Linear(3, width, bias=False)
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_hidden), nn.Linear(text_hidden, width)
        )
        self.roles = nn.Parameter(torch.zeros(2, width))
        self.context = nn.ModuleList(
            [EvidenceAttention(width, heads) for _ in range(layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.ff = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(width),
                    nn.Linear(width, width * 2),
                    nn.GELU(),
                    nn.Linear(width * 2, width),
                )
                for _ in range(layers)
            ]
        )
        self.comparison, self.sensor_attention = (
            EvidenceAttention(width, heads),
            EvidenceAttention(width, heads),
        )
        self.final_norm = nn.LayerNorm(width)
        self.answer_proj, self.choice_proj = (
            nn.Linear(width, width),
            nn.Linear(width, width),
        )
        self.evidence_score = nn.Sequential(
            nn.Linear(width * 4, width), nn.GELU(), nn.Linear(width, 1)
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(5.0)))
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def signal_memory(patches, time_bins=8):
        """Pool time while preserving CLS and the 13-scale patch grid.

        Convert [batch, sensors, patches, hidden] to
        [batch, sensors, 1 + min(time_bins, time_patches) * 13, hidden].
        The scale ordering and adaptive pooling are part of the cache contract.
        """
        batch, sensors, count, hidden = patches.shape
        if count < 2 or (count - 1) % 13:
            raise ValueError("Expected CLS and a 13-scale signal patch grid")
        grid = (
            patches[:, :, 1:]
            .reshape(batch * sensors, -1, 13, hidden)
            .permute(0, 3, 1, 2)
        )
        bins = min(time_bins, grid.shape[2])
        content = F.adaptive_avg_pool2d(grid, (bins, 13)).permute(0, 2, 3, 1)
        content = content.reshape(batch, sensors, bins * 13, hidden)
        return torch.cat([patches[:, :, :1], content], dim=2)

    def forward(
        self,
        memory,
        sensor_metadata,
        sensor_mask,
        question,
        choices,
        ending,
        choice_mask,
    ):
        """Return [batch, choices] scores from sensor/time/scale evidence.

        memory is [batch, sensors, tokens, hidden]; sensor metadata is
        [batch, sensors, METADATA_DIM]. question and ending are [batch, 768],
        choices is [batch, choices, 768]. Boolean masks mark real sensors and
        candidates. Padded candidates receive negative-infinity scores.
        """
        batch, sensors, tokens, hidden = memory.shape
        if (
            (tokens - 1) % 13
            or not sensor_mask.any(-1).all()
            or not choice_mask.any(-1).all()
        ):
            raise ValueError("Invalid evidence grid, sensors or candidates")
        bins = (tokens - 1) // 13
        positions = memory.new_zeros(tokens, 3)
        positions[0, 2] = 1
        positions[1:, 0] = torch.linspace(
            0, 1, bins, device=memory.device
        ).repeat_interleave(13)
        positions[1:, 1] = torch.linspace(0, 1, 13, device=memory.device).repeat(bins)
        projected = (
            self.signal_proj(memory.float())
            + self.sensor_proj(sensor_metadata)[:, :, None]
        )
        projected = projected + self.position_proj(positions)[None, None]
        sensor_vectors = projected.mean(2)
        valid = sensor_mask[:, :, None].expand(-1, -1, tokens).flatten(1, 2)
        evidence_memory = self.drop(projected.flatten(1, 2))
        q, c, e = (
            self.text_proj(question),
            self.text_proj(choices),
            self.text_proj(ending),
        )
        queries = torch.cat(
            [(q + e + self.roles[0])[:, None], c + q[:, None] + self.roles[1]], dim=1
        )
        for attention, norm, ff in zip(self.context, self.norms, self.ff):
            queries = queries + self.drop(
                attention(norm(queries), evidence_memory, valid)
            )
            queries = queries + self.drop(ff(queries))
        query_valid = torch.cat(
            [torch.ones_like(choice_mask[:, :1]), choice_mask], dim=1
        )
        queries = self.final_norm(
            queries
            + self.comparison(
                self.final_norm(queries), self.final_norm(queries), query_valid
            )
        )
        evidence = self.sensor_attention(c + q[:, None], sensor_vectors, sensor_mask)
        direct = self.evidence_score(
            self.drop(torch.cat([evidence, c, evidence * c, evidence * q[:, None]], -1))
        ).squeeze(-1)
        answer = F.normalize(self.answer_proj(queries[:, :1]), dim=-1)
        candidates = F.normalize(self.choice_proj(queries[:, 1:]), dim=-1)
        similarity = (answer * candidates).sum(-1) * self.logit_scale.clamp(
            max=math.log(20)
        ).exp()
        return (similarity + direct).masked_fill(~choice_mask, -torch.inf)


class SignalDecision(nn.Module):
    """Combine a frozen NormWear encoder with a shared choice head and temperature."""

    def __init__(
        self, core, width=256, layers=2, heads=4, dropout=0.1, memory_time_bins=8
    ):
        super().__init__()
        self.core = core.requires_grad_(False)
        self.head = GroundedChoiceHead(
            core.embed_dim, width=width, layers=layers, heads=heads, dropout=dropout
        )
        self.memory_time_bins = memory_time_bins
        self.register_buffer("temperature", torch.tensor(1.0))

    def train(self, mode=True):
        """Set head training mode while keeping the frozen encoder in evaluation mode."""
        super().train(mode)
        self.core.eval()
        return self

    def forward(
        self,
        question,
        choices,
        ending,
        choice_mask,
        sensor_metadata,
        sensor_mask,
        signals=None,
        signal_memory=None,
    ):
        """Score a batch from either raw signals or frozen signal memory.

        Supply exactly one of signals [batch, sensors, samples] and signal_memory
        [batch, sensors, tokens, 768]. Return masked choice logits divided by the
        model temperature. Raw batches contain only real channels; the encoder
        runs without gradients while the head remains differentiable.
        """
        if (signals is None) == (signal_memory is None):
            raise ValueError("Supply exactly one waveform or frozen evidence input")
        if signal_memory is None:
            if not sensor_mask.all():
                raise ValueError(
                    "Raw batches must contain real channels only; group by channel count"
                )
            if not torch.isfinite(signals).all():
                raise ValueError("Waveforms must be finite")
            with torch.no_grad():
                signal_memory = self.head.signal_memory(
                    self.core(signals), self.memory_time_bins
                )
        if (
            not torch.isfinite(signal_memory).all()
            or not torch.isfinite(sensor_metadata).all()
        ):
            raise ValueError("Evidence and physical metadata must be finite")
        with torch.autocast(device_type=question.device.type, enabled=False):
            logits = self.head(
                signal_memory.float(),
                sensor_metadata.float(),
                sensor_mask,
                question.float(),
                choices.float(),
                ending.float(),
                choice_mask,
            )
        return logits / self.temperature.clamp_min(0.05)


def build_model(asset_dir, config):
    """Verify pinned encoder files, load their tensors strictly, and build the head.

    asset_dir contains source.json, config.json and encoder-only weights.
    config supplies the head settings; no decision-head weights are loaded here.
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    asset_dir = Path(asset_dir)
    source = json.loads((asset_dir / "source.json").read_text())
    if (source["model_id"], source["revision"], source.get("format")) != (
        NORMWEAR_ID,
        NORMWEAR_REVISION,
        "encoder_only",
    ):
        raise ValueError("Signal encoder does not match the pinned encoder-only asset")
    for entry in source["files"]:
        if sha256(asset_dir / entry["file"]) != entry["sha256"]:
            raise ValueError("Signal encoder asset checksum changed")
    core = NormWear(**json.loads((asset_dir / "config.json").read_text()))
    core.load_state_dict(load_file(str(asset_dir / "model.safetensors")), strict=True)
    return SignalDecision(core, **config.get("head", {}))
