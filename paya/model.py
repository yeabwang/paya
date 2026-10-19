"""Prepare native signals, encode with Mantis and score separate candidate answers."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .encoders import EncodedSignal, MantisEncoder
from .head import DecisionHead
from .inputs import validate_text
from .preprocessing import SignalPreprocessor, PreparationConfig, PreparedState
from .provenance import RUNTIME_CONTRACT, ARCHITECTURE, verify_assets
from .text import TextEncoder


@dataclass
class DecisionBatch:
    states: list[PreparedState]
    question: torch.Tensor
    criteria: torch.Tensor
    criteria_mask: torch.Tensor
    descriptions: list[list[str]]

    def to(self, device):
        return DecisionBatch(self.states, self.question.to(device),
                             self.criteria.to(device),
                             self.criteria_mask.to(device), self.descriptions)


class PayaModel(nn.Module):
    """Fuse frozen Mantis signal embeddings with separate BioLORD text embeddings."""

    def __init__(self, mantis, text_encoder, preparation=None, head=None, asset_identity=None):
        super().__init__()
        self.encoder = mantis.requires_grad_(False).eval()
        self.head = DecisionHead(**(head or {}))
        self.text_encoder = text_encoder
        self.preprocessor = SignalPreprocessor(preparation)
        self.asset_identity = asset_identity or {"injected_test_encoders": True}
        self.register_buffer("temperature", torch.tensor(1.0))
        self.calibration = None
        self.train(False)

    @classmethod
    def from_assets(cls, assets, preparation=None, head=None, device="cpu"):
        sources = verify_assets(assets)
        settings = PreparationConfig(**(preparation or {}))
        text = TextEncoder(device=device, model_path=Path(assets["text"]))
        model = cls(MantisEncoder.load(assets["mantis"], settings.max_samples), text, settings, head, sources)
        return model.to(device)

    @property
    def device(self):
        return self.temperature.device

    def identity(self):
        return {"contract": RUNTIME_CONTRACT, "architecture": ARCHITECTURE,
                "assets": self.asset_identity, "preparation": self.preprocessor.identity(),
                "head": self.head.config}

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        if mode and hasattr(self, "temperature"):
            self.temperature.fill_(1.0)
            self.calibration = None
        if getattr(self, "text_encoder", None) is not None and self.text_encoder.model is not None:
            self.text_encoder.model.eval().requires_grad_(False)
        return self

    def trainable_parameters(self):
        return self.head.parameters()

    def make_batch(self, states, questions, criteria):
        """Prepare native signals and encode each question and choice separately."""
        if not states or len(states) != len(questions) or len(states) != len(criteria):
            raise ValueError("Supply matching nonempty state/question/criteria batches")
        for q, c in zip(questions, criteria):
            validate_text(q, c)
        prepared = [s if isinstance(s, PreparedState) else self.preprocessor.prepare(s) for s in states]
        if any(s.audit.get("preparation") != self.preprocessor.identity() for s in prepared):
            raise ValueError("Prepared signal does not match current preprocessing")
        strings = [s for q, c in zip(questions, criteria) for s in [q, *c]]
        vectors = self.text_encoder.encode(strings).to(self.device).detach().clone()
        if vectors.shape != (len(strings), 768) or not torch.isfinite(vectors).all():
            raise ValueError("BioLORD must return finite, separate 768-dimensional vectors")
        width, offset = max(map(len, criteria)), 0
        qs, cs = [], vectors.new_zeros(len(states), width, 768)
        mask = torch.zeros(len(states), width, dtype=torch.bool, device=self.device)
        for i, choices in enumerate(criteria):
            qs.append(vectors[offset])
            cs[i, :len(choices)] = vectors[offset+1:offset+1+len(choices)]
            mask[i, :len(choices)] = True
            offset += 1 + len(choices)
        return DecisionBatch(prepared, torch.stack(qs), cs,
                             mask, [list(c) for c in criteria]).to(self.device)

    def encode(self, prepared):
        encoder_input = self.preprocessor.format(prepared)
        with torch.no_grad():
            encoded = self.encoder(encoder_input.to(self.device))
        if encoded.values.ndim != 2 or encoded.values.shape[-1] != 512 or not torch.isfinite(encoded.values).all():
            raise ValueError("Encoder must return finite Mantis vectors [M,512]")
        intervals = encoded.intervals_s
        if (intervals.shape != (len(encoded.values), 2) or not torch.isfinite(intervals).all()
                or (intervals[:, 0] < 0).any() or (intervals[:, 1] <= intervals[:, 0]).any()):
            raise ValueError("Invalid elapsed interval times")
        if not 1 <= len(encoded.values) <= self.preprocessor.config.max_signal_tokens:
            raise ValueError("Invalid encoded token budget")
        return EncodedSignal(encoded.values.float().detach(), intervals.float().detach())

    def forward(self, batch, cache=None):
        encoded = [cache.get_or_encode(self, state) if cache else self.encode(state) for state in batch.states]
        tokens = max(len(e.values) for e in encoded)
        signal = batch.question.new_zeros(len(encoded), tokens, 512)
        intervals = batch.question.new_zeros(len(encoded), tokens, 2)
        mask = torch.zeros(len(encoded), tokens, dtype=torch.bool, device=self.device)
        for i, value in enumerate(encoded):
            length = len(value.values)
            signal[i, :length] = value.values.to(self.device)
            intervals[i, :length] = value.intervals_s.to(self.device)
            mask[i, :length] = True
        logits = self.head(signal, mask, intervals, batch.question, batch.criteria, batch.criteria_mask)
        return {"logits": logits, "criteria_mask": batch.criteria_mask}

    @torch.no_grad()
    def predict(self, state, question, criteria):
        """Return a choice, probabilities in criteria order and the chosen probability."""
        self.eval()
        if not torch.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Prediction temperature must be finite and positive")
        batch = self.make_batch([state], [question], [criteria])
        logits = self(batch)["logits"][0]
        probabilities = (logits / self.temperature).softmax(-1).cpu().tolist()
        chosen = min(range(len(criteria)), key=lambda i: (-probabilities[i], criteria[i]))
        return {"choice": criteria[chosen], "probabilities": probabilities,
                "confidence": probabilities[chosen]}
