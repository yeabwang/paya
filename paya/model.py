"""Preprocess -> two-expert router -> one frozen encoder -> one shared head."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .encoders import EncodedSignal, FixedWidthAdapter, NormWearEncoder, MantisEncoder
from .head import DecisionHead
from .inputs import ENDING, validate_text
from .preprocessing import SignalPreprocessor, PreparationConfig, PreparedState
from .provenance import RUNTIME_CONTRACT, ARCHITECTURE, verify_assets
from .router import ExpertRouter
from .text import TextEncoder


@dataclass
class DecisionBatch:
    states: list[PreparedState]
    question: torch.Tensor
    criteria: torch.Tensor
    ending: torch.Tensor
    criteria_mask: torch.Tensor
    descriptions: list[list[str]]

    def to(self, device):
        return DecisionBatch([s.to(device) for s in self.states], self.question.to(device),
                             self.criteria.to(device), self.ending.to(device),
                             self.criteria_mask.to(device), self.descriptions)


class PayaModel(nn.Module):
    def __init__(self, normwear, mantis, text_encoder, preparation=None, head=None, asset_identity=None):
        super().__init__()
        self.encoders = nn.ModuleList((normwear, mantis)).requires_grad_(False)
        self.adapters = nn.ModuleList((FixedWidthAdapter(768), FixedWidthAdapter(512)))
        self.router = ExpertRouter()
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
        model = cls(NormWearEncoder.load(assets["normwear"], settings.time_bins),
                    MantisEncoder.load(assets["mantis"]), text, settings, head, sources)
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
        self.encoders.eval()
        if mode and hasattr(self, "temperature"):
            self.temperature.fill_(1.0)
            self.calibration = None
        if getattr(self, "text_encoder", None) is not None and self.text_encoder.model is not None:
            self.text_encoder.model.eval().requires_grad_(False)
        return self

    def trainable_parameters(self):
        return (p for module in (self.router, self.head) for p in module.parameters())

    def make_batch(self, states, questions, criteria):
        if not states or len(states) != len(questions) or len(states) != len(criteria):
            raise ValueError("Supply matching nonempty state/question/criteria batches")
        for q, c in zip(questions, criteria):
            validate_text(q, c)
        prepared = [s if isinstance(s, PreparedState) else self.preprocessor.prepare(s) for s in states]
        strings = [s for q, c in zip(questions, criteria) for s in [q, *c]] + [ENDING]
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
        return DecisionBatch(prepared, torch.stack(qs), cs, vectors[-1:].expand(len(states), -1),
                             mask, [list(c) for c in criteria]).to(self.device)

    @staticmethod
    def router_inputs(batch):
        frames, owners = [], []
        compatible = torch.stack([s.compatible for s in batch.states]).to(batch.question.device)
        for expert, length in enumerate((390, 512)):
            pairs = [(i, s.views[expert]) for i, s in enumerate(batch.states) if s.views[expert] is not None]
            frames.append(torch.cat([view.frames for _, view in pairs]) if pairs else batch.question.new_empty(0, 1, length))
            owners.append(torch.cat([torch.full((len(view.frames),), i, device=batch.question.device, dtype=torch.long)
                                     for i, view in pairs]) if pairs else torch.empty(0, device=batch.question.device, dtype=torch.long))
        return tuple(frames), tuple(owners), compatible

    def encode(self, prepared, expert):
        view = prepared.views[expert]
        if view is None:
            raise ValueError("Cannot encode an unavailable preparation")
        with torch.no_grad():
            encoded = self.encoders[expert](view.to(self.device))
        expected_width = (768, 512)[expert]
        if encoded.values.ndim != 2 or encoded.values.shape[-1] != expected_width:
            raise ValueError("Encoder output width differs from its fixed adapter")
        if encoded.positions.shape != (len(encoded.values), 3) or not torch.isfinite(encoded.positions).all():
            raise ValueError("Invalid structural positions")
        if not 1 <= len(encoded.values) <= self.preprocessor.config.max_signal_tokens:
            raise ValueError("Invalid encoded token budget")
        return EncodedSignal(self.adapters[expert](encoded.values.float()).detach(), encoded.positions.float().detach())

    def forward(self, batch, routes=None, cache=None):
        frames, owners, compatible = self.router_inputs(batch)
        route_logits = self.router(frames, owners, batch.question, compatible)
        if routes is None:
            routes = route_logits.argmax(-1)
        if routes.shape != (len(batch.states),) or routes.dtype != torch.long:
            raise ValueError("Routes must be integer expert indices [B]")
        if ((routes < 0) | (routes > 1)).any() or not compatible.gather(1, routes[:, None]).all():
            raise ValueError("An expert route is incompatible")
        encoded = []
        for state, route in zip(batch.states, routes.tolist()):
            value = cache.get_or_encode(self, state, route) if cache else self.encode(state, route)
            encoded.append(value)
        tokens = max(len(e.values) for e in encoded)
        signal = batch.question.new_zeros(len(encoded), tokens, 768)
        positions = batch.question.new_zeros(len(encoded), tokens, 3)
        mask = torch.zeros(len(encoded), tokens, dtype=torch.bool, device=self.device)
        for i, value in enumerate(encoded):
            length = len(value.values)
            signal[i, :length] = value.values.to(self.device)
            positions[i, :length] = value.positions.to(self.device)
            mask[i, :length] = True
        logits = self.head(signal, mask, positions, batch.question, batch.criteria, batch.ending, batch.criteria_mask)
        return {"logits": logits, "router_logits": route_logits, "routes": routes,
                "criteria_mask": batch.criteria_mask}

    @torch.no_grad()
    def predict(self, state, question, criteria):
        self.eval()
        if not torch.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Prediction temperature must be finite and positive")
        batch = self.make_batch([state], [question], [criteria])
        logits = self(batch)["logits"][0]
        probabilities = (logits / self.temperature).softmax(-1).cpu().tolist()
        chosen = min(range(len(criteria)), key=lambda i: (-probabilities[i], criteria[i]))
        return {"choice": criteria[chosen], "probabilities": probabilities,
                "confidence": probabilities[chosen]}
