"""Clean native signals, route, format the chosen encoder input and score choices."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .encoders import EncodedSignal, FixedWidthAdapter, NormWearEncoder, MantisEncoder
from .head import DecisionHead
from .inputs import validate_text
from .preprocessing import SignalPreprocessor, PreparationConfig, CleanedState
from .provenance import RUNTIME_CONTRACT, ARCHITECTURE, verify_assets
from .router import ExpertRouter
from .text import TextEncoder


@dataclass
class DecisionBatch:
    states: list[CleanedState]
    question: torch.Tensor
    criteria: torch.Tensor
    criteria_mask: torch.Tensor
    descriptions: list[list[str]]

    def to(self, device):
        return DecisionBatch([s.to(device) for s in self.states], self.question.to(device),
                             self.criteria.to(device),
                             self.criteria_mask.to(device), self.descriptions)


class PayaModel(nn.Module):
    """Choose a frozen encoder from native cleaned signals and score candidate answers."""

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
        """Clean native signals and encode each question and choice separately."""
        if not states or len(states) != len(questions) or len(states) != len(criteria):
            raise ValueError("Supply matching nonempty state/question/criteria batches")
        for q, c in zip(questions, criteria):
            validate_text(q, c)
        cleaned = [s if isinstance(s, CleanedState) else self.preprocessor.clean(s) for s in states]
        if any(s.audit.get("preparation") != self.preprocessor.identity() for s in cleaned):
            raise ValueError("Cleaned signal does not match current physiological preprocessing")
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
        return DecisionBatch(cleaned, torch.stack(qs), cs,
                             mask, [list(c) for c in criteria]).to(self.device)

    @staticmethod
    def router_inputs(batch):
        signals, owners = [], []
        compatible = torch.stack([s.compatible for s in batch.states]).to(batch.question.device)
        for index, state in enumerate(batch.states):
            for channel in state.channels:
                signals.append(torch.from_numpy(channel.samples).to(batch.question.device))
                owners.append(index)
        return signals, torch.tensor(owners, device=batch.question.device, dtype=torch.long), compatible

    def encode(self, cleaned, expert):
        encoder_input = self.preprocessor.format_for_encoder(cleaned, expert)
        with torch.no_grad():
            encoded = self.encoders[expert](encoder_input.to(self.device))
        expected_width = (768, 512)[expert]
        if encoded.values.ndim != 2 or encoded.values.shape[-1] != expected_width:
            raise ValueError("Encoder output width differs from its fixed adapter")
        if (encoded.times.shape != (len(encoded.values),) or not torch.isfinite(encoded.times).all()
                or ((encoded.times < 0) | (encoded.times > 1)).any()):
            raise ValueError("Invalid relative times")
        if not 1 <= len(encoded.values) <= self.preprocessor.config.max_signal_tokens:
            raise ValueError("Invalid encoded token budget")
        return EncodedSignal(self.adapters[expert](encoded.values.float()).detach(), encoded.times.float().detach())

    def forward(self, batch, routes=None, cache=None):
        signals, owners, compatible = self.router_inputs(batch)
        route_logits = self.router(signals, owners, batch.question, compatible)
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
        times = batch.question.new_zeros(len(encoded), tokens)
        mask = torch.zeros(len(encoded), tokens, dtype=torch.bool, device=self.device)
        for i, value in enumerate(encoded):
            length = len(value.values)
            signal[i, :length] = value.values.to(self.device)
            times[i, :length] = value.times.to(self.device)
            mask[i, :length] = True
        logits = self.head(signal, mask, times, batch.question, batch.criteria, batch.criteria_mask)
        return {"logits": logits, "router_logits": route_logits, "routes": routes,
                "criteria_mask": batch.criteria_mask}

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
