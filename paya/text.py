"""Load pinned BioLORD weights and cache normalized text embeddings."""

import argparse
import json
import os
from pathlib import Path

import torch

from .provenance import TEXT_ID, TEXT_REVISION, TEXT_IDENTITY, VERSION, sha256


class TextEncoder:
    """Load frozen BioLORD on demand and cache normalized sentence vectors on CPU."""

    def __init__(self, cache=None, device="cpu", model_path=None):
        self.device, self.model_path, self.model = device, model_path, None
        self.embeddings = {}
        if cache is not None and Path(cache).exists():
            payload = torch.load(cache, map_location="cpu", weights_only=True)
            if payload.get("identity") != TEXT_IDENTITY:
                raise ValueError("Text bank does not match pinned BioLORD processing")
            for text, vector in payload["embeddings"].items():
                if vector.shape != (768,) or not torch.isfinite(vector).all():
                    raise ValueError("Invalid text bank vector")
                self.embeddings[text] = vector.float()

    def encode(self, strings):
        """Return separate [N,768] CPU vectors; reject texts above the token limit."""
        if not isinstance(strings, list) or not strings or any(not isinstance(s, str) or not s.strip() for s in strings):
            raise ValueError("Encode a nonempty list of separate, nonempty text strings")
        missing = list(dict.fromkeys(s for s in strings if s not in self.embeddings))
        if missing:
            if self.model is None:
                from sentence_transformers import SentenceTransformer

                local = self.model_path or (
                    Path("third-party/biolord")
                    if Path("third-party/biolord").exists()
                    else None
                )
                if local is not None:
                    local = Path(local)
                    source = json.loads((local / "source.json").read_text())
                    if (
                        source.get("model_id") != TEXT_ID
                        or source.get("revision") != TEXT_REVISION
                    ):
                        raise ValueError(
                            "Local text weights do not match pinned BioLORD"
                        )
                    for name, expected in source["files"].items():
                        path = (local / name.replace("\\", "/")).resolve()
                        if (
                            not path.is_relative_to(local.resolve())
                            or sha256(path) != expected
                        ):
                            raise ValueError(
                                "Local text weight/config checksum changed"
                            )
                self.model_path = local
                self.model = SentenceTransformer(
                    str(local) if local else TEXT_ID,
                    revision=None if local else TEXT_REVISION,
                    device=self.device,
                    trust_remote_code=False,
                    cache_folder=os.environ.get("HF_HOME", "cache/huggingface"),
                )
                self.model.eval().requires_grad_(False)
            for text in missing:
                if (
                    len(self.model.tokenizer.encode(text, add_special_tokens=True))
                    > self.model.max_seq_length
                ):
                    raise ValueError(
                        "Text exceeds BioLORD context; do not silently truncate decisive wording"
                    )
            with torch.inference_mode():
                vectors = self.model.encode(
                    missing,
                    normalize_embeddings=True,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    batch_size=32,
                )
            for text, vector in zip(missing, vectors):
                if vector.shape != (768,) or not torch.isfinite(vector).all():
                    raise ValueError("Text encoder produced an invalid vector")
                self.embeddings[text] = vector.detach().cpu().float().clone()
        return torch.stack([self.embeddings[s] for s in strings])

    def save(self, path):
        """Save cached vectors with their encoder identity."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"identity": TEXT_IDENTITY, "embeddings": self.embeddings}, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texts", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(f"cache/{VERSION}/text.pt"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    encoder = TextEncoder(args.output, args.device)
    encoder.encode(json.loads(args.texts.read_text()))
    encoder.save(args.output)
    print(json.dumps({"texts": len(encoder.embeddings), "identity": TEXT_IDENTITY}))


if __name__ == "__main__":
    main()
