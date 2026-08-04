"""HuggingFace Transformers embedding provider (local, GPU).

Requires: pip install mnemostack[huggingface]
"""

from __future__ import annotations

from .base import EmbeddingProvider

try:
    import torch
    from transformers import AutoModel, AutoTokenizer

    _AVAILABLE = True
except ImportError:  # pragma: no cover
    _AVAILABLE = False


class HuggingFaceProvider(EmbeddingProvider):
    """Embedding via HuggingFace Transformers — runs locally, supports GPU.

    Default: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, fast).
    For higher quality try `BAAI/bge-large-en-v1.5` (1024-dim).
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str | None = None,
        pooling: str = "mean",
    ):
        if not _AVAILABLE:
            raise ImportError("HuggingFaceProvider requires `pip install mnemostack[huggingface]`")
        self.model_name = model
        # Normalized and validated: pooling participates in the space
        # fingerprints, so "CLS" silently mean-pooling while fingerprinting
        # as a distinct space would corrupt both sides.
        self.pooling = pooling.lower()
        if self.pooling not in ("mean", "cls"):
            raise ValueError(f"pooling must be 'mean' or 'cls', got {pooling!r}")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).to(self.device).eval()
        # Infer dimension from a probe embedding
        self._dim = len(self.embed("dim probe"))

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"huggingface:{self.model_name}"

    def _fingerprint_extras(self) -> dict[str, str]:
        # Pooling changes the vector space for the same model, so it must
        # participate in the embedding-space fingerprints.
        return {"pooling": self.pooling}

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not _AVAILABLE:
            return [[] for _ in texts]
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt", max_length=512
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Mean pooling over the token dim
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        token_embs = outputs.last_hidden_state
        if self.pooling == "cls":
            pooled = token_embs[:, 0]
        else:
            pooled = (token_embs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        # L2 normalize
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().tolist()
