"""HuggingFace Transformers embedding provider (local, GPU).

Requires: pip install mnemostack[huggingface]
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .base import EmbeddingProvider

# Small text assets that change tokenization — and therefore the vectors —
# without touching config.json or the weight files. Hashed by CONTENT.
_TOKENIZER_ASSETS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "spiece.model",
    "sentencepiece.bpe.model",
)


# Per weight file, hash this many bytes from the head and from the tail in
# addition to (name, size). A same-architecture checkpoint swap keeps names
# and often sizes, but tensor data differs from the first bytes on — sampled
# content catches it without reading gigabytes at startup.
_WEIGHT_SAMPLE_BYTES = 1024 * 1024


def _local_weights_signature(path: Path) -> str:
    """Cheap identity of a local model directory for space fingerprints.

    A local path has no hub commit hash, yet its contents can be swapped in
    place while provider/model/pooling/dimension all stay equal. Hashing
    gigabytes of weights on every startup is not viable, so this pins the
    config and tokenizer asset BYTES in full (small files whose changes
    alter the vectors) plus, per weight file, its name, size and a sampled
    head+tail megabyte of content — catching config/tokenizer changes and
    same-size checkpoint swaps. Residual: a weight edit confined strictly
    to the unsampled middle of a file is not detected.
    """
    h = hashlib.sha256()
    for asset in ("config.json", *_TOKENIZER_ASSETS):
        f = path / asset
        if f.is_file():
            h.update(asset.encode())
            h.update(f.read_bytes())
    for f in sorted(path.iterdir()):
        if f.suffix in (".safetensors", ".bin", ".pt") and f.is_file():
            size = f.stat().st_size
            h.update(f"{f.name}:{size}".encode())
            with f.open("rb") as fh:
                if size <= 2 * _WEIGHT_SAMPLE_BYTES:
                    h.update(fh.read())  # small file: hash it whole
                else:
                    h.update(fh.read(_WEIGHT_SAMPLE_BYTES))
                    fh.seek(-_WEIGHT_SAMPLE_BYTES, 2)
                    h.update(fh.read(_WEIGHT_SAMPLE_BYTES))
    return h.hexdigest()[:32]

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

    # Decoder-based embedding families derive the sentence vector from the
    # final non-padding token — mean/CLS silently produce the wrong
    # representation for them, so they get a different pooling DEFAULT.
    _LAST_TOKEN_FAMILIES = ("qwen3-embedding", "e5-mistral")

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str | None = None,
        pooling: str | None = None,
        revision: str | None = None,
    ):
        if not _AVAILABLE:
            raise ImportError("HuggingFaceProvider requires `pip install mnemostack[huggingface]`")
        self.model_name = model
        if pooling is None:
            pooling = (
                "last"
                if any(f in model.lower() for f in self._LAST_TOKEN_FAMILIES)
                else "mean"
            )
        # Normalized and validated: pooling participates in the space
        # fingerprints, so "CLS" silently mean-pooling while fingerprinting
        # as a distinct space would corrupt both sides.
        self.pooling = pooling.lower()
        if self.pooling not in ("mean", "cls", "last"):
            raise ValueError(f"pooling must be 'mean', 'cls' or 'last', got {pooling!r}")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.model = AutoModel.from_pretrained(model, revision=revision).to(self.device).eval()
        # The RESOLVED weights identity: a mutable branch label ("main") can
        # be repointed to different weights, so the space fingerprint pins
        # the commit hash when the hub provides one — and a content signature
        # when the model is a LOCAL directory. For local directories the
        # signature wins over a caller-supplied `revision`: Transformers
        # ignores revisions for local files, so an arbitrary label would
        # mask an in-place file swap.
        commit = getattr(self.model.config, "_commit_hash", None)
        model_dir = Path(model)
        self.revision: str | None
        if commit is None and model_dir.is_dir():
            self.revision = f"local:{_local_weights_signature(model_dir)}"
        else:
            self.revision = commit or revision
        # Infer dimension from a probe embedding
        self._dim = len(self.embed("dim probe"))

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"huggingface:{self.model_name}"

    def _legacy_space_compatible(self) -> bool:
        # Pre-fingerprint mnemostack always mean-pooled: only an active mean
        # configuration reproduces those legacy vectors. The last-token
        # auto-default (correct for decoder families) must NOT silently adopt
        # a collection that was mean-pooled before the upgrade.
        return self.pooling == "mean"

    def _fingerprint_extras(self) -> dict[str, str]:
        # Pooling changes the vector space for the same model, so it must
        # participate in the embedding-space fingerprints — as must the
        # resolved weights revision (mutable labels can be repointed).
        extras = {"pooling": self.pooling}
        if self.revision:
            extras["revision"] = str(self.revision)
        return extras

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
        elif self.pooling == "last":
            # Last non-padding token per sequence. Computed as the position
            # of the mask's last 1 so it is correct for BOTH padding sides —
            # decoder tokenizers (Qwen, E5-mistral) often pad left, where
            # `mask.sum()-1` would land on padding or a mid-sequence token.
            att = inputs["attention_mask"]
            positions = torch.arange(att.size(1), device=token_embs.device).unsqueeze(0)
            last_idx = (att * positions).argmax(dim=1)
            rows = torch.arange(token_embs.size(0), device=token_embs.device)
            pooled = token_embs[rows, last_idx]
        else:
            pooled = (token_embs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        # L2 normalize
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().tolist()
