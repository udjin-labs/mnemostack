"""Configuration loader for mnemostack.

Priority order (later overrides earlier):
    1. Built-in defaults
    2. Config file (~/.config/mnemostack/config.yaml or explicit path)
    3. Environment variables (MNEMOSTACK_*)
    4. Explicit arguments

Example config file:

    embedding:
      provider: gemini
      model: gemini-embedding-001

    vector:
      host: http://localhost:6333
      collection: my-memory
      chunk_size: 800

    llm:
      provider: gemini
      model: gemini-2.5-flash

    graph:
      uri: bolt://localhost:7687

    recall:
      rrf_k: 60
      confidence_threshold: 0.5
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATHS = [
    Path.home() / ".config" / "mnemostack" / "config.yaml",
    Path.home() / ".config" / "mnemostack" / "config.yml",
    Path.home() / ".mnemostack.yaml",
]


def model_kwargs(model: str | None) -> dict[str, str]:
    """Return provider kwargs for an optional model override."""
    return {"model": model} if model else {}


@dataclass
class EmbeddingConfig:
    provider: str = "gemini"
    model: str | None = None  # uses provider default if None
    api_key_env: str = "GEMINI_API_KEY"
    ollama_host: str = "http://localhost:11434"


@dataclass
class VectorConfig:
    host: str = "http://localhost:6333"
    collection: str = "mnemostack"
    chunk_size: int = 800
    overlap: int = 100
    window_size: int = 1
    # Short timeout (whole seconds) for the HTTP server's liveness/readiness
    # Qdrant ping — kept separate from recall so a slow Qdrant fails a probe
    # promptly instead of hanging a worker for the recall client's full timeout.
    health_timeout: int = 2


@dataclass
class LLMConfig:
    provider: str = "gemini"
    model: str | None = None


@dataclass
class GraphConfig:
    uri: str | None = None  # None = graph disabled
    user: str = ""
    password: str = ""
    database: str | None = None
    timeout: float = 5.0
    health_timeout: float = 1.0


#: Valid values for ``recall.text_search`` (see the field's docstring).
TEXT_SEARCH_MODES = ("auto", "off", "bm25", "qdrant_bm25", "lexical", "sparse")


def resolve_text_search_mode(mode: str, bm25_paths: list[str] | None) -> str:
    """Concrete lexical-arm choice: ``auto`` keeps historical behavior
    (file-corpus BM25 when ``bm25_paths`` is configured, else no lexical arm);
    anything else passes through. Unknown modes fail loud at build time."""
    if mode not in TEXT_SEARCH_MODES:
        raise ValueError(
            f"text_search must be one of {TEXT_SEARCH_MODES}, got {mode!r}"
        )
    if mode == "auto":
        return "bm25" if bm25_paths else "off"
    return mode


def parse_text_search_fields(value: Any) -> dict[str, float]:
    """Normalize ``recall.text_search_fields`` into ``{payload_field: weight}``.

    Accepts the YAML mapping form (``{title: 2.0, text: 1.0}``) and the env
    string form (``"title:2.0,text"`` — omitted weight means 1.0). Weights are
    fusion-level (they weigh the arm's ranked LIST in RRF; the MatchText gate
    itself cannot score). Invalid shapes fail loud: a typo here silently
    dropping a lexical arm is exactly the misconfiguration this feature makes
    expressible.
    """
    if value is None:
        return {}
    fields: dict[str, float] = {}
    if isinstance(value, str):
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            key, _, w = part.partition(":")
            key = key.strip()
            if key in fields:
                # Last-wins would silently drop a weight — the very class of
                # absorbed typo (e.g. an env var appended twice by templating)
                # this parser exists to reject.
                raise ValueError(
                    f"text_search_fields lists field {key!r} more than once"
                )
            fields[key] = _text_field_weight(key, w.strip() or "1.0")
    elif isinstance(value, dict):
        for key, w in value.items():
            norm = str(key).strip()
            if norm in fields:
                # Keys distinct only by whitespace collapse after trimming —
                # last-wins would silently drop a weight, same as the env form.
                raise ValueError(
                    f"text_search_fields lists field {norm!r} more than once"
                )
            fields[norm] = _text_field_weight(norm, w)
    else:
        raise ValueError(
            "text_search_fields must be a mapping of payload field -> weight "
            f"(or a 'field:weight,...' string), got {type(value).__name__}"
        )
    if any(not k for k in fields):
        raise ValueError("text_search_fields contains an empty field name")
    return fields


def _text_field_weight(key: str, raw: Any) -> float:
    if isinstance(raw, bool):
        # bool is a float subclass, so YAML `title: yes` would silently become
        # weight 1.0 — almost certainly a typo for a number, not an intent.
        raise ValueError(
            f"text_search_fields weight for {key!r} must be a number, got {raw!r}"
        )
    try:
        w = float(raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"text_search_fields weight for {key!r} must be a number, got {raw!r}"
        ) from None
    if not math.isfinite(w) or w <= 0:
        raise ValueError(
            f"text_search_fields weight for {key!r} must be a positive finite "
            f"number, got {raw!r}"
        )
    return w


def ensure_text_fields_mode(resolved_mode: str, fields: dict[str, float]) -> None:
    """Fail loud when ``text_search_fields`` is configured but the resolved
    lexical mode is not ``lexical`` — the fields would be silently ignored,
    and a deployment that configured a title boost deserves an error, not a
    quietly missing arm."""
    if fields and resolved_mode != "lexical":
        raise ValueError(
            "recall.text_search_fields requires text_search=lexical "
            f"(resolved mode is {resolved_mode!r})"
        )


@dataclass
class RecallConfig:
    rrf_k: int = 60
    top_k: int = 10
    confidence_threshold: float = 0.5
    bm25_paths: list[str] = field(default_factory=list)
    vector_floor: int = 0
    rerank_mode: str = "relevant_only"
    #: Default token budget applied to recall results on the HTTP/MCP/CLI
    #: surfaces when the caller does not pass one. None = no budget.
    token_budget: int | None = None
    #: Payload schema of the (possibly pre-existing) collection: which payload
    #: keys hold the chunk text and its timestamp, and how timestamps are
    #: stored — "iso" (RFC3339 strings, the mnemostack-written default),
    #: "epoch" (numeric seconds), or "epoch_ms" (numeric milliseconds). Lets
    #: recall mount a foreign collection without rewriting its payloads.
    text_key: str = "text"
    timestamp_key: str = "timestamp"
    timestamp_format: str = "iso"
    #: Which lexical arm joins the recall fleet. "auto" (default) keeps the
    #: historical behavior: in-process BM25 over bm25_paths when set, else no
    #: lexical arm. "bm25" = markdown-file corpus; "qdrant_bm25" = in-process
    #: BM25 loaded FROM the Qdrant payloads (fine under ~100K chunks);
    #: "lexical" = lexical-gated dense over a Qdrant full-text index (any
    #: size, no reindex); "sparse" = server-side sparse tf·idf scoring (any
    #: size, needs the sparse space at ingest); "off" = none.
    text_search: str = "auto"
    #: Optional multi-field layout for the ``lexical`` arm: a mapping of
    #: payload field -> fusion weight (e.g. ``{title: 2.0, text: 1.0}``).
    #: Each field becomes its OWN lexically-gated arm (gating on that field,
    #: returning chunk text from ``text_key``), fused with the given weight —
    #: title/heading fields are typically far more precise lexical signals
    #: than chunk bodies. Empty (default) = the single arm over ``text_key``.
    #: When set, this REPLACES the arm set — list ``text_key`` explicitly if
    #: the body arm should stay. Weight 1.0 is left to the adaptive profile;
    #: any other weight is a static fusion override. Requires
    #: ``text_search: lexical`` (anything else fails loud at build time).
    text_search_fields: dict[str, float] = field(default_factory=dict)


@dataclass
class Config:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector: VectorConfig = field(default_factory=VectorConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    recall: RecallConfig = field(default_factory=RecallConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def embedding_provider_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for the configured embedding provider."""
        return model_kwargs(self.embedding.model)

    def llm_provider_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for the configured LLM provider."""
        return model_kwargs(self.llm.model)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        """Load config from file + env vars. If path=None, search default locations."""
        cfg = cls()

        # 1. File
        file_path = _resolve_config_path(path)
        if file_path and file_path.exists():
            with open(file_path) as f:
                data = yaml.safe_load(f) or {}
            cfg = _merge_dict_into_config(cfg, data)

        # 2. Env vars (MNEMOSTACK_*)
        cfg = _apply_env_overrides(cfg)

        # A configured budget of 0 or less means "no budget", not a broken
        # deployment: apply_token_budget requires >= 1, so passing a bad
        # file/env value through verbatim would turn a config typo into a
        # ValueError on every recall (HTTP 500s / CLI crashes).
        budget = cfg.recall.token_budget
        if budget is not None and int(budget) <= 0:
            cfg.recall.token_budget = None

        # Normalize whichever shape arrived (YAML mapping or env string) into
        # {field: weight}; malformed values fail here, at load, not at recall.
        cfg.recall.text_search_fields = parse_text_search_fields(
            cfg.recall.text_search_fields
        )

        return cfg

    def save(self, path: str | Path) -> None:
        """Write config to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


def _resolve_config_path(path: str | Path | None) -> Path | None:
    if path is not None:
        return Path(path).expanduser()
    # Check env override first
    env_path = os.environ.get("MNEMOSTACK_CONFIG")
    if env_path:
        return Path(env_path).expanduser()
    # Search default paths
    for candidate in DEFAULT_CONFIG_PATHS:
        if candidate.exists():
            return candidate
    return None


def _merge_dict_into_config(cfg: Config, data: dict[str, Any]) -> Config:
    """Overlay dict onto Config. Unknown keys are ignored."""
    for section_name, section_data in data.items():
        if not isinstance(section_data, dict):
            continue
        section = getattr(cfg, section_name, None)
        if section is None:
            continue
        for key, value in section_data.items():
            if hasattr(section, key):
                setattr(section, key, value)
    return cfg


def _apply_env_overrides(cfg: Config) -> Config:
    """Apply MNEMOSTACK_* env vars to config.

    Supported:
        MNEMOSTACK_EMBEDDING_PROVIDER
        MNEMOSTACK_PROVIDER          (alias for EMBEDDING_PROVIDER)
        MNEMOSTACK_EMBEDDING        (alias for EMBEDDING_PROVIDER)
        MNEMOSTACK_EMBEDDING_MODEL
        MNEMOSTACK_VECTOR_HOST
        MNEMOSTACK_QDRANT_URL       (alias for VECTOR_HOST)
        MNEMOSTACK_VECTOR_COLLECTION
        MNEMOSTACK_LLM_PROVIDER
        MNEMOSTACK_LLM              (alias for LLM_PROVIDER)
        MNEMOSTACK_LLM_MODEL
        MNEMOSTACK_GRAPH_URI
        MNEMOSTACK_GRAPH_USER
        MNEMOSTACK_GRAPH_PASSWORD
        MNEMOSTACK_GRAPH_DATABASE
        MNEMOSTACK_GRAPH_TIMEOUT
        MNEMOSTACK_GRAPH_HEALTH_TIMEOUT
        MNEMOSTACK_BM25_PATHS       (os.pathsep-separated)
        MNEMOSTACK_VECTOR_FLOOR
        MNEMOSTACK_RERANK_MODE      (relevant_only | full_reorder)
        MNEMOSTACK_TOKEN_BUDGET     (default recall token budget; unset = none)
        MNEMOSTACK_TEXT_KEY         (payload key holding chunk text)
        MNEMOSTACK_TIMESTAMP_KEY    (payload key holding the timestamp)
        MNEMOSTACK_TIMESTAMP_FORMAT (iso | epoch | epoch_ms)
        MNEMOSTACK_TEXT_SEARCH      (auto | off | bm25 | qdrant_bm25 | lexical | sparse)
        MNEMOSTACK_QDRANT_HOST  (alias for VECTOR_HOST)
        MNEMOSTACK_COLLECTION   (alias for VECTOR_COLLECTION)
        MNEMOSTACK_MEMGRAPH_URI (alias for GRAPH_URI)
    """
    env = os.environ

    # Embedding
    embedding_provider = (
        env.get("MNEMOSTACK_EMBEDDING_PROVIDER")
        or env.get("MNEMOSTACK_PROVIDER")
        or env.get("MNEMOSTACK_EMBEDDING")
    )
    if embedding_provider:
        cfg.embedding.provider = embedding_provider
    if v := env.get("MNEMOSTACK_EMBEDDING_MODEL"):
        cfg.embedding.model = v

    # Vector (with aliases)
    host = (
        env.get("MNEMOSTACK_VECTOR_HOST")
        or env.get("MNEMOSTACK_QDRANT_URL")
        or env.get("MNEMOSTACK_QDRANT_HOST")
    )
    if host:
        cfg.vector.host = host
    collection = env.get("MNEMOSTACK_VECTOR_COLLECTION") or env.get("MNEMOSTACK_COLLECTION")
    if collection:
        cfg.vector.collection = collection
    if v := env.get("MNEMOSTACK_VECTOR_HEALTH_TIMEOUT"):
        cfg.vector.health_timeout = max(1, int(v))

    # LLM
    llm_provider = env.get("MNEMOSTACK_LLM_PROVIDER") or env.get("MNEMOSTACK_LLM")
    if llm_provider:
        cfg.llm.provider = llm_provider
    if v := env.get("MNEMOSTACK_LLM_MODEL"):
        cfg.llm.model = v

    # Graph (with alias)
    graph_uri = env.get("MNEMOSTACK_GRAPH_URI") or env.get("MNEMOSTACK_MEMGRAPH_URI")
    if graph_uri:
        cfg.graph.uri = graph_uri
    if v := env.get("MNEMOSTACK_GRAPH_USER"):
        cfg.graph.user = v
    if v := env.get("MNEMOSTACK_GRAPH_PASSWORD"):
        cfg.graph.password = v
    if v := env.get("MNEMOSTACK_GRAPH_DATABASE"):
        cfg.graph.database = v
    if v := env.get("MNEMOSTACK_GRAPH_TIMEOUT"):
        cfg.graph.timeout = float(v)
    if v := env.get("MNEMOSTACK_GRAPH_HEALTH_TIMEOUT"):
        cfg.graph.health_timeout = float(v)

    # Recall
    if v := env.get("MNEMOSTACK_BM25_PATHS"):
        cfg.recall.bm25_paths = [p for p in v.split(os.pathsep) if p]
    if v := env.get("MNEMOSTACK_VECTOR_FLOOR"):
        cfg.recall.vector_floor = max(0, int(v))
    if v := env.get("MNEMOSTACK_RERANK_MODE"):
        cfg.recall.rerank_mode = v
    if v := env.get("MNEMOSTACK_TOKEN_BUDGET"):
        # <= 0 is normalized to "no budget" by Config.load
        cfg.recall.token_budget = int(v)
    if v := env.get("MNEMOSTACK_TEXT_KEY"):
        cfg.recall.text_key = v
    if v := env.get("MNEMOSTACK_TIMESTAMP_KEY"):
        cfg.recall.timestamp_key = v
    if v := env.get("MNEMOSTACK_TIMESTAMP_FORMAT"):
        cfg.recall.timestamp_format = v
    if v := env.get("MNEMOSTACK_TEXT_SEARCH"):
        cfg.recall.text_search = v
    if v := env.get("MNEMOSTACK_TEXT_SEARCH_FIELDS"):
        # Raw string here; Config.load normalizes (parse_text_search_fields).
        cfg.recall.text_search_fields = v  # type: ignore[assignment]

    return cfg


def generate_example_config() -> str:
    """Return a YAML string with all defaults + comments. Useful for `mnemostack init`."""
    return """# mnemostack configuration file
# See https://github.com/udjin-labs/mnemostack/blob/main/docs/config.md

embedding:
  provider: gemini        # gemini | ollama | huggingface
  model: null             # null = provider default
  api_key_env: GEMINI_API_KEY
  ollama_host: http://localhost:11434

vector:
  host: http://localhost:6333
  collection: mnemostack
  chunk_size: 800
  overlap: 100
  window_size: 1
  health_timeout: 2         # seconds; HTTP server's Qdrant liveness/readiness ping

llm:
  provider: gemini
  model: null             # null = provider default (gemini-2.5-flash)

graph:
  uri: null               # e.g. bolt://localhost:7687 to enable graph
  user: ""
  password: ""
  database: null
  timeout: 5.0
  health_timeout: 1.0

recall:
  rrf_k: 60
  top_k: 10
  confidence_threshold: 0.5
  bm25_paths: []
  vector_floor: 0
  rerank_mode: relevant_only  # relevant_only | full_reorder
  token_budget: null          # e.g. 2000 = trim recall results to ~2000 text tokens; 0/null = off
  # Payload schema of the collection recall reads (a pre-existing collection
  # keeps its own field names; timestamps may be numeric epochs):
  text_key: text
  timestamp_key: timestamp
  timestamp_format: iso       # iso | epoch | epoch_ms
  text_search: auto           # auto | off | bm25 | qdrant_bm25 | lexical | sparse
  # Multi-field lexical arms (text_search: lexical only): one gated arm per
  # payload field, fused with the given weight — title/heading fields are
  # usually far more precise lexical signals than chunk bodies. When set,
  # this REPLACES the arm set (list text_key explicitly to keep the body arm).
  # text_search_fields:
  #   title: 2.0
  #   text: 1.0
"""
