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


@dataclass
class RecallConfig:
    rrf_k: int = 60
    top_k: int = 10
    confidence_threshold: float = 0.5


@dataclass
class Config:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector: VectorConfig = field(default_factory=VectorConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    recall: RecallConfig = field(default_factory=RecallConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

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
        MNEMOSTACK_EMBEDDING_MODEL
        MNEMOSTACK_VECTOR_HOST
        MNEMOSTACK_VECTOR_COLLECTION
        MNEMOSTACK_LLM_PROVIDER
        MNEMOSTACK_LLM_MODEL
        MNEMOSTACK_GRAPH_URI
        MNEMOSTACK_QDRANT_HOST  (alias for VECTOR_HOST)
        MNEMOSTACK_COLLECTION   (alias for VECTOR_COLLECTION)
        MNEMOSTACK_MEMGRAPH_URI (alias for GRAPH_URI)
    """
    env = os.environ

    # Embedding
    if v := env.get("MNEMOSTACK_EMBEDDING_PROVIDER"):
        cfg.embedding.provider = v
    if v := env.get("MNEMOSTACK_EMBEDDING_MODEL"):
        cfg.embedding.model = v

    # Vector (with aliases)
    host = env.get("MNEMOSTACK_VECTOR_HOST") or env.get("MNEMOSTACK_QDRANT_HOST")
    if host:
        cfg.vector.host = host
    collection = env.get("MNEMOSTACK_VECTOR_COLLECTION") or env.get("MNEMOSTACK_COLLECTION")
    if collection:
        cfg.vector.collection = collection

    # LLM
    if v := env.get("MNEMOSTACK_LLM_PROVIDER"):
        cfg.llm.provider = v
    if v := env.get("MNEMOSTACK_LLM_MODEL"):
        cfg.llm.model = v

    # Graph (with alias)
    graph_uri = env.get("MNEMOSTACK_GRAPH_URI") or env.get("MNEMOSTACK_MEMGRAPH_URI")
    if graph_uri:
        cfg.graph.uri = graph_uri

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

llm:
  provider: gemini
  model: null             # null = provider default (gemini-2.5-flash)

graph:
  uri: null               # e.g. bolt://localhost:7687 to enable graph
  user: ""
  password: ""

recall:
  rrf_k: 60
  top_k: 10
  confidence_threshold: 0.5
"""
