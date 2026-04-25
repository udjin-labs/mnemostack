"""Tests for Config loader."""
import os

import pytest

from mnemostack.config import Config, generate_example_config


@pytest.fixture
def isolated_env(monkeypatch):
    """Remove all MNEMOSTACK_* env vars to avoid pollution."""
    for key in list(os.environ.keys()):
        if key.startswith("MNEMOSTACK_"):
            monkeypatch.delenv(key)
    monkeypatch.delenv("MNEMOSTACK_CONFIG", raising=False)
    yield monkeypatch


def test_default_config_values(isolated_env, tmp_path, monkeypatch):
    # Disable default file search by pointing HOME to an empty dir
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = Config.load()
    assert cfg.embedding.provider == "gemini"
    assert cfg.vector.collection == "mnemostack"
    assert cfg.llm.provider == "gemini"
    assert cfg.graph.uri is None
    assert cfg.recall.rrf_k == 60


def test_load_from_yaml(isolated_env, tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
embedding:
  provider: ollama
  model: nomic-embed-text

vector:
  collection: my-memory
  chunk_size: 400

graph:
  uri: bolt://localhost:7687
"""
    )
    cfg = Config.load(cfg_path)
    assert cfg.embedding.provider == "ollama"
    assert cfg.embedding.model == "nomic-embed-text"
    assert cfg.vector.collection == "my-memory"
    assert cfg.vector.chunk_size == 400
    assert cfg.graph.uri == "bolt://localhost:7687"
    # Unspecified values keep defaults
    assert cfg.recall.rrf_k == 60


def test_env_overrides_file(isolated_env, tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("vector:\n  collection: from-file\n")
    isolated_env.setenv("MNEMOSTACK_COLLECTION", "from-env")
    cfg = Config.load(cfg_path)
    assert cfg.vector.collection == "from-env"


def test_env_vars_direct(isolated_env, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    isolated_env.setenv("MNEMOSTACK_EMBEDDING_PROVIDER", "huggingface")
    isolated_env.setenv("MNEMOSTACK_GRAPH_URI", "bolt://foo:7687")
    cfg = Config.load()
    assert cfg.embedding.provider == "huggingface"
    assert cfg.graph.uri == "bolt://foo:7687"


def test_env_aliases_cover_cli_http_mcp_surfaces(isolated_env, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    isolated_env.setenv("MNEMOSTACK_PROVIDER", "ollama")
    isolated_env.setenv("MNEMOSTACK_QDRANT_URL", "http://qdrant:6333")
    isolated_env.setenv("MNEMOSTACK_COLLECTION", "memory")
    isolated_env.setenv("MNEMOSTACK_LLM", "ollama")
    isolated_env.setenv("MNEMOSTACK_MEMGRAPH_URI", "bolt://memgraph:7687")
    isolated_env.setenv("MNEMOSTACK_GRAPH_TIMEOUT", "2.5")
    isolated_env.setenv("MNEMOSTACK_GRAPH_HEALTH_TIMEOUT", "0.5")
    isolated_env.setenv("MNEMOSTACK_BM25_PATHS", f"/a{os.pathsep}/b")

    cfg = Config.load()

    assert cfg.embedding.provider == "ollama"
    assert cfg.vector.host == "http://qdrant:6333"
    assert cfg.vector.collection == "memory"
    assert cfg.llm.provider == "ollama"
    assert cfg.graph.uri == "bolt://memgraph:7687"
    assert cfg.graph.timeout == 2.5
    assert cfg.graph.health_timeout == 0.5
    assert cfg.recall.bm25_paths == ["/a", "/b"]


def test_cli_parser_defaults_from_config_env(isolated_env, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    isolated_env.setenv("MNEMOSTACK_PROVIDER", "ollama")
    isolated_env.setenv("MNEMOSTACK_QDRANT_URL", "http://qdrant:6333")
    isolated_env.setenv("MNEMOSTACK_COLLECTION", "memory")
    isolated_env.setenv("MNEMOSTACK_MEMGRAPH_URI", "bolt://memgraph:7687")
    isolated_env.setenv("MNEMOSTACK_BM25_PATHS", "/notes")

    from mnemostack.cli import build_parser

    args = build_parser().parse_args(["search", "hello"])

    assert args.provider == "ollama"
    assert args.qdrant == "http://qdrant:6333"
    assert args.collection == "memory"
    assert args.memgraph_uri == "bolt://memgraph:7687"
    assert args.bm25_path == ["/notes"]


def test_server_config_from_env_uses_config_aliases(isolated_env, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    isolated_env.setenv("MNEMOSTACK_PROVIDER", "ollama")
    isolated_env.setenv("MNEMOSTACK_LLM", "ollama")
    isolated_env.setenv("MNEMOSTACK_QDRANT_URL", "http://qdrant:6333")
    isolated_env.setenv("MNEMOSTACK_COLLECTION", "memory")
    isolated_env.setenv("MNEMOSTACK_GRAPH_URI", "bolt://memgraph:7687")
    isolated_env.setenv("MNEMOSTACK_GRAPH_TIMEOUT", "2.5")
    isolated_env.setenv("MNEMOSTACK_GRAPH_HEALTH_TIMEOUT", "0.5")
    isolated_env.setenv("MNEMOSTACK_BM25_PATHS", "/notes")

    from mnemostack.server import ServerConfig

    cfg = ServerConfig.from_env()

    assert cfg.provider_name == "ollama"
    assert cfg.llm_name == "ollama"
    assert cfg.qdrant_url == "http://qdrant:6333"
    assert cfg.collection == "memory"
    assert cfg.graph_uri == "bolt://memgraph:7687"
    assert cfg.graph_timeout == 2.5
    assert cfg.graph_health_timeout == 0.5
    assert cfg.bm25_paths == ["/notes"]


def test_save_roundtrip(isolated_env, tmp_path):
    cfg = Config()
    cfg.embedding.provider = "ollama"
    cfg.vector.collection = "roundtrip-test"
    cfg_path = tmp_path / "out.yaml"
    cfg.save(cfg_path)
    loaded = Config.load(cfg_path)
    assert loaded.embedding.provider == "ollama"
    assert loaded.vector.collection == "roundtrip-test"


def test_generate_example_config_is_valid_yaml():
    import yaml
    data = yaml.safe_load(generate_example_config())
    assert "embedding" in data
    assert "vector" in data
    assert "recall" in data


def test_mnemostack_config_env_points_to_file(isolated_env, tmp_path):
    cfg_path = tmp_path / "custom.yaml"
    cfg_path.write_text("llm:\n  provider: ollama\n")
    isolated_env.setenv("MNEMOSTACK_CONFIG", str(cfg_path))
    cfg = Config.load()
    assert cfg.llm.provider == "ollama"


def test_unknown_keys_ignored(isolated_env, tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
embedding:
  provider: ollama
  nonexistent_key: whatever

nonexistent_section:
  foo: bar
"""
    )
    cfg = Config.load(cfg_path)
    assert cfg.embedding.provider == "ollama"  # known key applied
    assert not hasattr(cfg.embedding, "nonexistent_key")
