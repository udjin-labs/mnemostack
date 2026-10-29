"""Secret-policy refusals must happen before provider and store side effects."""

import base64

import pytest

from mnemostack.ingest import IngestItem, Ingestor, expand_remote_items, ingest_remote_items
from mnemostack.ingest_policy import IngestPolicyError, SecretPolicy


def _token():
    # Synthetic format only; never a credential associated with an account.
    return "gh" + "p_" + "test" + "Ab7cDe9f" * 4


class _Embedding:
    dimension = 3

    def __init__(self):
        self.calls = []

    def embed(self, text):
        self.calls.append(text)
        return [1.0, 0.0, 0.0]


class _Store:
    def __init__(self):
        self.points = []

    def upsert_batch(self, points):
        self.points.extend(points)
        return len(points)


def test_off_is_a_noop_and_mode_is_strict():
    SecretPolicy().check(_token())
    for value in ("warn", "", False, None):
        with pytest.raises(ValueError, match="off.*block"):
            SecretPolicy(value)


@pytest.mark.parametrize("placement", ["text", "source", "metadata", "tags"])
def test_sdk_refuses_before_embedding_enrichment_storage_or_logs(placement, caplog):
    emb, store = _Embedding(), _Store()
    enriched = []
    item = IngestItem(text="ordinary text", source="fixture")
    if placement == "metadata":
        item.metadata = {"nested": [{_token(): "value"}]}
    elif placement == "tags":
        item.tags = [_token()]
    else:
        setattr(item, placement, _token())
    ing = Ingestor(emb, store, enrich=lambda item: enriched.append(item), secret_policy="block")
    with pytest.raises(IngestPolicyError) as exc:
        ing.ingest([item])
    assert str(exc.value) == "ingest blocked by secret policy"
    assert not emb.calls and not store.points and not enriched
    assert _token() not in caplog.text


def test_stream_preserves_completed_safe_batches_and_stops_before_secret():
    emb, store = _Embedding(), _Store()
    ing = Ingestor(emb, store, secret_policy="block", batch_size=1, window_size=2)
    stream = ing.stream([IngestItem(text="safe", source="s"), IngestItem(text=_token(), source="s")])
    assert next(stream).upserted == 1
    with pytest.raises(IngestPolicyError):
        next(stream)
    assert emb.calls == ["safe"] and len(store.points) == 1


@pytest.mark.asyncio
async def test_async_refusal_has_no_side_effects():
    emb, store = _Embedding(), _Store()
    with pytest.raises(IngestPolicyError):
        await Ingestor(emb, store, secret_policy="block").ingest_one_async(IngestItem(text=_token()))
    assert not emb.calls and not store.points


def test_remote_batch_preflight_covers_all_items_before_first_write():
    emb, store = _Embedding(), _Store()
    items = [IngestItem(text="safe"), IngestItem(text=_token())]
    with pytest.raises(IngestPolicyError):
        ingest_remote_items(emb, store, items, secret_policy="block")
    assert not emb.calls and not store.points


def test_sdk_chunk_helper_checks_whole_input_before_splitting():
    with pytest.raises(IngestPolicyError):
        expand_remote_items([(IngestItem(text=_token()), True)], chunk_size=10,
                            secret_policy="block")


def test_enrichment_output_is_checked_before_storage():
    emb, store = _Embedding(), _Store()
    ing = Ingestor(emb, store, enrich=lambda _: {"nested": _token()}, secret_policy="block")
    with pytest.raises(IngestPolicyError):
        ing.ingest_one(IngestItem(text="safe"))
    assert emb.calls == ["safe"] and not store.points


def test_private_key_supports_legacy_encryption_headers_and_nested_parsed_json():
    label = "RSA " + "PRIVATE KEY"
    body = base64.b64encode(bytes(range(128))).decode()
    value = (
        f"-----BEGIN {label}-----\r\nProc-Type: 4,ENCRYPTED\r\n"
        f"DEK-Info: AES-256-CBC,ABCDEF0123456789\r\n\r\n{body}\r\n-----END {label}-----"
    )
    with pytest.raises(IngestPolicyError):
        SecretPolicy("block").check({"metadata": [value]})
    SecretPolicy("block").check(value.replace("PRIVATE KEY", "PUBLIC KEY"))
    SecretPolicy("block").check(value.split("-----END")[0])


def test_policy_handles_cycles_without_disabling_nested_checks():
    value = []
    value.append(value)
    SecretPolicy("block").check(value)
    value.append(_token())
    with pytest.raises(IngestPolicyError):
        SecretPolicy("block").check(value)


def test_formats_examples_are_not_exempt_and_ordinary_text_is_allowed():
    policy = SecretPolicy("block")
    policy.check("Password rotation policy. API_KEY=${TOKEN}. Example public certificate.")
    with pytest.raises(IngestPolicyError):
        policy.check("Documentation example, do not use: " + _token())


def test_default_ingestor_preserves_existing_behavior():
    emb, store = _Embedding(), _Store()
    assert Ingestor(emb, store).ingest_one(IngestItem(text=_token())).upserted == 1


@pytest.mark.parametrize("suffix", [".", ",", "!", "?", ";", ")"])
@pytest.mark.parametrize("routable", [False, True])
def test_gitlab_token_at_sentence_end_is_blocked(suffix, routable):
    token = "gl" + "pat-" + "Ab7cDe9f" * (5 if routable else 2)
    token += ".ab1234567" if routable else "Wxyz"
    emb, store = _Embedding(), _Store()
    with pytest.raises(IngestPolicyError):
        Ingestor(emb, store, secret_policy="block").ingest_one(
            IngestItem(text="Credential: " + token + suffix)
        )
    assert not emb.calls and not store.points


def test_config_env_and_cli_share_policy(monkeypatch, tmp_path):
    from mnemostack import cli, config
    from mnemostack.server import ServerConfig

    monkeypatch.setattr(config, "DEFAULT_CONFIG_PATHS", [])
    monkeypatch.setenv("MNEMOSTACK_SECRET_POLICY", "block")
    assert config.Config.load().ingest.secret_policy == "block"
    assert ServerConfig.from_env().secret_policy == "block"
    assert cli.build_parser().parse_args(["index", str(tmp_path)]).secret_policy == "block"
    assert cli.build_parser().parse_args(["serve", "--secret-policy", "off"]).secret_policy == "off"
    monkeypatch.setenv("MNEMOSTACK_SECRET_POLICY", "typo")
    with pytest.raises(ValueError):
        config.Config.load()


def test_extractor_refuses_before_llm():
    from mnemostack.graph.extractor import TripleExtractor

    class LLM:
        def generate(self, *_a, **_kw):
            pytest.fail("blocked text reached the LLM")

    with pytest.raises(IngestPolicyError):
        TripleExtractor(LLM(), secret_policy="block").extract(_token())
