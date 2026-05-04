from mnemostack.recall import BM25Doc, Recaller, RecallResult
from mnemostack.recall.bm25 import BM25
from mnemostack.recall.mca_prefilter import extract_exact_tokens, mca_prefilter


def test_extract_exact_tokens_finds_technical_shapes():
    query = (
        "Find snake_case_id in /var/log/app.log and src/mnemostack/recall/recaller.py "
        "from 10.0.0.42, uuid 123e4567-e89b-12d3-a456-426614174000, "
        "branch feat-123-fix or feat/retrieval-improvements from main/develop, module "
        "mnemostack.recall.pipeline, and camelCaseIdentifier"
    )

    tokens = extract_exact_tokens(query)

    assert "snake_case_id" in tokens
    assert "/var/log/app.log" in tokens
    assert "src/mnemostack/recall/recaller.py" in tokens
    assert "10.0.0.42" in tokens
    assert "123e4567-e89b-12d3-a456-426614174000" in tokens
    assert "feat-123-fix" in tokens
    assert "feat/retrieval-improvements" in tokens
    assert "main" in tokens
    assert "develop" in tokens
    assert "mnemostack.recall.pipeline" in tokens
    assert "camelCaseIdentifier" in tokens


def test_extract_exact_tokens_skips_common_path_and_dot_false_positives():
    query = "choose и/или maybe e.g. or i.e. but not technical tokens"

    tokens = extract_exact_tokens(query)

    assert "и/или" not in tokens
    assert "e.g." not in tokens
    assert "i.e." not in tokens


def test_extract_exact_tokens_does_not_duplicate_absolute_paths_as_relative_paths():
    tokens = extract_exact_tokens("look at /etc/hosts")

    assert "/etc/hosts" in tokens
    assert "etc/hosts" not in tokens


def test_extract_exact_tokens_does_not_treat_common_branch_words_as_exact_tokens():
    assert extract_exact_tokens("what is the main issue") == []


def test_mca_prefilter_returns_bm25_matches_as_recall_results():
    bm25 = BM25([
        BM25Doc("rare", "Deployment mentions snake_case_id and /var/log/app.log"),
        BM25Doc("other", "ordinary semantic text"),
    ])

    results = mca_prefilter("where is snake_case_id?", bm25, limit=10)

    assert [result.id for result in results] == ["rare"]
    assert isinstance(results[0], RecallResult)
    assert results[0].sources == ["mca"]


class FixedRetriever:
    name = "semantic"

    def search(self, query, limit=20, filters=None):
        return [RecallResult("semantic", "semantic result", 1.0, sources=["semantic"])]


def test_mca_results_join_rrf_pool_in_retriever_mode():
    bm25_docs = [
        BM25Doc("mca", "Relevant branch feat-123-fix is here"),
        BM25Doc("noise", "unrelated"),
    ]
    bm25_retriever = type("BM25Like", (), {
        "name": "bm25",
        "bm25": BM25(bm25_docs),
        "search": lambda self, query, limit=20, filters=None: [],
    })()
    recaller = Recaller(
        retrievers=[FixedRetriever(), bm25_retriever],
        mca_prefilter=True,
        fallback_threshold=0.0,
    )

    results = recaller.recall("what about feat-123-fix", limit=5)

    assert "mca" in [result.id for result in results]
    assert results[0].id == "mca"
    assert results[0].sources == ["mca"]
