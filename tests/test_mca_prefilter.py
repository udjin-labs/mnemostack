from mnemostack.recall import BM25Doc, Recaller, RecallResult
from mnemostack.recall.bm25 import BM25
from mnemostack.recall.mca_prefilter import extract_exact_tokens, mca_prefilter


def test_extract_exact_tokens_finds_technical_shapes():
    query = (
        "Find snake_case_id in /var/log/app.log from 10.0.0.42, "
        "uuid 123e4567-e89b-12d3-a456-426614174000, branch feat-123-fix, "
        "and camelCaseIdentifier"
    )

    tokens = extract_exact_tokens(query)

    assert "snake_case_id" in tokens
    assert "/var/log/app.log" in tokens
    assert "10.0.0.42" in tokens
    assert "123e4567-e89b-12d3-a456-426614174000" in tokens
    assert "feat-123-fix" in tokens
    assert "camelCaseIdentifier" in tokens


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
