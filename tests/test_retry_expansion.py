from mnemostack.embeddings.base import EmbeddingProvider
from mnemostack.llm.base import LLMProvider, LLMResponse
from mnemostack.recall import AnswerGenerator, RecallResult
from mnemostack.recall.recaller import Recaller
from mnemostack.vector.qdrant import Hit


class SequenceLLM(LLMProvider):
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    @property
    def name(self):
        return "sequence"

    def generate(self, prompt, max_tokens=200, temperature=0.0):
        self.prompts.append(prompt)
        text = self.responses.pop(0)
        return LLMResponse(text=text, tokens_used=1)


class FakeEmbedding(EmbeddingProvider):
    def __init__(self):
        self.embed_calls = []
        self.embed_batch_calls = []

    @property
    def dimension(self):
        return 1

    @property
    def name(self):
        return "fake"

    def embed(self, text):
        self.embed_calls.append(text)
        return [float(len(text))]

    def embed_batch(self, texts):
        self.embed_batch_calls.append(list(texts))
        return [[float(len(text))] for text in texts]


class FakeVectorStore:
    def __init__(self):
        self.search_calls = []
        self.search_many_calls = []

    def search(self, query_vector, limit=10, filters=None):
        self.search_calls.append((query_vector, limit, filters))
        return [Hit(id=str(query_vector[0]), score=1.0, payload={"text": f"hit {query_vector[0]}"})]

    def search_many(self, query_vectors, limit=10, filters=None):
        self.search_many_calls.append((query_vectors, limit, filters))
        return [self.search(vec, limit=limit, filters=filters) for vec in query_vectors]


def memory(id_=1):
    return RecallResult(id=id_, text="existing memory", score=0.9, payload={"source": "s"}, sources=["vector"])


def test_weak_answer_triggers_retry_with_expansion():
    answer_llm = SequenceLLM([
        "Not in memory.\nCONFIDENCE: 0.0",
        "Expanded answer\nCONFIDENCE: 0.8",
    ])
    expansion_llm = SequenceLLM(["variant one\nvariant two"])
    embedding = FakeEmbedding()
    vector = FakeVectorStore()
    recaller = Recaller(embedding_provider=embedding, vector_store=vector, expansion_llm=expansion_llm)

    gen = AnswerGenerator(
        llm=answer_llm,
        recaller=recaller,
        retry_with_expansion=True,
        expansion_llm=expansion_llm,
    )

    answer = gen.generate("original question", [memory()])

    assert answer.text == "Expanded answer"
    assert len(answer_llm.prompts) == 2
    assert expansion_llm.prompts
    assert embedding.embed_batch_calls == [["original question", "variant one", "variant two"]]
    assert len(vector.search_many_calls) == 1


def test_strong_answer_skips_retry_with_expansion():
    answer_llm = SequenceLLM(["Strong answer\nCONFIDENCE: 0.9"])
    expansion_llm = SequenceLLM(["variant one\nvariant two"])
    embedding = FakeEmbedding()
    vector = FakeVectorStore()
    recaller = Recaller(embedding_provider=embedding, vector_store=vector, expansion_llm=expansion_llm)

    gen = AnswerGenerator(
        llm=answer_llm,
        recaller=recaller,
        retry_with_expansion=True,
        expansion_llm=expansion_llm,
    )

    answer = gen.generate("original question", [memory()])

    assert answer.text == "Strong answer"
    assert len(answer_llm.prompts) == 1
    assert expansion_llm.prompts == []
    assert embedding.embed_batch_calls == []
    assert vector.search_many_calls == []


def test_expanded_vector_retry_batch_matches_individual_vector_results():
    expansion_llm = SequenceLLM(["variant one\nvariant two"])
    embedding = FakeEmbedding()
    vector = FakeVectorStore()
    recaller = Recaller(embedding_provider=embedding, vector_store=vector, expansion_llm=expansion_llm)

    batch_results = recaller.recall_with_expanded_vectors("original", limit=10, vector_limit=5)

    queries = ["original", "variant one", "variant two"]
    individual_ids = set()
    individual_vectors = []
    for query in queries:
        vec = embedding.embed(query)
        individual_vectors.append(vec)
        individual_ids.update(hit.id for hit in vector.search(vec, limit=5))

    assert {result.id for result in batch_results} == individual_ids
    assert vector.search_many_calls[0][0] == individual_vectors
    assert embedding.embed_batch_calls == [queries]
    assert len(vector.search_many_calls) == 1
