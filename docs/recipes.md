# Recipes: reranking and BM25 language analyzers

Concrete, runnable recipes for two extension points mnemostack ships but
deliberately keeps dependency-free: the score-based reranker
(`ScoringReranker` / `RelevanceScorer`) and the pluggable BM25 analyzer
(`BM25Retriever(tokenizer=...)`). The core installs none of the models or
NLP libraries below — they live in *your* application, so you pick the
licensing and the language coverage you need.

Both stages are complementary: a BM25 analyzer improves **which** candidates
enter the pool; a reranker improves the **order** of candidates already
retrieved. Large or multilingual corpora usually want both.

---

## Reranking with a cross-encoder

`ScoringReranker` needs a backend with a single method — `score(query,
documents) -> Iterable[float]`, one score per document, higher = more
relevant. A cross-encoder fits this interface directly.

### Reference model: `bge-reranker-v2-m3`

For a self-hosted default, `BAAI/bge-reranker-v2-m3` is a strong pick:
**Apache-2.0** (usable in commercial self-hosting), ~0.6 B params (CPU-viable),
and a well-tested small multilingual reranker — reported MIRACL nDCG@10 ≈ 69
overall and ≈ 68 on Russian, near the top of RusBEIR among small models. A
same-size Apache-2.0 alternative worth benchmarking is `Qwen/Qwen3-Reranker-0.6B`,
which reports higher reranking scores on some public tables but is more
instruction/format-sensitive — validate it on your own data before switching.
Run either through `sentence-transformers`' `CrossEncoder`:

```python
# pip install sentence-transformers   (your app's dependency, not mnemostack's)
from sentence_transformers import CrossEncoder

from mnemostack.recall import ScoringReranker


class CrossEncoderScorer:
    """RelevanceScorer backed by a sentence-transformers CrossEncoder."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model = CrossEncoder(model_name)

    def score(self, query: str, documents: list[str]) -> list[float]:
        # CrossEncoder scores (query, doc) pairs; returns one float per pair.
        return self.model.predict([(query, doc) for doc in documents]).tolist()


reranker = ScoringReranker(CrossEncoderScorer(), max_items=100)
reranked = reranker.rerank(query, results)   # results: list[RecallResult]
```

`max_items` bounds the cross-encoder to the top-N candidates (it is far more
expensive than the retrievers); everything past `max_items` keeps its fused
order. Reranking is fail-open — if the model errors or returns the wrong number
of scores, the original order is preserved (see the `ScoringReranker`
docstring for the identity contract that keeps this thread-safe).

Wire it into the shared flow like any other reranker — but feed it a **wider
pool than you return**, or the reranker only reorders the handful you asked for.
`recall_flow` widens its internal pool (to `3 × limit`) only when a `pipeline=`
is passed; with no pipeline it recalls exactly `limit`. So either recall the
wide pool yourself and slice after reranking:

```python
pool = recaller.recall(query, limit=100)      # candidates for the reranker
reranked = reranker.rerank(query, pool)[:10]  # rerank the 100, keep the top 10
```

or let `recall_flow` widen the pool by giving it the ranking pipeline (which
also applies reranking internally, so you don't pass `reranker=` twice):

```python
from mnemostack.recall import recall_flow
from mnemostack.recall.pipeline import InMemoryStateStore, build_full_pipeline

# The stateful stages (Q-learning / inhibition-of-return / curiosity) read and
# write a state store; use InMemoryStateStore for a stateless process, or
# FileStateStore(path) to persist feedback across restarts.
pipeline = build_full_pipeline(state_store=InMemoryStateStore(), graph_uri=None)
results = recall_flow(recaller, query, limit=10, pipeline=pipeline, reranker=reranker)
# recall_flow fetches ~30 candidates, reranks, then cuts to limit=10
```

### Faster CPU / GPU inference

`sentence-transformers` can run the model through an OpenVINO (CPU) or ONNX
(GPU) backend instead of plain PyTorch. Each backend needs its own extras
(they pull in Optimum + the OpenVINO / ONNX Runtime):

```bash
pip install "sentence-transformers[openvino]"   # CPU
pip install "sentence-transformers[onnx]"        # ONNX Runtime (CPU)
pip install "sentence-transformers[onnx-gpu]"    # ONNX Runtime (GPU)
```

```python
CrossEncoder("BAAI/bge-reranker-v2-m3", backend="openvino")  # CPU
CrossEncoder("BAAI/bge-reranker-v2-m3", backend="onnx")      # GPU
```

Note that `backend=` alone loads (or exports) the *unoptimized* backend model.
The larger wins — OpenVINO INT8 quantization (~2–3× on CPU) and ONNX O4
optimization (GPU) — require a separate one-time export step, after which you
point the loader at the saved artifact via
`model_kwargs={"file_name": "openvino/openvino_model_qint8_quantized.xml"}`
(or the ONNX equivalent). See the sentence-transformers "Speeding up Inference"
guide for the exact export commands; benchmark the optimized file, not the
plain `backend=` load, or you'll measure the wrong model.

### Quality-first (larger, still Apache-2.0)

If latency budget allows a 4 B-class model, `Qwen/Qwen3-Reranker-4B` and
`mixedbread-ai/mxbai-rerank-large-v2` are battle-tested Apache-2.0 options that
score higher on multilingual benchmarks. Qwen3-Reranker is instruction- and
harness-sensitive — validate your prompt/formatting against your own data
before trusting a headline number.

### Hosted rerankers (near-free at small scale)

If you would rather not self-host weights, TEI-compatible endpoints and hosted
rerankers (Cohere, Voyage, …) fit the same `RelevanceScorer` interface — wrap
the HTTP call in `score()`. At small volumes these are close to free (e.g.
Voyage's free tier covers a large monthly token allowance). This sends your
query and candidate text to a third party; keep that in mind for private
corpora.

### What NOT to use

- **Jina's multilingual rerankers.** The multilingual `jina-reranker-v2`/`v3`
  weights are **CC-BY-NC** — non-commercial only, so don't ship them in a
  commercial self-hosted product. (Jina's older English-only v1 models such as
  `jina-reranker-v1-tiny-en` / `v1-turbo-en` are Apache-2.0, but being
  English-only they aren't a multilingual option anyway — use `bge-reranker-v2-m3`
  for that.)
- **Pre-2024 rerankers on non-English text.** Older models (e.g.
  `bge-reranker-large`) collapse on Russian and other non-English corpora
  (reported RU nDCG in the mid-40s vs. ~68 for `bge-reranker-v2-m3`). Never
  recommend them as a multilingual default.

> Benchmark figures above are as **reported** by the respective model cards and
> public leaderboards (MIRACL, RusBEIR). Treat them as directional and measure
> on your own corpus — mnemostack does not tune for any single benchmark.

---

## BM25 language analyzers

`BM25Retriever` applies one `tokenizer(text) -> list[str]` to both corpus and
query. The default is deliberately minimal — lowercase + Unicode word split —
which is ideal for exact identifiers, filenames and hashes but rigid for
morphologically rich languages, where the same word appears in many surface
forms. A custom analyzer normalizes those variants so relevant documents enter
the candidate pool in the first place.

> **Reality check.** Stemming's payoff scales with morphology: reported MAP
> gains are large for Finnish (~+0.25) and German (~+0.16) but small for
> Russian (~+0.06), and lemmatization is **not** justified for English. BM25's
> absolute ceiling on Russian is low (MIRACL ~0.33) regardless of analyzer —
> which is exactly why mnemostack fuses BM25 with dense vectors rather than
> leaning on lexical search alone. Analyzers help; they are not a silver bullet.

Core stays dependency-free — every library below is your application's choice.

### Default recipe: regex + Snowball stemming

A stemmer folds inflected forms to a common root. `PyStemmer` (BSD) wraps the
Snowball stemmers for ~20 languages:

```python
# pip install PyStemmer
import re

import Stemmer

from mnemostack.recall import BM25Retriever

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def snowball_analyzer(language: str = "english"):
    stemmer = Stemmer.Stemmer(language)

    def analyze(text: str) -> list[str]:
        tokens = _WORD_RE.findall(text.lower())
        return stemmer.stemWords(tokens)

    return analyze


retriever = BM25Retriever(docs, tokenizer=snowball_analyzer("english"))
```

The same analyzer is applied to queries automatically. When you build the BM25
corpus straight from Qdrant, thread it through `from_qdrant` so corpus and
query stay consistent (it also skips a redundant re-tokenization pass):

```python
retriever = BM25Retriever.from_qdrant(client, "memory",
                                      tokenizer=snowball_analyzer("german"))
```

### Russian-first recipe: lemmatization

For Russian, a lemmatizer (surface form → dictionary form) is often a better
fit than a stemmer. `simplemma` is a good default — MIT-licensed and
**zero heavy dependencies**, covering many languages from bundled word lists:

```python
# pip install simplemma
import re

import simplemma

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def lemma_analyzer(lang: str = "ru"):
    def analyze(text: str) -> list[str]:
        return [simplemma.lemmatize(tok, lang=lang) for tok in _WORD_RE.findall(text.lower())]

    return analyze


retriever = BM25Retriever(docs, tokenizer=lemma_analyzer("ru"))
```

If you need morphological analysis beyond dictionary lemmatization,
`pymorphy3` (MIT) is the heavier, more accurate Russian option — same wiring,
swap the `analyze` body for a `MorphAnalyzer().parse(tok)[0].normal_form` call.

### Mixed-language recipe: detect then route

When a single corpus mixes languages per document, detect the language and
route to the matching analyzer. `langdetect` is a common choice; cache the
per-language analyzers so you build each stemmer once:

```python
# pip install langdetect PyStemmer
import re

import Stemmer
from langdetect import DetectorFactory, LangDetectException, detect

# langdetect is nondeterministic on short/ambiguous text unless seeded — and an
# unseeded detector could stem the same text as different languages at index
# time vs. query time, silently breaking lexical matches. Seed once at import.
DetectorFactory.seed = 0

_WORD_RE = re.compile(r"\w+", re.UNICODE)
# langdetect codes -> Snowball language names you support
_LANGS = {"en": "english", "de": "german", "ru": "russian", "fi": "finnish"}
_STEMMERS = {name: Stemmer.Stemmer(name) for name in set(_LANGS.values())}


def routing_analyzer(default: str = "english"):
    def analyze(text: str) -> list[str]:
        try:
            lang = _LANGS.get(detect(text), default)
        except LangDetectException:
            lang = default
        tokens = _WORD_RE.findall(text.lower())
        return _STEMMERS[lang].stemWords(tokens)

    return analyze


retriever = BM25Retriever(docs, tokenizer=routing_analyzer())
```

Per-document detection is a pragmatic default; for short or code-heavy text
detection is unreliable, so fall back to a fixed `default` language rather than
mis-routing. If your corpus is single-language, skip detection entirely and use
the fixed-language recipe above — it is cheaper and never mis-detects.

### Pre-tokenized documents

If you already hold `BM25Doc` objects whose `tokens` came from the same
analyzer, pass `retokenize=False` to avoid a second pass:

```python
BM25Retriever(pretokenized_docs, tokenizer=my_analyzer, retokenize=False)
```
