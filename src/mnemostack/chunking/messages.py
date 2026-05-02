"""MessagePairChunker — sliding window for chat/dialogue transcripts.

Dialogue Q/A pairs are frequently split across consecutive messages:
    [2022-01-23] Joanna: How long have you had them?
    [2022-01-23] Nate: I've had them for 3 years now!

A per-message chunker lets vector/BM25 find Joanna's question but NOT Nate's
answer (different speaker, different wording). This chunker pairs each message
with its successor so the QA pair lives in a single chunk, boosting recall
for conversational data.

Input is either:
- a list of pre-formatted message strings, via `chunk_messages(...)`, or
- a single text with a line-based delimiter, via `chunk(...)` (one message per
  non-empty line).

The chunker emits both solo chunks (each message on its own) AND paired chunks
(msg[i] + msg[i+1]) so retrieval works for direct-match and context-bridging
queries.
"""
from __future__ import annotations

from .base import Chunk, Chunker


class MessagePairChunker(Chunker):
    """Sliding window chunker for dialogue transcripts.

    Args:
        include_solo: if True (default), also emit each message as its own chunk
                       so direct lookups still work.
        window: how many messages to include in a pair (2 = current + next).
        separator: joiner between messages inside a paired chunk.
    """

    def __init__(
        self,
        include_solo: bool = True,
        window: int | None = 2,
        separator: str = "\n",
        window_size: int | None = None,
    ):
        if window_size is not None:
            window = window_size
        if window is None:
            window = 2
        if window < 1:
            raise ValueError("window_size must be >= 1")
        if window == 1 and not include_solo:
            raise ValueError("window_size=1 requires include_solo=True")
        self.include_solo = include_solo
        self.window = window
        self.separator = separator

    def chunk_messages(
        self,
        messages: list[str],
        metadata: list[dict] | None = None,
    ) -> list[Chunk]:
        """Chunk a pre-split list of messages (preferred for structured input).

        Args:
            messages: ordered list of message strings
            metadata: optional per-message metadata; paired chunks inherit
                      the metadata of the middle message in the window.
        """
        chunks: list[Chunk] = []
        md = metadata or [{} for _ in messages]
        offset = 0
        for i, msg in enumerate(messages):
            if self.include_solo:
                chunks.append(Chunk(text=msg, offset=offset, metadata=dict(md[i])))
            if self.window > 1 and i + self.window - 1 < len(messages):
                window = messages[i : i + self.window]
                middle = i + self.window // 2
                combined = self.separator.join(window)
                meta = dict(md[middle])
                meta["chunk_window"] = self.window
                chunks.append(Chunk(text=combined, offset=offset, metadata=meta))
            offset += len(msg) + 1
        return chunks

    def chunk(self, text: str) -> list[Chunk]:
        """Fallback: treat each non-empty line as one message."""
        messages = [line for line in text.split("\n") if line.strip()]
        return self.chunk_messages(messages)
