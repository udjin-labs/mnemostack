"""Tests for MessagePairChunker — dialogue/transcript sliding window."""
import pytest

from mnemostack.chunking import MessagePairChunker


def test_pair_chunker_basic():
    msgs = ["A says hi", "B says hello", "A says bye"]
    chunks = MessagePairChunker().chunk_messages(msgs)
    # 3 solo + 2 pairs = 5
    assert len(chunks) == 5
    texts = [c.text for c in chunks]
    assert "A says hi" in texts
    assert "A says hi\nB says hello" in texts
    assert "B says hello\nA says bye" in texts


def test_pair_chunker_no_solo():
    msgs = ["A", "B", "C"]
    chunks = MessagePairChunker(include_solo=False).chunk_messages(msgs)
    # only 2 pairs
    assert len(chunks) == 2
    assert [c.text for c in chunks] == ["A\nB", "B\nC"]


def test_pair_chunker_window_3():
    msgs = ["A", "B", "C", "D"]
    chunks = MessagePairChunker(include_solo=False, window=3).chunk_messages(msgs)
    # windows starting at 0,1: A-B-C, B-C-D
    assert len(chunks) == 2
    assert chunks[0].text == "A\nB\nC"
    assert chunks[1].text == "B\nC\nD"


def test_pair_chunker_metadata_passthrough():
    msgs = ["m1", "m2", "m3"]
    meta = [{"speaker": "A"}, {"speaker": "B"}, {"speaker": "A"}]
    chunks = MessagePairChunker().chunk_messages(msgs, metadata=meta)
    # each chunk should carry speaker of its starting message
    for c in chunks:
        assert "speaker" in c.metadata
    pair_chunks = [c for c in chunks if "\n" in c.text]
    for c in pair_chunks:
        assert c.metadata["chunk_window"] == 2


def test_pair_chunker_single_message():
    msgs = ["lonely"]
    chunks = MessagePairChunker().chunk_messages(msgs)
    # only solo, no pair possible
    assert len(chunks) == 1
    assert chunks[0].text == "lonely"


def test_pair_chunker_empty():
    chunks = MessagePairChunker().chunk_messages([])
    assert chunks == []


def test_pair_chunker_from_text():
    text = "line1\nline2\n\nline3"
    chunks = MessagePairChunker().chunk(text)
    # 3 messages -> 3 solo + 2 pairs
    assert len(chunks) == 5


def test_pair_chunker_invalid_window():
    with pytest.raises(ValueError):
        MessagePairChunker(window=1)
