"""Text offsets refer to original line starts without changing chunk contents."""
import pytest

from mnemostack.chunking import MessagePairChunker


@pytest.mark.parametrize("prefix,gap", [("", "\n\n"), ("\n\n", "\n \t\n"), ("\r\n", "\r\n\r\n"), ("\n", "\n\n\n")])
@pytest.mark.parametrize("solo,window", [(True, 1), (True, 2), (False, 2), (True, 3)])
def test_offsets_use_original_text_while_contents_and_order_stay_unchanged(prefix, gap, solo, window):
    text = prefix + "Первый" + gap + "Second" + gap + "Третий"
    chunker = MessagePairChunker(include_solo=solo, window=window)
    messages = [line for line in text.split("\n") if line.strip()]
    expected = chunker.chunk_messages(messages)
    actual = chunker.chunk(text)
    assert [(c.text, c.metadata) for c in actual] == [(c.text, c.metadata) for c in expected]
    for chunk in actual:
        first_message = chunk.text.split(chunker.separator)[0]
        assert text.startswith(first_message, chunk.offset)


def test_repeated_message_values_keep_distinct_original_positions():
    chunks = MessagePairChunker().chunk("a\n\na\na")
    assert [c.offset for c in chunks] == [0, 0, 3, 3, 5]


def test_custom_separator_and_middle_metadata_contract_are_unchanged():
    chunker = MessagePairChunker(separator=" | ")
    chunks = chunker.chunk("\nA\n\nB\nC")
    assert [(c.text, c.offset) for c in chunks] == [("A", 1), ("A | B", 1), ("B", 4), ("B | C", 4), ("C", 6)]
    structured = chunker.chunk_messages(["A", "B"], [{"speaker": "one"}, {"speaker": "two"}])
    assert structured[1].metadata == {"speaker": "two", "chunk_window": 2}
    assert [c.offset for c in structured] == [0, 0, 2]


@pytest.mark.parametrize("text", ["", "\n", " \t\n\r\n"])
def test_empty_or_blank_text_produces_no_chunks(text):
    assert MessagePairChunker().chunk(text) == []
