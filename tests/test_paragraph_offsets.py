"""Regression tests for paragraph buffer offsets and unchanged chunk content."""

import pytest

from mnemostack.chunking import ParagraphChunker
from mnemostack.ingest import stable_chunk_id


@pytest.mark.parametrize(
    "parts,groups",
    [
        (["A" * 900, "B" * 900, "Tail."], [(0, 1), (1, 2), (2, 3)]),
        (["A" * 900, "Tail."], [(0, 1), (1, 2)]),
        (["A" * 900, "B" * 900], [(0, 1), (1, 2)]),
        (["Start.", "A" * 900, "Tail."], [(0, 1), (1, 2), (2, 3)]),
        (["Start.", "A" * 900, "B" * 900, "Tail."], [(0, 1), (1, 2), (2, 3), (3, 4)]),
        (["One.", "Two.", "A" * 900, "Tail.", "End."], [(0, 2), (2, 3), (3, 5)]),
        (["One.", "Two.", "A" * 900], [(0, 2), (2, 3)]),
        (["A" * 900, "A" * 900, "Tail.", "A" * 900, "Tail."], [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]),
        (["尾" * 900, "🙂" * 900, "e\u0301."], [(0, 1), (1, 2), (2, 3)]),
    ],
)
def test_buffer_offsets_follow_actual_paragraph_groups(parts, groups):
    source = "\n\n".join(parts)
    chunks = ParagraphChunker(chunk_size=800, min_chunk=200).chunk(source)
    starts = [sum(len(p) + 2 for p in parts[:i]) for i in range(len(parts))]
    assert [(c.text, c.offset) for c in chunks] == [
        ("\n\n".join(parts[a:b]), starts[a]) for a, b in groups
    ]
    assert all(source[c.offset : c.offset + len(c.text)] == c.text for c in chunks)
    assert all(c.metadata == {} for c in chunks)


@pytest.mark.parametrize("size", [8, 800])
@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_target_boundary_then_short_tail(size, delta):
    source = "A" * (size + delta) + "\n\nt"
    chunks = ParagraphChunker(chunk_size=size, min_chunk=2).chunk(source)
    assert [(c.text, c.offset) for c in chunks] == [
        ("A" * (size + delta), 0),
        ("t", size + delta + 2),
    ]


@pytest.mark.parametrize("separator", ["\n\n", "\r\n\r\n", "\n\n\n", "\n \t\n"])
def test_long_paragraph_separator_variants(separator):
    source = separator.join(["A" * 900, "B" * 900, "Tail."])
    chunks = ParagraphChunker().chunk(source)
    assert [c.offset for c in chunks] == [0, 900 + len(separator), 1800 + 2 * len(separator)]
    assert all(source[c.offset : c.offset + len(c.text)] == c.text for c in chunks)


@pytest.mark.parametrize("source", ["", " ", "\n\n", "\r\n\r\n", "\t  \n\n\n \t"])
def test_empty_inputs_remain_empty(source):
    assert ParagraphChunker().chunk(source) == []


@pytest.mark.parametrize(
    "source,expected,exact",
    [
        ("Alpha.\n\nBeta.", "Alpha.\n\nBeta.", True),
        ("  Alpha.\n\nBeta.", "Alpha.\n\nBeta.", False),
        ("Alpha.\n\n\nBeta.", "Alpha.\n\nBeta.", False),
        ("Alpha.\r\n\r\nBeta.", "Alpha.\n\nBeta.", False),
        ("Alpha.\n \t\nBeta.", "Alpha.\n\nBeta.", False),
        ("Alpha.  \n\n Beta. ", "Alpha.\n\nBeta.", False),
        ("Alpha.\r\nnext.\n\nBeta.", "Alpha.\r\nnext.\n\nBeta.", True),
    ],
)
def test_existing_normalization_is_not_redefined_as_exact_alignment(source, expected, exact):
    chunks = ParagraphChunker().chunk(source)
    assert len(chunks) == 1
    assert chunks[0].text == expected
    assert chunks[0].offset == 0
    assert (source[chunks[0].offset : chunks[0].offset + len(expected)] == expected) is exact


@pytest.mark.parametrize("min_chunk", [0, 1, 200, 1000])
def test_long_then_short_does_not_inherit_long_offset(min_chunk):
    chunks = ParagraphChunker(chunk_size=800, min_chunk=min_chunk).chunk("A" * 900 + "\n\nTail.")
    assert [(c.text, c.offset) for c in chunks] == [("A" * 900, 0), ("Tail.", 902)]


def test_corrected_offset_changes_only_the_affected_stable_id():
    source = "A" * 900 + "\n\n" + "B" * 900 + "\n\nTail."
    chunks = ParagraphChunker().chunk(source)
    ids = [stable_chunk_id("public.txt", c.offset, c.text, tenant="test") for c in chunks]
    assert ids[:2] == [
        stable_chunk_id("public.txt", 0, "A" * 900, tenant="test"),
        stable_chunk_id("public.txt", 902, "B" * 900, tenant="test"),
    ]
    assert ids[2] == stable_chunk_id("public.txt", 1804, "Tail.", tenant="test")
    assert ids[2] != stable_chunk_id("public.txt", 0, "Tail.", tenant="test")
    assert ids == [
        stable_chunk_id("public.txt", c.offset, c.text, tenant="test")
        for c in ParagraphChunker().chunk(source)
    ]
