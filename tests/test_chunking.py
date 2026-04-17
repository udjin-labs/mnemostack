"""Tests for chunking strategies."""
import pytest

from mnemostack.chunking import (
    CharChunker,
    Chunk,
    MarkdownChunker,
    ParagraphChunker,
)


# ---------- CharChunker ----------

def test_char_chunker_basic():
    chunker = CharChunker(chunk_size=10, overlap=2)
    chunks = chunker.chunk("abcdefghijklmnop")
    assert len(chunks) >= 2
    assert all(isinstance(c, Chunk) for c in chunks)
    assert chunks[0].text == "abcdefghij"
    assert chunks[0].offset == 0


def test_char_chunker_empty_text():
    chunker = CharChunker(chunk_size=100, overlap=10)
    assert chunker.chunk("") == []


def test_char_chunker_invalid_overlap():
    with pytest.raises(ValueError, match="overlap"):
        CharChunker(chunk_size=10, overlap=15)


def test_char_chunker_preserves_offsets():
    chunker = CharChunker(chunk_size=5, overlap=0)
    chunks = chunker.chunk("abcdeFGHIJ")
    assert chunks[0].offset == 0
    assert chunks[1].offset == 5


# ---------- ParagraphChunker ----------

def test_paragraph_chunker_basic():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunker = ParagraphChunker(chunk_size=1000)
    chunks = chunker.chunk(text)
    # Small text, all paragraphs merge into one chunk
    assert len(chunks) == 1
    assert "First" in chunks[0].text
    assert "Third" in chunks[0].text


def test_paragraph_chunker_splits_by_size():
    text = "\n\n".join(["A" * 100] * 10)  # 10 paragraphs of 100 chars
    chunker = ParagraphChunker(chunk_size=250, min_chunk=100)
    chunks = chunker.chunk(text)
    assert len(chunks) >= 4  # roughly 4-5 chunks
    assert all(len(c) <= 300 for c in chunks)  # within target + slack


def test_paragraph_chunker_single_huge_paragraph():
    text = "A" * 2000
    chunker = ParagraphChunker(chunk_size=500)
    chunks = chunker.chunk(text)
    # Huge single para is emitted as-is (no splitting mid-paragraph)
    assert len(chunks) == 1
    assert len(chunks[0].text) == 2000


def test_paragraph_chunker_empty():
    assert ParagraphChunker().chunk("") == []
    assert ParagraphChunker().chunk("   \n\n  ") == []


# ---------- MarkdownChunker ----------

def test_markdown_chunker_splits_on_headers():
    text = """# Title

Intro paragraph.

## Section A

Content A.

## Section B

Content B.
"""
    chunker = MarkdownChunker(chunk_size=10000)
    chunks = chunker.chunk(text)
    # Three sections: Title (h1 + intro), Section A, Section B
    assert len(chunks) == 3
    assert chunks[0].metadata["heading_path"] == ["Title"]
    assert chunks[1].metadata["heading_path"] == ["Title", "Section A"]
    assert chunks[2].metadata["heading_path"] == ["Title", "Section B"]


def test_markdown_chunker_includes_heading_path_in_text():
    text = "# Top\n\n## Sub\n\nContent here."
    chunker = MarkdownChunker(include_heading_in_text=True)
    chunks = chunker.chunk(text)
    # Sub section should have parent path prepended
    sub = [c for c in chunks if c.metadata["heading_path"] == ["Top", "Sub"]][0]
    assert "[Top]" in sub.text


def test_markdown_chunker_no_headers_returns_whole_text():
    text = "Just plain text without any headers."
    chunker = MarkdownChunker()
    chunks = chunker.chunk(text)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].metadata["heading_path"] == []


def test_markdown_chunker_ignores_headers_in_code_blocks():
    text = """# Real Header

Some text.

```python
# This is a comment, not a header
def foo(): pass
```

## Another Real Header
"""
    chunker = MarkdownChunker()
    chunks = chunker.chunk(text)
    headings = [c.metadata["heading_path"] for c in chunks]
    # Should NOT treat '# This is a comment' as a header
    assert ["Real Header", "This is a comment, not a header"] not in headings
    # Should still find Another Real Header
    titles = [h[-1] for h in headings if h]
    assert "Real Header" in titles
    assert "Another Real Header" in titles


def test_markdown_chunker_splits_large_sections():
    body = "A" * 2500
    text = f"# Big Section\n\n{body}"
    chunker = MarkdownChunker(chunk_size=1000)
    chunks = chunker.chunk(text)
    # Single section, but body > chunk_size → split
    assert len(chunks) > 1
    # All chunks preserve heading path
    assert all(c.metadata["heading_path"] == ["Big Section"] for c in chunks)


def test_markdown_chunker_heading_hierarchy_tracked():
    text = """# Level 1

## Level 2a

### Level 3a

Content 3a.

### Level 3b

Content 3b.

## Level 2b

Content 2b.
"""
    chunker = MarkdownChunker(chunk_size=10000)
    chunks = chunker.chunk(text)
    paths = [c.metadata["heading_path"] for c in chunks]
    assert ["Level 1", "Level 2a", "Level 3a"] in paths
    assert ["Level 1", "Level 2a", "Level 3b"] in paths
    # Level 2b should NOT be nested under Level 2a
    assert ["Level 1", "Level 2b"] in paths
    assert ["Level 1", "Level 2a", "Level 2b"] not in paths
