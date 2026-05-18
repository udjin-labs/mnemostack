"""
Tests for MCP tool input Pydantic validation.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pydantic import ValidationError
from mnemostack.mcp_models import (
    SearchInput,
    RecallInput,
    GraphQueryInput,
    StoreMemoryInput,
    GetContextInput,
    validate_tool_input,
)


class TestSearchValidation:
    def test_valid_defaults(self):
        """Valid input with defaults."""
        m = validate_tool_input(SearchInput, {"query": "hello"})
        assert m.query == "hello"
        assert m.limit == 10

    def test_missing_query(self):
        """Missing required field raises."""
        with pytest.raises(ValidationError):
            validate_tool_input(SearchInput, {})

    def test_blank_query(self):
        """Blank query is rejected."""
        with pytest.raises(ValidationError):
            validate_tool_input(SearchInput, {"query": ""})

    def test_whitespace_query(self):
        """Whitespace-only query is rejected."""
        with pytest.raises(ValidationError):
            validate_tool_input(SearchInput, {"query": "   "})

    def test_negative_limit(self):
        """Negative limit is rejected."""
        with pytest.raises(ValidationError):
            validate_tool_input(SearchInput, {"query": "test", "limit": -1})

    def test_limit_too_high(self):
        """Limit > 50 is rejected."""
        with pytest.raises(ValidationError):
            validate_tool_input(SearchInput, {"query": "test", "limit": 100})

    def test_extra_fields_ignored(self):
        """Extra fields are silently ignored."""
        m = validate_tool_input(SearchInput, {"query": "test", "extra": "x"})
        assert not hasattr(m, "extra")


class TestRecallValidation:
    def test_valid_with_rerank_false(self):
        m = validate_tool_input(RecallInput, {"query": "test", "rerank": False})
        assert m.rerank is False

    def test_defaults(self):
        m = validate_tool_input(RecallInput, {"query": "test"})
        assert m.limit == 10
        assert m.rerank is True


class TestGraphQueryValidation:
    def test_depth_too_high(self):
        with pytest.raises(ValidationError):
            validate_tool_input(GraphQueryInput, {"query": "test", "depth": 10})

    def test_depth_zero(self):
        with pytest.raises(ValidationError):
            validate_tool_input(GraphQueryInput, {"query": "test", "depth": 0})

    def test_valid_depth(self):
        m = validate_tool_input(GraphQueryInput, {"query": "test", "depth": 3})
        assert m.depth == 3


class TestStoreMemoryValidation:
    def test_empty_content(self):
        with pytest.raises(ValidationError):
            validate_tool_input(StoreMemoryInput, {"content": ""})

    def test_blank_content(self):
        with pytest.raises(ValidationError):
            validate_tool_input(StoreMemoryInput, {"content": "  "})

    def test_valid_with_metadata(self):
        m = validate_tool_input(StoreMemoryInput, {
            "content": "important fact",
            "metadata": {"source": "test"},
        })
        assert m.metadata == {"source": "test"}


class TestGetContextValidation:
    def test_limit_too_high(self):
        with pytest.raises(ValidationError):
            validate_tool_input(GetContextInput, {"query": "test", "limit": 100})

    def test_valid_defaults(self):
        m = validate_tool_input(GetContextInput, {"query": "test"})
        assert m.limit == 5
