"""
Pydantic validation models for MCP tool inputs.

Provides type-safe validation with clear error messages for all MCP tools.
Accepts both dict and typed input (extra fields ignored).
"""

from pydantic import BaseModel, Field, field_validator
from typing import Any


class SearchInput(BaseModel, extra="ignore"):
    """Input schema for the 'search' MCP tool."""
    query: str = Field(..., min_length=1, description="Natural language search query")
    limit: int = Field(default=10, ge=1, le=50, description="Max results (1-50)")

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v


class RecallInput(BaseModel, extra="ignore"):
    """Input schema for the 'recall' MCP tool."""
    query: str = Field(..., min_length=1, description="Natural language query")
    limit: int = Field(default=10, ge=1, le=50, description="Max results (1-50)")
    rerank: bool = Field(default=True, description="Apply Gemini reranking")

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v


class GraphQueryInput(BaseModel, extra="ignore"):
    """Input schema for the 'graph_query' MCP tool."""
    query: str = Field(..., min_length=1, description="Entity or concept to explore")
    depth: int = Field(default=2, ge=1, le=5, description="Relationship hops (1-5)")

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v


class StoreMemoryInput(BaseModel, extra="ignore"):
    """Input schema for the 'store_memory' MCP tool."""
    content: str = Field(..., min_length=1, description="Text content to store")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional metadata")

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be blank")
        return v


class GetContextInput(BaseModel, extra="ignore"):
    """Input schema for the 'get_context' MCP tool."""
    query: str = Field(..., min_length=1, description="Topic to build context around")
    limit: int = Field(default=5, ge=1, le=20, description="Max chunks (1-20)")

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v


def validate_tool_input(model_class: type[BaseModel], args: dict) -> BaseModel:
    """Validate tool input against a Pydantic model.

    Raises pydantic.ValidationError with clear messages on invalid input.
    """
    return model_class.model_validate(args)
