"""
Abstract base class for mnemostack retrievers.

All retrievers implement the same interface:
    retrieve(query: str, **kwargs) -> list[Hit]

Adding a new retriever = one subclass, no edits elsewhere.
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from .vector.qdrant import Hit


class RetrieverConfig(BaseModel):
    """Base configuration for all retrievers."""
    enabled: bool = Field(default=True, description="Whether the retriever is active")
    weight: float = Field(default=1.0, ge=0.0, description="Weight in RRF fusion")


class Retriever(ABC):
    """Abstract base for all retrievers.

    Subclasses must define:
        name: str           — identifier used in Hit.source
        retrieve(query, **kwargs) -> list[Hit]
    """

    name: str = "base"
    config: RetrieverConfig = RetrieverConfig()

    @abstractmethod
    def retrieve(self, query: str, **kwargs: Any) -> list[Hit]:
        """Retrieve hits for a query."""
        ...

    @property
    def is_enabled(self) -> bool:
        return self.config.enabled
