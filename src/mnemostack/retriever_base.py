"""
Abstract base class for mnemostack retrievers.

All retrievers implement the same interface:
    retrieve(query: str, **kwargs) -> list[Hit]

Adding a new retriever = one subclass, no edits elsewhere.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .vector.qdrant import Hit


@dataclass
class RetrieverConfig:
    """Base configuration for all retrievers."""

    enabled: bool = True
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.weight < 0.0:
            raise ValueError("weight must be >= 0.0")


class Retriever(ABC):
    """Abstract base for all retrievers.

    Subclasses must define:
        name: str           — identifier used in Hit.source
        retrieve(query, **kwargs) -> list[Hit]
    """

    name: str = "base"

    def __init__(self, config: RetrieverConfig | None = None):
        self.config = config or RetrieverConfig()

    @abstractmethod
    def retrieve(self, query: str, **kwargs: Any) -> list[Hit]:
        """Retrieve hits for a query."""
        ...

    @property
    def is_enabled(self) -> bool:
        return self.config.enabled
