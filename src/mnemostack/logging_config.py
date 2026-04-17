"""Structured logging setup for mnemostack.

Uses stdlib logging. Callers can configure format/level via standard
logging.basicConfig, or use our helper for a sensible default.

Usage:
    import logging
    from mnemostack.logging_config import configure_logging

    configure_logging(level='INFO')  # or 'DEBUG' for verbose

    logger = logging.getLogger('mnemostack.your_module')
    logger.info('something happened', extra={'count': 42})

All mnemostack modules use namespaced loggers: mnemostack.embeddings,
mnemostack.vector, mnemostack.recall, etc.
"""
from __future__ import annotations

import logging
import sys


def configure_logging(
    level: str | int = "INFO",
    fmt: str | None = None,
    stream=None,
) -> None:
    """Configure root logger for mnemostack.

    Safe to call multiple times — existing handlers on the mnemostack
    namespace are removed before re-adding.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    if fmt is None:
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter(fmt))

    logger = logging.getLogger("mnemostack")
    # Clear any previously attached handlers to avoid duplicate output
    for h in list(logger.handlers):
        logger.removeHandler(h)
    logger.addHandler(handler)
    logger.setLevel(level)
    # Don't propagate to root logger — we already have our handler
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Convenience: return a mnemostack-namespaced logger.

    If `name` doesn't start with 'mnemostack.', it's prefixed automatically.
    """
    if not name.startswith("mnemostack"):
        name = f"mnemostack.{name}"
    return logging.getLogger(name)
