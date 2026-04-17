"""Tests for logging configuration."""
import io
import logging

import pytest

from mnemostack.logging_config import configure_logging, get_logger


@pytest.fixture(autouse=True)
def cleanup_handlers():
    """Ensure each test starts with a clean mnemostack logger."""
    yield
    logger = logging.getLogger("mnemostack")
    for h in list(logger.handlers):
        logger.removeHandler(h)
    logger.setLevel(logging.NOTSET)


def test_configure_logging_sets_level():
    configure_logging(level="DEBUG")
    logger = logging.getLogger("mnemostack")
    assert logger.level == logging.DEBUG


def test_configure_logging_handler_attached():
    stream = io.StringIO()
    configure_logging(level="INFO", stream=stream)
    logger = get_logger("test")
    logger.info("hello world")
    output = stream.getvalue()
    assert "hello world" in output
    assert "INFO" in output


def test_configure_logging_idempotent():
    """Calling configure twice shouldn't create duplicate handlers."""
    configure_logging(level="INFO")
    configure_logging(level="DEBUG")
    logger = logging.getLogger("mnemostack")
    assert len(logger.handlers) == 1


def test_get_logger_namespaces_automatically():
    logger = get_logger("embeddings")
    assert logger.name == "mnemostack.embeddings"


def test_get_logger_preserves_full_mnemostack_name():
    logger = get_logger("mnemostack.recall.answer")
    assert logger.name == "mnemostack.recall.answer"


def test_logger_does_not_propagate_to_root():
    """Our logger should not trigger root logger handlers."""
    configure_logging(level="INFO")
    logger = logging.getLogger("mnemostack")
    assert not logger.propagate


def test_level_as_int():
    configure_logging(level=logging.WARNING)
    logger = logging.getLogger("mnemostack")
    assert logger.level == logging.WARNING
