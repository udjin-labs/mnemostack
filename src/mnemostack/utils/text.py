"""Text utilities for cleaning agent transcripts before embedding/indexing.

AI agent runtimes (OpenClaw, Claude Code, etc.) wrap user/assistant messages
in metadata envelopes (active_memory_plugin, untrusted context blocks, sender
JSON, replied-message JSON, system-reminder tags). Embedding the envelope
drowns the actual message body in boilerplate, making all wrapped messages
look semantically similar to each other.

Use strip_metadata_blocks() before chunking to keep only the actual body.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

# Built-in profiles (can be combined). Each profile contributes a list of regexes.
_PROFILES: dict[str, list[re.Pattern]] = {
    "openclaw_webchat": [
        re.compile(
            r"^Untrusted context[^\n]*:\s*\n<active_memory_plugin>.*?</active_memory_plugin>\s*\n?",
            re.DOTALL | re.MULTILINE,
        ),
        re.compile(r"<active_memory_plugin>.*?</active_memory_plugin>\s*\n?", re.DOTALL),
        re.compile(r"<system-reminder>.*?</system-reminder>\s*\n?", re.DOTALL),
        re.compile(r"^Untrusted context[^\n]*:\s*\n", re.MULTILINE),
    ],
    "telegram_envelope": [
        re.compile(
            r"^Sender \(untrusted metadata\):\s*\n```json\s*\n.*?\n```\s*\n?",
            re.DOTALL | re.MULTILINE,
        ),
        re.compile(
            r"^Conversation info \(untrusted metadata\):\s*\n```json\s*\n.*?\n```\s*\n?",
            re.DOTALL | re.MULTILINE,
        ),
        re.compile(
            r"^Replied message \(untrusted[^)]*\):\s*\n```json\s*\n.*?\n```\s*\n?",
            re.DOTALL | re.MULTILINE,
        ),
        re.compile(r"^Sender \(untrusted metadata\):\s*\n", re.MULTILINE),
        re.compile(r"^Conversation info \(untrusted metadata\):\s*\n", re.MULTILINE),
        re.compile(r"^Replied message \(untrusted[^)]*\):\s*\n", re.MULTILINE),
    ],
}

_HEARTBEAT_RX = re.compile(
    r"Read HEARTBEAT\.md if it exists|HEARTBEAT_OK|HEALTH_CHECK_OK",
    re.IGNORECASE,
)


def strip_metadata_blocks(
    content: str,
    profiles: Iterable[str] = ("openclaw_webchat", "telegram_envelope"),
    extra_patterns: Iterable[re.Pattern | str] | None = None,
) -> str:
    """Remove agent-runtime metadata envelopes from a transcript message.

    Args:
        content: raw message content (may contain wrapping blocks).
        profiles: built-in profile names to apply. Defaults to OpenClaw webchat
            + Telegram envelopes (most common combination).
        extra_patterns: optional additional regex patterns (compiled or strings)
            to apply on top of the chosen profiles.

    Returns:
        Cleaned content with metadata blocks removed and excessive blank lines
        collapsed. Returns original input if it's empty.

    Example:
        >>> raw = '''Untrusted context (metadata):
        ... <active_memory_plugin>...</active_memory_plugin>
        ...
        ... [Sat 2026-04-25 21:20 UTC] MERGED!
        ... '''
        >>> strip_metadata_blocks(raw)
        '[Sat 2026-04-25 21:20 UTC] MERGED!'
    """
    if not content:
        return content
    text = content
    for prof in profiles:
        for rx in _PROFILES.get(prof, ()):  # unknown profiles are ignored intentionally
            text = rx.sub("", text)
    if extra_patterns:
        for pattern in extra_patterns:
            rx = re.compile(pattern, re.DOTALL | re.MULTILINE) if isinstance(pattern, str) else pattern
            text = rx.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_heartbeat_poll(content: str, body_threshold: int = 80) -> bool:
    """Return True if content is a scheduled heartbeat poll with no real body.

    Heartbeat polls (HEARTBEAT.md / HEARTBEAT_OK / HEALTH_CHECK_OK) add no
    recall value — they're scheduled prompts, not actual conversation turns.
    """
    if not content or not _HEARTBEAT_RX.search(content):
        return False
    body = re.sub(
        r"Read HEARTBEAT\.md if it exists.*?(reply HEARTBEAT_OK\.|HEALTH_CHECK_OK\.)",
        "",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )
    body = re.sub(r"HEARTBEAT_OK|HEALTH_CHECK_OK", "", body, flags=re.IGNORECASE).strip()
    return len(body) < body_threshold
