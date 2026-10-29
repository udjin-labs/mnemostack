"""Opt-in, format-based secret blocking before ingestion side effects.

No network verification, content logging, redaction, or placeholder bypasses.
This is a bounded set of formats, not a guarantee that input contains no secrets.
"""

from __future__ import annotations

import base64
import binascii
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

_TOKEN_RULES = tuple(
    re.compile(pattern, re.ASCII)
    for pattern in (
        r"(?<![A-Za-z0-9_])gh[pousr]_[A-Za-z0-9]{36}(?![A-Za-z0-9_])",
        r"(?<![A-Za-z0-9_])github_pat_[A-Za-z0-9_]{82}(?![A-Za-z0-9_])",
        r"(?<![A-Za-z0-9_])glpat-(?:[A-Za-z0-9_-]{27,300}\.[a-z0-9]{9}|[A-Za-z0-9_-]{20})(?![A-Za-z0-9_-])",
        r"(?<![A-Za-z0-9_])xoxb-[0-9]{10,13}-[0-9]{10,13}-[A-Za-z0-9]{24}(?![A-Za-z0-9_])",
        r"(?<![A-Za-z0-9_])hvs\.[A-Za-z0-9_-]{24,}(?![A-Za-z0-9_])",
    )
)
_BOUNDARY = re.compile(r"-----(BEGIN|END) ((?:RSA |EC |OPENSSH |ENCRYPTED )?PRIVATE KEY)-----")


class IngestPolicyError(ValueError):
    """A safe refusal: never carries the input, path, match, or exception cause."""

    def __init__(self) -> None:
        super().__init__("ingest blocked by secret policy")


def _private_block(text: str) -> bool:
    start: int | None = None
    label: str | None = None
    for match in _BOUNDARY.finditer(text):
        if match[1] == "BEGIN":
            start, label = match.end(), match[2]
            continue
        if start is None or label != match[2]:
            start, label = None, None
            continue
        body = text[start : match.start()]
        start, label = None, None
        lines = [line.strip() for line in body.strip().splitlines()]
        if lines and re.fullmatch(r"Proc-Type:\s*4,ENCRYPTED", lines[0]):
            if len(lines) < 3 or not re.fullmatch(
                r"DEK-Info:\s*[A-Z0-9-]+,[0-9A-Fa-f]+", lines[1]
            ):
                continue
            lines = lines[2:]
        value = "".join(lines)
        if len(value) < 64:
            continue
        try:
            if len(base64.b64decode(value, validate=True)) >= 48:
                return True
        except (ValueError, binascii.Error):
            continue
    return False


@dataclass(frozen=True)
class SecretPolicy:
    """Deployment-controlled policy. SDK callers opt in explicitly."""

    mode: str = "off"

    def __post_init__(self) -> None:
        if self.mode not in ("off", "block"):
            raise ValueError("ingest.secret_policy must be 'off' or 'block'")

    def check(self, value: Any) -> None:
        if self.mode == "off":
            return
        # Iterative traversal also handles nested metadata without recursion
        # failures. Ignore cycles on SDK objects; JSON inputs cannot have them.
        pending = [value]
        visited: set[int] = set()
        while pending:
            item = pending.pop()
            if isinstance(item, str):
                if any(rule.search(item) for rule in _TOKEN_RULES) or _private_block(item):
                    raise IngestPolicyError()
            elif isinstance(item, (dict, list, tuple)):
                if id(item) in visited:
                    continue
                visited.add(id(item))
                if isinstance(item, dict):
                    pending.extend(item.keys())
                    pending.extend(item.values())
                else:
                    pending.extend(item)

    def check_item(self, item: Any) -> None:
        self.check((item.text, item.source, item.metadata, item.tags, item.timestamp))

    def checked_items(self, items: Iterable[Any]) -> Iterator[Any]:
        for item in items:
            self.check_item(item)
            yield item
