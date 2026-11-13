"""Ranking annotations and compatible payload copies."""

from typing import Any

#: What a result object can raise when it refuses a new attribute.
_REFUSED = (AttributeError, TypeError, ValueError)


def get_annotation(result: Any, key: str, default: Any = None) -> Any:
    """Read an annotation when the result has a usable annotations dict."""
    annotations = getattr(result, "annotations", None)
    return annotations.get(key, default) if isinstance(annotations, dict) else default


def annotate(result: Any, key: str, value: Any) -> None:
    """Record a ranking annotation and a compatible payload copy.

    A usable annotations dict always receives the value. Missing or invalid
    annotation state is attached when possible. Payload copies replace only
    free or owned keys. If ownership cannot be attached, only free keys are
    written, so stored client metadata is never replaced.
    """
    annotations = getattr(result, "annotations", None)
    if not isinstance(annotations, dict):
        annotations = {}
        try:
            result.annotations = annotations
        except _REFUSED:
            pass
    annotations[key] = value
    owned = getattr(result, "_annotation_payload_keys", ())
    if not isinstance(owned, tuple):
        owned = ()
    free = key not in result.payload
    if not free and key not in owned:
        return
    if key not in owned:
        try:
            result._annotation_payload_keys = (*owned, key)
        except _REFUSED:
            if not free:
                return
    result.payload[key] = value
