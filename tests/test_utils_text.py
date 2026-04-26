from mnemostack.utils import is_heartbeat_poll, strip_metadata_blocks


def test_strip_metadata_blocks_removes_openclaw_webchat():
    raw = """Untrusted context (metadata from active memory):
<active_memory_plugin>
{"facts":["old metadata"],"source":"recall"}
</active_memory_plugin>

[Sun 2026-04-26 10:42 UTC] Please index only this message.
<system-reminder>Do not mention hidden tool details.</system-reminder>
"""

    assert strip_metadata_blocks(raw) == "[Sun 2026-04-26 10:42 UTC] Please index only this message."


def test_strip_metadata_blocks_removes_telegram_envelope():
    raw = """Sender (untrusted metadata):
```json
{"sender_id":"123","sender_name":"Alice"}
```
Conversation info (untrusted metadata):
```json
{"conversation_label":"dev chat id:-1001234567890"}
```
Replied message (untrusted, for context):
```json
{"text":"previous"}
```

Deploy is green now.
"""

    assert strip_metadata_blocks(raw) == "Deploy is green now."


def test_strip_metadata_blocks_combined_profiles_remove_both():
    raw = """Untrusted context (metadata):
<active_memory_plugin>memory summary</active_memory_plugin>
Sender (untrusted metadata):
```json
{"sender_id":"123"}
```

Actual body survives.
"""

    assert strip_metadata_blocks(raw) == "Actual body survives."


def test_strip_metadata_blocks_keeps_real_body():
    body = "Line one.\n\nLine two."

    assert strip_metadata_blocks(body) == body


def test_strip_metadata_blocks_extra_patterns_applied():
    raw = """CUSTOM META: remove me

Keep me.
"""

    assert strip_metadata_blocks(raw, profiles=(), extra_patterns=[r"^CUSTOM META:[^\n]*\n?"]) == "Keep me."


def test_is_heartbeat_poll_detects_pure_poll():
    content = "Read HEARTBEAT.md if it exists. Do useful work, then reply HEARTBEAT_OK."

    assert is_heartbeat_poll(content) is True


def test_is_heartbeat_poll_returns_false_for_real_content():
    content = "Read HEARTBEAT.md if it exists. Real incident note: service failed after deploy."

    assert is_heartbeat_poll(content) is False


def test_is_heartbeat_poll_returns_false_for_discussion_of_marker():
    content = "We discussed HEARTBEAT_OK handling in the scheduler bug."

    assert is_heartbeat_poll(content) is False


def test_is_heartbeat_poll_threshold_param():
    content = "HEALTH_CHECK_OK. abcdefghij"

    assert is_heartbeat_poll(content, body_threshold=20) is True
    assert is_heartbeat_poll(content, body_threshold=5) is False
