"""State storage for stateful stages (Q-learning, IOR, mood, curiosity)."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from threading import Lock
from typing import Any


class StateStore(ABC):
    """Key-value persistence for pipeline stage state.

    Keys are strings (stage names usually), values are JSON-serializable dicts.
    Concrete implementations choose backend: in-memory, file, Redis, SQLite.
    """

    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any: ...

    @abstractmethod
    def set(self, key: str, value: Any) -> None: ...

    @abstractmethod
    def update(self, key: str, updater) -> Any:
        """Atomically read key, apply updater(value) → new_value, write back."""

    def get_dict(self, key: str) -> dict:
        value = self.get(key)
        return dict(value) if isinstance(value, dict) else {}


class InMemoryStateStore(StateStore):
    """Thread-safe in-memory state. Lost on restart."""

    def __init__(self):
        self._data: dict[str, Any] = {}
        self._lock = Lock()

    def get(self, key, default=None):
        with self._lock:
            return self._data.get(key, default)

    def set(self, key, value):
        with self._lock:
            self._data[key] = value

    def update(self, key, updater):
        with self._lock:
            current = self._data.get(key)
            new_value = updater(current)
            self._data[key] = new_value
            return new_value


_LEGACY_STATE_PATH = Path("/tmp/mnemostack-server-state.json")


def default_state_path() -> str:
    """Per-user persistent state location (XDG_STATE_HOME or ~/.local/state).

    The historical default lived in /tmp — world-shared and wiped on reboot.
    A legacy /tmp state file is migrated once so learned IOR/Q-learning state
    survives the upgrade.
    """
    base = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")
    target = base / "mnemostack" / "server-state.json"
    if not target.exists() and _LEGACY_STATE_PATH.exists():
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_LEGACY_STATE_PATH, target)
        except OSError:
            pass
    return str(target)


class FileStateStore(StateStore):
    """JSON-backed state. Safe for single-process use.

    Each update rewrites the full file — suitable for state files up to a
    few MB. For high-write workloads use a proper DB backend.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self._lock = Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    @contextlib.contextmanager
    def _file_lock(self):
        """Cross-process lock: CLI, HTTP and MCP can share one state file.

        Uses flock where available; degrades to the in-process lock only on
        platforms without fcntl.
        """
        try:
            import fcntl
        except ImportError:  # pragma: no cover - non-POSIX
            yield
            return
        with open(self._lock_path, "a+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def _read_all(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def _write_all(self, data: dict) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))
        tmp.replace(self.path)

    def get(self, key, default=None):
        with self._lock, self._file_lock():
            data = self._read_all()
            return data.get(key, default)

    def set(self, key, value):
        with self._lock, self._file_lock():
            data = self._read_all()
            data[key] = value
            self._write_all(data)

    def update(self, key, updater):
        with self._lock, self._file_lock():
            data = self._read_all()
            new_value = updater(data.get(key))
            data[key] = new_value
            self._write_all(data)
            return new_value
