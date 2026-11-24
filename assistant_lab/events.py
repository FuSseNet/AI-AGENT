"""Append-only event metadata; task and file payloads never belong in this log."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import threading
from typing import Iterable


_METADATA_FIELDS = frozenset(
    {
        "status", "step", "steps", "tool", "tool_name", "tool_calls", "ok",
        "bytes", "count", "input_bytes", "output_bytes", "duration_ms", "metadata", "skill", "stage",
    }
)
_EVENT_NAME = re.compile(r"[a-z][a-z0-9_.-]{0,63}\Z")


class EventLog:
    """Write JSONL containing only approved metadata and redacted known secrets.

    Arbitrary keyword arguments are deliberately not serialized. In particular,
    arguments, messages, content, responses, exceptions and tasks are excluded.
    Nested metadata uses the same allowlist as the top level.
    """

    def __init__(self, path: Path, secrets: Iterable[str] = ()) -> None:
        self.path = Path(path)
        self._secrets = tuple(
            sorted({item for item in secrets if isinstance(item, str) and item}, key=len, reverse=True)
        )
        self._lock = threading.Lock()

    def _redact(self, value: object) -> object:
        if isinstance(value, str):
            for secret in self._secrets:
                value = value.replace(secret, "[REDACTED]")
            return value
        if isinstance(value, dict):
            return {self._redact(key): self._redact(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._redact(item) for item in value]
        return value

    def _metadata(self, value: object) -> object:
        if isinstance(value, dict):
            return {
                key: self._metadata(item)
                for key, item in value.items()
                if key in _METADATA_FIELDS
            }
        if isinstance(value, (list, tuple)):
            return [self._metadata(item) for item in value]
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float) and math.isfinite(value):
            return value
        # Never call str() on an arbitrary object: it could be an exception
        # whose representation includes credentials or an HTTP response body.
        return None

    def emit(self, event: str, **fields: object) -> None:
        event_name = event if isinstance(event, str) and _EVENT_NAME.fullmatch(event) else "event"
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "event": event_name,
            **self._metadata(fields),
        }
        line = json.dumps(self._redact(record), ensure_ascii=False, allow_nan=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
