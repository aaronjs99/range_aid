"""Hash-chained JSONL event archive for reproducible estimator replays."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import threading
from typing import Dict, Optional
import uuid


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return {"nonfinite": "nan"}
        return {
            "nonfinite": "positive_infinity" if value > 0.0 else "negative_infinity"
        }
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _canonical_json(value: Dict[str, object]) -> str:
    return json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


class EventArchive:
    """Append immutable estimator inputs and state transitions to JSONL."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        directory: Path,
        *,
        session_id: Optional[str] = None,
        raw_bag_uri: str = "",
        extrinsic_revision: str = "",
    ) -> None:
        self.session_id = str(session_id or uuid.uuid4())
        self.raw_bag_uri = str(raw_bag_uri or "")
        self.extrinsic_revision = str(extrinsic_revision or "")
        directory = Path(directory).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = directory / "{}-{}.jsonl".format(stamp, self.session_id)
        self._handle = self.path.open("x", encoding="utf-8", newline="\n")
        self._lock = threading.Lock()
        self._sequence = 0
        self._previous_sha256 = "0" * 64
        self.append(
            "session_start",
            {
                "schema_version": self.SCHEMA_VERSION,
                "raw_bag_uri": self.raw_bag_uri,
                "extrinsic_revision": self.extrinsic_revision,
            },
            sync=True,
        )

    @property
    def event_count(self) -> int:
        return self._sequence

    def append(
        self, event_type: str, payload: Dict[str, object], *, sync: bool = False
    ) -> str:
        """Append one event and return its content hash."""
        with self._lock:
            envelope = {
                "schema_version": self.SCHEMA_VERSION,
                "session_id": self.session_id,
                "sequence": self._sequence,
                "event_type": str(event_type),
                "previous_sha256": self._previous_sha256,
                "payload": dict(payload),
            }
            digest = hashlib.sha256(
                _canonical_json(envelope).encode("utf-8")
            ).hexdigest()
            record = dict(envelope)
            record["event_sha256"] = digest
            self._handle.write(_canonical_json(record) + "\n")
            self._handle.flush()
            if sync:
                os.fsync(self._handle.fileno())
            self._sequence += 1
            self._previous_sha256 = digest
            return digest

    def close(self) -> None:
        with self._lock:
            if self._handle.closed:
                return
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()


def verify_archive(path: Path) -> Dict[str, object]:
    """Verify the sequence, previous-hash chain, and event content hashes."""
    path = Path(path)
    expected_previous = "0" * 64
    expected_sequence = 0
    session_id = None
    errors = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                errors.append("line_{}:invalid_json".format(line_number))
                continue
            actual_digest = str(record.pop("event_sha256", ""))
            if int(record.get("sequence", -1)) != expected_sequence:
                errors.append("line_{}:sequence".format(line_number))
            if str(record.get("previous_sha256", "")) != expected_previous:
                errors.append("line_{}:previous_sha256".format(line_number))
            record_session = str(record.get("session_id", ""))
            if session_id is None:
                session_id = record_session
            elif record_session != session_id:
                errors.append("line_{}:session_id".format(line_number))
            expected_digest = hashlib.sha256(
                _canonical_json(record).encode("utf-8")
            ).hexdigest()
            if actual_digest != expected_digest:
                errors.append("line_{}:event_sha256".format(line_number))
            expected_previous = actual_digest
            expected_sequence += 1
    if expected_sequence == 0:
        errors.append("empty_archive")
    return {
        "valid": not errors,
        "event_count": expected_sequence,
        "session_id": session_id or "",
        "terminal_sha256": expected_previous,
        "errors": errors,
    }
