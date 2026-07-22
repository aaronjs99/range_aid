"""Deterministic atomic JSON artifact I/O."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile


def atomic_json_write(path: Path, payload) -> None:
    """Atomically replace a finite JSON artifact with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=".{}-".format(path.name),
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            temporary = Path(handle.name)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
