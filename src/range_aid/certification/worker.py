"""Single-slot asynchronous worker for immutable certification snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
import time
from typing import Dict, Optional, Tuple

from range_aid.certification.cora_snapshot import certify_snapshot


@dataclass(frozen=True)
class CertificationResult:
    """Latest completed snapshot diagnostic."""

    epoch: int
    backend: str
    tight: bool
    formal_full_graph_certificate: bool
    reasons: Tuple[str, ...]
    completed_monotonic: float
    solve_duration_ms: float
    details: Dict[str, object]


class AsynchronousSnapshotCertifier:
    """Certify only the newest queued snapshot without blocking estimation."""

    def __init__(self, solver: str = "SCS") -> None:
        self.solver = solver
        self._queue = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._latest: Optional[CertificationResult] = None
        self._pending = False
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, name="range-aid-certifier")
        self._thread.daemon = True
        self._thread.start()

    def submit(self, snapshot: Dict[str, object]) -> None:
        """Replace any unstarted snapshot with the newest graph epoch."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put_nowait(dict(snapshot))
        with self._lock:
            self._pending = True

    def latest(self) -> Tuple[Optional[CertificationResult], bool]:
        """Return the latest result and whether newer work is pending."""
        with self._lock:
            return self._latest, self._pending

    def close(self) -> None:
        self._stopped.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stopped.is_set():
            try:
                snapshot = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            started = time.perf_counter()
            details = certify_snapshot(snapshot, solver=self.solver)
            result = CertificationResult(
                epoch=int(details.get("epoch", 0)),
                backend=str(details.get("backend", "unknown")),
                tight=bool(details.get("tight", False)),
                formal_full_graph_certificate=bool(
                    details.get("formal_full_graph_certificate", False)
                ),
                reasons=tuple(str(item) for item in details.get("reasons", [])),
                completed_monotonic=time.monotonic(),
                solve_duration_ms=(time.perf_counter() - started) * 1000.0,
                details=details,
            )
            with self._lock:
                self._latest = result
                self._pending = not self._queue.empty()
