# File Structure

| File | Relevance | Dependencies | Used by |
| --- | --- | --- | --- |
| __init__.py | Marks this directory as an importable Python package. | range_aid | None |
| external.py | Shared validation for pinned external assurance repositories. | __future__ | README.md |
| objective_parity.py | Independent evaluator for CORA's exported chordal pose-range objective. | __future__, gtsam, numpy, range_aid | None |
| pyfg_export.py | Deterministic adapter from immutable snapshots to official CORA PyFG text. | __future__, gtsam, numpy, range_aid | None |
| snapshot_sdp_diagnostic.py | Dense SDP rank diagnostic for immutable landmark snapshots. | __future__, cvxpy, numpy | None |
| worker.py | Single-slot asynchronous worker for immutable certification snapshots. | __future__, range_aid | None |
