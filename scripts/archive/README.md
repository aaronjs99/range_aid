# File Structure

| File | Relevance | Dependencies | Used by |
| --- | --- | --- | --- |
| __init__.py | Marks this directory as an importable Python package. | range_aid | None |
| events.py | Hash-chained JSONL event archive for reproducible estimator replays. | __future__, uuid | None |
| io.py | Deterministic atomic JSON artifact I/O. | __future__ | README.md, scripts/run_score_baseline.py |
| rebuild.py | Deterministic full-batch reconstruction from immutable range_aid events. | __future__, gtsam, numpy, range_aid | None |
