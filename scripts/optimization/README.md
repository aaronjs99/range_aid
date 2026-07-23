# File Structure

| File | Relevance | Dependencies | Used by |
| --- | --- | --- | --- |
| __init__.py | Marks this directory as an importable Python package. | __future__, scripts | None |
| common.py | Shared optimizer initialization helpers. | __future__, numpy, scripts | None |
| cora.py | CORA-style event-triggered range-aided SDP backend. | __future__, types, cvxpy, numpy | CMakeLists.txt, README.md |
| full.py | Moving-target range-aided estimation. | __future__, numpy, scipy, scripts | None |
