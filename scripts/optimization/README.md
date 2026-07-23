# File Structure

| File | Relevance | Dependencies | Used by |
| --- | --- | --- | --- |
| __init__.py | Selects the full nonlinear or explicitly diagnostic dense-SDP backend. | configuration/config.py | run.py |
| common.py | Shared optimizer initialization helpers. | __future__, numpy, scripts | None |
| full.py | Moving-target range-aided estimation. | __future__, numpy, scipy, scripts | None |
| snapshot_sdp_diagnostic.py | Runs the educational dense-CVXPY snapshot SDP diagnostic without claiming official CORA certification. | __future__, types, cvxpy, numpy | optimization/__init__.py, reports/summary.py |
