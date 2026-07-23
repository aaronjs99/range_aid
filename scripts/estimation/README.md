# File Structure

| File | Relevance | Dependencies | Used by |
| --- | --- | --- | --- |
| __init__.py | Marks this directory as an importable Python package. | range_aid | None |
| fixed_lag.py | Exact fixed-lag range-aided smoother with explicit graph identity. | __future__, gtsam, gtsam_unstable, numpy | None |
| rtabmap.py | Translate RTAB-Map closure links without importing odometry edges. | __future__, gtsam, numpy, range_aid | None |
