# File Structure

| File | Relevance | Dependencies | Used by |
| --- | --- | --- | --- |
| __init__.py | Marks this directory as an importable Python package. | range_aid.py | None |
| export_cora_snapshot.py | Export one immutable range_aid snapshot to official CORA's PyFG format. | __future__, range_aid | CMakeLists.txt, README.md |
| range_aid.py | Flat source-package bridge for Range Aid and generated ROS messages. | pkgutil | CMakeLists.txt, README.md, scripts/__init__.py |
| range_aid_archive.py | Verify a range_aid archive or extract an immutable certification snapshot. | __future__, range_aid | CMakeLists.txt, README.md |
| range_aid_node.py | Run range-aided estimation as a map-frame, non-authoritative shadow node. | __future__, gtsam, numpy, rospy | CMakeLists.txt, README.md, launch/range_aid.launch |
| run_official_cora.py | Run a pinned official CORA build and ingest its machine-readable result. | __future__, range_aid | CMakeLists.txt, README.md |
| run_score_baseline.py | Run the pinned official SCORE SOCP baseline on an exported PyFG snapshot. | __future__, numpy, range_aid, py_factor_graph | CMakeLists.txt, README.md |
| synthetic_range_source.py | Publish explicitly synthetic known-landmark ranges from simulator truth. | __future__, numpy, rospy, nav_msgs | CMakeLists.txt, launch/range_aid.launch |
