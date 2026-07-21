"""Repository-local range-aided simulation and research modules.

Online ROS tools expose selected sibling modules through ``range_aid.py``;
keeping this initializer lazy prevents the offline demo from importing ROS,
GTSAM, CVXPY, or plotting dependencies until each capability is selected.
"""
