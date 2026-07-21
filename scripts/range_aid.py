"""Flat source-package bridge for Range Aid and generated ROS messages.

The implementation modules live directly under ``scripts/`` by project
convention. Marking this module as a package lets ``range_aid.archive`` and the
other reusable modules resolve there, while ``extend_path`` also exposes
catkin's generated ``range_aid.msg`` package.
"""

from pathlib import Path
from pkgutil import extend_path

__path__ = [str(Path(__file__).resolve().parent)]
__path__ = extend_path(__path__, __name__)
