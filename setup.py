#!/usr/bin/env python3
"""Install reusable range-aid estimator modules into catkin spaces."""

from setuptools import setup

from catkin_pkg.python_setup import generate_distutils_setup

setup(
    **generate_distutils_setup(
        packages=[
            "range_aid",
            "range_aid.archive",
            "range_aid.certification",
            "range_aid.estimation",
            "range_aid.models",
        ],
        package_dir={"": "src"},
    )
)
