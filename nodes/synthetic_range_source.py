#!/usr/bin/env python3
"""Publish explicitly synthetic known-landmark ranges from simulator truth."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rospy
from nav_msgs.msg import Odometry

from range_aid.models import load_online_config
from range_aid.msg import RangeObservation


def _rotation_matrix(quaternion) -> np.ndarray:
    x, y, z, w = (
        float(quaternion.x),
        float(quaternion.y),
        float(quaternion.z),
        float(quaternion.w),
    )
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        return np.eye(3)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class SyntheticRangeSource:
    """Convert simulator ground truth into calibration-ineligible observations."""

    def __init__(self) -> None:
        rospy.init_node("synthetic_range_source")
        config = load_online_config(Path(rospy.get_param("~config_file")))
        self.landmark_id = str(
            rospy.get_param("~landmark_id", sorted(config.landmarks)[0])
        )
        self.landmark = np.asarray(config.landmarks[self.landmark_id].position_m)
        self.sensor_translation = np.asarray(config.sensor_translation_m)
        self.input_topic = str(rospy.get_param("~input_topic", "/pose_gt"))
        self.output_topic = str(
            rospy.get_param("~output_topic", "/range_aid/observations")
        )
        self.rate_hz = max(0.1, float(rospy.get_param("~rate_hz", 3.0)))
        self.sigma_m = max(1e-6, float(rospy.get_param("~sigma_m", 0.05)))
        self.dropout_probability = min(
            1.0, max(0.0, float(rospy.get_param("~dropout_probability", 0.05)))
        )
        self.outlier_probability = min(
            1.0, max(0.0, float(rospy.get_param("~outlier_probability", 0.01)))
        )
        self.outlier_m = float(rospy.get_param("~outlier_m", 1.0))
        self.rng = np.random.default_rng(int(rospy.get_param("~seed", 19)))
        self.last_stamp_sec = -math.inf
        self.publisher = rospy.Publisher(
            self.output_topic, RangeObservation, queue_size=20
        )
        rospy.Subscriber(self.input_topic, Odometry, self._callback, queue_size=1)
        rospy.loginfo(
            "synthetic range source truth=%s output=%s landmark=%s",
            self.input_topic,
            self.output_topic,
            self.landmark_id,
        )

    def _callback(self, message: Odometry) -> None:
        stamp = message.header.stamp if message.header.stamp != rospy.Time(0) else rospy.Time.now()
        stamp_sec = stamp.to_sec()
        if stamp_sec - self.last_stamp_sec < 1.0 / self.rate_hz:
            return
        self.last_stamp_sec = stamp_sec
        if self.rng.random() < self.dropout_probability:
            return
        position = np.asarray(
            [
                message.pose.pose.position.x,
                message.pose.pose.position.y,
                message.pose.pose.position.z,
            ],
            dtype=float,
        )
        sensor = position + _rotation_matrix(message.pose.pose.orientation) @ self.sensor_translation
        measured = float(np.linalg.norm(sensor - self.landmark))
        measured += float(self.rng.normal(0.0, self.sigma_m))
        if self.rng.random() < self.outlier_probability:
            measured += self.outlier_m
        observation = RangeObservation()
        observation.header.stamp = stamp
        observation.header.frame_id = message.header.frame_id
        observation.landmark_id = self.landmark_id
        observation.range_m = measured
        observation.variance_m2 = self.sigma_m * self.sigma_m
        observation.has_bearing = False
        observation.azimuth_rad = math.nan
        observation.elevation_rad = math.nan
        observation.synthetic = True
        observation.source = "heron_simulator_ground_truth_range"
        self.publisher.publish(observation)


if __name__ == "__main__":
    SyntheticRangeSource()
    rospy.spin()
