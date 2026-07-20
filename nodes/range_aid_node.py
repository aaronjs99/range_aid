#!/usr/bin/env python3
"""Run bounded range-aided estimation as a non-authoritative shadow ROS node."""

from __future__ import annotations

import copy
import math
from pathlib import Path
import threading
import time

import numpy as np
import rospy
from nav_msgs.msg import Odometry

from range_aid.certification import AsynchronousSnapshotCertifier
from range_aid.estimation import RangeMeasurement, RebuildingFixedLagSmoother
from range_aid.estimation.fixed_lag import pose3_from_components, pose3_to_components
from range_aid.models import load_online_config
from range_aid.msg import CorrectionProposal, RangeAidStatus, RangeObservation


def _finite(value: float, fallback: float = 0.0) -> float:
    return float(value) if math.isfinite(float(value)) else float(fallback)


def _ros_covariance(covariance: np.ndarray):
    """Convert GTSAM [rotation, translation] covariance to ROS pose order."""
    if covariance is None or covariance.shape != (6, 6):
        return [0.0] * 36
    order = [3, 4, 5, 0, 1, 2]
    converted = covariance[np.ix_(order, order)]
    return converted.reshape(-1).tolist()


class RangeAidNode:
    """ROS adaptation around the bounded smoother and asynchronous certifier."""

    def __init__(self) -> None:
        rospy.init_node("range_aid")
        config_file = Path(
            rospy.get_param("~config_file", "$(find range_aid)/config/online.yaml")
        ).expanduser()
        self.config = load_online_config(config_file)
        self.mode = str(rospy.get_param("~mode", "shadow") or "shadow").lower()
        if self.mode != "shadow":
            raise ValueError("range_aid currently permits only mode=shadow")
        self.odometry_topic = str(
            rospy.get_param("~odometry_topic", "/state/odometry_6dof")
        )
        self.observation_topic = str(
            rospy.get_param("~observation_topic", "/range_aid/observations")
        )
        self.shadow_topic = str(
            rospy.get_param("~shadow_odometry_topic", "/range_aid/odometry_shadow")
        )
        self.proposal_topic = str(
            rospy.get_param(
                "~correction_proposal_topic", "/range_aid/correction_proposal"
            )
        )
        self.status_topic = str(
            rospy.get_param("~status_topic", "/range_aid/status")
        )
        self.smoother = RebuildingFixedLagSmoother(self.config)
        self.certifier = AsynchronousSnapshotCertifier(
            solver=str(
                rospy.get_param(
                    "~certification_solver", self.config.certification_solver
                )
            )
        )
        self.lock = threading.RLock()
        self.latest_source_odometry = None
        self.latest_diagnostics = self.smoother.diagnostics()
        self.latest_odom_received_monotonic = -math.inf
        self.latest_range_received_monotonic = -math.inf
        self.last_state_stamp_sec = -math.inf
        self.last_certification_submit_monotonic = -math.inf
        self.last_published_certification_epoch = -1
        self.shadow_pub = rospy.Publisher(
            self.shadow_topic, Odometry, queue_size=20
        )
        self.proposal_pub = rospy.Publisher(
            self.proposal_topic, CorrectionProposal, queue_size=5
        )
        self.status_pub = rospy.Publisher(
            self.status_topic, RangeAidStatus, queue_size=1, latch=True
        )
        rospy.Subscriber(
            self.odometry_topic,
            Odometry,
            self._odometry_cb,
            queue_size=20,
            tcp_nodelay=True,
        )
        rospy.Subscriber(
            self.observation_topic,
            RangeObservation,
            self._range_cb,
            queue_size=50,
            tcp_nodelay=True,
        )
        rospy.Timer(rospy.Duration(0.5), self._status_timer)
        rospy.on_shutdown(self.certifier.close)
        rospy.loginfo(
            "range_aid shadow ready odometry=%s observations=%s output=%s lag=%.1fs",
            self.odometry_topic,
            self.observation_topic,
            self.shadow_topic,
            self.config.lag_sec,
        )

    @staticmethod
    def _stamp_sec(message) -> float:
        stamp = message.header.stamp
        return float(stamp.to_sec()) if stamp != rospy.Time(0) else float(rospy.Time.now().to_sec())

    def _odometry_cb(self, message: Odometry) -> None:
        stamp_sec = self._stamp_sec(message)
        min_interval = 1.0 / self.config.state_rate_hz
        with self.lock:
            self.latest_odom_received_monotonic = time.monotonic()
            if stamp_sec - self.last_state_stamp_sec < min_interval:
                return
            pose = message.pose.pose
            raw_pose = pose3_from_components(
                (pose.position.x, pose.position.y, pose.position.z),
                (
                    pose.orientation.w,
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                ),
            )
            self.latest_source_odometry = copy.deepcopy(message)
            self.latest_diagnostics = self.smoother.add_odometry(stamp_sec, raw_pose)
            self.last_state_stamp_sec = stamp_sec
            self._publish_shadow(message)

    def _range_cb(self, message: RangeObservation) -> None:
        sigma_m = math.sqrt(message.variance_m2) if message.variance_m2 > 0.0 else self.config.range_default_sigma_m
        measurement = RangeMeasurement(
            stamp_sec=self._stamp_sec(message),
            landmark_id=str(message.landmark_id),
            range_m=float(message.range_m),
            sigma_m=float(sigma_m),
            synthetic=bool(message.synthetic),
            source=str(message.source),
        )
        with self.lock:
            self.latest_range_received_monotonic = time.monotonic()
            started = time.perf_counter()
            self.smoother.add_range(measurement)
            self.latest_diagnostics = self.smoother.diagnostics(
                (time.perf_counter() - started) * 1000.0
            )

    def _publish_shadow(self, source: Odometry) -> None:
        pose = self.smoother.latest_pose()
        if pose is None:
            return
        position, quaternion = pose3_to_components(pose)
        output = copy.deepcopy(source)
        output.pose.pose.position.x = float(position[0])
        output.pose.pose.position.y = float(position[1])
        output.pose.pose.position.z = float(position[2])
        output.pose.pose.orientation.w = float(quaternion[0])
        output.pose.pose.orientation.x = float(quaternion[1])
        output.pose.pose.orientation.y = float(quaternion[2])
        output.pose.pose.orientation.z = float(quaternion[3])
        output.pose.covariance = _ros_covariance(self.smoother.latest_covariance())
        self.shadow_pub.publish(output)

    def _certification_state(self):
        result, pending = self.certifier.latest()
        age = (
            max(0.0, time.monotonic() - result.completed_monotonic)
            if result is not None
            else math.inf
        )
        current = result is not None and age <= self.config.certification_max_age_sec
        tight = bool(current and result.tight)
        return result, pending, age, tight

    def _status_timer(self, _event) -> None:
        with self.lock:
            now = time.monotonic()
            diagnostics = self.latest_diagnostics
            if (
                diagnostics.range_count >= self.config.min_measurements
                and now - self.last_certification_submit_monotonic
                >= self.config.certification_period_sec
            ):
                self.certifier.submit(self.smoother.snapshot())
                self.last_certification_submit_monotonic = now
            result, pending, certification_age, certification_tight = (
                self._certification_state()
            )
            candidate_gate = diagnostics.gate_passed and (
                certification_tight or not self.config.require_tight_certification
            )
            reasons = list(diagnostics.gate_reasons)
            if self.config.require_tight_certification and not certification_tight:
                reasons.append("tight_snapshot_certification_unavailable")
            status = RangeAidStatus()
            status.header.stamp = rospy.Time.now()
            status.header.frame_id = (
                self.latest_source_odometry.header.frame_id
                if self.latest_source_odometry is not None
                else ""
            )
            status.graph_epoch = diagnostics.epoch
            status.mode = self.mode
            status.estimator_backend = "bounded_rebuilding_isam2"
            status.certification_backend = (
                result.backend if result is not None else "cora_landmark_snapshot_diagnostic"
            )
            status.active_pose_count = diagnostics.pose_count
            status.active_range_count = diagnostics.range_count
            status.rejected_observation_count = diagnostics.rejected_observation_count
            status.translational_observability_rank = diagnostics.translational_rank
            status.observability_condition = _finite(diagnostics.observability_condition, 1e30)
            status.range_residual_rms_m = _finite(diagnostics.residual_rms_m, 1e30)
            status.update_duration_ms = diagnostics.update_duration_ms
            status.latest_odometry_age_sec = _finite(now - self.latest_odom_received_monotonic, 1e30)
            status.latest_range_age_sec = _finite(now - self.latest_range_received_monotonic, 1e30)
            status.estimate_available = self.smoother.latest_pose() is not None
            status.candidate_gate_passed = candidate_gate
            status.navigation_eligible = False
            status.synthetic_evidence = diagnostics.synthetic_evidence
            status.certification_pending = pending
            status.certification_tight = certification_tight
            status.certification_age_sec = _finite(certification_age, 1e30)
            status.gate_reasons = reasons
            self.status_pub.publish(status)
            if (
                result is not None
                and result.epoch != self.last_published_certification_epoch
            ):
                self._publish_proposal(result)
                self.last_published_certification_epoch = result.epoch

    def _publish_proposal(self, certification) -> None:
        details = certification.details
        position = list(details.get("latest_pose_position_m", []) or [])
        quaternion = list(details.get("latest_pose_quaternion_wxyz", []) or [])
        if len(position) != 3 or len(quaternion) != 4:
            return
        proposal = CorrectionProposal()
        proposal.header.stamp = rospy.Time.from_sec(
            float(details.get("latest_stamp_sec", 0.0))
        )
        proposal.header.frame_id = (
            self.latest_source_odometry.header.frame_id
            if self.latest_source_odometry is not None
            else ""
        )
        proposal.graph_epoch = certification.epoch
        proposal.pose.pose.position.x = position[0]
        proposal.pose.pose.position.y = position[1]
        proposal.pose.pose.position.z = position[2]
        proposal.pose.pose.orientation.w = quaternion[0]
        proposal.pose.pose.orientation.x = quaternion[1]
        proposal.pose.pose.orientation.y = quaternion[2]
        proposal.pose.pose.orientation.z = quaternion[3]
        snapshot_covariance = np.asarray(
            details.get("latest_pose_covariance", []) or [], dtype=float
        )
        proposal.pose.covariance = _ros_covariance(snapshot_covariance)
        proposal.source_odometry_topic = self.odometry_topic
        proposal.landmark_id = ",".join(sorted(self.config.landmarks))
        proposal.range_measurement_count = int(details.get("range_count", 0))
        proposal.translational_observability_rank = int(
            details.get("translational_rank", 0)
        )
        proposal.observability_condition = _finite(
            details.get("observability_condition", math.inf), 1e30
        )
        proposal.range_residual_rms_m = _finite(
            details.get("residual_rms_m", math.inf), 1e30
        )
        proposal.candidate_gate_passed = bool(
            details.get("candidate_gate_passed", False) and certification.tight
        )
        proposal.navigation_eligible = False
        proposal.synthetic_evidence = bool(details.get("synthetic_evidence", False))
        proposal.gate_reasons = list(details.get("gate_reasons", []) or []) + list(
            certification.reasons
        )
        self.proposal_pub.publish(proposal)


if __name__ == "__main__":
    RangeAidNode()
    rospy.spin()
