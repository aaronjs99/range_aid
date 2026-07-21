#!/usr/bin/env python3
"""Run range-aided estimation as a map-frame, non-authoritative shadow node."""

from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import threading
import time

import gtsam
import numpy as np
import rospy
from nav_msgs.msg import Odometry
from rtabmap_msgs.msg import MapData
from std_srvs.srv import Trigger, TriggerResponse
import tf2_ros

from range_aid.archive import EventArchive
from range_aid.certification import AsynchronousSnapshotCertifier
from range_aid.estimation import (
    FixedLagRangeSmoother,
    RangeMeasurement,
    translate_link,
)
from range_aid.estimation.fixed_lag import pose3_from_components, pose3_to_components
from range_aid.models import load_online_config
from range_aid.msg import CorrectionProposal, RangeAidStatus, RangeObservation


def _finite(value: float, fallback: float = 0.0) -> float:
    return float(value) if math.isfinite(float(value)) else float(fallback)


def _pose_from_ros(pose) -> gtsam.Pose3:
    return pose3_from_components(
        (pose.position.x, pose.position.y, pose.position.z),
        (
            pose.orientation.w,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
        ),
    )


def _transform_from_ros(transform) -> gtsam.Pose3:
    return pose3_from_components(
        (transform.translation.x, transform.translation.y, transform.translation.z),
        (
            transform.rotation.w,
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
        ),
    )


def _pose_payload(pose: gtsam.Pose3):
    position, quaternion = pose3_to_components(pose)
    return {
        "position_m": position.tolist(),
        "quaternion_wxyz": quaternion.tolist(),
    }


def _ros_covariance(covariance: np.ndarray, pose: gtsam.Pose3):
    """Map GTSAM local [rotation, translation] covariance into map/ROS order."""
    if covariance is None or covariance.shape != (6, 6):
        return [0.0] * 36
    rotation = np.asarray(pose.rotation().matrix(), dtype=float)
    local_to_map = np.zeros((6, 6), dtype=float)
    local_to_map[:3, :3] = rotation
    local_to_map[3:, 3:] = rotation
    map_covariance = local_to_map @ covariance @ local_to_map.T
    order = [3, 4, 5, 0, 1, 2]
    return map_covariance[np.ix_(order, order)].reshape(-1).tolist()


class RangeAidNode:
    """ROS adaptation around an exact fixed-lag smoother and immutable archive."""

    def __init__(self) -> None:
        rospy.init_node("range_aid")
        config_file = Path(
            rospy.get_param("~config_file", "$(find range_aid)/config/online.yaml")
        ).expanduser()
        self.config = load_online_config(config_file)
        self.mode = str(rospy.get_param("~mode", "shadow") or "shadow").lower()
        if self.mode != "shadow":
            raise ValueError("range_aid currently permits only mode=shadow")
        self.graph_frame = str(rospy.get_param("~graph_frame", "map") or "map")
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
        self.status_topic = str(rospy.get_param("~status_topic", "/range_aid/status"))
        self.rtabmap_map_data_topic = str(
            rospy.get_param(
                "~rtabmap_map_data_topic", self.config.rtabmap_map_data_topic
            )
            or ""
        )
        archive_directory = str(rospy.get_param("~archive_directory", "") or "")
        raw_bag_uri = str(
            rospy.get_param("~raw_bag_uri", self.config.raw_bag_uri) or ""
        )
        self.archive = None
        self.archive_error = ""
        try:
            self.archive = EventArchive(
                Path(archive_directory or self.config.archive_directory),
                raw_bag_uri=raw_bag_uri,
                extrinsic_revision=self.config.extrinsic_revision,
            )
        except Exception as exc:
            self.archive_error = "{}:{}".format(type(exc).__name__, exc)
            rospy.logerr("range_aid archive unavailable: %s", self.archive_error)
        archive_id = self.archive.session_id if self.archive is not None else ""
        self.smoother = FixedLagRangeSmoother(self.config, archive_id=archive_id)
        self.certifier = AsynchronousSnapshotCertifier(
            solver=str(
                rospy.get_param(
                    "~certification_solver", self.config.certification_solver
                )
            )
        )
        self.lock = threading.RLock()
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(60.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.epoch_graph_from_source = None
        self.epoch_source_frame = ""
        self.latest_source_odometry = None
        self.latest_diagnostics = self.smoother.diagnostics()
        self.latest_odom_received_monotonic = -math.inf
        self.latest_range_received_monotonic = -math.inf
        self.latest_range_accepted_monotonic = -math.inf
        self.last_state_stamp_sec = -math.inf
        self.last_certification_submit_monotonic = -math.inf
        self.last_certification_submit_snapshot_id = ""
        self.last_published_certification_snapshot_id = ""
        self._seen_rtab_payloads = set()
        self.shadow_pub = rospy.Publisher(self.shadow_topic, Odometry, queue_size=20)
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
        if self.rtabmap_map_data_topic:
            rospy.Subscriber(
                self.rtabmap_map_data_topic,
                MapData,
                self._rtabmap_cb,
                queue_size=1,
                tcp_nodelay=True,
            )
        rospy.Service("/range_aid/reset", Trigger, self._reset_service)
        rospy.Timer(rospy.Duration(0.5), self._status_timer)
        rospy.on_shutdown(self._shutdown)
        self._archive(
            "runtime_configuration",
            {
                "mode": self.mode,
                "graph_frame": self.graph_frame,
                "odometry_topic": self.odometry_topic,
                "observation_topic": self.observation_topic,
                "rtabmap_map_data_topic": self.rtabmap_map_data_topic,
                "config_file": str(config_file.resolve()),
                "config_sha256": hashlib.sha256(config_file.read_bytes()).hexdigest(),
                "resolved_online_config": asdict(self.config),
            },
            sync=True,
        )
        rospy.loginfo(
            "range_aid shadow ready graph_frame=%s odometry=%s observations=%s lag=%.1fs",
            self.graph_frame,
            self.odometry_topic,
            self.observation_topic,
            self.config.lag_sec,
        )

    def _archive(self, event_type, payload, *, sync=False) -> None:
        if self.archive is None:
            return
        try:
            self.archive.append(event_type, payload, sync=sync)
        except Exception as exc:
            self.archive_error = "{}:{}".format(type(exc).__name__, exc)
            rospy.logerr_throttle(
                5.0, "range_aid archive write failed: %s", self.archive_error
            )

    def _shutdown(self) -> None:
        self.certifier.close()
        if self.archive is not None:
            self.archive.close()

    @staticmethod
    def _stamp_sec(message) -> float:
        stamp = message.header.stamp
        return (
            float(stamp.to_sec())
            if stamp != rospy.Time(0)
            else float(rospy.Time.now().to_sec())
        )

    def _graph_pose(self, message: Odometry):
        source_frame = str(message.header.frame_id or "")
        if not source_frame:
            return None, "odometry_frame_missing"
        if self.epoch_source_frame and source_frame != self.epoch_source_frame:
            self._reset_locked("source_odometry_frame_changed")
        source_pose = _pose_from_ros(message.pose.pose)
        if self.epoch_graph_from_source is None:
            if source_frame == self.graph_frame:
                self.epoch_graph_from_source = gtsam.Pose3()
            else:
                try:
                    transform = self.tf_buffer.lookup_transform(
                        self.graph_frame,
                        source_frame,
                        message.header.stamp,
                        rospy.Duration(0.25),
                    )
                except (
                    tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException,
                ) as exc:
                    return None, "map_odom_transform_unavailable:{}".format(
                        type(exc).__name__
                    )
                self.epoch_graph_from_source = _transform_from_ros(transform.transform)
            self.epoch_source_frame = source_frame
        return self.epoch_graph_from_source.compose(source_pose), "accepted"

    def _reset_locked(self, reason: str) -> None:
        self.smoother.reset(reason)
        self.certifier.invalidate()
        self.epoch_graph_from_source = None
        self.epoch_source_frame = ""
        self.latest_source_odometry = None
        self.last_state_stamp_sec = -math.inf
        self.last_certification_submit_monotonic = -math.inf
        self.last_certification_submit_snapshot_id = ""
        self.last_published_certification_snapshot_id = ""
        self._seen_rtab_payloads.clear()
        self.latest_diagnostics = self.smoother.diagnostics()
        self._archive(
            "estimator_reset",
            {
                "reason": reason,
                "graph_epoch": self.smoother.epoch,
                "graph_revision": self.smoother.revision,
            },
            sync=True,
        )

    def _reset_service(self, _request) -> TriggerResponse:
        with self.lock:
            self._reset_locked("operator_reset")
            return TriggerResponse(
                success=True,
                message="range_aid reset to epoch {}".format(self.smoother.epoch),
            )

    def _odometry_cb(self, message: Odometry) -> None:
        stamp_sec = self._stamp_sec(message)
        min_interval = 1.0 / self.config.state_rate_hz
        with self.lock:
            self.latest_odom_received_monotonic = time.monotonic()
            if (
                math.isfinite(self.last_state_stamp_sec)
                and stamp_sec
                < self.last_state_stamp_sec - self.config.time_rollback_tolerance_sec
            ):
                self._reset_locked("odometry_time_rollback")
            if stamp_sec - self.last_state_stamp_sec < min_interval:
                return
            graph_pose, reason = self._graph_pose(message)
            if graph_pose is None:
                self._archive(
                    "odometry_rejected",
                    {
                        "stamp_sec": stamp_sec,
                        "source_frame": str(message.header.frame_id),
                        "child_frame": str(message.child_frame_id),
                        "reason": reason,
                    },
                )
                rospy.logwarn_throttle(5.0, "range_aid odometry rejected: %s", reason)
                return
            epoch_before = self.smoother.epoch
            self.latest_source_odometry = copy.deepcopy(message)
            self.latest_diagnostics = self.smoother.add_odometry(stamp_sec, graph_pose)
            if self.smoother.epoch != epoch_before:
                self.certifier.invalidate()
                self._archive(
                    "estimator_reset",
                    {
                        "reason": self.smoother.last_reset_reason,
                        "graph_epoch": self.smoother.epoch,
                        "graph_revision": self.smoother.revision,
                    },
                    sync=True,
                )
            self.last_state_stamp_sec = stamp_sec
            pose_index = self.smoother.records[-1].index
            self._archive(
                "odometry_factor",
                {
                    "stamp_sec": stamp_sec,
                    "pose_index": pose_index,
                    "source_frame": str(message.header.frame_id),
                    "graph_frame": self.graph_frame,
                    "child_frame": str(message.child_frame_id),
                    "graph_pose": _pose_payload(graph_pose),
                    "source_pose_covariance": list(message.pose.covariance),
                    "graph_epoch": self.smoother.epoch,
                    "graph_revision": self.smoother.revision,
                },
            )
            self._publish_shadow(message)

    def _range_cb(self, message: RangeObservation) -> None:
        variance_valid = bool(
            math.isfinite(message.variance_m2) and message.variance_m2 > 0.0
        )
        frame_valid = str(message.header.frame_id) == self.config.sensor_frame_id
        bearing_valid = bool(
            not message.has_bearing
            or (
                math.isfinite(message.azimuth_rad)
                and math.isfinite(message.elevation_rad)
                and math.isfinite(message.azimuth_variance_rad2)
                and message.azimuth_variance_rad2 > 0.0
                and math.isfinite(message.elevation_variance_rad2)
                and message.elevation_variance_rad2 > 0.0
            )
        )
        quality_valid = bool(
            math.isfinite(message.quality_score) and 0.0 <= message.quality_score <= 1.0
        )
        invalid_reason = str(message.invalid_reason)
        if message.valid and not variance_valid:
            invalid_reason = "range_variance_invalid"
        elif message.valid and not frame_valid:
            invalid_reason = "sensor_frame_mismatch"
        elif message.valid and not bearing_valid:
            invalid_reason = "bearing_uncertainty_invalid"
        elif message.valid and not quality_valid:
            invalid_reason = "provider_quality_invalid"
        effective_valid = bool(
            message.valid
            and variance_valid
            and frame_valid
            and bearing_valid
            and quality_valid
        )
        sigma_m = (
            math.sqrt(message.variance_m2)
            if variance_valid
            else self.config.range_default_sigma_m
        )
        measurement = RangeMeasurement(
            observation_id=str(message.observation_id),
            stamp_sec=self._stamp_sec(message),
            landmark_id=str(message.landmark_id),
            range_m=float(message.range_m),
            sigma_m=float(sigma_m),
            has_bearing=bool(message.has_bearing),
            azimuth_rad=float(message.azimuth_rad),
            elevation_rad=float(message.elevation_rad),
            azimuth_variance_rad2=float(message.azimuth_variance_rad2),
            elevation_variance_rad2=float(message.elevation_variance_rad2),
            valid=effective_valid,
            invalid_reason=invalid_reason,
            quality_score=float(message.quality_score),
            quality_flags=int(message.quality_flags),
            provider=str(message.provider),
            provenance_uri=str(message.provenance_uri),
            extrinsic_revision=str(message.extrinsic_revision),
            synthetic=bool(message.synthetic),
        )
        with self.lock:
            self.latest_range_received_monotonic = time.monotonic()
            self._archive(
                "range_observation_raw",
                {
                    "observation_id": measurement.observation_id,
                    "stamp_sec": measurement.stamp_sec,
                    "frame_id": str(message.header.frame_id),
                    "landmark_id": measurement.landmark_id,
                    "range_m": measurement.range_m,
                    "variance_m2": float(message.variance_m2),
                    "has_bearing": measurement.has_bearing,
                    "azimuth_rad": measurement.azimuth_rad,
                    "elevation_rad": measurement.elevation_rad,
                    "azimuth_variance_rad2": measurement.azimuth_variance_rad2,
                    "elevation_variance_rad2": measurement.elevation_variance_rad2,
                    "valid": measurement.valid,
                    "invalid_reason": measurement.invalid_reason,
                    "quality_score": measurement.quality_score,
                    "quality_flags": measurement.quality_flags,
                    "provider": measurement.provider,
                    "provenance_uri": measurement.provenance_uri,
                    "extrinsic_revision": measurement.extrinsic_revision,
                    "synthetic": measurement.synthetic,
                    "graph_epoch": self.smoother.epoch,
                    "graph_revision_at_receipt": self.smoother.revision,
                },
            )
            started = time.perf_counter()
            association = self.smoother.add_range(measurement)
            self.latest_diagnostics = self.smoother.diagnostics(
                (time.perf_counter() - started) * 1000.0
            )
            if association.accepted:
                self.latest_range_accepted_monotonic = time.monotonic()
            self._archive(
                "range_factor_association",
                {
                    "observation_id": measurement.observation_id,
                    "accepted": association.accepted,
                    "reason": association.reason,
                    "pose_index": association.pose_index,
                    "graph_epoch": self.smoother.epoch,
                    "graph_revision": self.smoother.revision,
                },
                sync=association.accepted,
            )

    def _rtabmap_cb(self, message: MapData) -> None:
        node_stamps = {int(node.id): float(node.stamp) for node in message.nodes}
        node_map_ids = {int(node.id): int(node.map_id) for node in message.nodes}
        with self.lock:
            for link in message.graph.links:
                measurement, translation_reason = translate_link(
                    link,
                    node_stamps=node_stamps,
                    node_map_ids=node_map_ids,
                    config=self.config,
                )
                raw_identity = "{}:{}:{}".format(
                    int(link.fromId), int(link.toId), int(link.type)
                )
                if measurement is None:
                    if int(link.type) in self.config.accepted_rtabmap_link_types:
                        self._archive(
                            "rtab_link_rejected",
                            {
                                "link_identity": raw_identity,
                                "reason": translation_reason,
                            },
                        )
                    continue
                seen_key = (measurement.closure_id, measurement.payload_sha256)
                if seen_key in self._seen_rtab_payloads:
                    continue
                self._seen_rtab_payloads.add(seen_key)
                association = self.smoother.add_loop_closure(measurement)
                self.latest_diagnostics = self.smoother.diagnostics()
                self._archive(
                    "rtab_loop_closure",
                    {
                        "closure_id": measurement.closure_id,
                        "graph_identity": measurement.graph_identity,
                        "from_rtab_id": measurement.from_rtab_id,
                        "to_rtab_id": measurement.to_rtab_id,
                        "from_stamp_sec": measurement.from_stamp_sec,
                        "to_stamp_sec": measurement.to_stamp_sec,
                        "link_type": measurement.link_type,
                        "relative_pose": _pose_payload(measurement.relative_pose),
                        "information_rotation_translation": measurement.information.tolist(),
                        "payload_sha256": measurement.payload_sha256,
                        "translation_reason": translation_reason,
                        "accepted": association.accepted,
                        "association_reason": association.reason,
                        "from_pose_index": association.pose_index,
                        "to_pose_index": association.second_pose_index,
                        "graph_epoch": self.smoother.epoch,
                        "graph_revision": self.smoother.revision,
                    },
                    sync=association.accepted,
                )

    def _publish_shadow(self, source: Odometry) -> None:
        pose = self.smoother.latest_pose()
        if pose is None:
            return
        position, quaternion = pose3_to_components(pose)
        output = copy.deepcopy(source)
        output.header.frame_id = self.graph_frame
        output.pose.pose.position.x = float(position[0])
        output.pose.pose.position.y = float(position[1])
        output.pose.pose.position.z = float(position[2])
        output.pose.pose.orientation.w = float(quaternion[0])
        output.pose.pose.orientation.x = float(quaternion[1])
        output.pose.pose.orientation.y = float(quaternion[2])
        output.pose.pose.orientation.z = float(quaternion[3])
        output.pose.covariance = _ros_covariance(
            self.smoother.latest_covariance(), pose
        )
        self.shadow_pub.publish(output)

    def _certification_state(self, diagnostics):
        result, pending = self.certifier.latest()
        age = (
            max(0.0, time.monotonic() - result.completed_monotonic)
            if result is not None
            else math.inf
        )
        identity_matches = bool(
            result is not None
            and result.epoch == diagnostics.epoch
            and result.revision == diagnostics.revision
            and result.snapshot_id == diagnostics.snapshot_id
        )
        current = identity_matches and age <= self.config.certification_max_age_sec
        return result, pending, age, current

    def _status_timer(self, _event) -> None:
        with self.lock:
            now = time.monotonic()
            diagnostics = self.latest_diagnostics
            snapshot = self.smoother.snapshot()
            if (
                diagnostics.range_count >= self.config.min_measurements
                and snapshot["snapshot_id"]
                != self.last_certification_submit_snapshot_id
                and now - self.last_certification_submit_monotonic
                >= self.config.certification_period_sec
            ):
                self.certifier.submit(snapshot)
                self.last_certification_submit_monotonic = now
                self.last_certification_submit_snapshot_id = str(
                    snapshot["snapshot_id"]
                )
                self._archive(
                    "certification_snapshot_submitted",
                    {
                        "snapshot_id": snapshot["snapshot_id"],
                        "graph_epoch": snapshot["epoch"],
                        "graph_revision": snapshot["revision"],
                        "snapshot": snapshot,
                    },
                    sync=True,
                )
            result, pending, certification_age, certification_current = (
                self._certification_state(diagnostics)
            )
            formal_tight = bool(
                certification_current
                and result.formal_full_graph_certificate
                and result.formal_certificate_tight
            )
            candidate_gate = diagnostics.gate_passed and (
                formal_tight or not self.config.require_tight_certification
            )
            reasons = list(diagnostics.gate_reasons)
            if self.archive is None or self.archive_error:
                reasons.append("archive_unavailable")
                candidate_gate = False
            if result is not None and not certification_current:
                reasons.append("certification_snapshot_stale")
            if self.config.require_tight_certification and not formal_tight:
                reasons.append("formal_full_graph_certificate_unavailable")
            status = RangeAidStatus()
            status.header.stamp = rospy.Time.now()
            status.header.frame_id = self.graph_frame
            status.graph_epoch = diagnostics.epoch
            status.graph_revision = diagnostics.revision
            status.snapshot_id = diagnostics.snapshot_id
            status.mode = self.mode
            status.estimator_backend = "incremental_fixed_lag_smoother"
            status.certification_backend = (
                result.backend if result is not None else "snapshot_sdp_diagnostic"
            )
            status.covariance_model = diagnostics.covariance_model
            status.covariance_calibrated = diagnostics.covariance_calibrated
            status.graph_frame = self.graph_frame
            status.source_odometry_frame = self.epoch_source_frame
            status.archive_id = (
                self.archive.session_id if self.archive is not None else ""
            )
            status.archive_path = (
                str(self.archive.path) if self.archive is not None else ""
            )
            status.archive_event_count = (
                self.archive.event_count if self.archive is not None else 0
            )
            status.active_pose_count = diagnostics.pose_count
            status.active_range_count = diagnostics.range_count
            status.active_loop_closure_count = diagnostics.loop_closure_count
            status.rejected_observation_count = diagnostics.rejected_observation_count
            status.rejected_loop_closure_count = diagnostics.rejected_loop_closure_count
            status.translational_observability_rank = diagnostics.translational_rank
            status.observability_condition = _finite(
                diagnostics.observability_condition, 1e30
            )
            status.range_residual_rms_m = _finite(diagnostics.residual_rms_m, 1e30)
            status.update_duration_ms = diagnostics.update_duration_ms
            status.latest_odometry_age_sec = _finite(
                now - self.latest_odom_received_monotonic, 1e30
            )
            status.latest_range_received_age_sec = _finite(
                now - self.latest_range_received_monotonic, 1e30
            )
            status.latest_range_accepted_age_sec = _finite(
                now - self.latest_range_accepted_monotonic, 1e30
            )
            status.estimate_available = self.smoother.latest_pose() is not None
            status.candidate_gate_passed = candidate_gate
            status.navigation_eligible = False
            status.synthetic_evidence = diagnostics.synthetic_evidence
            status.certification_pending = pending
            status.diagnostic_rank_tight = bool(
                result is not None and result.diagnostic_rank_tight
            )
            status.certification_tight = formal_tight
            status.formal_full_graph_certificate = bool(
                certification_current
                and result is not None
                and result.formal_full_graph_certificate
            )
            status.certification_age_sec = _finite(certification_age, 1e30)
            status.gate_reasons = sorted(set(reasons))
            self.status_pub.publish(status)
            if (
                result is not None
                and result.snapshot_id
                and result.snapshot_id != self.last_published_certification_snapshot_id
            ):
                self._publish_proposal(result)
                self.last_published_certification_snapshot_id = result.snapshot_id

    def _publish_proposal(self, certification) -> None:
        details = certification.details
        poses = list(details.get("poses", []) or [])
        if not poses:
            return
        estimate = dict(poses[-1].get("estimate", {}) or {})
        position = list(estimate.get("position_m", []) or [])
        quaternion = list(estimate.get("quaternion_wxyz", []) or [])
        if len(position) != 3 or len(quaternion) != 4:
            return
        proposal = CorrectionProposal()
        proposal.header.stamp = rospy.Time.from_sec(
            float(poses[-1].get("stamp_sec", 0.0))
        )
        proposal.header.frame_id = str(details.get("graph_frame", self.graph_frame))
        proposal.graph_epoch = certification.epoch
        proposal.graph_revision = certification.revision
        proposal.snapshot_id = certification.snapshot_id
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
        pose = pose3_from_components(tuple(position), tuple(quaternion))
        proposal.pose.covariance = _ros_covariance(snapshot_covariance, pose)
        proposal.child_frame_id = (
            self.latest_source_odometry.child_frame_id
            if self.latest_source_odometry is not None
            else ""
        )
        proposal.source_odometry_topic = self.odometry_topic
        proposal.covariance_model = str(details.get("covariance_model", ""))
        proposal.covariance_calibrated = bool(
            details.get("covariance_calibrated", False)
        )
        proposal.archive_id = str(details.get("archive_id", ""))
        proposal.archive_path = (
            str(self.archive.path) if self.archive is not None else ""
        )
        proposal.landmark_id = ",".join(sorted(self.config.landmarks))
        proposal.range_measurement_count = int(details.get("range_count", 0))
        proposal.loop_closure_count = int(details.get("loop_closure_count", 0))
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
            details.get("candidate_gate_passed", False)
            and certification.formal_full_graph_certificate
            and certification.formal_certificate_tight
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
