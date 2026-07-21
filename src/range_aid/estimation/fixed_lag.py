"""Exact fixed-lag range-aided smoother with explicit graph identity."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import time
from typing import Deque, Dict, List, Optional, Tuple

import gtsam
import gtsam_unstable
import numpy as np

from range_aid.models.config import OnlineConfig

COVARIANCE_MODEL = "local_linearized_robust_unvalidated"


@dataclass(frozen=True)
class RangeMeasurement:
    """One provider-neutral acoustic observation."""

    observation_id: str
    stamp_sec: float
    landmark_id: str
    range_m: float
    sigma_m: float
    has_bearing: bool
    azimuth_rad: float
    elevation_rad: float
    azimuth_variance_rad2: float
    elevation_variance_rad2: float
    valid: bool
    invalid_reason: str
    quality_score: float
    quality_flags: int
    provider: str
    provenance_uri: str
    extrinsic_revision: str
    synthetic: bool


@dataclass(frozen=True)
class LoopClosureMeasurement:
    """One accepted RTAB-Map closure expressed from one keyframe to another."""

    closure_id: str
    graph_identity: str
    from_rtab_id: int
    to_rtab_id: int
    from_stamp_sec: float
    to_stamp_sec: float
    link_type: int
    relative_pose: gtsam.Pose3
    information: np.ndarray
    payload_sha256: str
    used_information_fallback: bool = False


@dataclass(frozen=True)
class FactorAssociation:
    """Result of associating an asynchronous factor with active pose keys."""

    accepted: bool
    reason: str
    pose_index: int = -1
    second_pose_index: int = -1


@dataclass
class PoseRecord:
    """Raw graph-frame odometry pose retained in the active lag."""

    index: int
    stamp_sec: float
    raw_pose: gtsam.Pose3
    measurements: List[RangeMeasurement] = field(default_factory=list)


@dataclass(frozen=True)
class ActiveLoopClosure:
    measurement: LoopClosureMeasurement
    from_pose_index: int
    to_pose_index: int


@dataclass(frozen=True)
class EstimateDiagnostics:
    """Inspectable estimator state used by ROS status and promotion gates."""

    epoch: int
    revision: int
    snapshot_id: str
    pose_count: int
    range_count: int
    loop_closure_count: int
    rejected_observation_count: int
    rejected_loop_closure_count: int
    translational_rank: int
    observability_condition: float
    residual_rms_m: float
    correction_translation_m: float
    correction_rotation_rad: float
    update_duration_ms: float
    gate_passed: bool
    gate_reasons: Tuple[str, ...]
    synthetic_evidence: bool
    covariance_model: str
    covariance_calibrated: bool
    last_reset_reason: str


def pose3_from_components(
    position_xyz: Tuple[float, float, float],
    quaternion_wxyz: Tuple[float, float, float, float],
) -> gtsam.Pose3:
    """Construct a GTSAM pose from ROS-compatible position and quaternion data."""
    w, x, y, z = quaternion_wxyz
    return gtsam.Pose3(gtsam.Rot3.Quaternion(w, x, y, z), np.asarray(position_xyz))


def pose3_to_components(pose: gtsam.Pose3) -> Tuple[np.ndarray, np.ndarray]:
    """Return position xyz and quaternion wxyz."""
    quaternion = pose.rotation().toQuaternion()
    return (
        np.asarray(pose.translation(), dtype=float),
        np.asarray(
            [quaternion.w(), quaternion.x(), quaternion.y(), quaternion.z()],
            dtype=float,
        ),
    )


def _pose_payload(pose: gtsam.Pose3) -> Dict[str, object]:
    position, quaternion = pose3_to_components(pose)
    return {
        "position_m": position.tolist(),
        "quaternion_wxyz": quaternion.tolist(),
    }


def _finite_or_none(value: float):
    return float(value) if math.isfinite(float(value)) else None


class FixedLagRangeSmoother:
    """IncrementalFixedLagSmoother backend for known Point3 landmarks."""

    def __init__(self, config: OnlineConfig, *, archive_id: str = "") -> None:
        self.config = config
        self.archive_id = str(archive_id or "")
        self.records: Deque[PoseRecord] = deque()
        self.active_loop_closures: List[ActiveLoopClosure] = []
        self.rejected_observation_count = 0
        self.rejected_loop_closure_count = 0
        self.epoch = 0
        self.revision = 0
        self.last_reset_reason = "startup"
        self._next_index = 0
        self._smoother = self._new_smoother()
        self._initialized = False
        self._last_stamp_sec = -math.inf
        self._landmark_keys = {
            landmark_id: gtsam.symbol("l", index)
            for index, landmark_id in enumerate(sorted(config.landmarks))
        }
        self._sensor_pose = self._sensor_extrinsic_pose()
        self._closure_payloads: Dict[str, str] = {}
        self._observation_payloads: Dict[str, str] = {}

    def _new_smoother(self):
        params = gtsam.ISAM2Params()
        params.setRelinearizeThreshold(0.01)
        params.relinearizeSkip = 1
        return gtsam_unstable.IncrementalFixedLagSmoother(self.config.lag_sec, params)

    @staticmethod
    def _timestamp_map(*items: Tuple[int, float]):
        timestamps = gtsam_unstable.FixedLagSmootherKeyTimestampMap()
        for key, stamp_sec in items:
            timestamps.insert((int(key), float(stamp_sec)))
        return timestamps

    def _sensor_extrinsic_pose(self) -> gtsam.Pose3:
        roll, pitch, yaw = self.config.sensor_rotation_rpy_rad
        return gtsam.Pose3(
            gtsam.Rot3.RzRyRx(roll, pitch, yaw),
            np.asarray(self.config.sensor_translation_m),
        )

    @staticmethod
    def _pose_key(index: int) -> int:
        return gtsam.symbol("x", index)

    @staticmethod
    def _pose_noise(translation_sigma: float, rotation_sigma: float):
        return gtsam.noiseModel.Diagonal.Sigmas(
            np.asarray([rotation_sigma] * 3 + [translation_sigma] * 3)
        )

    def reset(self, reason: str) -> None:
        """Begin a new estimator epoch while preserving external archives."""
        self.epoch += 1
        self.revision += 1
        self.last_reset_reason = str(reason or "unspecified_reset")
        self.records.clear()
        self.active_loop_closures.clear()
        self._closure_payloads.clear()
        self._smoother = self._new_smoother()
        self._initialized = False
        self._last_stamp_sec = -math.inf

    def _initialize(self, stamp_sec: float, raw_pose: gtsam.Pose3) -> PoseRecord:
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        for landmark_id, landmark in self.config.landmarks.items():
            key = self._landmark_keys[landmark_id]
            point = np.asarray(landmark.position_m, dtype=float)
            values.insert(key, point)
            graph.add(
                gtsam.PriorFactorPoint3(
                    key,
                    point,
                    gtsam.noiseModel.Isotropic.Sigma(3, landmark.prior_sigma_m),
                )
            )
        index = self._next_index
        self._next_index += 1
        key = self._pose_key(index)
        values.insert(key, raw_pose)
        graph.add(
            gtsam.PriorFactorPose3(
                key,
                raw_pose,
                self._pose_noise(
                    self.config.boundary_translation_sigma_m,
                    self.config.boundary_rotation_sigma_rad,
                ),
            )
        )
        self._smoother.update(graph, values, self._timestamp_map((key, stamp_sec)))
        record = PoseRecord(index=index, stamp_sec=stamp_sec, raw_pose=raw_pose)
        self.records.append(record)
        self._initialized = True
        self._last_stamp_sec = stamp_sec
        self.revision += 1
        return record

    def add_odometry(
        self, stamp_sec: float, raw_pose: gtsam.Pose3
    ) -> EstimateDiagnostics:
        """Add one graph-frame pose and one relative odometry factor."""
        started = time.perf_counter()
        stamp_sec = float(stamp_sec)
        if (
            self._initialized
            and stamp_sec
            < self._last_stamp_sec - self.config.time_rollback_tolerance_sec
        ):
            self.reset("odometry_time_rollback")
        if not self._initialized:
            self._initialize(stamp_sec, raw_pose)
            return self.diagnostics((time.perf_counter() - started) * 1000.0)

        previous = self.records[-1]
        index = self._next_index
        self._next_index += 1
        key = self._pose_key(index)
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        values.insert(key, raw_pose)
        graph.add(
            gtsam.BetweenFactorPose3(
                self._pose_key(previous.index),
                key,
                previous.raw_pose.between(raw_pose),
                self._pose_noise(
                    self.config.odometry_translation_sigma_m,
                    self.config.odometry_rotation_sigma_rad,
                ),
            )
        )
        self._smoother.update(graph, values, self._timestamp_map((key, stamp_sec)))
        self.records.append(
            PoseRecord(index=index, stamp_sec=stamp_sec, raw_pose=raw_pose)
        )
        self._last_stamp_sec = stamp_sec
        self.revision += 1
        self._refresh_active_records()
        if len(self.records) > self.config.max_pose_count:
            self.reset("active_pose_watchdog_overflow")
            self._initialize(stamp_sec, raw_pose)
        return self.diagnostics((time.perf_counter() - started) * 1000.0)

    def _refresh_active_records(self) -> None:
        estimates = self._smoother.calculateEstimate()
        self.records = deque(
            record
            for record in self.records
            if estimates.exists(self._pose_key(record.index))
        )
        active_indices = {record.index for record in self.records}
        self.active_loop_closures = [
            closure
            for closure in self.active_loop_closures
            if closure.from_pose_index in active_indices
            and closure.to_pose_index in active_indices
        ]

    def _range_factor(self, pose_key: int, measurement: RangeMeasurement):
        base = gtsam.noiseModel.Isotropic.Sigma(1, measurement.sigma_m)
        robust = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(self.config.robust_range_huber_k),
            base,
        )
        return gtsam.RangeFactorWithTransform3D(
            pose_key,
            self._landmark_keys[measurement.landmark_id],
            measurement.range_m,
            robust,
            self._sensor_pose,
        )

    def _nearest_record(self, stamp_sec: float) -> Optional[PoseRecord]:
        if not self.records:
            return None
        return min(self.records, key=lambda item: abs(item.stamp_sec - stamp_sec))

    def add_range(self, measurement: RangeMeasurement) -> FactorAssociation:
        """Validate, associate, and insert a range factor."""
        invalid_reason = ""
        payload_sha256 = hashlib.sha256(
            json.dumps(
                asdict(measurement), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        previous_payload = self._observation_payloads.get(measurement.observation_id)
        if previous_payload == payload_sha256:
            return FactorAssociation(False, "duplicate_observation")
        if previous_payload is not None:
            invalid_reason = "observation_payload_changed"
        elif measurement.observation_id:
            self._observation_payloads[measurement.observation_id] = payload_sha256
        if not invalid_reason and not measurement.valid:
            invalid_reason = measurement.invalid_reason or "provider_marked_invalid"
        elif not invalid_reason and not measurement.observation_id:
            invalid_reason = "missing_observation_id"
        elif not invalid_reason and measurement.landmark_id not in self._landmark_keys:
            invalid_reason = "unknown_landmark"
        elif not invalid_reason and not math.isfinite(measurement.range_m):
            invalid_reason = "nonfinite_range"
        elif not invalid_reason and (
            not self.config.min_range_m
            <= measurement.range_m
            <= self.config.max_range_m
        ):
            invalid_reason = "range_out_of_bounds"
        elif not invalid_reason and (
            not math.isfinite(measurement.sigma_m) or measurement.sigma_m <= 0.0
        ):
            invalid_reason = "invalid_range_sigma"
        elif (
            not invalid_reason
            and measurement.extrinsic_revision != self.config.extrinsic_revision
        ):
            invalid_reason = "extrinsic_revision_mismatch"
        record = self._nearest_record(measurement.stamp_sec)
        if not invalid_reason and record is None:
            invalid_reason = "no_active_pose"
        if (
            not invalid_reason
            and abs(record.stamp_sec - measurement.stamp_sec)
            > self.config.max_sync_error_sec
        ):
            invalid_reason = "range_pose_sync_exceeded"
        if invalid_reason:
            self.rejected_observation_count += 1
            return FactorAssociation(False, invalid_reason)

        graph = gtsam.NonlinearFactorGraph()
        graph.add(self._range_factor(self._pose_key(record.index), measurement))
        self._smoother.update(graph, gtsam.Values(), self._timestamp_map())
        record.measurements.append(measurement)
        self.revision += 1
        return FactorAssociation(True, "accepted", pose_index=record.index)

    def add_loop_closure(
        self, measurement: LoopClosureMeasurement
    ) -> FactorAssociation:
        """Insert one non-odometric RTAB closure when both poses remain active."""
        if measurement.link_type not in self.config.accepted_rtabmap_link_types:
            self.rejected_loop_closure_count += 1
            return FactorAssociation(False, "rtab_link_type_rejected")
        previous_payload = self._closure_payloads.get(measurement.closure_id)
        if previous_payload == measurement.payload_sha256:
            return FactorAssociation(False, "duplicate_loop_closure")
        if previous_payload is not None:
            self.rejected_loop_closure_count += 1
            return FactorAssociation(False, "loop_closure_payload_changed")
        from_record = self._nearest_record(measurement.from_stamp_sec)
        to_record = self._nearest_record(measurement.to_stamp_sec)
        if from_record is None or to_record is None:
            self.rejected_loop_closure_count += 1
            return FactorAssociation(False, "closure_endpoint_outside_active_lag")
        if (
            abs(from_record.stamp_sec - measurement.from_stamp_sec)
            > self.config.max_sync_error_sec
            or abs(to_record.stamp_sec - measurement.to_stamp_sec)
            > self.config.max_sync_error_sec
        ):
            self.rejected_loop_closure_count += 1
            return FactorAssociation(False, "closure_endpoint_sync_exceeded")
        if from_record.index == to_record.index:
            self.rejected_loop_closure_count += 1
            return FactorAssociation(False, "closure_collapsed_to_one_pose")
        information = np.asarray(measurement.information, dtype=float)
        if information.shape != (6, 6) or not np.all(np.isfinite(information)):
            self.rejected_loop_closure_count += 1
            return FactorAssociation(False, "invalid_loop_information")
        information = 0.5 * (information + information.T)
        if np.min(np.linalg.eigvalsh(information)) <= 0.0:
            self.rejected_loop_closure_count += 1
            return FactorAssociation(False, "non_spd_loop_information")
        graph = gtsam.NonlinearFactorGraph()
        graph.add(
            gtsam.BetweenFactorPose3(
                self._pose_key(from_record.index),
                self._pose_key(to_record.index),
                measurement.relative_pose,
                gtsam.noiseModel.Gaussian.Information(information),
            )
        )
        self._smoother.update(graph, gtsam.Values(), self._timestamp_map())
        self._closure_payloads[measurement.closure_id] = measurement.payload_sha256
        self.active_loop_closures.append(
            ActiveLoopClosure(measurement, from_record.index, to_record.index)
        )
        self.revision += 1
        return FactorAssociation(
            True,
            (
                "accepted_with_information_fallback"
                if measurement.used_information_fallback
                else "accepted"
            ),
            pose_index=from_record.index,
            second_pose_index=to_record.index,
        )

    def _current_pose_estimates(self) -> Dict[int, gtsam.Pose3]:
        if not self._initialized:
            return {}
        values = self._smoother.calculateEstimate()
        return {
            record.index: values.atPose3(self._pose_key(record.index))
            for record in self.records
            if values.exists(self._pose_key(record.index))
        }

    def latest_pose(self) -> Optional[gtsam.Pose3]:
        if not self.records:
            return None
        values = self._smoother.calculateEstimate()
        key = self._pose_key(self.records[-1].index)
        return values.atPose3(key) if values.exists(key) else None

    def latest_covariance(self) -> Optional[np.ndarray]:
        """Return local linearized covariance; it is not empirically calibrated."""
        if not self.records:
            return None
        try:
            return np.asarray(
                self._smoother.getISAM2().marginalCovariance(
                    self._pose_key(self.records[-1].index)
                ),
                dtype=float,
            )
        except (IndexError, RuntimeError):
            return None

    def _measurement_geometry(self):
        estimates = self._current_pose_estimates()
        rows = []
        residuals = []
        synthetic = False
        for record in self.records:
            pose = estimates.get(record.index)
            if pose is None:
                continue
            sensor_position = np.asarray(pose.compose(self._sensor_pose).translation())
            for measurement in record.measurements:
                landmark = np.asarray(
                    self.config.landmarks[measurement.landmark_id].position_m
                )
                delta = sensor_position - landmark
                distance = float(np.linalg.norm(delta))
                if distance <= 1e-9:
                    continue
                rows.append(delta / distance)
                residuals.append(distance - measurement.range_m)
                synthetic = synthetic or measurement.synthetic
        return rows, residuals, synthetic

    def _snapshot_payload(self) -> Dict[str, object]:
        estimates = self._current_pose_estimates()
        covariance = self.latest_covariance()
        pose_rows = []
        observations = []
        for record in self.records:
            estimate = estimates.get(record.index)
            if estimate is None:
                continue
            pose_rows.append(
                {
                    "pose_index": record.index,
                    "stamp_sec": record.stamp_sec,
                    "raw_pose": _pose_payload(record.raw_pose),
                    "estimate": _pose_payload(estimate),
                }
            )
            sensor_position = np.asarray(
                estimate.compose(self._sensor_pose).translation(), dtype=float
            )
            for measurement in record.measurements:
                observations.append(
                    {
                        "observation_id": measurement.observation_id,
                        "pose_index": record.index,
                        "stamp_sec": measurement.stamp_sec,
                        "sensor_position_m": sensor_position.tolist(),
                        "landmark_id": measurement.landmark_id,
                        "range_m": measurement.range_m,
                        "sigma_m": measurement.sigma_m,
                        "provider": measurement.provider,
                        "provenance_uri": measurement.provenance_uri,
                        "extrinsic_revision": measurement.extrinsic_revision,
                        "synthetic": measurement.synthetic,
                    }
                )
        loops = [
            {
                "closure_id": item.measurement.closure_id,
                "graph_identity": item.measurement.graph_identity,
                "from_rtab_id": item.measurement.from_rtab_id,
                "to_rtab_id": item.measurement.to_rtab_id,
                "from_pose_index": item.from_pose_index,
                "to_pose_index": item.to_pose_index,
                "link_type": item.measurement.link_type,
                "relative_pose": _pose_payload(item.measurement.relative_pose),
                "information_rotation_translation": np.asarray(
                    item.measurement.information, dtype=float
                ).tolist(),
                "payload_sha256": item.measurement.payload_sha256,
                "used_information_fallback": item.measurement.used_information_fallback,
            }
            for item in self.active_loop_closures
        ]
        diagnostics = self._diagnostic_values()
        gate_reasons = self._gate_reasons(diagnostics)
        return {
            "schema_version": 2,
            "archive_id": self.archive_id,
            "epoch": self.epoch,
            "revision": self.revision,
            "graph_frame": "map",
            "covariance_model": COVARIANCE_MODEL,
            "covariance_calibrated": False,
            "objective_convention": {
                "pose_tangent_order": "rotation_xyz_then_translation_xyz",
                "rtab_information_input_order": "translation_xyz_then_rotation_rpy",
                "range_residual": "predicted_range_minus_measured_range",
                "robust_policy": "huber_on_whitened_range_residual",
                "odometry_translation_sigma_m": self.config.odometry_translation_sigma_m,
                "odometry_rotation_sigma_rad": self.config.odometry_rotation_sigma_rad,
                "boundary_translation_sigma_m": self.config.boundary_translation_sigma_m,
                "boundary_rotation_sigma_rad": self.config.boundary_rotation_sigma_rad,
                "range_huber_k": self.config.robust_range_huber_k,
            },
            "sensor_extrinsic": {
                "frame_id": self.config.sensor_frame_id,
                "revision": self.config.extrinsic_revision,
                "translation_m": list(self.config.sensor_translation_m),
                "rotation_rpy_rad": list(self.config.sensor_rotation_rpy_rad),
            },
            "landmarks": {
                key: {
                    "position_m": list(value.position_m),
                    "prior_sigma_m": value.prior_sigma_m,
                }
                for key, value in self.config.landmarks.items()
            },
            "poses": pose_rows,
            "observations": observations,
            "loop_closures": loops,
            "latest_pose_covariance": (
                covariance.tolist() if covariance is not None else []
            ),
            "candidate_gate_passed": not gate_reasons,
            "gate_reasons": list(gate_reasons),
            **diagnostics,
        }

    def snapshot(self) -> Dict[str, object]:
        """Create an immutable, content-addressed certification snapshot."""
        payload = self._snapshot_payload()
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        snapshot_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return {
            **payload,
            "snapshot_id": snapshot_id,
            "created_monotonic": time.monotonic(),
            "created_wall_sec": time.time(),
        }

    def _diagnostic_values(self) -> Dict[str, object]:
        rows, residuals, synthetic = self._measurement_geometry()
        if rows:
            singular = np.linalg.svd(np.asarray(rows), compute_uv=False)
            tolerance = max(singular[0] * 1e-6, 1e-9)
            positive = singular[singular > tolerance]
            rank = int(len(positive))
            condition = float(positive[0] / positive[-1]) if len(positive) else math.inf
        else:
            rank, condition = 0, math.inf
        residual_rms = (
            float(math.sqrt(np.mean(np.square(residuals)))) if residuals else math.inf
        )
        latest = self.latest_pose()
        correction_translation = 0.0
        correction_rotation = 0.0
        if latest is not None and self.records:
            correction = self.records[-1].raw_pose.between(latest)
            correction_translation = float(np.linalg.norm(correction.translation()))
            correction_rotation = float(
                np.linalg.norm(gtsam.Rot3.Logmap(correction.rotation()))
            )
        return {
            "range_count": sum(len(record.measurements) for record in self.records),
            "loop_closure_count": len(self.active_loop_closures),
            "translational_rank": rank,
            "observability_condition": _finite_or_none(condition),
            "residual_rms_m": _finite_or_none(residual_rms),
            "correction_translation_m": correction_translation,
            "correction_rotation_rad": correction_rotation,
            "synthetic_evidence": synthetic,
        }

    def diagnostics(self, update_duration_ms: float = 0.0) -> EstimateDiagnostics:
        values = self._diagnostic_values()
        count = int(values["range_count"])
        rank = int(values["translational_rank"])
        condition = values["observability_condition"]
        residual_rms = values["residual_rms_m"]
        reasons = list(self._gate_reasons(values))
        snapshot = self.snapshot()
        return EstimateDiagnostics(
            epoch=self.epoch,
            revision=self.revision,
            snapshot_id=str(snapshot["snapshot_id"]),
            pose_count=len(self.records),
            range_count=count,
            loop_closure_count=int(values["loop_closure_count"]),
            rejected_observation_count=self.rejected_observation_count,
            rejected_loop_closure_count=self.rejected_loop_closure_count,
            translational_rank=rank,
            observability_condition=(
                float(condition) if condition is not None else math.inf
            ),
            residual_rms_m=(
                float(residual_rms) if residual_rms is not None else math.inf
            ),
            correction_translation_m=float(values["correction_translation_m"]),
            correction_rotation_rad=float(values["correction_rotation_rad"]),
            update_duration_ms=float(update_duration_ms),
            gate_passed=not reasons,
            gate_reasons=tuple(reasons),
            synthetic_evidence=bool(values["synthetic_evidence"]),
            covariance_model=COVARIANCE_MODEL,
            covariance_calibrated=False,
            last_reset_reason=self.last_reset_reason,
        )

    def _gate_reasons(self, values: Dict[str, object]) -> Tuple[str, ...]:
        count = int(values["range_count"])
        rank = int(values["translational_rank"])
        condition = values["observability_condition"]
        residual_rms = values["residual_rms_m"]
        reasons = []
        if count < self.config.min_measurements:
            reasons.append("insufficient_range_measurements")
        if rank < self.config.min_translational_rank:
            reasons.append("insufficient_translational_observability")
        if condition is None or condition > self.config.max_observability_condition:
            reasons.append("ill_conditioned_geometry")
        if residual_rms is None or residual_rms > self.config.max_range_residual_rms_m:
            reasons.append("range_residual_too_large")
        if (
            values["correction_translation_m"]
            > self.config.max_correction_translation_m
        ):
            reasons.append("translation_correction_too_large")
        if values["correction_rotation_rad"] > self.config.max_correction_rotation_rad:
            reasons.append("rotation_correction_too_large")
        if any(
            item.measurement.used_information_fallback
            for item in self.active_loop_closures
        ):
            reasons.append("loop_information_fallback_active")
        return tuple(reasons)
