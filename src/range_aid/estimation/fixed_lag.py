"""Bounded iSAM2 range-aided smoother for known acoustic landmarks.

The GTSAM build on IG Handle does not expose ``IncrementalFixedLagSmoother``
through Python. This implementation therefore updates iSAM2 incrementally
inside the active window and rebuilds the graph whenever old states leave the
configured lag. The rebuilt boundary receives a conservative prior centered on
the previous estimate, which keeps computation and memory bounded without
claiming exact Bayes-tree marginalization.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import time
from typing import Deque, Dict, List, Optional, Tuple

import gtsam
import numpy as np

from range_aid.models.config import OnlineConfig


@dataclass(frozen=True)
class RangeMeasurement:
    """One validated range observation associated with a graph pose."""

    stamp_sec: float
    landmark_id: str
    range_m: float
    sigma_m: float
    synthetic: bool
    source: str


@dataclass
class PoseRecord:
    """Raw odometry pose and range measurements retained in the active lag."""

    index: int
    stamp_sec: float
    raw_pose: gtsam.Pose3
    measurements: List[RangeMeasurement] = field(default_factory=list)


@dataclass(frozen=True)
class EstimateDiagnostics:
    """Inspectable estimator output used by ROS status and promotion gates."""

    epoch: int
    pose_count: int
    range_count: int
    rejected_observation_count: int
    translational_rank: int
    observability_condition: float
    residual_rms_m: float
    correction_translation_m: float
    correction_rotation_rad: float
    update_duration_ms: float
    gate_passed: bool
    gate_reasons: Tuple[str, ...]
    synthetic_evidence: bool


def pose3_from_components(
    position_xyz: Tuple[float, float, float],
    quaternion_wxyz: Tuple[float, float, float, float],
) -> gtsam.Pose3:
    """Construct a GTSAM pose from ROS-compatible position and quaternion data."""
    w, x, y, z = quaternion_wxyz
    return gtsam.Pose3(gtsam.Rot3.Quaternion(w, x, y, z), np.asarray(position_xyz))


def pose3_to_components(pose: gtsam.Pose3) -> Tuple[np.ndarray, np.ndarray]:
    """Return position xyz and quaternion wxyz from a GTSAM pose."""
    quaternion = pose.rotation().toQuaternion()
    return (
        np.asarray(pose.translation(), dtype=float),
        np.asarray(
            [quaternion.w(), quaternion.x(), quaternion.y(), quaternion.z()],
            dtype=float,
        ),
    )


class RebuildingFixedLagSmoother:
    """Incremental range-aided pose graph with a bounded active time window."""

    def __init__(self, config: OnlineConfig) -> None:
        self.config = config
        self.records: Deque[PoseRecord] = deque()
        self.rejected_observation_count = 0
        self.epoch = 0
        self._next_index = 0
        self._isam = self._new_isam()
        self._landmark_keys = {
            landmark_id: gtsam.symbol("l", index)
            for index, landmark_id in enumerate(sorted(config.landmarks))
        }
        self._sensor_pose = self._sensor_extrinsic_pose()
        self._initialize_landmarks()

    @staticmethod
    def _new_isam() -> gtsam.ISAM2:
        params = gtsam.ISAM2Params()
        params.setRelinearizeThreshold(0.01)
        params.relinearizeSkip = 1
        return gtsam.ISAM2(params)

    def _sensor_extrinsic_pose(self) -> gtsam.Pose3:
        roll, pitch, yaw = self.config.sensor_rotation_rpy_rad
        rotation = gtsam.Rot3.RzRyRx(roll, pitch, yaw)
        return gtsam.Pose3(rotation, np.asarray(self.config.sensor_translation_m))

    def _pose_key(self, index: int) -> int:
        return gtsam.symbol("x", index)

    def _pose_noise(self, translation_sigma: float, rotation_sigma: float):
        return gtsam.noiseModel.Diagonal.Sigmas(
            np.asarray([rotation_sigma] * 3 + [translation_sigma] * 3)
        )

    def _initialize_landmarks(self) -> None:
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        for landmark_id, landmark in self.config.landmarks.items():
            key = self._landmark_keys[landmark_id]
            landmark_pose = gtsam.Pose3(
                gtsam.Rot3(), np.asarray(landmark.position_m, dtype=float)
            )
            values.insert(key, landmark_pose)
            graph.add(
                gtsam.PriorFactorPose3(
                    key,
                    landmark_pose,
                    self._pose_noise(landmark.prior_sigma_m, 1e-4),
                )
            )
        self._isam.update(graph, values)

    def add_odometry(self, stamp_sec: float, raw_pose: gtsam.Pose3) -> EstimateDiagnostics:
        """Append one odometry state, prune the lag, and update iSAM2."""
        started = time.perf_counter()
        index = self._next_index
        self._next_index += 1
        record = PoseRecord(index=index, stamp_sec=float(stamp_sec), raw_pose=raw_pose)
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        key = self._pose_key(index)
        values.insert(key, raw_pose)
        if not self.records:
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
        else:
            previous = self.records[-1]
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
        self.records.append(record)
        self._isam.update(graph, values)
        self.epoch += 1
        if self._prune_required(stamp_sec):
            self._prune_and_rebuild(stamp_sec)
        return self.diagnostics((time.perf_counter() - started) * 1000.0)

    def _prune_required(self, latest_stamp_sec: float) -> bool:
        if len(self.records) > self.config.max_pose_count:
            return True
        return bool(
            self.records
            and latest_stamp_sec - self.records[0].stamp_sec > self.config.lag_sec
        )

    def _prune_and_rebuild(self, latest_stamp_sec: float) -> None:
        prior_estimates = self._current_pose_estimates()
        while len(self.records) > 1 and (
            len(self.records) > self.config.max_pose_count
            or latest_stamp_sec - self.records[0].stamp_sec > self.config.lag_sec
        ):
            self.records.popleft()
        self._isam = self._new_isam()
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        for landmark_id, landmark in self.config.landmarks.items():
            key = self._landmark_keys[landmark_id]
            landmark_pose = gtsam.Pose3(
                gtsam.Rot3(), np.asarray(landmark.position_m, dtype=float)
            )
            values.insert(key, landmark_pose)
            graph.add(
                gtsam.PriorFactorPose3(
                    key,
                    landmark_pose,
                    self._pose_noise(landmark.prior_sigma_m, 1e-4),
                )
            )
        for position, record in enumerate(self.records):
            key = self._pose_key(record.index)
            initial = prior_estimates.get(record.index, record.raw_pose)
            values.insert(key, initial)
            if position == 0:
                graph.add(
                    gtsam.PriorFactorPose3(
                        key,
                        initial,
                        self._pose_noise(
                            self.config.boundary_translation_sigma_m,
                            self.config.boundary_rotation_sigma_rad,
                        ),
                    )
                )
            else:
                previous = self.records[position - 1]
                graph.add(
                    gtsam.BetweenFactorPose3(
                        self._pose_key(previous.index),
                        key,
                        previous.raw_pose.between(record.raw_pose),
                        self._pose_noise(
                            self.config.odometry_translation_sigma_m,
                            self.config.odometry_rotation_sigma_rad,
                        ),
                    )
                )
            for measurement in record.measurements:
                graph.add(self._range_factor(key, measurement))
        self._isam.update(graph, values)
        self.epoch += 1

    def _current_pose_estimates(self) -> Dict[int, gtsam.Pose3]:
        values = self._isam.calculateEstimate()
        estimates = {}
        for record in self.records:
            key = self._pose_key(record.index)
            if values.exists(key):
                estimates[record.index] = values.atPose3(key)
        return estimates

    def _range_factor(self, pose_key: int, measurement: RangeMeasurement):
        base = gtsam.noiseModel.Isotropic.Sigma(1, measurement.sigma_m)
        robust = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(
                self.config.robust_range_huber_k
            ),
            base,
        )
        return gtsam.RangeFactorWithTransformPose3(
            pose_key,
            self._landmark_keys[measurement.landmark_id],
            measurement.range_m,
            robust,
            self._sensor_pose,
        )

    def add_range(self, measurement: RangeMeasurement) -> bool:
        """Associate and add a range factor; return false for rejected input."""
        if (
            measurement.landmark_id not in self._landmark_keys
            or not math.isfinite(measurement.range_m)
            or not self.config.min_range_m <= measurement.range_m <= self.config.max_range_m
            or not math.isfinite(measurement.sigma_m)
            or measurement.sigma_m <= 0.0
            or not self.records
        ):
            self.rejected_observation_count += 1
            return False
        record = min(
            self.records, key=lambda item: abs(item.stamp_sec - measurement.stamp_sec)
        )
        if abs(record.stamp_sec - measurement.stamp_sec) > self.config.max_sync_error_sec:
            self.rejected_observation_count += 1
            return False
        record.measurements.append(measurement)
        graph = gtsam.NonlinearFactorGraph()
        graph.add(self._range_factor(self._pose_key(record.index), measurement))
        self._isam.update(graph, gtsam.Values())
        self.epoch += 1
        return True

    def latest_pose(self) -> Optional[gtsam.Pose3]:
        """Return the newest corrected pose, if the graph has a pose state."""
        if not self.records:
            return None
        return self._isam.calculateEstimate().atPose3(
            self._pose_key(self.records[-1].index)
        )

    def latest_covariance(self) -> Optional[np.ndarray]:
        """Return the newest Pose3 marginal covariance when available."""
        if not self.records:
            return None
        try:
            return np.asarray(
                self._isam.marginalCovariance(self._pose_key(self.records[-1].index)),
                dtype=float,
            )
        except RuntimeError:
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

    def diagnostics(self, update_duration_ms: float = 0.0) -> EstimateDiagnostics:
        """Evaluate observability, residual, and bounded-correction gates."""
        rows, residuals, synthetic = self._measurement_geometry()
        if rows:
            singular = np.linalg.svd(np.asarray(rows), compute_uv=False)
            tolerance = max(singular[0] * 1e-6, 1e-9)
            rank = int(np.sum(singular > tolerance))
            positive = singular[singular > tolerance]
            condition = float(positive[0] / positive[-1]) if len(positive) else math.inf
        else:
            rank, condition = 0, math.inf
        residual_rms = (
            float(math.sqrt(np.mean(np.square(residuals)))) if residuals else math.inf
        )
        correction_translation = 0.0
        correction_rotation = 0.0
        latest = self.latest_pose()
        if latest is not None and self.records:
            correction = self.records[-1].raw_pose.between(latest)
            correction_translation = float(np.linalg.norm(correction.translation()))
            correction_rotation = float(
                np.linalg.norm(gtsam.Rot3.Logmap(correction.rotation()))
            )
        count = sum(len(record.measurements) for record in self.records)
        reasons = []
        if count < self.config.min_measurements:
            reasons.append("insufficient_range_measurements")
        if rank < self.config.min_translational_rank:
            reasons.append("insufficient_translational_observability")
        if condition > self.config.max_observability_condition:
            reasons.append("ill_conditioned_geometry")
        if residual_rms > self.config.max_range_residual_rms_m:
            reasons.append("range_residual_too_large")
        if correction_translation > self.config.max_correction_translation_m:
            reasons.append("translation_correction_too_large")
        if correction_rotation > self.config.max_correction_rotation_rad:
            reasons.append("rotation_correction_too_large")
        return EstimateDiagnostics(
            epoch=self.epoch,
            pose_count=len(self.records),
            range_count=count,
            rejected_observation_count=self.rejected_observation_count,
            translational_rank=rank,
            observability_condition=condition,
            residual_rms_m=residual_rms,
            correction_translation_m=correction_translation,
            correction_rotation_rad=correction_rotation,
            update_duration_ms=float(update_duration_ms),
            gate_passed=not reasons,
            gate_reasons=tuple(reasons),
            synthetic_evidence=synthetic,
        )

    def snapshot(self) -> Dict[str, object]:
        """Create an immutable, serialization-friendly certification snapshot."""
        estimates = self._current_pose_estimates()
        diagnostics = self.diagnostics()
        covariance = self.latest_covariance()
        observations = []
        latest_position = []
        latest_quaternion = []
        latest_stamp_sec = 0.0
        for record in self.records:
            pose = estimates.get(record.index)
            if pose is None:
                continue
            position, quaternion = pose3_to_components(pose)
            latest_position = position.tolist()
            latest_quaternion = quaternion.tolist()
            latest_stamp_sec = record.stamp_sec
            sensor_position = np.asarray(
                pose.compose(self._sensor_pose).translation(), dtype=float
            )
            for measurement in record.measurements:
                observations.append(
                    {
                        "stamp_sec": measurement.stamp_sec,
                        "pose_position_m": position.tolist(),
                        "pose_quaternion_wxyz": quaternion.tolist(),
                        "sensor_position_m": sensor_position.tolist(),
                        "landmark_id": measurement.landmark_id,
                        "range_m": measurement.range_m,
                        "sigma_m": measurement.sigma_m,
                        "synthetic": measurement.synthetic,
                    }
                )
        return {
            "epoch": self.epoch,
            "created_monotonic": time.monotonic(),
            "sensor_translation_m": list(self.config.sensor_translation_m),
            "sensor_rotation_rpy_rad": list(self.config.sensor_rotation_rpy_rad),
            "latest_pose_position_m": latest_position,
            "latest_pose_quaternion_wxyz": latest_quaternion,
            "latest_stamp_sec": latest_stamp_sec,
            "latest_pose_covariance": (
                covariance.tolist() if covariance is not None else []
            ),
            "range_count": diagnostics.range_count,
            "translational_rank": diagnostics.translational_rank,
            "observability_condition": diagnostics.observability_condition,
            "residual_rms_m": diagnostics.residual_rms_m,
            "candidate_gate_passed": diagnostics.gate_passed,
            "gate_reasons": list(diagnostics.gate_reasons),
            "synthetic_evidence": diagnostics.synthetic_evidence,
            "landmarks": {
                key: list(value.position_m) for key, value in self.config.landmarks.items()
            },
            "observations": observations,
        }
