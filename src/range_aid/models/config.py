"""Validated configuration for shadow range-aided estimation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import yaml


@dataclass(frozen=True)
class LandmarkConfig:
    """One map-frame acoustic landmark with a Gaussian position prior."""

    landmark_id: str
    position_m: Tuple[float, float, float]
    prior_sigma_m: float


@dataclass(frozen=True)
class OnlineConfig:
    """Bounded smoother, synchronization, and promotion-gate settings."""

    lag_sec: float
    max_pose_count: int
    state_rate_hz: float
    max_sync_error_sec: float
    odometry_translation_sigma_m: float
    odometry_rotation_sigma_rad: float
    boundary_translation_sigma_m: float
    boundary_rotation_sigma_rad: float
    range_default_sigma_m: float
    robust_range_huber_k: float
    min_range_m: float
    max_range_m: float
    min_measurements: int
    min_translational_rank: int
    max_observability_condition: float
    max_range_residual_rms_m: float
    max_correction_translation_m: float
    max_correction_rotation_rad: float
    certification_period_sec: float
    certification_max_age_sec: float
    require_tight_certification: bool
    certification_solver: str
    sensor_translation_m: Tuple[float, float, float]
    sensor_rotation_rpy_rad: Tuple[float, float, float]
    landmarks: Dict[str, LandmarkConfig]


def _triple(value, key: str) -> Tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("{} must contain three numbers".format(key))
    return tuple(float(item) for item in value)


def load_online_config(path: Path) -> OnlineConfig:
    """Load and validate the online shadow-estimator configuration."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("{} must use schema_version 1".format(path))
    smoother = dict(payload.get("smoother", {}) or {})
    noise = dict(payload.get("noise", {}) or {})
    gates = dict(payload.get("gates", {}) or {})
    certification = dict(payload.get("certification", {}) or {})
    sensor = dict(payload.get("sensor_extrinsic", {}) or {})
    raw_landmarks = dict(payload.get("landmarks", {}) or {})
    if not raw_landmarks:
        raise ValueError("{} must define at least one landmark".format(path))
    landmarks = {
        str(name): LandmarkConfig(
            landmark_id=str(name),
            position_m=_triple(entry.get("position_m"), "landmarks.{}.position_m".format(name)),
            prior_sigma_m=float(entry.get("prior_sigma_m", 0.05)),
        )
        for name, entry in raw_landmarks.items()
    }
    config = OnlineConfig(
        lag_sec=float(smoother.get("lag_sec", 12.0)),
        max_pose_count=int(smoother.get("max_pose_count", 240)),
        state_rate_hz=float(smoother.get("state_rate_hz", 10.0)),
        max_sync_error_sec=float(smoother.get("max_sync_error_sec", 0.15)),
        odometry_translation_sigma_m=float(noise.get("odometry_translation_sigma_m", 0.08)),
        odometry_rotation_sigma_rad=float(noise.get("odometry_rotation_sigma_rad", 0.035)),
        boundary_translation_sigma_m=float(noise.get("boundary_translation_sigma_m", 0.15)),
        boundary_rotation_sigma_rad=float(noise.get("boundary_rotation_sigma_rad", 0.07)),
        range_default_sigma_m=float(noise.get("range_default_sigma_m", 0.20)),
        robust_range_huber_k=float(noise.get("robust_range_huber_k", 1.5)),
        min_range_m=float(gates.get("min_range_m", 0.5)),
        max_range_m=float(gates.get("max_range_m", 1000.0)),
        min_measurements=int(gates.get("min_measurements", 6)),
        min_translational_rank=int(gates.get("min_translational_rank", 2)),
        max_observability_condition=float(gates.get("max_observability_condition", 100.0)),
        max_range_residual_rms_m=float(gates.get("max_range_residual_rms_m", 0.75)),
        max_correction_translation_m=float(gates.get("max_correction_translation_m", 2.0)),
        max_correction_rotation_rad=float(gates.get("max_correction_rotation_rad", 0.35)),
        certification_period_sec=float(certification.get("period_sec", 5.0)),
        certification_max_age_sec=float(certification.get("max_age_sec", 15.0)),
        require_tight_certification=bool(
            certification.get("require_tight_for_candidate", True)
        ),
        certification_solver=str(certification.get("solver", "SCS") or "SCS"),
        sensor_translation_m=_triple(sensor.get("translation_m", [0.6, 0.0, -0.75]), "sensor_extrinsic.translation_m"),
        sensor_rotation_rpy_rad=_triple(sensor.get("rotation_rpy_rad", [0.0, 0.0, 0.0]), "sensor_extrinsic.rotation_rpy_rad"),
        landmarks=landmarks,
    )
    if config.lag_sec <= 0.0 or config.max_pose_count < 2 or config.state_rate_hz <= 0.0:
        raise ValueError("smoother lag and pose count must be positive")
    if config.min_measurements < 1 or config.min_translational_rank not in (1, 2, 3):
        raise ValueError("invalid observability gate configuration")
    return config
