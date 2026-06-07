"""Shared boat-mounted USBL measurement helpers."""

from __future__ import annotations

import numpy as np

from scripts.configuration.config import SimConfig
from scripts.math.geometry import (
    relative_vectors_in_sensor_frame,
    sensor_positions_from_boat_poses,
    spherical_from_sensor_vectors,
)


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    """Wrap angle arrays to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def predict_usbl_observations(
    a_pose: np.ndarray,
    b_xyz: np.ndarray,
    cfg: SimConfig,
    *,
    use_true_mount: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict boat-mounted USBL range, azimuth, and elevation measurements."""
    mount_offset = np.array(cfg.boat_usbl_mount_offset_m)
    mount_rpy = np.array(cfg.boat_usbl_mount_rpy_rad)
    if use_true_mount:
        mount_offset = mount_offset + np.array(cfg.usbl_mount_bias_m)
        mount_rpy = mount_rpy + np.array(cfg.usbl_mount_bias_rpy_rad)
    q_sensor = relative_vectors_in_sensor_frame(a_pose, b_xyz, mount_offset, mount_rpy)
    return spherical_from_sensor_vectors(q_sensor)


def estimate_position_covariance(
    b_est_xyz: np.ndarray, a_pose: np.ndarray, cfg: SimConfig
) -> np.ndarray:
    """Approximate local position covariance for display."""
    cov_blocks = np.zeros((len(b_est_xyz), 3, 3))
    radial_sigma2 = cfg.range_sigma_m**2
    sensor_positions = sensor_positions_from_boat_poses(
        a_pose,
        np.array(cfg.boat_usbl_mount_offset_m),
        np.array(cfg.boat_usbl_mount_rpy_rad),
    )
    for k, point in enumerate(b_est_xyz):
        if not np.all(np.isfinite(point)):
            cov_blocks[k] = np.full((3, 3), np.nan)
            continue
        radial = point - sensor_positions[k]
        norm = np.linalg.norm(radial)
        if norm < 1e-9:
            cov_blocks[k] = np.eye(3) * cfg.tangent_uncertainty_sigma_m**2
            continue
        u = radial / norm
        radial_projector = np.outer(u, u)
        tangent_projector = np.eye(3) - radial_projector
        tangent_sigma = (
            max(norm * cfg.angular_accuracy_rad, cfg.range_sigma_m)
            if cfg.use_usbl_angles
            else cfg.tangent_uncertainty_sigma_m
        )
        cov_blocks[k] = (
            radial_sigma2 * radial_projector + tangent_sigma**2 * tangent_projector
        )
        if cfg.use_depth_factor:
            cov_blocks[k, 2, 2] = min(cov_blocks[k, 2, 2], cfg.depth_sigma_m**2)
    return cov_blocks
