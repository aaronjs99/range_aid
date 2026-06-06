"""Moving-target range-aided estimation."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from scripts.config import SimConfig
from scripts.simulation import make_known_a_pose, make_true_b_pose


def initial_guess(a_xyz: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """One plausible B_i path used only to start optimization."""
    guess = np.zeros_like(a_xyz)
    guess[0] = a_xyz[0] + np.array(cfg.initial_guess_offset_m)
    velocity = np.array(cfg.initial_guess_velocity_mps)
    for k in range(1, len(guess)):
        guess[k] = guess[k - 1] + velocity * cfg.dt
    return guess


def range_residuals(
    flat_b_xyz: np.ndarray,
    a_xyz: np.ndarray,
    measured_ranges: np.ndarray,
    measured_azimuths: np.ndarray,
    measured_elevations: np.ndarray,
    measured_depths: np.ndarray,
    cfg: SimConfig,
) -> np.ndarray:
    b_xyz = flat_b_xyz.reshape((-1, 3))
    delta = b_xyz - a_xyz
    horizontal = np.linalg.norm(delta[:, :2], axis=1)
    predicted_ranges = np.linalg.norm(delta, axis=1)

    residual_blocks = [
        (predicted_ranges - measured_ranges) / cfg.range_sigma_m,
    ]
    if cfg.use_usbl_angles:
        predicted_azimuths = np.arctan2(delta[:, 1], delta[:, 0])
        predicted_elevations = np.arctan2(delta[:, 2], horizontal)
        residual_blocks.extend(
            [
                wrap_angle(predicted_azimuths - measured_azimuths)
                / cfg.angular_accuracy_rad,
                (predicted_elevations - measured_elevations) / cfg.angular_accuracy_rad,
            ]
        )
    if cfg.use_depth_factor:
        residual_blocks.append((b_xyz[:, 2] - measured_depths) / cfg.depth_sigma_m)
    if cfg.use_smoothness_factor and len(b_xyz) >= 3:
        accel = (b_xyz[2:] - 2.0 * b_xyz[1:-1] + b_xyz[:-2]) / (cfg.dt**2)
        residual_blocks.append((accel / cfg.smoothness_accel_sigma_mps2).ravel())
    return np.concatenate(residual_blocks)


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def estimate_position_covariance(
    b_est_xyz: np.ndarray, a_xyz: np.ndarray, cfg: SimConfig
) -> np.ndarray:
    """Approximate local position covariance for display.

    Range constrains the radial direction. USBL angle factors constrain tangent
    directions with uncertainty increasing with range. If angles are disabled,
    tangent directions remain weakly observable and use the configured display
    sigma. If depth is enabled, vertical variance is capped by the depth sigma.
    """
    cov_blocks = np.zeros((len(b_est_xyz), 3, 3))
    radial_sigma2 = cfg.range_sigma_m**2
    for k, point in enumerate(b_est_xyz):
        radial = point - a_xyz[k]
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


def simulate_and_estimate(cfg: SimConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)
    t = cfg.dt * np.arange(cfg.steps)
    a_pose = make_known_a_pose(t)
    b_true_pose = make_true_b_pose(t, cfg)

    a_xyz = a_pose[:, :3]
    b_true_xyz = b_true_pose[:, :3]
    clean_ranges = np.linalg.norm(b_true_xyz - a_xyz, axis=1)
    if np.any(clean_ranges > cfg.max_range_m):
        raise ValueError(
            f"simulated A-B range exceeds {cfg.sensor_name} max range "
            f"({cfg.max_range_m:.1f} m)"
        )
    if np.max(np.abs(b_true_xyz[:, 2])) > cfg.depth_rating_m:
        raise ValueError(
            f"simulated B depth exceeds {cfg.sensor_name} depth rating "
            f"({cfg.depth_rating_m:.1f} m)"
        )

    delta_true = b_true_xyz - a_xyz
    true_azimuths = np.arctan2(delta_true[:, 1], delta_true[:, 0])
    true_elevations = np.arctan2(
        delta_true[:, 2], np.linalg.norm(delta_true[:, :2], axis=1)
    )
    true_depths = b_true_xyz[:, 2]
    measured_ranges = clean_ranges + rng.normal(0.0, cfg.range_sigma_m, cfg.steps)
    measured_azimuths = wrap_angle(
        true_azimuths + rng.normal(0.0, cfg.angular_accuracy_rad, cfg.steps)
    )
    measured_elevations = true_elevations + rng.normal(
        0.0, cfg.angular_accuracy_rad, cfg.steps
    )
    measured_depths = true_depths + rng.normal(0.0, cfg.depth_sigma_m, cfg.steps)
    b_init_xyz = initial_guess(a_xyz, cfg)
    result = least_squares(
        range_residuals,
        b_init_xyz.ravel(),
        args=(
            a_xyz,
            measured_ranges,
            measured_azimuths,
            measured_elevations,
            measured_depths,
            cfg,
        ),
        x_scale="jac",
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        max_nfev=1000,
    )
    b_est_xyz = result.x.reshape((-1, 3))
    b_cov_xyz = estimate_position_covariance(b_est_xyz, a_xyz, cfg)
    b_est_pose = np.column_stack([b_est_xyz, np.zeros((cfg.steps, 3))])

    return {
        "cfg": cfg,
        "t": t,
        "a_pose": a_pose,
        "b_true_pose": b_true_pose,
        "b_init_xyz": b_init_xyz,
        "b_est_pose": b_est_pose,
        "b_cov_xyz": b_cov_xyz,
        "measured_ranges": measured_ranges,
        "measured_azimuths": measured_azimuths,
        "measured_elevations": measured_elevations,
        "measured_depths": measured_depths,
        "clean_ranges": clean_ranges,
        "true_azimuths": true_azimuths,
        "true_elevations": true_elevations,
        "true_depths": true_depths,
        "optimizer": result,
    }
