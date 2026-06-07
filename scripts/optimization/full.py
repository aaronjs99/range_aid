"""Moving-target range-aided estimation."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from scripts.configuration.config import SimConfig
from scripts.math.usbl import (
    estimate_position_covariance,
    predict_usbl_observations,
    wrap_angle,
)
from scripts.optimization.common import initial_guess
from scripts.sim.trajectories import (
    make_estimated_a_pose,
    make_known_a_pose,
    make_true_b_pose,
)


def usbl_smoothness_residuals(
    flat_b_xyz: np.ndarray,
    a_pose: np.ndarray,
    measured_ranges: np.ndarray,
    measured_azimuths: np.ndarray,
    measured_elevations: np.ndarray,
    measured_depths: np.ndarray,
    cfg: SimConfig,
) -> np.ndarray:
    b_xyz = flat_b_xyz.reshape((-1, 3))
    predicted_ranges, predicted_azimuths, predicted_elevations = (
        predict_usbl_observations(a_pose, b_xyz, cfg)
    )

    residual_blocks = [
        (predicted_ranges - measured_ranges) / cfg.range_sigma_m,
    ]
    if cfg.use_usbl_angles:
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


def simulate_and_estimate(cfg: SimConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)
    t = cfg.dt * np.arange(cfg.steps)
    a_true_pose = make_known_a_pose(t)
    a_pose = make_estimated_a_pose(a_true_pose, cfg, rng)
    b_true_pose = make_true_b_pose(t, cfg)

    a_xyz = a_pose[:, :3]
    b_true_xyz = b_true_pose[:, :3]
    clean_ranges, true_azimuths, true_elevations = predict_usbl_observations(
        a_true_pose, b_true_xyz, cfg, use_true_mount=True
    )
    if np.any(clean_ranges > cfg.max_range_m):
        raise ValueError(
            f"simulated USBL-transponder range exceeds {cfg.boat_usbl_name} "
            f"and {cfg.target_transponder_name} max range "
            f"({cfg.max_range_m:.1f} m)"
        )
    if np.max(np.abs(b_true_xyz[:, 2])) > cfg.target_depth_rating_m:
        raise ValueError(
            f"simulated B depth exceeds {cfg.target_transponder_name} depth rating "
            f"({cfg.target_depth_rating_m:.1f} m)"
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
        usbl_smoothness_residuals,
        b_init_xyz.ravel(),
        args=(
            a_pose,
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
    b_cov_xyz = estimate_position_covariance(b_est_xyz, a_pose, cfg)
    b_est_pose = np.column_stack([b_est_xyz, np.zeros((cfg.steps, 3))])

    return {
        "cfg": cfg,
        "t": t,
        "a_true_pose": a_true_pose,
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
