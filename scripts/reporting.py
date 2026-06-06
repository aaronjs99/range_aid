"""Text result reporting."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.config import SimConfig
from scripts.plot_utils import sigma_radius_from_cov


def write_summary(path: Path, data) -> None:
    cfg: SimConfig = data["cfg"]
    b_true = data["b_true_pose"][:, :3]
    b_init = data["b_init_xyz"]
    b_est = data["b_est_pose"][:, :3]
    clean_ranges = data["clean_ranges"]
    init_rmse = float(np.sqrt(np.mean(np.sum((b_init - b_true) ** 2, axis=1))))
    est_rmse = float(np.sqrt(np.mean(np.sum((b_est - b_true) ** 2, axis=1))))
    range_rmse = float(
        np.sqrt(
            np.mean(
                (np.linalg.norm(b_est - data["a_pose"][:, :3], axis=1) - clean_ranges)
                ** 2
            )
        )
    )
    delta_est = b_est - data["a_pose"][:, :3]
    est_azimuths = np.arctan2(delta_est[:, 1], delta_est[:, 0])
    est_elevations = np.arctan2(
        delta_est[:, 2], np.linalg.norm(delta_est[:, :2], axis=1)
    )
    azimuth_rmse = float(
        np.sqrt(np.mean(wrap_angle(est_azimuths - data["true_azimuths"]) ** 2))
    )
    elevation_rmse = float(
        np.sqrt(np.mean((est_elevations - data["true_elevations"]) ** 2))
    )
    depth_rmse = float(np.sqrt(np.mean((b_est[:, 2] - data["true_depths"]) ** 2)))
    smoothness_rms = float(
        np.sqrt(
            np.mean(
                np.sum(
                    ((b_est[2:] - 2.0 * b_est[1:-1] + b_est[:-2]) / (cfg.dt**2)) ** 2,
                    axis=1,
                )
            )
        )
    )
    mean_2sigma = float(np.mean([sigma_radius_from_cov(c) for c in data["b_cov_xyz"]]))

    path.write_text(
        "\n".join(
            [
                "3D moving range-aided pose estimation demo",
                "",
                "Setup:",
                "- Object A pose is known from a DLiO-like VLP-16 + Xsens stack.",
                "- Object B is a moving underwater handheld-sonar/diver-carried target.",
                "- Measurements are A pose plus enabled diver range/USBL/depth factors.",
                "- The range residual at time i is ||B_i - A_i|| - measured_range_i.",
                f"- smoothness factor enabled: {cfg.use_smoothness_factor}",
                f"- depth factor enabled: {cfg.use_depth_factor}",
                f"- USBL angle factors enabled: {cfg.use_usbl_angles}",
                "- The optimizer estimates B_i position per timestamp with the active factor set.",
                "- Range alone does not estimate attitude; outputs focus on position.",
                "",
                f"steps: {cfg.steps}",
                f"sensor: {cfg.sensor_name}",
                f"frequency_band_khz: {cfg.frequency_band_khz}",
                f"max_range_m: {cfg.max_range_m}",
                f"depth_rating_m: {cfg.depth_rating_m}",
                f"acoustic_coverage_deg: {cfg.acoustic_coverage_deg}",
                f"range_sigma_m: {cfg.range_sigma_m}",
                f"angular_accuracy_deg: {np.rad2deg(cfg.angular_accuracy_rad):.2f}",
                f"smoothness_accel_sigma_mps2: {cfg.smoothness_accel_sigma_mps2}",
                f"depth_sigma_m: {cfg.depth_sigma_m}",
                f"initial_B_first_position_m: {tuple(data['b_init_xyz'][0])}",
                f"estimated_B_first_position_m: {tuple(data['b_est_pose'][0, :3])}",
                f"max_clean_range_m: {float(np.max(clean_ranges)):.4f}",
                f"max_abs_B_depth_m: {float(np.max(np.abs(b_true[:, 2]))):.4f}",
                f"A position sigma xyz m: {cfg.a_position_sigma_m}",
                f"A orientation sigma rpy deg: {tuple(np.rad2deg(cfg.a_orientation_sigma_rad))}",
                f"initial trajectory RMSE: {init_rmse:.4f} m",
                f"range-factor trajectory RMSE: {est_rmse:.4f} m",
                f"range consistency RMSE: {range_rmse:.4f} m",
                f"azimuth consistency RMSE deg: {np.rad2deg(azimuth_rmse):.4f}",
                f"elevation consistency RMSE deg: {np.rad2deg(elevation_rmse):.4f}",
                f"depth consistency RMSE: {depth_rmse:.4f} m",
                f"smoothness acceleration RMS: {smoothness_rms:.4f} m/s^2",
                f"mean B 2-sigma position sphere radius: {mean_2sigma:.4f} m",
                f"optimizer success: {data['optimizer'].success}",
                f"optimizer cost: {data['optimizer'].cost:.4f}",
                f"optimizer iterations: {data['optimizer'].nfev}",
                "",
            ]
        )
    )


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi
