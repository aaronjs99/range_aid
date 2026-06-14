"""Configuration loading for the range-aided pose estimation demo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml


@dataclass(frozen=True)
class SimConfig:
    steps: int
    dt: float
    seed: int
    optimizer_backend: str
    initial_guess_offset_m: tuple[float, float, float]
    initial_guess_velocity_mps: tuple[float, float, float]
    use_smoothness_factor: bool
    smoothness_accel_sigma_mps2: float
    use_depth_factor: bool
    depth_sigma_m: float
    use_usbl_angles: bool
    tangent_uncertainty_sigma_m: float
    cora_window_size: int
    cora_solve_stride: int
    cora_anchor_mode: str
    cora_bound_xy_m: float
    cora_bound_z_boat_m: float
    cora_bound_z_target_min_m: float
    cora_bound_z_target_max_m: float
    cora_second_moment_bounds: bool
    cora_range_slack_weight: float
    cora_boat_prior_weight: float
    cora_boat_displacement_weight: float
    cora_surface_prior_weight: float
    cora_surface_z_m: float
    cora_surface_sigma_m: float
    cora_rank_tightness_tol: float
    cora_sdp_feasibility_tol: float
    cora_sdp_objective_tol: float
    cora_sdp_psd_tol: float
    cora_refine_with_full: bool
    cora_solver: str
    usbl_mount_bias_m: tuple[float, float, float]
    usbl_mount_bias_rpy_rad: tuple[float, float, float]
    usbl_mount_prior_sigma_m: float
    usbl_mount_prior_sigma_rad: tuple[float, float, float]
    boat_usbl_name: str
    boat_usbl_role: str
    boat_usbl_frequency_band_khz: tuple[float, float]
    boat_usbl_max_range_m: float
    boat_usbl_depth_rating_m: float
    boat_usbl_acoustic_coverage_deg: float
    boat_usbl_update_rate_hz: float
    boat_usbl_weight_air_kg: float
    boat_usbl_weight_water_kg: float
    boat_usbl_typical_power_w: float
    boat_usbl_max_power_w: float
    boat_usbl_mount_offset_m: tuple[float, float, float]
    boat_usbl_mount_rpy_rad: tuple[float, float, float]
    target_transponder_name: str
    target_transponder_role: str
    target_transponder_frequency_band_khz: tuple[float, float]
    target_transponder_max_range_m: float
    target_transponder_depth_rating_m: float
    target_transponder_beam_shape_deg: float
    target_transponder_range_precision_m: float
    target_transponder_weight_air_kg: float
    target_transponder_weight_water_kg: float
    max_range_m: float
    target_depth_rating_m: float
    acoustic_coverage_deg: float
    range_sigma_m: float
    angular_accuracy_rad: float
    a_position_sigma_m: tuple[float, float, float]
    a_orientation_sigma_rad: tuple[float, float, float]
    stationary_ping_windows: tuple[tuple[int, int], ...]
    b_event_start_offset_m: tuple[float, float, float]
    b_event_stationary_position_m: tuple[float, float, float]
    b_event_end_position_m: tuple[float, float, float]
    b_start_position_m: tuple[float, float, float]
    b_velocity_mps: tuple[float, float, float]
    b_motion_amplitude_m: tuple[float, float, float]
    b_motion_frequency_radps: tuple[float, float, float]
    b_motion_phase_rad: tuple[float, float, float]


@dataclass(frozen=True)
class OutputConfig:
    output_dir: Path
    summary_file: str
    summary_plot_file: str
    position_plot_file: str
    video_file: str


def _tuple(values: Iterable[float]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _windows(
    values: Iterable[Iterable[int]], steps: int
) -> tuple[tuple[int, int], ...]:
    windows = []
    for raw_start, raw_end in values:
        start = int(raw_start)
        end = int(raw_end)
        if start < 0 or end > steps or start >= end:
            raise ValueError(
                f"invalid stationary ping window [{start}, {end}) for {steps} steps"
            )
        windows.append((start, end))
    if not windows:
        raise ValueError("at least one stationary ping window is required")
    return tuple(windows)


def load_config(path: Path) -> tuple[SimConfig, OutputConfig]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping: {path}")

    simulation = raw["simulation"]
    estimator = raw["estimator"]
    sensors = raw["sensors"]
    boat_usbl = sensors.get("boat", {}).get("usbl")
    target_transponder = sensors.get("target", {}).get("transponder", {})
    if boat_usbl is None:
        boat_usbl = sensors["diver"]["range"]
    covariance = raw["covariance"]
    scenario = raw["scenario"]
    output = raw["output"]

    usbl_weight_air = boat_usbl.get("weight_air_kg")
    usbl_weight_water = boat_usbl.get("weight_water_kg")
    if usbl_weight_air is None or usbl_weight_water is None:
        weight_pair = boat_usbl.get("weight_air_water_kg", [0.0, 0.0])
        usbl_weight_air = weight_pair[0]
        usbl_weight_water = weight_pair[1]
    usbl_typical_power = boat_usbl.get(
        "typical_power_w", boat_usbl.get("electrical", {}).get("typical_power_w", 0.0)
    )
    usbl_max_power = boat_usbl.get(
        "max_power_w", boat_usbl.get("electrical", {}).get("max_power_w", 0.0)
    )
    target_max_range = float(
        target_transponder.get("max_range_m", boat_usbl["max_range_m"])
    )
    target_depth_rating = float(
        target_transponder.get("depth_rating_m", boat_usbl["depth_rating_m"])
    )

    cora_anchor_mode = str(estimator.get("cora_anchor_mode", "fixed_sensor_positions"))
    if cora_anchor_mode not in {"fixed_sensor_positions", "boat_variables"}:
        raise ValueError(
            "estimator.cora_anchor_mode must be 'fixed_sensor_positions' "
            f"or 'boat_variables', got {cora_anchor_mode!r}"
        )

    sim_cfg = SimConfig(
        steps=int(simulation["steps"]),
        dt=float(simulation["dt"]),
        seed=int(simulation["seed"]),
        optimizer_backend=str(estimator.get("optimizer_backend", "full")),
        initial_guess_offset_m=_tuple(estimator["initial_guess_offset_m"]),  # type: ignore[arg-type]
        initial_guess_velocity_mps=_tuple(estimator["initial_guess_velocity_mps"]),  # type: ignore[arg-type]
        use_smoothness_factor=bool(estimator["use_smoothness_factor"]),
        smoothness_accel_sigma_mps2=float(estimator["smoothness_accel_sigma_mps2"]),
        use_depth_factor=bool(estimator["use_depth_factor"]),
        depth_sigma_m=float(estimator["depth_sigma_m"]),
        use_usbl_angles=bool(estimator["use_usbl_angles"]),
        tangent_uncertainty_sigma_m=float(estimator["tangent_uncertainty_sigma_m"]),
        cora_window_size=int(estimator.get("cora_window_size", 25)),
        cora_solve_stride=int(estimator.get("cora_solve_stride", 5)),
        cora_anchor_mode=cora_anchor_mode,
        cora_bound_xy_m=float(estimator.get("cora_bound_xy_m", 30.0)),
        cora_bound_z_boat_m=float(estimator.get("cora_bound_z_boat_m", 2.0)),
        cora_bound_z_target_min_m=float(
            estimator.get("cora_bound_z_target_min_m", -20.0)
        ),
        cora_bound_z_target_max_m=float(
            estimator.get("cora_bound_z_target_max_m", 0.0)
        ),
        cora_second_moment_bounds=bool(
            estimator.get("cora_second_moment_bounds", True)
        ),
        cora_range_slack_weight=float(estimator.get("cora_range_slack_weight", 1.0)),
        cora_boat_prior_weight=float(estimator.get("cora_boat_prior_weight", 100.0)),
        cora_boat_displacement_weight=float(
            estimator.get("cora_boat_displacement_weight", 25.0)
        ),
        cora_surface_prior_weight=float(
            estimator.get("cora_surface_prior_weight", 4.0)
        ),
        cora_surface_z_m=float(estimator.get("cora_surface_z_m", 0.0)),
        cora_surface_sigma_m=float(estimator.get("cora_surface_sigma_m", 2.0)),
        cora_rank_tightness_tol=float(estimator.get("cora_rank_tightness_tol", 1e-3)),
        cora_sdp_feasibility_tol=float(estimator.get("cora_sdp_feasibility_tol", 1e-3)),
        cora_sdp_objective_tol=float(estimator.get("cora_sdp_objective_tol", 1e-6)),
        cora_sdp_psd_tol=float(estimator.get("cora_sdp_psd_tol", 1e-5)),
        cora_refine_with_full=bool(estimator.get("cora_refine_with_full", False)),
        cora_solver=str(estimator.get("cora_solver", "SCS")),
        usbl_mount_bias_m=_tuple(estimator.get("usbl_mount_bias_m", [0.0, 0.0, 0.0])),  # type: ignore[arg-type]
        usbl_mount_bias_rpy_rad=tuple(
            np.deg2rad(
                _tuple(estimator.get("usbl_mount_bias_rpy_deg", [0.0, 0.0, 0.0]))
            )
        ),
        usbl_mount_prior_sigma_m=float(estimator.get("usbl_mount_prior_sigma_m", 0.02)),
        usbl_mount_prior_sigma_rad=tuple(
            np.deg2rad(
                _tuple(estimator.get("usbl_mount_prior_sigma_deg", [0.25, 0.25, 0.5]))
            )
        ),
        boat_usbl_name=str(boat_usbl["name"]),
        boat_usbl_role=str(boat_usbl.get("role", "Boat-mounted USBL transceiver")),
        boat_usbl_frequency_band_khz=_tuple(boat_usbl["frequency_band_khz"]),  # type: ignore[arg-type]
        boat_usbl_max_range_m=float(boat_usbl["max_range_m"]),
        boat_usbl_depth_rating_m=float(boat_usbl["depth_rating_m"]),
        boat_usbl_acoustic_coverage_deg=float(boat_usbl["acoustic_coverage_deg"]),
        boat_usbl_update_rate_hz=float(boat_usbl.get("update_rate_hz", 0.0)),
        boat_usbl_weight_air_kg=float(usbl_weight_air),
        boat_usbl_weight_water_kg=float(usbl_weight_water),
        boat_usbl_typical_power_w=float(usbl_typical_power),
        boat_usbl_max_power_w=float(usbl_max_power),
        boat_usbl_mount_offset_m=_tuple(boat_usbl.get("mount_offset_m", [0.0, 0.0, 0.0])),  # type: ignore[arg-type]
        boat_usbl_mount_rpy_rad=tuple(
            np.deg2rad(_tuple(boat_usbl.get("mount_rpy_deg", [0.0, 0.0, 0.0])))
        ),
        target_transponder_name=str(
            target_transponder.get("name", "Underwater transponder")
        ),
        target_transponder_role=str(
            target_transponder.get("role", "Underwater tracked target transponder")
        ),
        target_transponder_frequency_band_khz=_tuple(
            target_transponder.get(
                "frequency_band_khz", boat_usbl["frequency_band_khz"]
            )
        ),  # type: ignore[arg-type]
        target_transponder_max_range_m=target_max_range,
        target_transponder_depth_rating_m=target_depth_rating,
        target_transponder_beam_shape_deg=float(
            target_transponder.get("beam_shape_deg", 0.0)
        ),
        target_transponder_range_precision_m=float(
            target_transponder.get("range_precision_m", boat_usbl["range_sigma_m"])
        ),
        target_transponder_weight_air_kg=float(
            target_transponder.get("weight_air_kg", 0.0)
        ),
        target_transponder_weight_water_kg=float(
            target_transponder.get("weight_water_kg", 0.0)
        ),
        max_range_m=min(float(boat_usbl["max_range_m"]), target_max_range),
        target_depth_rating_m=target_depth_rating,
        acoustic_coverage_deg=float(boat_usbl["acoustic_coverage_deg"]),
        range_sigma_m=float(boat_usbl["range_sigma_m"]),
        angular_accuracy_rad=np.deg2rad(float(boat_usbl["angular_accuracy_deg"])),
        a_position_sigma_m=_tuple(covariance["a_position_sigma_m"]),  # type: ignore[arg-type]
        a_orientation_sigma_rad=tuple(
            np.deg2rad(_tuple(covariance["a_orientation_sigma_deg"]))
        ),
        stationary_ping_windows=_windows(
            scenario.get("stationary_ping_windows", [[0, int(simulation["steps"])]]),
            int(simulation["steps"]),
        ),
        b_event_start_offset_m=_tuple(scenario.get("b_event_start_offset_m", [-4.0, 3.0, -2.5])),  # type: ignore[arg-type]
        b_event_stationary_position_m=_tuple(
            scenario.get("b_event_stationary_position_m", [-7.0, 6.3, -8.0])
        ),  # type: ignore[arg-type]
        b_event_end_position_m=_tuple(scenario.get("b_event_end_position_m", [2.0, -4.0, -10.0])),  # type: ignore[arg-type]
        b_start_position_m=_tuple(scenario["b_start_position_m"]),  # type: ignore[arg-type]
        b_velocity_mps=_tuple(scenario["b_velocity_mps"]),  # type: ignore[arg-type]
        b_motion_amplitude_m=_tuple(scenario["b_motion_amplitude_m"]),  # type: ignore[arg-type]
        b_motion_frequency_radps=_tuple(scenario["b_motion_frequency_radps"]),  # type: ignore[arg-type]
        b_motion_phase_rad=tuple(np.deg2rad(_tuple(scenario["b_motion_phase_deg"]))),
    )

    output_dir = Path(output["output_dir"])
    if not output_dir.is_absolute():
        output_dir = path.parent.parent / output_dir
    output_cfg = OutputConfig(
        output_dir=output_dir,
        summary_file=str(output["summary_file"]),
        summary_plot_file=str(output["summary_plot_file"]),
        position_plot_file=str(output["position_plot_file"]),
        video_file=str(output["video_file"]),
    )
    return sim_cfg, output_cfg
