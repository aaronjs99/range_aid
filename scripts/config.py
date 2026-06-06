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
    initial_guess_offset_m: tuple[float, float, float]
    initial_guess_velocity_mps: tuple[float, float, float]
    use_smoothness_factor: bool
    smoothness_accel_sigma_mps2: float
    use_depth_factor: bool
    depth_sigma_m: float
    use_usbl_angles: bool
    tangent_uncertainty_sigma_m: float
    sensor_name: str
    frequency_band_khz: tuple[float, float]
    max_range_m: float
    depth_rating_m: float
    acoustic_coverage_deg: float
    range_sigma_m: float
    angular_accuracy_rad: float
    a_position_sigma_m: tuple[float, float, float]
    a_orientation_sigma_rad: tuple[float, float, float]
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


def load_config(path: Path) -> tuple[SimConfig, OutputConfig]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping: {path}")

    simulation = raw["simulation"]
    estimator = raw["estimator"]
    range_sensor = raw["sensors"]["diver"]["range"]
    covariance = raw["covariance"]
    scenario = raw["scenario"]
    output = raw["output"]

    sim_cfg = SimConfig(
        steps=int(simulation["steps"]),
        dt=float(simulation["dt"]),
        seed=int(simulation["seed"]),
        initial_guess_offset_m=_tuple(estimator["initial_guess_offset_m"]),  # type: ignore[arg-type]
        initial_guess_velocity_mps=_tuple(estimator["initial_guess_velocity_mps"]),  # type: ignore[arg-type]
        use_smoothness_factor=bool(estimator["use_smoothness_factor"]),
        smoothness_accel_sigma_mps2=float(estimator["smoothness_accel_sigma_mps2"]),
        use_depth_factor=bool(estimator["use_depth_factor"]),
        depth_sigma_m=float(estimator["depth_sigma_m"]),
        use_usbl_angles=bool(estimator["use_usbl_angles"]),
        tangent_uncertainty_sigma_m=float(estimator["tangent_uncertainty_sigma_m"]),
        sensor_name=str(range_sensor["name"]),
        frequency_band_khz=_tuple(range_sensor["frequency_band_khz"]),  # type: ignore[arg-type]
        max_range_m=float(range_sensor["max_range_m"]),
        depth_rating_m=float(range_sensor["depth_rating_m"]),
        acoustic_coverage_deg=float(range_sensor["acoustic_coverage_deg"]),
        range_sigma_m=float(range_sensor["range_sigma_m"]),
        angular_accuracy_rad=np.deg2rad(float(range_sensor["angular_accuracy_deg"])),
        a_position_sigma_m=_tuple(covariance["a_position_sigma_m"]),  # type: ignore[arg-type]
        a_orientation_sigma_rad=tuple(
            np.deg2rad(_tuple(covariance["a_orientation_sigma_deg"]))
        ),
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
