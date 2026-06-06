"""Synthetic A/B trajectory generation."""

from __future__ import annotations

import numpy as np

from scripts.config import SimConfig


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def make_known_a_pose(t: np.ndarray) -> np.ndarray:
    xyz = np.column_stack(
        [
            6.0 * np.cos(0.35 * t),
            5.0 * np.sin(0.35 * t),
            1.2 + 0.35 * np.sin(0.8 * t),
        ]
    )
    rpy = np.column_stack(
        [
            0.04 * np.sin(0.6 * t),
            0.03 * np.cos(0.5 * t),
            wrap_angle(0.35 * t + np.pi / 2.0),
        ]
    )
    return np.column_stack([xyz, rpy])


def make_true_b_pose(t: np.ndarray, cfg: SimConfig) -> np.ndarray:
    start = np.array(cfg.b_start_position_m)
    velocity = np.array(cfg.b_velocity_mps)
    amplitude = np.array(cfg.b_motion_amplitude_m)
    frequency = np.array(cfg.b_motion_frequency_radps)
    phase = np.array(cfg.b_motion_phase_rad)
    xyz = (
        start.reshape(1, 3)
        + t.reshape(-1, 1) * velocity.reshape(1, 3)
        + amplitude.reshape(1, 3) * np.sin(t.reshape(-1, 1) * frequency + phase)
    )
    dx = np.gradient(xyz[:, 0], t)
    dy = np.gradient(xyz[:, 1], t)
    dz = np.gradient(xyz[:, 2], t)
    yaw = np.arctan2(dy, dx)
    pitch = np.arctan2(-dz, np.hypot(dx, dy))
    roll = 0.1 * np.sin(0.8 * t)
    return np.column_stack([xyz, roll, pitch, yaw])
