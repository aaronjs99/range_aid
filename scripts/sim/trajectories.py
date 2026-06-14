"""Synthetic A/B trajectory generation."""

from __future__ import annotations

import numpy as np

from scripts.configuration.config import SimConfig
from scripts.math.usbl import wrap_angle


def make_known_a_pose(t: np.ndarray) -> np.ndarray:
    xyz = np.column_stack(
        [
            6.0 * np.cos(0.35 * t),
            5.0 * np.sin(0.35 * t),
            0.35 * np.sin(0.8 * t),
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


def make_estimated_a_pose(
    true_pose: np.ndarray, cfg: SimConfig, rng: np.random.Generator
) -> np.ndarray:
    """Simulate the DLiO-like boat pose estimate used by the optimizer."""
    estimated_pose = true_pose.copy()
    estimated_pose[:, :3] += rng.normal(
        0.0, np.array(cfg.a_position_sigma_m), size=(len(true_pose), 3)
    )
    estimated_pose[:, 3:6] = wrap_angle(
        estimated_pose[:, 3:6]
        + rng.normal(
            0.0, np.array(cfg.a_orientation_sigma_rad), size=(len(true_pose), 3)
        )
    )
    return estimated_pose


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


def _smoothstep(alpha: np.ndarray) -> np.ndarray:
    clipped = np.clip(alpha, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def make_event_b_pose(
    t: np.ndarray, cfg: SimConfig, a_true_pose: np.ndarray
) -> np.ndarray:
    """B trajectory for event-triggered CORA localization.

    The target starts near the boat, moves underwater, holds position during
    each stationary ping window, then moves again. CORA only uses the stationary
    ping measurements; the rest of this trajectory exists to make the simulation
    physically honest.
    """
    start_xyz = a_true_pose[0, :3] + np.array(cfg.b_event_start_offset_m)
    end_xyz = np.array(cfg.b_event_end_position_m)
    xyz = np.empty((len(t), 3), dtype=float)

    base_stationary = np.array(cfg.b_event_stationary_position_m)
    event_positions = [
        base_stationary + np.array([1.7, -1.2, -0.6]) * event_index
        for event_index, _ in enumerate(cfg.stationary_ping_windows)
    ]

    first_start = cfg.stationary_ping_windows[0][0]
    first_start_time = t[first_start]
    before_first = np.arange(len(t)) < first_start
    if np.any(before_first):
        alpha = _smoothstep(t[before_first] / max(first_start_time, cfg.dt))
        xyz[before_first] = start_xyz + alpha[:, None] * (
            event_positions[0] - start_xyz
        )
        wiggle = np.column_stack(
            [
                0.35 * np.sin(2.2 * t[before_first]),
                -0.25 * np.sin(1.7 * t[before_first] + 0.6),
                0.20 * np.sin(2.8 * t[before_first] + 1.1),
            ]
        )
        xyz[before_first] += (1.0 - alpha)[:, None] * wiggle

    for event_index, (start, end) in enumerate(cfg.stationary_ping_windows):
        xyz[start:end] = event_positions[event_index]

        next_start = (
            cfg.stationary_ping_windows[event_index + 1][0]
            if event_index + 1 < len(cfg.stationary_ping_windows)
            else None
        )
        if next_start is None:
            continue
        between = np.arange(len(t))
        between = between[(between >= end) & (between < next_start)]
        if len(between) == 0:
            continue
        segment_start_time = t[end - 1]
        segment_end_time = t[next_start]
        alpha = _smoothstep(
            (t[between] - segment_start_time)
            / max(segment_end_time - segment_start_time, cfg.dt)
        )
        segment_target = event_positions[event_index + 1]
        xyz[between] = event_positions[event_index] + alpha[:, None] * (
            segment_target - event_positions[event_index]
        )
        wiggle = np.column_stack(
            [
                0.35 * np.sin(1.8 * (t[between] - segment_start_time)),
                -0.25 * np.sin(2.1 * (t[between] - segment_start_time) + 0.8),
                0.12 * np.sin(2.7 * (t[between] - segment_start_time)),
            ]
        )
        xyz[between] += np.sin(np.pi * alpha)[:, None] * wiggle

    last_end = cfg.stationary_ping_windows[-1][1]
    after_last = np.arange(len(t)) >= last_end
    if np.any(after_last):
        last_event_time = t[last_end - 1]
        after_duration = max(t[-1] - last_event_time, cfg.dt)
        alpha = _smoothstep((t[after_last] - last_event_time) / after_duration)
        last_stationary_xyz = event_positions[-1]
        xyz[after_last] = last_stationary_xyz + alpha[:, None] * (
            end_xyz - last_stationary_xyz
        )
        wiggle = np.column_stack(
            [
                0.45 * np.sin(1.8 * (t[after_last] - last_event_time) + 0.4),
                0.35 * np.sin(2.4 * (t[after_last] - last_event_time) + 1.2),
                -0.18 * np.sin(2.0 * (t[after_last] - last_event_time)),
            ]
        )
        xyz[after_last] += alpha[:, None] * wiggle

    dx = np.gradient(xyz[:, 0], t)
    dy = np.gradient(xyz[:, 1], t)
    dz = np.gradient(xyz[:, 2], t)
    yaw = np.arctan2(dy, dx)
    pitch = np.arctan2(-dz, np.hypot(dx, dy))
    roll = 0.14 * np.sin(1.1 * t)
    return np.column_stack([xyz, roll, pitch, yaw])
