"""Shared optimizer initialization helpers."""

from __future__ import annotations

import numpy as np

from scripts.configuration.config import SimConfig


def initial_guess(a_xyz: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """One plausible B_i path used only to start local optimization."""
    guess = np.zeros_like(a_xyz)
    guess[0] = a_xyz[0] + np.array(cfg.initial_guess_offset_m)
    velocity = np.array(cfg.initial_guess_velocity_mps)
    for k in range(1, len(guess)):
        guess[k] = guess[k - 1] + velocity * cfg.dt
    return guess
