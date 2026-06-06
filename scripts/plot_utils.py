"""Shared plotting helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from matplotlib.patches import Circle


def axis_limits(points: Iterable[np.ndarray], pad: float = 1.5):
    xyz = np.vstack(list(points))
    return xyz.min(axis=0) - pad, xyz.max(axis=0) + pad


def set_equal_3d(ax, lo: np.ndarray, hi: np.ndarray) -> None:
    center = 0.5 * (lo + hi)
    radius = 0.5 * float(np.max(hi - lo))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def sigma_radius_from_cov(cov: np.ndarray, scale: float = 2.0) -> float:
    eig = np.linalg.eigvalsh(cov)
    return scale * float(np.sqrt(max(0.0, eig.max())))


def draw_sphere(
    ax, center: np.ndarray, radius: float, color: str, alpha: float = 0.12
) -> None:
    u = np.linspace(0, 2 * np.pi, 18)
    v = np.linspace(0, np.pi, 10)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color=color, alpha=alpha, linewidth=0.5)


def draw_circle(
    ax, xy: tuple[float, float], radius: float, color: str, alpha: float = 0.16
) -> None:
    ax.add_patch(
        Circle(
            xy, radius=radius, fill=False, edgecolor=color, alpha=alpha, linewidth=1.2
        )
    )
