"""MP4 rendering for the range-aided pose estimation demo."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

from scripts.config import SimConfig
from scripts.plot_utils import (
    axis_limits,
    draw_circle,
    draw_sphere,
    set_equal_3d,
    sigma_radius_from_cov,
)


def plot_view_2d(
    ax,
    data,
    frame: int,
    dims: tuple[int, int],
    title: str,
    labels: tuple[str, str],
    limits: tuple[tuple[float, float], tuple[float, float]],
    equal_aspect: bool,
) -> None:
    cfg: SimConfig = data["cfg"]
    a = data["a_pose"][:, :3]
    bt = data["b_true_pose"][:, :3]
    be = data["b_est_pose"][:, :3]
    cov = data["b_cov_xyz"]
    d0, d1 = dims

    ax.plot(a[: frame + 1, d0], a[: frame + 1, d1], color="tab:blue", label="known A")
    ax.scatter(bt[frame, d0], bt[frame, d1], color="tab:green", s=36, label="true B")
    ax.scatter(be[frame, d0], be[frame, d1], color="tab:red", s=36, label="estimated B")
    ax.scatter(a[frame, d0], a[frame, d1], color="tab:blue", s=20)

    a_radius = 2.0 * max(cfg.a_position_sigma_m[d0], cfg.a_position_sigma_m[d1])
    b_radius = sigma_radius_from_cov(cov[frame])
    draw_circle(ax, (a[frame, d0], a[frame, d1]), a_radius, "tab:blue")
    draw_circle(ax, (be[frame, d0], be[frame, d1]), b_radius, "tab:red")

    ax.set_title(title)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.grid(True, alpha=0.25)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")


def render_video(path: Path, data) -> None:
    cfg: SimConfig = data["cfg"]
    a = data["a_pose"][:, :3]
    bt = data["b_true_pose"][:, :3]
    be = data["b_est_pose"][:, :3]
    cov = data["b_cov_xyz"]
    lo, hi = axis_limits([a, bt, be], pad=2.0)
    uncertainty_pad = max(
        2.0 * max(cfg.a_position_sigma_m), max(sigma_radius_from_cov(c) for c in cov)
    )
    view_lo, view_hi = axis_limits([a, bt, be], pad=uncertainty_pad + 0.8)
    view_limits = {
        (0, 1): ((view_lo[0], view_hi[0]), (view_lo[1], view_hi[1])),
        (0, 2): ((view_lo[0], view_hi[0]), (view_lo[2], view_hi[2])),
        (1, 2): ((view_lo[1], view_hi[1]), (view_lo[2], view_hi[2])),
    }

    writer = FFMpegWriter(fps=10, metadata={"title": "range aided pose estimation"})
    fig = plt.figure(figsize=(12, 9))
    with writer.saving(fig, str(path), dpi=140):
        for frame in range(0, cfg.steps, 2):
            fig.clear()
            ax_top = fig.add_subplot(2, 2, 1)
            ax_side = fig.add_subplot(2, 2, 2)
            ax_front = fig.add_subplot(2, 2, 3)
            ax_iso = fig.add_subplot(2, 2, 4, projection="3d")

            plot_view_2d(
                ax_top,
                data,
                frame,
                (0, 1),
                "Top view (x-y)",
                ("x [m]", "y [m]"),
                view_limits[(0, 1)],
                True,
            )
            plot_view_2d(
                ax_side,
                data,
                frame,
                (0, 2),
                "Side view (x-z)",
                ("x [m]", "z [m]"),
                view_limits[(0, 2)],
                False,
            )
            plot_view_2d(
                ax_front,
                data,
                frame,
                (1, 2),
                "Front view (y-z)",
                ("y [m]", "z [m]"),
                view_limits[(1, 2)],
                False,
            )

            ax_iso.plot(
                a[: frame + 1, 0],
                a[: frame + 1, 1],
                a[: frame + 1, 2],
                color="tab:blue",
                label="known A",
            )
            ax_iso.scatter(
                bt[frame, 0],
                bt[frame, 1],
                bt[frame, 2],
                color="tab:green",
                s=36,
                label="true B",
            )
            ax_iso.scatter(
                be[frame, 0],
                be[frame, 1],
                be[frame, 2],
                color="tab:red",
                s=36,
                label="estimated B",
            )
            ax_iso.scatter(
                a[frame, 0], a[frame, 1], a[frame, 2], color="tab:blue", s=22
            )
            draw_sphere(ax_iso, a[frame], 2.0 * max(cfg.a_position_sigma_m), "tab:blue")
            draw_sphere(ax_iso, be[frame], sigma_radius_from_cov(cov[frame]), "tab:red")

            set_equal_3d(ax_iso, lo, hi)
            ax_iso.view_init(elev=24, azim=-45)
            ax_iso.set_title("Isometric 3D")
            ax_iso.set_xlabel("x [m]")
            ax_iso.set_ylabel("y [m]")
            ax_iso.set_zlabel("z [m]")
            ax_iso.legend(fontsize=7)
            fig.suptitle(
                f"Moving B range-factor fit from known A pose, t={data['t'][frame]:.1f}s"
            )
            fig.tight_layout()
            writer.grab_frame()
