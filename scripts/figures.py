"""Static PNG figures for the range-aided pose estimation demo."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_position_timeseries(path: Path, data) -> None:
    cfg = data["cfg"]
    t = data["t"]
    a_pos = data["a_pose"][:, :3]
    b_true = data["b_true_pose"][:, :3]
    b_est = data["b_est_pose"][:, :3]
    b_cov = data["b_cov_xyz"]
    a_sigma = np.tile(np.array(cfg.a_position_sigma_m), (len(t), 1))
    b_sigma = np.sqrt(np.maximum(0.0, np.array([b_cov[:, i, i] for i in range(3)]).T))

    series = [
        ("A x [m]", a_pos[:, 0], a_pos[:, 0], a_sigma[:, 0], "tab:blue"),
        ("A y [m]", a_pos[:, 1], a_pos[:, 1], a_sigma[:, 1], "tab:blue"),
        ("A z [m]", a_pos[:, 2], a_pos[:, 2], a_sigma[:, 2], "tab:blue"),
        ("B x [m]", b_true[:, 0], b_est[:, 0], b_sigma[:, 0], "tab:red"),
        ("B y [m]", b_true[:, 1], b_est[:, 1], b_sigma[:, 1], "tab:red"),
        ("B z [m]", b_true[:, 2], b_est[:, 2], b_sigma[:, 2], "tab:red"),
    ]

    fig, axes = plt.subplots(6, 1, figsize=(10, 14), sharex=True)
    for i, (ax, (ylabel, truth, estimate, sigma, color)) in enumerate(
        zip(axes, series)
    ):
        upper = estimate + 2.0 * sigma
        lower = estimate - 2.0 * sigma
        ax.fill_between(t, lower, upper, color=color, alpha=0.14)
        ax.plot(t, truth, color="tab:green", linewidth=1.5, label="truth")
        ax.plot(t, estimate, color=color, linewidth=1.4, label="estimate")
        ax.plot(
            t,
            upper,
            ":",
            color=color,
            linewidth=1.0,
            label="+/- 2 sigma" if i == 0 else None,
        )
        ax.plot(t, lower, ":", color=color, linewidth=1.0)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Position estimates for moving A and moving B")
    fig.tight_layout(rect=[0, 0.0, 1, 0.98])
    fig.savefig(path, dpi=180)


def plot_summary(path: Path, data) -> None:
    t = data["t"]
    a_xyz = data["a_pose"][:, :3]
    b_true = data["b_true_pose"][:, :3]
    b_init = data["b_init_xyz"]
    b_est = data["b_est_pose"][:, :3]
    measured_ranges = data["measured_ranges"]

    fig = plt.figure(figsize=(12, 8))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax3d.plot(
        a_xyz[:, 0],
        a_xyz[:, 1],
        a_xyz[:, 2],
        label="known A trajectory",
        color="tab:blue",
    )
    ax3d.plot(
        b_true[:, 0],
        b_true[:, 1],
        b_true[:, 2],
        label="true moving B",
        color="tab:green",
    )
    ax3d.plot(
        b_init[:, 0],
        b_init[:, 1],
        b_init[:, 2],
        "--",
        label="initial B path",
        color="tab:orange",
        alpha=0.8,
    )
    ax3d.plot(
        b_est[:, 0], b_est[:, 1], b_est[:, 2], label="range-fit B path", color="tab:red"
    )
    ax3d.set_title("3D moving range-factor geometry")
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    ax3d.legend(fontsize=8)

    ax_xy = fig.add_subplot(2, 2, 2)
    ax_xy.plot(a_xyz[:, 0], a_xyz[:, 1], label="known A", color="tab:blue")
    ax_xy.plot(b_true[:, 0], b_true[:, 1], label="true B", color="tab:green")
    ax_xy.plot(b_est[:, 0], b_est[:, 1], label="range-fit B", color="tab:red")
    ax_xy.axis("equal")
    ax_xy.set_title("Top-down view")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(fontsize=8)

    ax_err = fig.add_subplot(2, 2, 3)
    ax_err.plot(
        t,
        np.linalg.norm(b_init - b_true, axis=1),
        "--",
        label="initial error",
        color="tab:orange",
    )
    ax_err.plot(
        t,
        np.linalg.norm(b_est - b_true, axis=1),
        label="estimated error",
        color="tab:red",
    )
    ax_err.set_title("B position error over time")
    ax_err.set_xlabel("time [s]")
    ax_err.set_ylabel("position error [m]")
    ax_err.grid(True, alpha=0.3)
    ax_err.legend(fontsize=8)

    ax_range = fig.add_subplot(2, 2, 4)
    true_ranges = np.linalg.norm(b_true - a_xyz, axis=1)
    est_ranges = np.linalg.norm(b_est - a_xyz, axis=1)
    ax_range.plot(t, true_ranges, label="true range", color="tab:green")
    ax_range.scatter(
        t, measured_ranges, s=10, label="measured range", color="black", alpha=0.5
    )
    ax_range.plot(t, est_ranges, label="estimated range", color="tab:red")
    ax_range.set_title("Range measurements")
    ax_range.set_xlabel("time [s]")
    ax_range.set_ylabel("range [m]")
    ax_range.grid(True, alpha=0.3)
    ax_range.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
