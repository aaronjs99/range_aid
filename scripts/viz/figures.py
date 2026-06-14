"""Static PNG figures for the range-aided pose estimation demo."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.math.usbl import predict_usbl_observations


def _ping_window_edges(data) -> list[tuple[float, float]]:
    cfg = data["cfg"]
    return [
        (start * cfg.dt, end * cfg.dt) for start, end in cfg.stationary_ping_windows
    ]


def _mark_ping_windows(axes, data) -> None:
    for ax in np.ravel(axes):
        for start_t, end_t in _ping_window_edges(data):
            ax.axvline(start_t, color="0.25", linestyle=":", linewidth=1.0)
            ax.axvline(end_t, color="0.25", linestyle=":", linewidth=1.0)


def _event_landmarks(data) -> list[dict]:
    return data.get("event_landmark_estimates", [])


def plot_position_timeseries(path: Path, data) -> None:
    cfg = data["cfg"]
    t = data["t"]
    a_true = data.get("a_true_pose", data["a_pose"])[:, :3]
    a_est = data["a_pose"][:, :3]
    b_true = data["b_true_pose"][:, :3]
    b_est = data["b_est_pose"][:, :3]
    b_cov = data["b_cov_xyz"]
    a_sigma = np.tile(np.array(cfg.a_position_sigma_m), (len(t), 1))
    b_sigma = np.sqrt(np.maximum(0.0, np.array([b_cov[:, i, i] for i in range(3)]).T))

    series = [
        ("A x [m]", a_true[:, 0], a_est[:, 0], a_sigma[:, 0], "tab:blue"),
        ("A y [m]", a_true[:, 1], a_est[:, 1], a_sigma[:, 1], "tab:blue"),
        ("A z [m]", a_true[:, 2], a_est[:, 2], a_sigma[:, 2], "tab:blue"),
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
    _mark_ping_windows(axes, data)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Position estimates for moving A and moving B")
    fig.tight_layout(rect=[0, 0.0, 1, 0.98])
    fig.savefig(path, dpi=180)


def plot_summary(path: Path, data) -> None:
    t = data["t"]
    a_true_xyz = data.get("a_true_pose", data["a_pose"])[:, :3]
    a_xyz = data["a_pose"][:, :3]
    b_true = data["b_true_pose"][:, :3]
    b_init = data["b_init_xyz"]
    b_est = data["b_est_pose"][:, :3]
    measured_ranges = data["measured_ranges"]
    cfg = data["cfg"]
    cora_feed_mask = data.get("cora_feed_mask", np.ones_like(t, dtype=bool))

    fig = plt.figure(figsize=(12, 8))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax3d.plot(
        a_true_xyz[:, 0],
        a_true_xyz[:, 1],
        a_true_xyz[:, 2],
        "--",
        label="true A trajectory",
        color="tab:cyan",
        alpha=0.7,
    )
    ax3d.plot(
        a_xyz[:, 0],
        a_xyz[:, 1],
        a_xyz[:, 2],
        label="estimated A trajectory",
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
        b_est[:, 0],
        b_est[:, 1],
        b_est[:, 2],
        label="active event B estimate",
        color="tab:red",
    )
    for i, event in enumerate(_event_landmarks(data)):
        point = event["l_hat"]
        ax3d.scatter(
            point[0],
            point[1],
            point[2],
            marker="x",
            s=70,
            color="tab:purple",
            label="archived event landmark" if i == 0 else None,
        )
    active_indices = np.flatnonzero(cora_feed_mask & np.all(np.isfinite(b_est), axis=1))
    if len(active_indices):
        stride = max(1, len(active_indices) // 10)
        for k in active_indices[::stride]:
            ax3d.plot(
                [a_xyz[k, 0], b_est[k, 0]],
                [a_xyz[k, 1], b_est[k, 1]],
                [a_xyz[k, 2], b_est[k, 2]],
                color="0.35",
                alpha=0.25,
                linewidth=0.8,
            )
    ax3d.set_title("3D moving USBL-factor geometry")
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    ax3d.legend(fontsize=8)

    ax_xy = fig.add_subplot(2, 2, 2)
    ax_xy.plot(
        a_true_xyz[:, 0],
        a_true_xyz[:, 1],
        "--",
        label="true A",
        color="tab:cyan",
        alpha=0.7,
    )
    ax_xy.plot(a_xyz[:, 0], a_xyz[:, 1], label="estimated A", color="tab:blue")
    ax_xy.plot(b_true[:, 0], b_true[:, 1], label="true B", color="tab:green")
    ax_xy.plot(
        b_est[:, 0], b_est[:, 1], label="active event B estimate", color="tab:red"
    )
    for i, event in enumerate(_event_landmarks(data)):
        point = event["l_hat"]
        ax_xy.scatter(
            point[0],
            point[1],
            marker="x",
            s=55,
            color="tab:purple",
            label="archived event landmark" if i == 0 else None,
        )
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
    _mark_ping_windows([ax_err], data)

    ax_range = fig.add_subplot(2, 2, 4)
    true_ranges = data["clean_ranges"]
    est_ranges, _, _ = predict_usbl_observations(data["a_pose"], b_est, cfg)
    active_true_ranges = np.where(cora_feed_mask, true_ranges, np.nan)
    active_measured_ranges = np.where(cora_feed_mask, measured_ranges, np.nan)
    active_est_ranges = np.where(cora_feed_mask, est_ranges, np.nan)
    ax_range.plot(t, active_true_ranges, label="true range", color="tab:green")
    ax_range.scatter(
        t,
        active_measured_ranges,
        s=10,
        label="measured range",
        color="black",
        alpha=0.5,
    )
    ax_range.plot(t, active_est_ranges, label="estimated range", color="tab:red")
    ax_range.set_title("Boat-mounted USBL range measurements")
    ax_range.set_xlabel("time [s]")
    ax_range.set_ylabel("range [m]")
    ax_range.grid(True, alpha=0.3)
    ax_range.legend(fontsize=8)
    _mark_ping_windows([ax_range], data)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
