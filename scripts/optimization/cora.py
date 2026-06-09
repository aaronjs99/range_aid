"""CORA-style event-triggered range-aided SDP backend.

This is an educational dense-CVXPY implementation of the CORA modeling idea:
build stationary acoustic ping windows, lift the range-aided QCQP to an SDP,
solve the relaxation, recover a primal estimate, and report rank/tightness
diagnostics plus explicit SDP feasibility checks. It is not the performant CORA
Riemannian Staircase backend yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import cvxpy as cp
import numpy as np
from scipy.optimize import least_squares

from scripts.configuration.config import SimConfig
from scripts.math.geometry import rotation_from_rpy, sensor_pose_from_boat_pose
from scripts.math.usbl import (
    estimate_position_covariance,
    predict_usbl_observations,
    wrap_angle,
)
from scripts.optimization.common import initial_guess
from scripts.sim.trajectories import (
    make_estimated_a_pose,
    make_event_b_pose,
    make_known_a_pose,
)


@dataclass(frozen=True)
class CoraWindowResult:
    event_index: int
    start: int
    end: int
    indices: tuple[int, ...]
    status: str
    sdp_valid: bool
    rank_tight: bool
    certified_tight: bool
    invalid_reasons: tuple[str, ...]
    z00_error: float
    min_eig_z: float
    psd_violation: float
    max_range_constraint_residual: float
    min_slack_plus: float
    min_slack_minus: float
    min_slack: float
    sdp_objective_nonnegative: bool
    sdp_objective: float
    sdp_primal_objective: float
    sdp_gap: float
    sdp_relative_gap: float
    slack_sum: float
    rank_ratio: float
    recovery_method: str
    l_hat_sdp: np.ndarray
    a_hat_window_sdp: np.ndarray
    l_hat_refined: np.ndarray
    l_hat_final: np.ndarray
    refinement_used: bool
    refined_local_objective: float

    @property
    def objective(self) -> float:
        return self.sdp_objective

    @property
    def primal_objective(self) -> float:
        return self.sdp_primal_objective

    @property
    def certificate_gap(self) -> float:
        return self.sdp_gap

    @property
    def relative_certificate_gap(self) -> float:
        return self.sdp_relative_gap

    @property
    def l_hat(self) -> np.ndarray:
        """Published estimate used by plots and scoring."""
        return self.l_hat_final

    @property
    def a_hat_window(self) -> np.ndarray:
        return self.a_hat_window_sdp

    @property
    def refined(self) -> bool:
        return self.refinement_used

    @property
    def sdp_status(self) -> str:
        return self.status

    @property
    def tight(self) -> bool:
        """Backward-compatible alias for a valid and rank-tight SDP result."""
        return self.certified_tight


def state_dim(window_size: int) -> int:
    return 3 * window_size + 3


def a_slice(k: int) -> slice:
    return slice(3 * k, 3 * k + 3)


def l_slice(window_size: int) -> slice:
    start = 3 * window_size
    return slice(start, start + 3)


def _indices(slc: slice) -> np.ndarray:
    return np.arange(slc.start, slc.stop)


def _weighted_affine_norm_matrix(
    terms: list[tuple[slice, float]],
    offset: np.ndarray,
    axis_weights: np.ndarray,
    n: int,
) -> np.ndarray:
    """Lifted Q for sum_j w_j * (sum_i c_i x_i[j] - offset[j])^2."""
    q = np.zeros((n + 1, n + 1))
    for axis in range(3):
        coeff = np.zeros(n + 1)
        coeff[0] = -float(offset[axis])
        for slc, scale in terms:
            coeff[1 + _indices(slc)[axis]] += float(scale)
        q += float(axis_weights[axis]) * np.outer(coeff, coeff)
    return q


def _coordinate_prior_matrix(
    slc: slice, axis: int, prior: float, weight: float, n: int
) -> np.ndarray:
    q = np.zeros((n + 1, n + 1))
    coeff = np.zeros(n + 1)
    coeff[0] = -float(prior)
    coeff[1 + _indices(slc)[axis]] = 1.0
    q += float(weight) * np.outer(coeff, coeff)
    return q


def _position_bounds(kind: str, cfg: SimConfig) -> tuple[tuple[float, float], ...]:
    xy = float(cfg.cora_bound_xy_m)
    if kind == "boat":
        z = float(cfg.cora_bound_z_boat_m)
        return ((-xy, xy), (-xy, xy), (-z, z))
    if kind == "target":
        return (
            (-xy, xy),
            (-xy, xy),
            (
                float(cfg.cora_bound_z_target_min_m),
                float(cfg.cora_bound_z_target_max_m),
            ),
        )
    raise ValueError(f"unknown CORA bound kind {kind!r}")


def _add_lifted_position_bounds(
    constraints: list,
    z: cp.Variable,
    slc: slice,
    bounds: tuple[tuple[float, float], ...],
    cfg: SimConfig,
) -> None:
    for axis, (lower, upper) in enumerate(bounds):
        matrix_idx = 1 + _indices(slc)[axis]
        constraints.append(z[0, matrix_idx] >= lower)
        constraints.append(z[0, matrix_idx] <= upper)
        if cfg.cora_second_moment_bounds:
            second_moment_bound = max(abs(lower), abs(upper)) ** 2
            constraints.append(z[matrix_idx, matrix_idx] <= second_moment_bound)


def _sensor_offsets_from_boat_poses(a_pose_window: np.ndarray, cfg: SimConfig):
    mount_offset = np.array(cfg.boat_usbl_mount_offset_m)
    return np.array(
        [rotation_from_rpy(*pose[3:6]) @ mount_offset for pose in a_pose_window]
    )


def _sensor_positions_from_boat_poses(
    a_pose_window: np.ndarray, cfg: SimConfig
) -> np.ndarray:
    mount_offset = np.array(cfg.boat_usbl_mount_offset_m)
    mount_rpy = np.array(cfg.boat_usbl_mount_rpy_rad)
    return np.array(
        [
            sensor_pose_from_boat_pose(pose, mount_offset, mount_rpy)[0]
            for pose in a_pose_window
        ]
    )


def _primal_objective(
    a_hat_window: np.ndarray,
    l_hat: np.ndarray,
    measurement_indices: np.ndarray,
    a_pose_window: np.ndarray,
    sensor_offsets: np.ndarray,
    measured_ranges: np.ndarray,
    measured_depths: np.ndarray,
    cfg: SimConfig,
) -> tuple[float, float]:
    """Evaluate the original QCQP-style objective at a recovered primal point."""
    a_priors = a_pose_window[:, :3]
    position_weights = (
        cfg.cora_boat_prior_weight
        / np.maximum(np.array(cfg.a_position_sigma_m), 1e-9) ** 2
    )
    displacement_sigma = np.maximum(
        np.array(cfg.a_position_sigma_m) * np.sqrt(2.0), 1e-9
    )
    displacement_weights = cfg.cora_boat_displacement_weight / displacement_sigma**2
    surface_weight = (
        cfg.cora_surface_prior_weight / max(cfg.cora_surface_sigma_m, 1e-9) ** 2
    )

    range_slack_sum = 0.0
    objective = 0.0
    for local_k, global_k in enumerate(measurement_indices):
        diff = l_hat - a_hat_window[local_k] - sensor_offsets[local_k]
        range_error = float(diff @ diff - measured_ranges[global_k] ** 2)
        range_slack_sum += abs(range_error)
        a_delta = a_hat_window[local_k] - a_priors[local_k]
        objective += float(np.sum(position_weights * a_delta**2))
        surface_error = a_hat_window[local_k, 2] - cfg.cora_surface_z_m
        objective += float(surface_weight * surface_error**2)
        if cfg.use_depth_factor:
            depth_error = l_hat[2] - measured_depths[global_k]
            objective += float(depth_error**2 / max(cfg.depth_sigma_m, 1e-9) ** 2)

    for local_k in range(len(a_hat_window) - 1):
        delta_prior = a_priors[local_k + 1] - a_priors[local_k]
        delta = a_hat_window[local_k + 1] - a_hat_window[local_k] - delta_prior
        objective += float(np.sum(displacement_weights * delta**2))

    objective += cfg.cora_range_slack_weight * range_slack_sum
    return objective, range_slack_sum


def _certificate_gap(
    primal_objective: float, sdp_objective: float
) -> tuple[float, float]:
    if not np.isfinite(primal_objective) or not np.isfinite(sdp_objective):
        return float("nan"), float("nan")
    gap = primal_objective - sdp_objective
    scale = max(1.0, abs(primal_objective), abs(sdp_objective))
    return float(gap), float(gap / scale)


def _sdp_validity(
    status: str,
    z00_error: float,
    psd_violation: float,
    max_range_constraint_residual: float,
    min_slack: float,
    sdp_objective: float,
    cfg: SimConfig,
) -> tuple[bool, bool, tuple[str, ...]]:
    sdp_objective_nonnegative = bool(
        np.isfinite(sdp_objective)
        and sdp_objective >= -float(cfg.cora_sdp_objective_tol)
    )
    reasons = []
    if status != "optimal":
        reasons.append(f"status={status}")
    if not np.isfinite(z00_error) or z00_error >= cfg.cora_sdp_feasibility_tol:
        reasons.append(f"z00_error={z00_error:.6g}")
    if not np.isfinite(psd_violation) or psd_violation >= cfg.cora_sdp_psd_tol:
        reasons.append(f"psd_violation={psd_violation:.6g}")
    if (
        not np.isfinite(max_range_constraint_residual)
        or max_range_constraint_residual >= cfg.cora_sdp_feasibility_tol
    ):
        reasons.append(
            f"max_range_constraint_residual={max_range_constraint_residual:.6g}"
        )
    if not np.isfinite(min_slack) or min_slack < -cfg.cora_sdp_feasibility_tol:
        reasons.append(f"min_slack={min_slack:.6g}")
    if not sdp_objective_nonnegative:
        reasons.append(f"sdp_objective={sdp_objective:.6g}")
    return len(reasons) == 0, sdp_objective_nonnegative, tuple(reasons)


def _failed_cora_result(
    event_index: int,
    event_start: int,
    event_end: int,
    measurement_indices: np.ndarray,
    status: str,
    window_size: int,
) -> CoraWindowResult:
    fallback_l = np.full(3, np.nan)
    fallback_a = np.full((window_size, 3), np.nan)
    return CoraWindowResult(
        event_index=event_index,
        start=event_start,
        end=event_end,
        indices=tuple(int(i) for i in measurement_indices),
        status=status,
        sdp_valid=False,
        rank_tight=False,
        certified_tight=False,
        invalid_reasons=(f"status={status}", "no_sdp_solution"),
        z00_error=float("nan"),
        min_eig_z=float("nan"),
        psd_violation=float("nan"),
        max_range_constraint_residual=float("nan"),
        min_slack_plus=float("nan"),
        min_slack_minus=float("nan"),
        min_slack=float("nan"),
        sdp_objective_nonnegative=False,
        sdp_objective=float("nan"),
        sdp_primal_objective=float("nan"),
        sdp_gap=float("nan"),
        sdp_relative_gap=float("nan"),
        slack_sum=float("nan"),
        rank_ratio=float("inf"),
        recovery_method="not_recovered",
        l_hat_sdp=fallback_l,
        a_hat_window_sdp=fallback_a,
        l_hat_refined=fallback_l,
        l_hat_final=fallback_l,
        refinement_used=False,
        refined_local_objective=float("nan"),
    )


def _sdp_solution_metrics(
    z_value: np.ndarray,
    slack_plus_value: np.ndarray,
    slack_minus_value: np.ndarray,
    range_q_matrices: list[tuple[np.ndarray, float]],
    status: str,
    sdp_objective: float,
    cfg: SimConfig,
) -> dict:
    sym_z = 0.5 * (z_value + z_value.T)
    eigvals_raw = np.sort(np.linalg.eigvalsh(sym_z))
    min_eig_z = float(eigvals_raw[0]) if len(eigvals_raw) else float("nan")
    psd_violation = (
        float(max(0.0, -min_eig_z)) if np.isfinite(min_eig_z) else float("nan")
    )
    eigvals = np.sort(np.maximum(eigvals_raw, 0.0))[::-1]
    rank_ratio = (
        float(eigvals[1] / eigvals[0])
        if len(eigvals) > 1 and eigvals[0] > 1e-12
        else 0.0
    )
    z00_error = float(abs(z_value[0, 0] - 1.0))
    min_slack_plus = (
        float(np.min(slack_plus_value)) if slack_plus_value.size else float("nan")
    )
    min_slack_minus = (
        float(np.min(slack_minus_value)) if slack_minus_value.size else float("nan")
    )
    min_slack = float(min(min_slack_plus, min_slack_minus))
    range_constraint_residuals = [
        abs(float(np.trace(q_range @ z_value) - range_squared - plus + minus))
        for (q_range, range_squared), plus, minus in zip(
            range_q_matrices, slack_plus_value, slack_minus_value
        )
    ]
    max_range_constraint_residual = (
        float(np.max(range_constraint_residuals))
        if range_constraint_residuals
        else float("nan")
    )
    sdp_valid, sdp_objective_nonnegative, invalid_reasons = _sdp_validity(
        status,
        z00_error,
        psd_violation,
        max_range_constraint_residual,
        min_slack,
        sdp_objective,
        cfg,
    )
    rank_tight = rank_ratio < cfg.cora_rank_tightness_tol
    certified_tight = sdp_valid and rank_tight
    return {
        "sdp_valid": sdp_valid,
        "rank_tight": rank_tight,
        "certified_tight": certified_tight,
        "invalid_reasons": invalid_reasons,
        "z00_error": z00_error,
        "min_eig_z": min_eig_z,
        "psd_violation": psd_violation,
        "max_range_constraint_residual": max_range_constraint_residual,
        "min_slack_plus": min_slack_plus,
        "min_slack_minus": min_slack_minus,
        "min_slack": min_slack,
        "sdp_objective_nonnegative": sdp_objective_nonnegative,
        "rank_ratio": rank_ratio,
    }


def _recover_primal(
    z_value: np.ndarray, certified_tight: bool
) -> tuple[np.ndarray, str, float]:
    sym_z = 0.5 * (z_value + z_value.T)
    eigvals, eigvecs = np.linalg.eigh(sym_z)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]
    rank_ratio = (
        float(eigvals[1] / eigvals[0])
        if len(eigvals) > 1 and eigvals[0] > 1e-12
        else 0.0
    )
    if certified_tight:
        return np.asarray(z_value[1:, 0]).reshape(-1), "column", rank_ratio

    leading = eigvecs[:, 0] * np.sqrt(max(float(eigvals[0]), 0.0))
    if abs(leading[0]) > 1e-9:
        return leading[1:] / leading[0], "leading_eigenvector", rank_ratio
    return np.asarray(z_value[1:, 0]).reshape(-1), "column_fallback", rank_ratio


def _fixed_anchor_primal_objective(
    l_hat: np.ndarray,
    sensor_positions: np.ndarray,
    measured_ranges_window: np.ndarray,
    cfg: SimConfig,
) -> tuple[float, float]:
    range_errors = [
        float((l_hat - sensor_position) @ (l_hat - sensor_position) - range_m**2)
        for sensor_position, range_m in zip(sensor_positions, measured_ranges_window)
    ]
    range_slack_sum = float(np.sum(np.abs(range_errors)))
    return cfg.cora_range_slack_weight * range_slack_sum, range_slack_sum


def solve_cora_window_fixed_anchors(
    event_index: int,
    event_start: int,
    event_end: int,
    measurement_indices: np.ndarray,
    a_pose: np.ndarray,
    measured_ranges: np.ndarray,
    cfg: SimConfig,
) -> CoraWindowResult:
    window_size = len(measurement_indices)
    n = 3
    lifted_dim = n + 1
    z = cp.Variable((lifted_dim, lifted_dim), symmetric=True)
    slack_plus = cp.Variable(window_size, nonneg=True)
    slack_minus = cp.Variable(window_size, nonneg=True)

    a_pose_window = a_pose[measurement_indices]
    sensor_positions = _sensor_positions_from_boat_poses(a_pose_window, cfg)
    constraints = [z >> 0, z[0, 0] == 1.0]
    range_terms = []
    range_q_matrices = []
    l_slc = slice(0, 3)
    _add_lifted_position_bounds(
        constraints, z, l_slc, _position_bounds("target", cfg), cfg
    )

    for local_k, global_k in enumerate(measurement_indices):
        q_range = _weighted_affine_norm_matrix(
            [(l_slc, 1.0)],
            sensor_positions[local_k],
            np.ones(3),
            n,
        )
        range_expr = cp.trace(q_range @ z)
        range_q_matrices.append((q_range, float(measured_ranges[global_k] ** 2)))
        constraints.append(
            range_expr - measured_ranges[global_k] ** 2
            == slack_plus[local_k] - slack_minus[local_k]
        )
        range_terms.append(slack_plus[local_k] + slack_minus[local_k])

    objective = cp.Minimize(
        cfg.cora_range_slack_weight * cp.sum(cp.hstack(range_terms))
    )
    problem = cp.Problem(objective, constraints)
    solve_kwargs = {"solver": cfg.cora_solver, "verbose": False}
    if cfg.cora_solver.upper() == "SCS":
        solve_kwargs.update({"eps": 1e-4, "max_iters": 5000})

    try:
        problem.solve(**solve_kwargs)
    except Exception as exc:  # pragma: no cover - defensive reporting path
        return _failed_cora_result(
            event_index,
            event_start,
            event_end,
            measurement_indices,
            f"error:{type(exc).__name__}",
            0,
        )

    if z.value is None:
        return _failed_cora_result(
            event_index,
            event_start,
            event_end,
            measurement_indices,
            str(problem.status),
            0,
        )

    slack_plus_value = (
        np.asarray(slack_plus.value, dtype=float)
        if slack_plus.value is not None
        else np.full(window_size, np.nan)
    )
    slack_minus_value = (
        np.asarray(slack_minus.value, dtype=float)
        if slack_minus.value is not None
        else np.full(window_size, np.nan)
    )
    sdp_objective = float(problem.value) if problem.value is not None else float("nan")
    metrics = _sdp_solution_metrics(
        z.value,
        slack_plus_value,
        slack_minus_value,
        range_q_matrices,
        str(problem.status),
        sdp_objective,
        cfg,
    )
    recovered, recovery_method, rank_ratio = _recover_primal(
        z.value, metrics["certified_tight"]
    )
    l_hat_sdp = recovered[l_slc]
    sdp_primal_objective, recovered_slack_sum = _fixed_anchor_primal_objective(
        l_hat_sdp,
        sensor_positions,
        measured_ranges[measurement_indices],
        cfg,
    )
    gap, relative_gap = _certificate_gap(sdp_primal_objective, sdp_objective)
    slack_sum = float(np.sum(slack_plus_value + slack_minus_value))

    return CoraWindowResult(
        event_index=event_index,
        start=event_start,
        end=event_end,
        indices=tuple(int(i) for i in measurement_indices),
        status=str(problem.status),
        sdp_valid=metrics["sdp_valid"],
        rank_tight=metrics["rank_tight"],
        certified_tight=metrics["certified_tight"],
        invalid_reasons=metrics["invalid_reasons"],
        z00_error=metrics["z00_error"],
        min_eig_z=metrics["min_eig_z"],
        psd_violation=metrics["psd_violation"],
        max_range_constraint_residual=metrics["max_range_constraint_residual"],
        min_slack_plus=metrics["min_slack_plus"],
        min_slack_minus=metrics["min_slack_minus"],
        min_slack=metrics["min_slack"],
        sdp_objective_nonnegative=metrics["sdp_objective_nonnegative"],
        sdp_objective=sdp_objective,
        sdp_primal_objective=sdp_primal_objective,
        sdp_gap=gap,
        sdp_relative_gap=relative_gap,
        slack_sum=slack_sum,
        rank_ratio=rank_ratio,
        recovery_method=recovery_method,
        l_hat_sdp=l_hat_sdp,
        a_hat_window_sdp=a_pose_window[:, :3].copy(),
        l_hat_refined=np.full(3, np.nan),
        l_hat_final=l_hat_sdp,
        refinement_used=False,
        refined_local_objective=float("nan"),
    )


def solve_cora_window(
    event_index: int,
    event_start: int,
    event_end: int,
    measurement_indices: np.ndarray,
    a_pose: np.ndarray,
    measured_ranges: np.ndarray,
    measured_depths: np.ndarray,
    cfg: SimConfig,
) -> CoraWindowResult:
    if cfg.cora_anchor_mode == "fixed_sensor_positions":
        return solve_cora_window_fixed_anchors(
            event_index,
            event_start,
            event_end,
            measurement_indices,
            a_pose,
            measured_ranges,
            cfg,
        )
    if cfg.cora_anchor_mode != "boat_variables":
        raise ValueError(f"unknown CORA anchor mode {cfg.cora_anchor_mode!r}")

    window_size = len(measurement_indices)
    n = state_dim(window_size)
    lifted_dim = n + 1
    z = cp.Variable((lifted_dim, lifted_dim), symmetric=True)
    slack_plus = cp.Variable(window_size, nonneg=True)
    slack_minus = cp.Variable(window_size, nonneg=True)

    a_pose_window = a_pose[measurement_indices]
    a_priors = a_pose_window[:, :3]
    sensor_offsets = _sensor_offsets_from_boat_poses(a_pose_window, cfg)
    position_weights = (
        cfg.cora_boat_prior_weight
        / np.maximum(np.array(cfg.a_position_sigma_m), 1e-9) ** 2
    )
    displacement_sigma = np.maximum(
        np.array(cfg.a_position_sigma_m) * np.sqrt(2.0), 1e-9
    )
    displacement_weights = cfg.cora_boat_displacement_weight / displacement_sigma**2
    surface_weight = (
        cfg.cora_surface_prior_weight / max(cfg.cora_surface_sigma_m, 1e-9) ** 2
    )

    constraints = [z >> 0, z[0, 0] == 1.0]
    range_terms = []
    range_q_matrices = []
    objective_terms = []
    l_slc = l_slice(window_size)
    _add_lifted_position_bounds(
        constraints, z, l_slc, _position_bounds("target", cfg), cfg
    )

    for local_k, global_k in enumerate(measurement_indices):
        a_slc = a_slice(local_k)
        _add_lifted_position_bounds(
            constraints, z, a_slc, _position_bounds("boat", cfg), cfg
        )
        q_range = _weighted_affine_norm_matrix(
            [(l_slc, 1.0), (a_slc, -1.0)],
            sensor_offsets[local_k],
            np.ones(3),
            n,
        )
        range_expr = cp.trace(q_range @ z)
        range_q_matrices.append((q_range, float(measured_ranges[global_k] ** 2)))
        constraints.append(
            range_expr - measured_ranges[global_k] ** 2
            == slack_plus[local_k] - slack_minus[local_k]
        )
        range_terms.append(slack_plus[local_k] + slack_minus[local_k])

        q_prior = _weighted_affine_norm_matrix(
            [(a_slc, 1.0)], a_priors[local_k], position_weights, n
        )
        objective_terms.append(cp.trace(q_prior @ z))
        q_surface = _coordinate_prior_matrix(
            a_slc, 2, cfg.cora_surface_z_m, surface_weight, n
        )
        objective_terms.append(cp.trace(q_surface @ z))

    for local_k in range(window_size - 1):
        delta_prior = a_priors[local_k + 1] - a_priors[local_k]
        q_delta = _weighted_affine_norm_matrix(
            [(a_slice(local_k + 1), 1.0), (a_slice(local_k), -1.0)],
            delta_prior,
            displacement_weights,
            n,
        )
        objective_terms.append(cp.trace(q_delta @ z))

    if cfg.use_depth_factor:
        depth_weight = 1.0 / max(cfg.depth_sigma_m, 1e-9) ** 2
        for global_k in measurement_indices:
            q_depth = _coordinate_prior_matrix(
                l_slc, 2, measured_depths[global_k], depth_weight, n
            )
            objective_terms.append(cp.trace(q_depth @ z))

    objective = cp.Minimize(
        cfg.cora_range_slack_weight * cp.sum(cp.hstack(range_terms))
        + cp.sum(cp.hstack(objective_terms))
    )
    problem = cp.Problem(objective, constraints)
    solve_kwargs = {"solver": cfg.cora_solver, "verbose": False}
    if cfg.cora_solver.upper() == "SCS":
        solve_kwargs.update({"eps": 1e-4, "max_iters": 5000})

    try:
        problem.solve(**solve_kwargs)
    except Exception as exc:  # pragma: no cover - defensive reporting path
        return _failed_cora_result(
            event_index,
            event_start,
            event_end,
            measurement_indices,
            f"error:{type(exc).__name__}",
            window_size,
        )

    if z.value is None:
        return _failed_cora_result(
            event_index,
            event_start,
            event_end,
            measurement_indices,
            str(problem.status),
            window_size,
        )

    slack_plus_value = (
        np.asarray(slack_plus.value, dtype=float)
        if slack_plus.value is not None
        else np.full(window_size, np.nan)
    )
    slack_minus_value = (
        np.asarray(slack_minus.value, dtype=float)
        if slack_minus.value is not None
        else np.full(window_size, np.nan)
    )
    sdp_objective = float(problem.value) if problem.value is not None else float("nan")
    metrics = _sdp_solution_metrics(
        z.value,
        slack_plus_value,
        slack_minus_value,
        range_q_matrices,
        str(problem.status),
        sdp_objective,
        cfg,
    )
    recovered, recovery_method, rank_ratio = _recover_primal(
        z.value, metrics["certified_tight"]
    )
    a_hat_window = recovered[: 3 * window_size].reshape(window_size, 3)
    l_hat_sdp = recovered[l_slc]
    sdp_primal_objective, recovered_slack_sum = _primal_objective(
        a_hat_window,
        l_hat_sdp,
        measurement_indices,
        a_pose_window,
        sensor_offsets,
        measured_ranges,
        measured_depths,
        cfg,
    )
    gap, relative_gap = _certificate_gap(sdp_primal_objective, sdp_objective)
    slack_sum = float(np.sum(slack_plus.value + slack_minus.value))
    return CoraWindowResult(
        event_index=event_index,
        start=event_start,
        end=event_end,
        indices=tuple(int(i) for i in measurement_indices),
        status=str(problem.status),
        sdp_valid=metrics["sdp_valid"],
        rank_tight=metrics["rank_tight"],
        certified_tight=metrics["certified_tight"],
        invalid_reasons=metrics["invalid_reasons"],
        z00_error=metrics["z00_error"],
        min_eig_z=metrics["min_eig_z"],
        psd_violation=metrics["psd_violation"],
        max_range_constraint_residual=metrics["max_range_constraint_residual"],
        min_slack_plus=metrics["min_slack_plus"],
        min_slack_minus=metrics["min_slack_minus"],
        min_slack=metrics["min_slack"],
        sdp_objective_nonnegative=metrics["sdp_objective_nonnegative"],
        sdp_objective=sdp_objective,
        sdp_primal_objective=sdp_primal_objective,
        sdp_gap=gap,
        sdp_relative_gap=relative_gap,
        slack_sum=slack_sum,
        rank_ratio=rank_ratio,
        recovery_method=recovery_method,
        l_hat_sdp=l_hat_sdp,
        a_hat_window_sdp=a_hat_window,
        l_hat_refined=np.full(3, np.nan),
        l_hat_final=l_hat_sdp,
        refinement_used=False,
        refined_local_objective=float("nan"),
    )


def static_landmark_local_residuals(
    l_xyz: np.ndarray,
    a_pose_window: np.ndarray,
    measured_ranges_window: np.ndarray,
    measured_azimuths_window: np.ndarray,
    measured_elevations_window: np.ndarray,
    measured_depths_window: np.ndarray,
    cfg: SimConfig,
) -> np.ndarray:
    l_path = np.repeat(l_xyz.reshape(1, 3), len(a_pose_window), axis=0)
    ranges, azimuths, elevations = predict_usbl_observations(a_pose_window, l_path, cfg)
    blocks = [(ranges - measured_ranges_window) / cfg.range_sigma_m]
    if cfg.use_usbl_angles:
        blocks.extend(
            [
                wrap_angle(azimuths - measured_azimuths_window)
                / cfg.angular_accuracy_rad,
                (elevations - measured_elevations_window) / cfg.angular_accuracy_rad,
            ]
        )
    if cfg.use_depth_factor:
        blocks.append((l_path[:, 2] - measured_depths_window) / cfg.depth_sigma_m)
    return np.concatenate(blocks)


def static_landmark_local_objective(
    l_xyz: np.ndarray,
    a_pose_window: np.ndarray,
    measured_ranges_window: np.ndarray,
    measured_azimuths_window: np.ndarray,
    measured_elevations_window: np.ndarray,
    measured_depths_window: np.ndarray,
    cfg: SimConfig,
) -> float:
    residuals = static_landmark_local_residuals(
        l_xyz,
        a_pose_window,
        measured_ranges_window,
        measured_azimuths_window,
        measured_elevations_window,
        measured_depths_window,
        cfg,
    )
    return float(residuals @ residuals)


def refine_static_landmark(
    l_initial: np.ndarray,
    a_pose_window: np.ndarray,
    measured_ranges_window: np.ndarray,
    measured_azimuths_window: np.ndarray,
    measured_elevations_window: np.ndarray,
    measured_depths_window: np.ndarray,
    cfg: SimConfig,
) -> tuple[np.ndarray, float]:
    """Local static-landmark refinement using enabled nonlinear factors."""

    result = least_squares(
        static_landmark_local_residuals,
        l_initial,
        args=(
            a_pose_window,
            measured_ranges_window,
            measured_azimuths_window,
            measured_elevations_window,
            measured_depths_window,
            cfg,
        ),
        x_scale="jac",
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        max_nfev=200,
    )
    return result.x, float(2.0 * result.cost)


def usbl_static_initial_guess(
    a_pose_window: np.ndarray,
    measured_ranges_window: np.ndarray,
    measured_azimuths_window: np.ndarray,
    measured_elevations_window: np.ndarray,
    cfg: SimConfig,
) -> np.ndarray:
    """Average USBL range/bearing points into a stationary-window seed."""
    mount_offset = np.array(cfg.boat_usbl_mount_offset_m)
    mount_rpy = np.array(cfg.boat_usbl_mount_rpy_rad)
    points = []
    for pose, range_m, azimuth, elevation in zip(
        a_pose_window,
        measured_ranges_window,
        measured_azimuths_window,
        measured_elevations_window,
    ):
        sensor_position, sensor_rotation = sensor_pose_from_boat_pose(
            pose, mount_offset, mount_rpy
        )
        q_sensor = np.array(
            [
                range_m * np.cos(elevation) * np.cos(azimuth),
                range_m * np.cos(elevation) * np.sin(azimuth),
                range_m * np.sin(elevation),
            ]
        )
        points.append(sensor_position + sensor_rotation @ q_sensor)
    return np.mean(points, axis=0)


def _stationary_mask(cfg: SimConfig) -> np.ndarray:
    mask = np.zeros(cfg.steps, dtype=bool)
    for start, end in cfg.stationary_ping_windows:
        mask[start:end] = True
    return mask


def _usbl_update_mask(t: np.ndarray, cfg: SimConfig) -> np.ndarray:
    mask = np.zeros(cfg.steps, dtype=bool)
    if cfg.boat_usbl_update_rate_hz <= 0.0:
        mask[:] = True
        return mask

    sample_interval = 1.0 / cfg.boat_usbl_update_rate_hz
    for start, end in cfg.stationary_ping_windows:
        next_sample_time = t[start]
        for k in range(start, end):
            if t[k] + 0.5 * cfg.dt >= next_sample_time:
                mask[k] = True
                next_sample_time += sample_interval
    return mask


def _event_measurement_groups(active_mask: np.ndarray, cfg: SimConfig) -> list[dict]:
    groups = []
    for event_index, (start, end) in enumerate(cfg.stationary_ping_windows):
        indices = np.flatnonzero(active_mask[start:end]) + start
        if len(indices) < 4:
            continue
        groups.append(
            {
                "event_index": event_index,
                "event_start": start,
                "event_end": end,
                "measurement_indices": indices,
            }
        )
    return groups


def _active_range_coverage(
    clean_ranges: np.ndarray, true_azimuths: np.ndarray, cfg: SimConfig
) -> tuple[np.ndarray, np.ndarray]:
    out_of_range = clean_ranges > cfg.max_range_m
    half_coverage = np.deg2rad(0.5 * cfg.acoustic_coverage_deg)
    out_of_coverage = np.abs(wrap_angle(true_azimuths)) > half_coverage
    return out_of_range, out_of_coverage


def _stitch_window_estimates(
    cfg: SimConfig, window_results: list[CoraWindowResult]
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    b_est_xyz = np.full((cfg.steps, 3), np.nan)
    event_landmarks = []
    for result in window_results:
        if np.all(np.isfinite(result.l_hat_final)):
            # The event estimate is valid only while the configured ping event
            # declares B stationary. It is archived afterward, not treated as
            # current live B tracking.
            b_est_xyz[result.start : result.end] = result.l_hat_final
            event_landmarks.append(
                {
                    "event_index": result.event_index,
                    "start": result.start,
                    "end": result.end,
                    "indices": result.indices,
                    "l_hat_sdp": result.l_hat_sdp.copy(),
                    "l_hat_refined": result.l_hat_refined.copy(),
                    "l_hat_final": result.l_hat_final.copy(),
                    "l_hat": result.l_hat_final.copy(),
                    "status": result.status,
                    "sdp_status": result.sdp_status,
                    "sdp_valid": result.sdp_valid,
                    "rank_tight": result.rank_tight,
                    "certified_tight": result.certified_tight,
                    "invalid_reasons": result.invalid_reasons,
                    "z00_error": result.z00_error,
                    "min_eig_z": result.min_eig_z,
                    "min_eig_Z": result.min_eig_z,
                    "psd_violation": result.psd_violation,
                    "max_range_constraint_residual": (
                        result.max_range_constraint_residual
                    ),
                    "min_slack_plus": result.min_slack_plus,
                    "min_slack_minus": result.min_slack_minus,
                    "min_slack": result.min_slack,
                    "sdp_objective_nonnegative": result.sdp_objective_nonnegative,
                    "sdp_objective": result.sdp_objective,
                    "sdp_primal_objective": result.sdp_primal_objective,
                    "sdp_gap": result.sdp_gap,
                    "sdp_relative_gap": result.sdp_relative_gap,
                    "slack_sum": result.slack_sum,
                    "rank_ratio": result.rank_ratio,
                    "tight": result.certified_tight,
                    "recovery_method": result.recovery_method,
                    "refinement_used": result.refinement_used,
                    "refined": result.refinement_used,
                    "refined_local_objective": result.refined_local_objective,
                }
            )
    available = np.all(np.isfinite(b_est_xyz), axis=1)
    return b_est_xyz, available, event_landmarks


def simulate_and_estimate(cfg: SimConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)
    t = cfg.dt * np.arange(cfg.steps)
    a_true_pose = make_known_a_pose(t)
    a_pose = make_estimated_a_pose(a_true_pose, cfg, rng)
    b_true_pose = make_event_b_pose(t, cfg, a_true_pose)
    b_true_xyz = b_true_pose[:, :3]

    clean_ranges, true_azimuths, true_elevations = predict_usbl_observations(
        a_true_pose, b_true_xyz, cfg, use_true_mount=True
    )
    if np.max(np.abs(b_true_xyz[:, 2])) > cfg.target_depth_rating_m:
        raise ValueError(
            f"simulated target depth exceeds {cfg.target_transponder_name} "
            f"depth rating ({cfg.target_depth_rating_m:.1f} m)"
        )

    measured_ranges = clean_ranges + rng.normal(0.0, cfg.range_sigma_m, cfg.steps)
    measured_azimuths = wrap_angle(
        true_azimuths + rng.normal(0.0, cfg.angular_accuracy_rad, cfg.steps)
    )
    measured_elevations = true_elevations + rng.normal(
        0.0, cfg.angular_accuracy_rad, cfg.steps
    )
    true_depths = b_true_xyz[:, 2]
    measured_depths = true_depths + rng.normal(0.0, cfg.depth_sigma_m, cfg.steps)

    stationary_mask = _stationary_mask(cfg)
    update_mask = _usbl_update_mask(t, cfg)
    out_of_range_mask, out_of_coverage_mask = _active_range_coverage(
        clean_ranges, true_azimuths, cfg
    )
    detection_valid_mask = ~(out_of_range_mask | out_of_coverage_mask)
    cora_feed_mask = stationary_mask & update_mask & detection_valid_mask
    event_measurement_groups = _event_measurement_groups(cora_feed_mask, cfg)

    window_results = []
    for event_group in event_measurement_groups:
        indices = event_group["measurement_indices"]
        result = solve_cora_window(
            event_group["event_index"],
            event_group["event_start"],
            event_group["event_end"],
            indices,
            a_pose,
            measured_ranges,
            measured_depths,
            cfg,
        )
        l_hat_sdp = result.l_hat_sdp
        if (
            cfg.cora_refine_with_full
            and np.all(np.isfinite(l_hat_sdp))
            and result.status in {"optimal", "optimal_inaccurate"}
        ):
            refine_initial = (
                usbl_static_initial_guess(
                    a_pose[indices],
                    measured_ranges[indices],
                    measured_azimuths[indices],
                    measured_elevations[indices],
                    cfg,
                )
                if cfg.use_usbl_angles
                else l_hat_sdp
            )
            l_hat_refined, refined_local_objective = refine_static_landmark(
                refine_initial,
                a_pose[indices],
                measured_ranges[indices],
                measured_azimuths[indices],
                measured_elevations[indices],
                measured_depths[indices],
                cfg,
            )
            result = CoraWindowResult(
                event_index=result.event_index,
                start=result.start,
                end=result.end,
                indices=result.indices,
                status=result.status,
                sdp_valid=result.sdp_valid,
                rank_tight=result.rank_tight,
                certified_tight=result.certified_tight,
                invalid_reasons=result.invalid_reasons,
                z00_error=result.z00_error,
                min_eig_z=result.min_eig_z,
                psd_violation=result.psd_violation,
                max_range_constraint_residual=result.max_range_constraint_residual,
                min_slack_plus=result.min_slack_plus,
                min_slack_minus=result.min_slack_minus,
                min_slack=result.min_slack,
                sdp_objective_nonnegative=result.sdp_objective_nonnegative,
                sdp_objective=result.sdp_objective,
                sdp_primal_objective=result.sdp_primal_objective,
                sdp_gap=result.sdp_gap,
                sdp_relative_gap=result.sdp_relative_gap,
                slack_sum=result.slack_sum,
                rank_ratio=result.rank_ratio,
                recovery_method=result.recovery_method,
                l_hat_sdp=result.l_hat_sdp,
                a_hat_window_sdp=result.a_hat_window_sdp,
                l_hat_refined=l_hat_refined,
                l_hat_final=l_hat_refined,
                refinement_used=True,
                refined_local_objective=refined_local_objective,
            )
        window_results.append(result)

    b_est_xyz, b_est_available_mask, event_landmark_estimates = (
        _stitch_window_estimates(cfg, window_results)
    )
    b_init_xyz = initial_guess(a_pose[:, :3], cfg)
    b_cov_xyz = estimate_position_covariance(b_est_xyz, a_pose, cfg)
    b_est_pose = np.column_stack([b_est_xyz, np.zeros((cfg.steps, 3))])

    statuses = [result.status for result in window_results]
    sdp_objectives = np.array(
        [result.sdp_objective for result in window_results], dtype=float
    )
    sdp_primal_objectives = np.array(
        [result.sdp_primal_objective for result in window_results], dtype=float
    )
    sdp_gaps = np.array([result.sdp_gap for result in window_results], dtype=float)
    sdp_relative_gaps = np.array(
        [result.sdp_relative_gap for result in window_results], dtype=float
    )
    sdp_valid_flags = np.array(
        [result.sdp_valid for result in window_results], dtype=bool
    )
    rank_tight_flags = np.array(
        [result.rank_tight for result in window_results], dtype=bool
    )
    certified_tight_flags = np.array(
        [result.certified_tight for result in window_results], dtype=bool
    )
    refined_local_objectives = np.array(
        [result.refined_local_objective for result in window_results], dtype=float
    )
    slack_sums = np.array([result.slack_sum for result in window_results], dtype=float)
    rank_ratios = np.array(
        [result.rank_ratio for result in window_results], dtype=float
    )
    sdp_valid_count = int(np.sum(sdp_valid_flags))
    rank_tight_count = int(np.sum(rank_tight_flags))
    certified_tight_count = int(np.sum(certified_tight_flags))
    sdp_all_valid = bool(window_results) and bool(np.all(sdp_valid_flags))
    sdp_diagnostic_success = sdp_all_valid
    local_refinement_success = bool(window_results) and all(
        result.refinement_used
        and np.all(np.isfinite(result.l_hat_final))
        and np.isfinite(result.refined_local_objective)
        for result in window_results
    )
    pipeline_success = bool(window_results) and (
        sdp_all_valid or local_refinement_success
    )
    invalid_reasons_by_event = {
        int(result.event_index): result.invalid_reasons
        for result in window_results
        if not result.sdp_valid
    }
    if window_results and all(result.refinement_used for result in window_results):
        published_estimate_source = "local_refinement"
    elif window_results and all(
        not result.refinement_used for result in window_results
    ):
        published_estimate_source = "sdp_recovery"
    elif window_results:
        published_estimate_source = "mixed"
    else:
        published_estimate_source = "none"
    sdp_objective_sum = float(np.nansum(sdp_objectives))
    refined_local_objective_sum = float(np.nansum(refined_local_objectives))
    sdp_primal_objective_sum = float(np.nansum(sdp_primal_objectives))
    if published_estimate_source in {"local_refinement", "mixed"}:
        published_cost = refined_local_objective_sum
    else:
        published_cost = sdp_primal_objective_sum
    optimizer = SimpleNamespace(
        success=pipeline_success,
        cost=published_cost,
        nfev=len(window_results),
        status=",".join(sorted(set(statuses))) if statuses else "not_run",
        sdp_all_valid=sdp_all_valid,
        sdp_diagnostic_success=sdp_diagnostic_success,
        local_refinement_success=local_refinement_success,
        pipeline_success=pipeline_success,
        published_estimate_source=published_estimate_source,
        published_cost=published_cost,
        sdp_objective_sum=sdp_objective_sum,
        refined_local_objective_sum=refined_local_objective_sum,
    )

    return {
        "cfg": cfg,
        "t": t,
        "a_true_pose": a_true_pose,
        "a_pose": a_pose,
        "b_true_pose": b_true_pose,
        "b_init_xyz": b_init_xyz,
        "b_est_pose": b_est_pose,
        "b_est_available_mask": b_est_available_mask,
        "b_cov_xyz": b_cov_xyz,
        "measured_ranges": measured_ranges,
        "measured_azimuths": measured_azimuths,
        "measured_elevations": measured_elevations,
        "measured_depths": measured_depths,
        "clean_ranges": clean_ranges,
        "true_azimuths": true_azimuths,
        "true_elevations": true_elevations,
        "true_depths": true_depths,
        "stationary_ping_mask": stationary_mask,
        "usbl_update_mask": update_mask,
        "usbl_detection_valid_mask": detection_valid_mask,
        "usbl_out_of_range_mask": out_of_range_mask,
        "usbl_out_of_coverage_mask": out_of_coverage_mask,
        "cora_feed_mask": cora_feed_mask,
        "optimizer": optimizer,
        "optimizer_backend_note": "cora dense SDP backend; no artificial target smoothness",
        "event_landmark_estimates": event_landmark_estimates,
        "num_events": len(event_landmark_estimates),
        "cora_window_results": window_results,
        "cora_statuses": statuses,
        "cora_sdp_objectives": sdp_objectives,
        "cora_sdp_primal_objectives": sdp_primal_objectives,
        "cora_sdp_gaps": sdp_gaps,
        "cora_sdp_relative_gaps": sdp_relative_gaps,
        "cora_sdp_valid": sdp_valid_flags,
        "cora_rank_tight": rank_tight_flags,
        "cora_certified_tight": certified_tight_flags,
        "cora_sdp_valid_count": sdp_valid_count,
        "cora_rank_tight_count": rank_tight_count,
        "cora_certified_tight_count": certified_tight_count,
        "cora_sdp_all_valid": sdp_all_valid,
        "cora_sdp_diagnostic_success": sdp_diagnostic_success,
        "cora_local_refinement_success": local_refinement_success,
        "cora_pipeline_success": pipeline_success,
        "cora_published_estimate_source": published_estimate_source,
        "cora_published_cost": published_cost,
        "cora_sdp_objective_sum": sdp_objective_sum,
        "cora_sdp_primal_objective_sum": sdp_primal_objective_sum,
        "cora_refined_local_objective_sum": refined_local_objective_sum,
        "cora_invalid_reasons_by_event": invalid_reasons_by_event,
        "cora_refined_local_objectives": refined_local_objectives,
        "cora_objectives": sdp_objectives,
        "cora_primal_objectives": sdp_primal_objectives,
        "cora_certificate_gaps": sdp_gaps,
        "cora_relative_certificate_gaps": sdp_relative_gaps,
        "cora_slack_sums": slack_sums,
        "cora_rank_ratios": rank_ratios,
        "cora_tight_count": certified_tight_count,
        "cora_window_count": len(window_results),
        "cora_measurement_windows": [
            group["measurement_indices"] for group in event_measurement_groups
        ],
        "cora_event_measurement_groups": event_measurement_groups,
    }
