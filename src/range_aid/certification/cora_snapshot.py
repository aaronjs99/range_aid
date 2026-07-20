"""CORA-style SDP diagnostics for immutable known-landmark snapshots.

This certifies the tightness and feasibility of a fixed-pose landmark
subproblem. It is deliberately not described as a certificate for the complete
online GTSAM pose graph; that graph contains additional odometry and boundary
priors with a different objective.
"""

from __future__ import annotations

import math
from typing import Dict, List

import cvxpy as cp
import numpy as np


def _quadratic_distance_matrix(anchor: np.ndarray) -> np.ndarray:
    matrix = np.zeros((4, 4), dtype=float)
    matrix[0, 0] = float(anchor @ anchor)
    matrix[0, 1:] = -anchor
    matrix[1:, 0] = -anchor
    matrix[1:, 1:] = np.eye(3)
    return matrix


def certify_snapshot(
    snapshot: Dict[str, object],
    *,
    solver: str = "SCS",
    rank_tolerance: float = 1e-3,
    feasibility_tolerance: float = 1e-3,
) -> Dict[str, object]:
    """Solve one small lifted landmark problem per observed landmark."""
    observations = list(snapshot.get("observations", []) or [])
    landmarks = dict(snapshot.get("landmarks", {}) or {})
    by_landmark: Dict[str, List[dict]] = {}
    for observation in observations:
        by_landmark.setdefault(str(observation["landmark_id"]), []).append(observation)
    reports = []
    for landmark_id, group in sorted(by_landmark.items()):
        if landmark_id not in landmarks or len(group) < 4:
            continue
        known = np.asarray(landmarks[landmark_id], dtype=float)
        z = cp.Variable((4, 4), symmetric=True)
        slack_plus = cp.Variable(len(group), nonneg=True)
        slack_minus = cp.Variable(len(group), nonneg=True)
        constraints = [z >> 0, z[0, 0] == 1.0]
        terms = []
        matrices = []
        for index, observation in enumerate(group):
            sensor = np.asarray(observation["sensor_position_m"], dtype=float)
            q = _quadratic_distance_matrix(sensor)
            measured_squared = float(observation["range_m"]) ** 2
            constraints.append(
                cp.trace(q @ z) - measured_squared
                == slack_plus[index] - slack_minus[index]
            )
            matrices.append((q, measured_squared))
            terms.append(slack_plus[index] + slack_minus[index])
        prior_q = _quadratic_distance_matrix(known)
        objective = cp.Minimize(cp.trace(prior_q @ z) + cp.sum(cp.hstack(terms)))
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=solver, verbose=False)
        except Exception as exc:
            reports.append(
                {
                    "landmark_id": landmark_id,
                    "status": "error:{}".format(type(exc).__name__),
                    "tight": False,
                    "reasons": ["solver_error"],
                }
            )
            continue
        if z.value is None:
            reports.append(
                {
                    "landmark_id": landmark_id,
                    "status": str(problem.status),
                    "tight": False,
                    "reasons": ["no_sdp_solution"],
                }
            )
            continue
        z_value = np.asarray(z.value, dtype=float)
        eigenvalues = np.linalg.eigvalsh(0.5 * (z_value + z_value.T))
        positive = np.sort(np.maximum(eigenvalues, 0.0))[::-1]
        rank_ratio = (
            float(positive[1] / positive[0])
            if len(positive) > 1 and positive[0] > 1e-12
            else 0.0
        )
        residuals = [
            abs(float(np.trace(q @ z_value) - measured_squared - plus + minus))
            for (q, measured_squared), plus, minus in zip(
                matrices,
                np.asarray(slack_plus.value, dtype=float),
                np.asarray(slack_minus.value, dtype=float),
            )
        ]
        max_residual = max(residuals) if residuals else math.inf
        reasons = []
        if str(problem.status) not in {"optimal", "optimal_inaccurate"}:
            reasons.append("solver_status")
        if rank_ratio > rank_tolerance:
            reasons.append("relaxation_not_rank_tight")
        if max_residual > feasibility_tolerance:
            reasons.append("sdp_constraint_residual")
        reports.append(
            {
                "landmark_id": landmark_id,
                "status": str(problem.status),
                "tight": not reasons,
                "reasons": reasons,
                "measurement_count": len(group),
                "rank_ratio": rank_ratio,
                "max_constraint_residual": max_residual,
                "objective": float(problem.value),
            }
        )
    reasons = []
    if not reports:
        reasons.append("no_certifiable_landmark_window")
    for report in reports:
        reasons.extend(
            "{}:{}".format(report["landmark_id"], reason)
            for reason in report.get("reasons", [])
        )
    return {
        "epoch": int(snapshot.get("epoch", 0)),
        "backend": "cora_landmark_snapshot_diagnostic",
        "tight": bool(reports) and all(bool(report.get("tight")) for report in reports),
        "formal_full_graph_certificate": False,
        "reasons": reasons,
        "landmarks": reports,
        "latest_pose_position_m": list(
            snapshot.get("latest_pose_position_m", []) or []
        ),
        "latest_pose_quaternion_wxyz": list(
            snapshot.get("latest_pose_quaternion_wxyz", []) or []
        ),
        "latest_stamp_sec": float(snapshot.get("latest_stamp_sec", 0.0)),
        "latest_pose_covariance": list(
            snapshot.get("latest_pose_covariance", []) or []
        ),
        "range_count": int(snapshot.get("range_count", 0)),
        "translational_rank": int(snapshot.get("translational_rank", 0)),
        "observability_condition": float(
            snapshot.get("observability_condition", math.inf)
        ),
        "residual_rms_m": float(snapshot.get("residual_rms_m", math.inf)),
        "candidate_gate_passed": bool(
            snapshot.get("candidate_gate_passed", False)
        ),
        "gate_reasons": list(snapshot.get("gate_reasons", []) or []),
        "synthetic_evidence": bool(snapshot.get("synthetic_evidence", False)),
    }
