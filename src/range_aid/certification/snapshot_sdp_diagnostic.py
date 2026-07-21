"""Dense SDP rank diagnostic for immutable landmark snapshots.

This objective is intentionally narrower than the exported online factor graph.
Rank tightness here is diagnostic only and is never a formal CORA certificate.
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


def evaluate_snapshot_sdp(
    snapshot: Dict[str, object],
    *,
    solver: str = "SCS",
    rank_tolerance: float = 1e-3,
    feasibility_tolerance: float = 1e-3,
) -> Dict[str, object]:
    """Solve one lifted landmark diagnostic per sufficiently observed landmark."""
    observations = list(snapshot.get("observations", []) or [])
    landmarks = dict(snapshot.get("landmarks", {}) or {})
    by_landmark: Dict[str, List[dict]] = {}
    for observation in observations:
        by_landmark.setdefault(str(observation["landmark_id"]), []).append(observation)
    reports = []
    for landmark_id, group in sorted(by_landmark.items()):
        if landmark_id not in landmarks or len(group) < 4:
            continue
        landmark_entry = landmarks[landmark_id]
        known = np.asarray(landmark_entry["position_m"], dtype=float)
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
        problem = cp.Problem(
            cp.Minimize(cp.trace(prior_q @ z) + cp.sum(cp.hstack(terms))),
            constraints,
        )
        try:
            problem.solve(solver=solver, verbose=False)
        except Exception as exc:
            reports.append(
                {
                    "landmark_id": landmark_id,
                    "status": "error:{}".format(type(exc).__name__),
                    "diagnostic_rank_tight": False,
                    "reasons": ["solver_error"],
                }
            )
            continue
        if z.value is None:
            reports.append(
                {
                    "landmark_id": landmark_id,
                    "status": str(problem.status),
                    "diagnostic_rank_tight": False,
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
                "diagnostic_rank_tight": not reasons,
                "reasons": reasons,
                "measurement_count": len(group),
                "rank_ratio": rank_ratio,
                "max_constraint_residual": max_residual,
                "objective": float(problem.value),
            }
        )
    reasons = []
    if not reports:
        reasons.append("no_diagnostic_landmark_window")
    for report in reports:
        reasons.extend(
            "{}:{}".format(report["landmark_id"], reason)
            for reason in report.get("reasons", [])
        )
    diagnostic_rank_tight = bool(reports) and all(
        bool(report.get("diagnostic_rank_tight")) for report in reports
    )
    return {
        **snapshot,
        "backend": "snapshot_sdp_diagnostic",
        "diagnostic_rank_tight": diagnostic_rank_tight,
        "formal_full_graph_certificate": False,
        "formal_certificate_tight": False,
        "reasons": reasons,
        "landmark_diagnostics": reports,
    }
