"""Independent evaluator for CORA's exported chordal pose-range objective."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import gtsam
import numpy as np

from range_aid.estimation.fixed_lag import pose3_from_components


def _snapshot_pose(payload) -> Tuple[np.ndarray, np.ndarray]:
    pose = pose3_from_components(
        tuple(float(item) for item in payload["position_m"]),
        tuple(float(item) for item in payload["quaternion_wxyz"]),
    )
    return np.asarray(pose.rotation().matrix()), np.asarray(pose.translation())


def _result_pose(payload) -> Tuple[np.ndarray, np.ndarray]:
    rotation = np.asarray(payload["rotation"], dtype=float)
    translation = np.asarray(payload["translation"], dtype=float)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("pose state has invalid dimensions")
    if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
        raise ValueError("pose state is non-finite")
    return rotation, translation


def _relative_pose(first, second) -> Tuple[np.ndarray, np.ndarray]:
    first_pose = pose3_from_components(
        tuple(float(item) for item in first["position_m"]),
        tuple(float(item) for item in first["quaternion_wxyz"]),
    )
    second_pose = pose3_from_components(
        tuple(float(item) for item in second["position_m"]),
        tuple(float(item) for item in second["quaternion_wxyz"]),
    )
    relative = first_pose.between(second_pose)
    return np.asarray(relative.rotation().matrix()), np.asarray(relative.translation())


def _scalar_precisions(covariance: np.ndarray) -> Tuple[float, float]:
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (6, 6):
        raise ValueError("pose covariance must be 6x6")
    translation_trace = float(np.trace(covariance[:3, :3]))
    rotation_trace = float(np.trace(covariance[3:, 3:]))
    if translation_trace <= 0.0 or rotation_trace <= 0.0:
        raise ValueError("pose covariance traces must be positive")
    return 3.0 / translation_trace, 1.5 / rotation_trace


def _isotropic_pose_covariance(
    translation_sigma: float, rotation_sigma: float
) -> np.ndarray:
    return np.diag([translation_sigma**2] * 3 + [rotation_sigma**2] * 3)


def _state_from_official(result: Dict[str, object]):
    poses = {
        str(name): _result_pose(payload)
        for name, payload in dict(result.get("poses", {}) or {}).items()
    }
    landmarks = {
        str(name): np.asarray(payload["translation"], dtype=float)
        for name, payload in dict(result.get("landmarks", {}) or {}).items()
    }
    if "O0" not in poses:
        raise ValueError("official CORA state is missing its origin pose O0")
    return poses, landmarks


def _state_from_snapshot(snapshot: Dict[str, object]):
    rows = sorted(snapshot["poses"], key=lambda item: int(item["pose_index"]))
    poses = {
        "A{}".format(index): _snapshot_pose(entry["estimate"])
        for index, entry in enumerate(rows)
    }
    poses["O0"] = (np.eye(3), np.zeros(3))
    landmarks = {
        "L{}".format(index): np.asarray(entry[1]["position_m"], dtype=float)
        for index, entry in enumerate(sorted(dict(snapshot["landmarks"]).items()))
    }
    return poses, landmarks


def _evaluate(snapshot: Dict[str, object], poses, landmarks) -> Dict[str, float]:
    rows = sorted(snapshot["poses"], key=lambda item: int(item["pose_index"]))
    pose_names = {
        int(entry["pose_index"]): "A{}".format(index)
        for index, entry in enumerate(rows)
    }
    landmark_names = {
        name: "L{}".format(index)
        for index, name in enumerate(sorted(dict(snapshot["landmarks"])))
    }
    objective = dict(snapshot["objective_convention"])
    components = {
        "pose_rotation": 0.0,
        "pose_translation": 0.0,
        "range": 0.0,
    }

    def add_pose_factor(
        first_name: str,
        second_name: str,
        measured_rotation: np.ndarray,
        measured_translation: np.ndarray,
        covariance: np.ndarray,
    ) -> None:
        first_rotation, first_translation = poses[first_name]
        second_rotation, second_translation = poses[second_name]
        translation_precision, rotation_precision = _scalar_precisions(covariance)
        rotation_residual = second_rotation - first_rotation @ measured_rotation
        translation_residual = (
            second_translation
            - first_translation
            - first_rotation @ measured_translation
        )
        components["pose_rotation"] += (
            0.5 * rotation_precision * float(np.sum(rotation_residual**2))
        )
        components["pose_translation"] += (
            0.5
            * translation_precision
            * float(translation_residual @ translation_residual)
        )

    boundary_covariance = _isotropic_pose_covariance(
        float(objective["boundary_translation_sigma_m"]),
        float(objective["boundary_rotation_sigma_rad"]),
    )
    first = rows[0]
    prior_rotation, prior_translation = _snapshot_pose(first["raw_pose"])
    add_pose_factor("O0", "A0", prior_rotation, prior_translation, boundary_covariance)

    odometry_covariance = _isotropic_pose_covariance(
        float(objective["odometry_translation_sigma_m"]),
        float(objective["odometry_rotation_sigma_rad"]),
    )
    for index, (previous, current) in enumerate(zip(rows, rows[1:])):
        rotation, translation = _relative_pose(
            previous["raw_pose"], current["raw_pose"]
        )
        add_pose_factor(
            "A{}".format(index),
            "A{}".format(index + 1),
            rotation,
            translation,
            odometry_covariance,
        )

    for closure in snapshot.get("loop_closures", []) or []:
        information = np.asarray(
            closure["information_rotation_translation"], dtype=float
        )
        covariance_gtsam = np.linalg.inv(0.5 * (information + information.T))
        order = [3, 4, 5, 0, 1, 2]
        covariance_pyfg = covariance_gtsam[np.ix_(order, order)]
        rotation, translation = _snapshot_pose(closure["relative_pose"])
        add_pose_factor(
            pose_names[int(closure["from_pose_index"])],
            pose_names[int(closure["to_pose_index"])],
            rotation,
            translation,
            covariance_pyfg,
        )

    origin_rotation, origin_translation = poses["O0"]
    for landmark_id, entry in sorted(dict(snapshot["landmarks"]).items()):
        landmark_name = landmark_names[landmark_id]
        measured = np.asarray(entry["position_m"], dtype=float)
        residual = (
            landmarks[landmark_name] - origin_translation - origin_rotation @ measured
        )
        precision = 1.0 / float(entry["prior_sigma_m"]) ** 2
        components["pose_translation"] += 0.5 * precision * float(residual @ residual)

    for observation in snapshot.get("observations", []) or []:
        pose_name = pose_names[int(observation["pose_index"])]
        landmark_name = landmark_names[str(observation["landmark_id"])]
        distance = float(np.linalg.norm(landmarks[landmark_name] - poses[pose_name][1]))
        residual = distance - float(observation["range_m"])
        variance = float(observation["sigma_m"]) ** 2
        if not math.isfinite(variance) or variance <= 0.0:
            raise ValueError("range variance must be positive and finite")
        components["range"] += 0.5 * residual**2 / variance

    components["total"] = sum(components.values())
    return components


def evaluate_objective_parity(
    snapshot: Dict[str, object],
    official_result: Dict[str, object],
    *,
    absolute_tolerance: float = 1e-6,
    relative_tolerance: float = 1e-5,
) -> Dict[str, object]:
    """Validate an official result and score the GTSAM state identically."""
    official_components = _evaluate(snapshot, *_state_from_official(official_result))
    gtsam_components = _evaluate(snapshot, *_state_from_snapshot(snapshot))
    official_reported = float(official_result["objective"])
    independent = float(official_components["total"])
    gtsam_state_objective = float(gtsam_components["total"])
    absolute_error = abs(independent - official_reported)
    tolerance = max(
        float(absolute_tolerance),
        float(relative_tolerance) * max(1.0, abs(independent), abs(official_reported)),
    )
    return {
        "convention": "cora_half_weighted_chordal_scalar_precision_objective",
        "official_cora_reported": official_reported,
        "official_cora_independent": independent,
        "official_components": official_components,
        "gtsam_state_same_objective": gtsam_state_objective,
        "gtsam_state_components": gtsam_components,
        "gtsam_cora_state_objective_delta": gtsam_state_objective - independent,
        "gtsam_cora_state_relative_gap": abs(gtsam_state_objective - independent)
        / max(1.0, abs(gtsam_state_objective), abs(independent)),
        "absolute_error": absolute_error,
        "tolerance": tolerance,
        "passed": bool(absolute_error <= tolerance),
    }
