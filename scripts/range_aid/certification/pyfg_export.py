"""Deterministic adapter from immutable snapshots to official CORA PyFG text."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import gtsam
import numpy as np

from range_aid.estimation.fixed_lag import pose3_from_components, pose3_to_components

OFFICIAL_CORA_REPOSITORY = "https://github.com/MarineRoboticsGroup/cora.git"
OFFICIAL_CORA_COMMIT = "015dc43340ca3aed07226bee1727ea929536fd01"


class PyfgExportError(ValueError):
    """Snapshot cannot be represented without changing its mathematical model."""


def _pose(entry: Dict[str, object]) -> gtsam.Pose3:
    return pose3_from_components(
        tuple(float(item) for item in entry["position_m"]),
        tuple(float(item) for item in entry["quaternion_wxyz"]),
    )


def _pose_fields(pose: gtsam.Pose3) -> str:
    position, quaternion = pose3_to_components(pose)
    w, x, y, z = quaternion
    return "{:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f}".format(
        position[0], position[1], position[2], x, y, z, w
    )


def _symmetric_fields(matrix: np.ndarray) -> str:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape[0] != matrix.shape[1] or not np.allclose(matrix, matrix.T):
        raise PyfgExportError("covariance must be finite and symmetric")
    if not np.all(np.isfinite(matrix)) or np.min(np.linalg.eigvalsh(matrix)) <= 0.0:
        raise PyfgExportError("covariance must be positive definite")
    values = []
    for row in range(matrix.shape[0]):
        for column in range(row, matrix.shape[1]):
            values.append(matrix[row, column])
    return " ".join("{:.9f}".format(value) for value in values)


def _pose_covariance(translation_sigma: float, rotation_sigma: float) -> np.ndarray:
    """PyFG pose covariance order is translation xyz then rotation xyz."""
    return np.diag(
        [float(translation_sigma) ** 2] * 3 + [float(rotation_sigma) ** 2] * 3
    )


def _information_to_pyfg_covariance(information) -> np.ndarray:
    information = np.asarray(information, dtype=float)
    if information.shape != (6, 6) or not np.all(np.isfinite(information)):
        raise PyfgExportError("loop information must be finite 6x6")
    information = 0.5 * (information + information.T)
    if np.min(np.linalg.eigvalsh(information)) <= 0.0:
        raise PyfgExportError("loop information must be positive definite")
    covariance_gtsam = np.linalg.inv(information)
    order = [3, 4, 5, 0, 1, 2]
    return covariance_gtsam[np.ix_(order, order)]


def _identity_extrinsic(snapshot: Dict[str, object], tolerance: float = 1e-12) -> bool:
    extrinsic = dict(snapshot.get("sensor_extrinsic", {}) or {})
    values = list(extrinsic.get("translation_m", []) or []) + list(
        extrinsic.get("rotation_rpy_rad", []) or []
    )
    return len(values) == 6 and all(abs(float(value)) <= tolerance for value in values)


def _content_snapshot_id(snapshot: Dict[str, object]) -> str:
    payload = {
        key: value
        for key, value in snapshot.items()
        if key not in {"snapshot_id", "created_monotonic", "created_wall_sec"}
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_snapshot(snapshot: Dict[str, object]) -> None:
    if int(snapshot.get("schema_version", 0)) != 2:
        raise PyfgExportError("snapshot schema_version 2 is required")
    if not snapshot.get("snapshot_id"):
        raise PyfgExportError("content-addressed snapshot_id is required")
    if str(snapshot["snapshot_id"]) != _content_snapshot_id(snapshot):
        raise PyfgExportError("snapshot content does not match snapshot_id")
    if not snapshot.get("poses"):
        raise PyfgExportError("at least one active pose is required")
    if not _identity_extrinsic(snapshot):
        raise PyfgExportError(
            "official CORA PyFG has no sensor lever-arm field; export is blocked "
            "until an equivalent reviewed reparameterization is validated"
        )


def render_pyfg(snapshot: Dict[str, object]) -> Tuple[str, Dict[str, object]]:
    """Render the bounded non-robust audit instance accepted by official CORA."""
    _validate_snapshot(snapshot)
    poses = sorted(snapshot["poses"], key=lambda item: int(item["pose_index"]))
    landmarks = dict(snapshot.get("landmarks", {}) or {})
    observations = list(snapshot.get("observations", []) or [])
    loops = list(snapshot.get("loop_closures", []) or [])
    objective = dict(snapshot.get("objective_convention", {}) or {})
    pose_names = {
        int(entry["pose_index"]): "A{}".format(index)
        for index, entry in enumerate(poses)
    }
    landmark_names = {
        name: "L{}".format(index) for index, name in enumerate(sorted(landmarks))
    }
    lines: List[str] = []
    for entry in poses:
        pose = _pose(dict(entry["estimate"]))
        lines.append(
            "VERTEX_SE3:QUAT {:.9f} {} {}".format(
                float(entry["stamp_sec"]),
                pose_names[int(entry["pose_index"])],
                _pose_fields(pose),
            )
        )
    for landmark_id in sorted(landmarks):
        point = [float(item) for item in landmarks[landmark_id]["position_m"]]
        lines.append(
            "VERTEX_XYZ {} {:.9f} {:.9f} {:.9f}".format(
                landmark_names[landmark_id], *point
            )
        )

    first = poses[0]
    boundary_covariance = _pose_covariance(
        float(objective["boundary_translation_sigma_m"]),
        float(objective["boundary_rotation_sigma_rad"]),
    )
    lines.append(
        "VERTEX_SE3:QUAT:PRIOR {:.9f} {} {} {}".format(
            float(first["stamp_sec"]),
            pose_names[int(first["pose_index"])],
            _pose_fields(_pose(dict(first["raw_pose"]))),
            _symmetric_fields(boundary_covariance),
        )
    )
    for landmark_id in sorted(landmarks):
        entry = landmarks[landmark_id]
        point = [float(item) for item in entry["position_m"]]
        covariance = np.eye(3) * float(entry["prior_sigma_m"]) ** 2
        lines.append(
            "VERTEX_XYZ:PRIOR 0.000000000 {} {:.9f} {:.9f} {:.9f} {}".format(
                landmark_names[landmark_id], *point, _symmetric_fields(covariance)
            )
        )

    odometry_covariance = _pose_covariance(
        float(objective["odometry_translation_sigma_m"]),
        float(objective["odometry_rotation_sigma_rad"]),
    )
    for previous, current in zip(poses, poses[1:]):
        relative = _pose(dict(previous["raw_pose"])).between(
            _pose(dict(current["raw_pose"]))
        )
        lines.append(
            "EDGE_SE3:QUAT {:.9f} {} {} {} {}".format(
                float(current["stamp_sec"]),
                pose_names[int(previous["pose_index"])],
                pose_names[int(current["pose_index"])],
                _pose_fields(relative),
                _symmetric_fields(odometry_covariance),
            )
        )
    for closure in sorted(loops, key=lambda item: str(item["closure_id"])):
        from_index = int(closure["from_pose_index"])
        to_index = int(closure["to_pose_index"])
        if from_index not in pose_names or to_index not in pose_names:
            raise PyfgExportError("loop closure endpoint is outside the snapshot")
        covariance = _information_to_pyfg_covariance(
            closure["information_rotation_translation"]
        )
        lines.append(
            "EDGE_SE3:QUAT 0.000000000 {} {} {} {}".format(
                pose_names[from_index],
                pose_names[to_index],
                _pose_fields(_pose(dict(closure["relative_pose"]))),
                _symmetric_fields(covariance),
            )
        )
    for observation in sorted(
        observations,
        key=lambda item: (float(item["stamp_sec"]), item["observation_id"]),
    ):
        pose_index = int(observation["pose_index"])
        landmark_id = str(observation["landmark_id"])
        if pose_index not in pose_names or landmark_id not in landmark_names:
            raise PyfgExportError("range endpoint is outside the snapshot")
        variance = float(observation["sigma_m"]) ** 2
        if not math.isfinite(variance) or variance <= 0.0:
            raise PyfgExportError("range variance must be positive and finite")
        lines.append(
            "EDGE_RANGE {:.9f} {} {} {:.9f} {:.9f}".format(
                float(observation["stamp_sec"]),
                pose_names[pose_index],
                landmark_names[landmark_id],
                float(observation["range_m"]),
                variance,
            )
        )
    text = "\n".join(lines) + "\n"
    manifest = {
        "schema_version": 1,
        "snapshot_id": str(snapshot["snapshot_id"]),
        "snapshot_epoch": int(snapshot["epoch"]),
        "snapshot_revision": int(snapshot["revision"]),
        "graph_frame": str(snapshot["graph_frame"]),
        "pyfg_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "official_cora_repository": OFFICIAL_CORA_REPOSITORY,
        "required_official_cora_commit": OFFICIAL_CORA_COMMIT,
        "pose_count": len(poses),
        "landmark_count": len(landmarks),
        "odometry_factor_count": max(0, len(poses) - 1),
        "loop_closure_count": len(loops),
        "range_factor_count": len(observations),
        "audit_objective": "nonrobust_gaussian_pose_range_subgraph",
        "online_objective": str(objective.get("robust_policy", "unknown")),
        "formal_certificate_claimed": False,
        "formal_gate_requirements": [
            "official CORA executable commit matches required pin",
            "official parser accepts this exact pyfg_sha256",
            "GTSAM and CORA evaluate the same non-robust exported objective",
            "machine-readable rank and objective result passes schema validation",
        ],
    }
    return text, manifest


def export_snapshot(
    snapshot: Dict[str, object], destination: Path
) -> Dict[str, object]:
    """Write a PyFG file and content-addressed manifest outside the repository."""
    destination = Path(destination)
    if destination.suffix != ".pyfg":
        raise PyfgExportError("destination must use the .pyfg suffix")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pyfg_text, manifest = render_pyfg(snapshot)
    destination.write_text(pyfg_text, encoding="utf-8")
    manifest_path = destination.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        **manifest,
        "pyfg_path": str(destination),
        "manifest_path": str(manifest_path),
    }
