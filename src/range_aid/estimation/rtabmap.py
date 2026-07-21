"""Translate RTAB-Map closure links without importing odometry edges."""

from __future__ import annotations

import hashlib
import json
from typing import Dict, Optional, Tuple

import gtsam
import numpy as np

from range_aid.estimation.fixed_lag import LoopClosureMeasurement
from range_aid.models.config import OnlineConfig

ACCEPTED_CLOSURE_TYPES = (1, 2, 3, 4)
REJECTED_STRUCTURAL_TYPES = (0, 5, 6)
RTAB_TO_GTSAM_ORDER = (3, 4, 5, 0, 1, 2)


def convert_rtab_information(values) -> Tuple[Optional[np.ndarray], str]:
    """Convert RTAB [translation, rotation] information to GTSAM order."""
    matrix = np.asarray(values, dtype=float)
    if matrix.size != 36:
        return None, "rtab_information_wrong_size"
    matrix = matrix.reshape(6, 6)
    if not np.all(np.isfinite(matrix)):
        return None, "rtab_information_nonfinite"
    matrix = matrix[np.ix_(RTAB_TO_GTSAM_ORDER, RTAB_TO_GTSAM_ORDER)]
    matrix = 0.5 * (matrix + matrix.T)
    if np.min(np.linalg.eigvalsh(matrix)) <= 0.0:
        return None, "rtab_information_not_spd"
    return matrix, "accepted"


def _transform_pose(transform) -> gtsam.Pose3:
    rotation = transform.rotation
    translation = transform.translation
    return gtsam.Pose3(
        gtsam.Rot3.Quaternion(
            float(rotation.w),
            float(rotation.x),
            float(rotation.y),
            float(rotation.z),
        ),
        np.asarray([float(translation.x), float(translation.y), float(translation.z)]),
    )


def translate_link(
    link,
    *,
    node_stamps: Dict[int, float],
    node_map_ids: Dict[int, int],
    config: OnlineConfig,
) -> Tuple[Optional[LoopClosureMeasurement], str]:
    """Return an accepted closure candidate or an explicit rejection reason."""
    link_type = int(link.type)
    if link_type not in ACCEPTED_CLOSURE_TYPES:
        return None, "rtab_structural_link_rejected"
    if link_type not in config.accepted_rtabmap_link_types:
        return None, "rtab_link_type_disabled"
    from_id, to_id = int(link.fromId), int(link.toId)
    if from_id not in node_stamps or to_id not in node_stamps:
        return None, "rtab_link_missing_node_stamp"
    information, reason = convert_rtab_information(link.information)
    fallback = False
    if information is None:
        if not config.allow_rtabmap_information_fallback:
            return None, reason
        sigmas = np.asarray(
            [config.loop_closure_rotation_sigma_rad] * 3
            + [config.loop_closure_translation_sigma_m] * 3
        )
        information = np.diag(1.0 / np.square(sigmas))
        fallback = True
    relative_pose = _transform_pose(link.transform)
    graph_identity = "rtabmap:{}:{}".format(
        node_map_ids.get(from_id, -1), node_map_ids.get(to_id, -1)
    )
    closure_id = "{}:{}:{}:{}".format(graph_identity, from_id, to_id, link_type)
    payload = {
        "closure_id": closure_id,
        "translation_m": np.asarray(relative_pose.translation()).tolist(),
        "rotation_matrix": np.asarray(relative_pose.rotation().matrix()).tolist(),
        "information_rotation_translation": information.tolist(),
    }
    payload_sha256 = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return (
        LoopClosureMeasurement(
            closure_id=closure_id,
            graph_identity=graph_identity,
            from_rtab_id=from_id,
            to_rtab_id=to_id,
            from_stamp_sec=float(node_stamps[from_id]),
            to_stamp_sec=float(node_stamps[to_id]),
            link_type=link_type,
            relative_pose=relative_pose,
            information=information,
            payload_sha256=payload_sha256,
            used_information_fallback=fallback,
        ),
        "accepted_with_information_fallback" if fallback else "accepted",
    )
