"""Deterministic full-batch reconstruction from immutable range_aid events."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import gtsam
import numpy as np

from range_aid.estimation.fixed_lag import pose3_from_components, pose3_to_components
from range_aid.models.config import OnlineConfig


def read_archive_records(path: Path) -> List[Dict[str, object]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _pose(payload) -> gtsam.Pose3:
    return pose3_from_components(
        tuple(float(item) for item in payload["position_m"]),
        tuple(float(item) for item in payload["quaternion_wxyz"]),
    )


def _pose_payload(pose: gtsam.Pose3) -> Dict[str, object]:
    position, quaternion = pose3_to_components(pose)
    return {
        "position_m": position.tolist(),
        "quaternion_wxyz": quaternion.tolist(),
    }


def _pose_noise(translation_sigma: float, rotation_sigma: float):
    return gtsam.noiseModel.Diagonal.Sigmas(
        np.asarray([rotation_sigma] * 3 + [translation_sigma] * 3)
    )


def _sensor_pose(config: OnlineConfig) -> gtsam.Pose3:
    roll, pitch, yaw = config.sensor_rotation_rpy_rad
    return gtsam.Pose3(
        gtsam.Rot3.RzRyRx(roll, pitch, yaw),
        np.asarray(config.sensor_translation_m),
    )


def _nearest_pose(poses, stamp_sec: float, max_error: float):
    if not poses:
        return None
    candidate = min(poses, key=lambda item: abs(item["stamp_sec"] - stamp_sec))
    return candidate if abs(candidate["stamp_sec"] - stamp_sec) <= max_error else None


def rebuild_full_batch(
    records: Iterable[Dict[str, object]],
    config: OnlineConfig,
    *,
    epoch: Optional[int] = None,
) -> Dict[str, object]:
    """Rebuild one epoch without fixed-lag marginalization."""
    records = list(records)
    odometry_events = [
        record for record in records if record.get("event_type") == "odometry_factor"
    ]
    if not odometry_events:
        raise ValueError("archive contains no odometry_factor events")
    available_epochs = sorted(
        {int(record["payload"]["graph_epoch"]) for record in odometry_events}
    )
    selected_epoch = available_epochs[-1] if epoch is None else int(epoch)
    if selected_epoch not in available_epochs:
        raise ValueError("requested epoch has no odometry events")
    poses = []
    for record in odometry_events:
        payload = dict(record["payload"])
        if int(payload["graph_epoch"]) != selected_epoch:
            continue
        poses.append(
            {
                "pose_index": int(payload["pose_index"]),
                "stamp_sec": float(payload["stamp_sec"]),
                "raw_pose": _pose(payload["graph_pose"]),
            }
        )
    poses.sort(key=lambda item: (item["stamp_sec"], item["pose_index"]))
    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()
    landmark_keys = {
        landmark_id: gtsam.symbol("l", index)
        for index, landmark_id in enumerate(sorted(config.landmarks))
    }
    for landmark_id, key in landmark_keys.items():
        landmark = config.landmarks[landmark_id]
        point = np.asarray(landmark.position_m, dtype=float)
        values.insert(key, point)
        graph.add(
            gtsam.PriorFactorPoint3(
                key,
                point,
                gtsam.noiseModel.Isotropic.Sigma(3, landmark.prior_sigma_m),
            )
        )
    for entry in poses:
        values.insert(gtsam.symbol("x", entry["pose_index"]), entry["raw_pose"])
    first = poses[0]
    graph.add(
        gtsam.PriorFactorPose3(
            gtsam.symbol("x", first["pose_index"]),
            first["raw_pose"],
            _pose_noise(
                config.boundary_translation_sigma_m,
                config.boundary_rotation_sigma_rad,
            ),
        )
    )
    for previous, current in zip(poses, poses[1:]):
        graph.add(
            gtsam.BetweenFactorPose3(
                gtsam.symbol("x", previous["pose_index"]),
                gtsam.symbol("x", current["pose_index"]),
                previous["raw_pose"].between(current["raw_pose"]),
                _pose_noise(
                    config.odometry_translation_sigma_m,
                    config.odometry_rotation_sigma_rad,
                ),
            )
        )

    online_accepted_ids = {
        str(record["payload"].get("observation_id", ""))
        for record in records
        if record.get("event_type") == "range_factor_association"
        and bool(record["payload"].get("accepted", False))
        and int(record["payload"].get("graph_epoch", -1)) == selected_epoch
    }
    accepted_ranges = []
    delayed_ranges = 0
    seen_observations = {}
    sensor_pose = _sensor_pose(config)
    for record in records:
        if record.get("event_type") != "range_observation_raw":
            continue
        payload = dict(record["payload"])
        if int(payload.get("graph_epoch", -1)) != selected_epoch:
            continue
        if not bool(payload.get("valid", False)):
            continue
        landmark_id = str(payload.get("landmark_id", ""))
        observation_id = str(payload.get("observation_id", ""))
        payload_identity = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        previous_payload = seen_observations.get(observation_id)
        if not observation_id or previous_payload is not None:
            continue
        seen_observations[observation_id] = payload_identity
        range_m = float(payload.get("range_m", math.nan))
        variance = float(payload.get("variance_m2", math.nan))
        if landmark_id not in landmark_keys or not math.isfinite(range_m):
            continue
        if not math.isfinite(variance) or variance <= 0.0:
            variance = config.range_default_sigma_m**2
        pose_entry = _nearest_pose(
            poses, float(payload["stamp_sec"]), config.max_sync_error_sec
        )
        if pose_entry is None:
            continue
        base = gtsam.noiseModel.Isotropic.Sigma(1, math.sqrt(variance))
        robust = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(config.robust_range_huber_k),
            base,
        )
        graph.add(
            gtsam.RangeFactorWithTransform3D(
                gtsam.symbol("x", pose_entry["pose_index"]),
                landmark_keys[landmark_id],
                range_m,
                robust,
                sensor_pose,
            )
        )
        accepted_ranges.append(observation_id)
        delayed_ranges += int(observation_id not in online_accepted_ids)

    accepted_closures = []
    seen_closures = set()
    for record in records:
        if record.get("event_type") != "rtab_loop_closure":
            continue
        payload = dict(record["payload"])
        if int(payload.get("graph_epoch", -1)) != selected_epoch:
            continue
        closure_id = str(payload.get("closure_id", ""))
        identity = (closure_id, str(payload.get("payload_sha256", "")))
        if not closure_id or identity in seen_closures:
            continue
        seen_closures.add(identity)
        from_pose = _nearest_pose(
            poses, float(payload["from_stamp_sec"]), config.max_sync_error_sec
        )
        to_pose = _nearest_pose(
            poses, float(payload["to_stamp_sec"]), config.max_sync_error_sec
        )
        information = np.asarray(
            payload.get("information_rotation_translation", []), dtype=float
        )
        if (
            from_pose is None
            or to_pose is None
            or from_pose["pose_index"] == to_pose["pose_index"]
            or information.shape != (6, 6)
            or not np.all(np.isfinite(information))
        ):
            continue
        information = 0.5 * (information + information.T)
        if np.min(np.linalg.eigvalsh(information)) <= 0.0:
            continue
        graph.add(
            gtsam.BetweenFactorPose3(
                gtsam.symbol("x", from_pose["pose_index"]),
                gtsam.symbol("x", to_pose["pose_index"]),
                _pose(payload["relative_pose"]),
                gtsam.noiseModel.Gaussian.Information(information),
            )
        )
        accepted_closures.append(closure_id)

    initial_error = float(graph.error(values))
    result = gtsam.LevenbergMarquardtOptimizer(graph, values).optimize()
    final_error = float(graph.error(result))
    trajectory = [
        {
            "pose_index": entry["pose_index"],
            "stamp_sec": entry["stamp_sec"],
            "raw_pose": _pose_payload(entry["raw_pose"]),
            "estimate": _pose_payload(
                result.atPose3(gtsam.symbol("x", entry["pose_index"]))
            ),
        }
        for entry in poses
    ]
    return {
        "schema_version": 1,
        "backend": "gtsam_full_batch_rebuild",
        "graph_epoch": selected_epoch,
        "available_epochs": available_epochs,
        "pose_count": len(poses),
        "range_factor_count": len(accepted_ranges),
        "delayed_range_factor_count": delayed_ranges,
        "loop_closure_count": len(accepted_closures),
        "initial_objective": initial_error,
        "final_objective": final_error,
        "covariance_model": "full_batch_local_linearized_robust_unvalidated",
        "covariance_calibrated": False,
        "trajectory": trajectory,
        "accepted_observation_ids": accepted_ranges,
        "accepted_closure_ids": accepted_closures,
    }
