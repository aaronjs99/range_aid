#!/usr/bin/env python3
"""Deterministic validation for range_aid graph identity and factor handling."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

import gtsam
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from range_aid.archive import atomic_json_write, rebuild_full_batch
from range_aid.archive.events import EventArchive, verify_archive
from range_aid.certification.worker import AsynchronousSnapshotCertifier
from range_aid.certification.objective_parity import evaluate_objective_parity
from range_aid.certification.pyfg_export import (
    PyfgExportError,
    export_snapshot,
    render_pyfg,
)
from range_aid.estimation.fixed_lag import (
    FixedLagRangeSmoother,
    RangeMeasurement,
    pose3_from_components,
)
from range_aid.estimation.rtabmap import convert_rtab_information, translate_link
from range_aid.models.config import load_online_config


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _pose(x: float, y: float = 0.0, z: float = 0.0) -> gtsam.Pose3:
    return pose3_from_components((x, y, z), (1.0, 0.0, 0.0, 0.0))


def _range(config, *, sequence: int, stamp_sec: float, valid: bool = True):
    sensor = np.asarray([stamp_sec, 0.0, 0.0]) + np.asarray(config.sensor_translation_m)
    landmark = np.asarray(config.landmarks["sim_transponder_0"].position_m)
    distance = float(np.linalg.norm(sensor - landmark))
    return RangeMeasurement(
        observation_id="validation-range-{:04d}".format(sequence),
        stamp_sec=stamp_sec,
        landmark_id="sim_transponder_0",
        range_m=distance,
        sigma_m=config.range_default_sigma_m,
        has_bearing=False,
        azimuth_rad=0.0,
        elevation_rad=0.0,
        azimuth_variance_rad2=0.0,
        elevation_variance_rad2=0.0,
        valid=valid,
        invalid_reason="provider_rejected" if not valid else "",
        quality_score=1.0 if valid else 0.0,
        quality_flags=0,
        provider="validation",
        provenance_uri="synthetic://estimator-consistency",
        extrinsic_revision=config.extrinsic_revision,
        synthetic=True,
    )


def _transform():
    return SimpleNamespace(
        translation=SimpleNamespace(x=1.0, y=0.0, z=0.0),
        rotation=SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0),
    )


def _link(link_type: int, information):
    return SimpleNamespace(
        fromId=1,
        toId=2,
        type=link_type,
        transform=_transform(),
        information=list(information),
    )


def _validate_smoother(config):
    smoother = FixedLagRangeSmoother(config, archive_id="validation-archive")
    initial_epoch = smoother.epoch
    for index in range(180):
        stamp_sec = index * 0.1
        smoother.add_odometry(stamp_sec, _pose(stamp_sec))
        if index % 10 == 0:
            association = smoother.add_range(
                _range(config, sequence=index, stamp_sec=stamp_sec)
            )
            _assert(association.accepted, association.reason)
    _assert(smoother.epoch == initial_epoch, "normal updates changed graph epoch")
    _assert(smoother.revision > 180, "factor updates did not advance revision")
    _assert(len(smoother.records) <= 123, "fixed lag did not bound active poses")
    _assert(len(smoother.records) >= 115, "fixed lag discarded too much history")
    _assert(smoother.latest_pose() is not None, "latest estimate missing")
    _assert(smoother.latest_covariance() is not None, "marginal covariance missing")
    snapshot_a = smoother.snapshot()
    snapshot_b = smoother.snapshot()
    _assert(snapshot_a["snapshot_id"] == snapshot_b["snapshot_id"], "unstable ID")
    _assert(
        snapshot_a["covariance_model"] == "local_linearized_robust_unvalidated",
        "covariance semantics were not exposed",
    )
    _assert(not snapshot_a["covariance_calibrated"], "covariance overclaimed")

    repeated = _range(config, sequence=170, stamp_sec=17.0)
    duplicate = smoother.add_range(repeated)
    _assert(
        not duplicate.accepted and duplicate.reason == "duplicate_observation",
        "duplicate observation was inserted twice",
    )
    changed = smoother.add_range(replace(repeated, range_m=repeated.range_m + 1.0))
    _assert(
        not changed.accepted and changed.reason == "observation_payload_changed",
        "changed payload reused an immutable observation ID",
    )

    rejected_before = smoother.rejected_observation_count
    rejected = smoother.add_range(
        _range(config, sequence=999, stamp_sec=17.9, valid=False)
    )
    _assert(not rejected.accepted, "provider-invalid range was accepted")
    _assert(
        smoother.rejected_observation_count == rejected_before + 1,
        "range rejection counter did not advance",
    )

    epoch_before_reset = smoother.epoch
    smoother.reset("validation_explicit_reset")
    _assert(smoother.epoch == epoch_before_reset + 1, "reset did not advance epoch")
    _assert(not smoother.records, "reset retained active poses")
    smoother.add_odometry(20.0, _pose(20.0))
    epoch_before_rollback = smoother.epoch
    smoother.add_odometry(18.0, _pose(18.0))
    _assert(
        smoother.epoch == epoch_before_rollback + 1,
        "time rollback did not advance epoch",
    )
    return snapshot_a


def _validate_rtab(config):
    diagonal = np.arange(1.0, 7.0)
    source = np.diag(diagonal)
    converted, reason = convert_rtab_information(source.reshape(-1))
    _assert(reason == "accepted", reason)
    _assert(
        np.allclose(np.diag(converted), [4.0, 5.0, 6.0, 1.0, 2.0, 3.0]),
        "RTAB-to-GTSAM covariance order is wrong",
    )
    invalid, reason = convert_rtab_information(np.zeros(36))
    _assert(invalid is None and reason == "rtab_information_not_spd", reason)

    stamps = {1: 1.0, 2: 2.0}
    map_ids = {1: 7, 2: 7}
    accepted, reason = translate_link(
        _link(1, np.eye(6).reshape(-1)),
        node_stamps=stamps,
        node_map_ids=map_ids,
        config=config,
    )
    _assert(accepted is not None and reason == "accepted", reason)
    rejected, reason = translate_link(
        _link(0, np.eye(6).reshape(-1)),
        node_stamps=stamps,
        node_map_ids=map_ids,
        config=config,
    )
    _assert(rejected is None and reason == "rtab_structural_link_rejected", reason)


def _validate_archive():
    with tempfile.TemporaryDirectory(prefix="range-aid-archive-") as directory:
        archive = EventArchive(
            Path(directory),
            session_id="validation",
            raw_bag_uri="bag://immutable-input",
            extrinsic_revision="validation-extrinsic",
        )
        archive.append("range", {"value": math.inf, "id": "r0"}, sync=True)
        path = archive.path
        archive.close()
        report = verify_archive(path)
        _assert(report["valid"], "archive chain verification failed")
        _assert(report["event_count"] == 2, "archive event count mismatch")
        lines = path.read_text(encoding="utf-8").splitlines()
        tampered = json.loads(lines[1])
        tampered["payload"]["id"] = "changed"
        lines[1] = json.dumps(tampered, sort_keys=True, separators=(",", ":"))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        _assert(not verify_archive(path)["valid"], "archive tampering was missed")


def _validate_certifier(snapshot):
    certifier = AsynchronousSnapshotCertifier()
    try:
        certifier.submit(snapshot)
        deadline = time.monotonic() + 20.0
        result = None
        while time.monotonic() < deadline:
            result, pending = certifier.latest()
            if result is not None and not pending:
                break
            time.sleep(0.05)
        _assert(result is not None, "snapshot diagnostic did not finish")
        _assert(result.snapshot_id == snapshot["snapshot_id"], "snapshot ID changed")
        _assert(
            not result.formal_full_graph_certificate,
            "dense snapshot diagnostic claimed formal CORA certification",
        )
        certifier.invalidate()
        latest, pending = certifier.latest()
        _assert(latest is None and not pending, "invalidation retained stale result")
    finally:
        certifier.close()


def _validate_cora_export(config):
    identity_config = replace(
        config,
        sensor_translation_m=(0.0, 0.0, 0.0),
        sensor_rotation_rpy_rad=(0.0, 0.0, 0.0),
    )
    smoother = FixedLagRangeSmoother(identity_config, archive_id="export-validation")
    for index in range(8):
        stamp_sec = index * 0.1
        smoother.add_odometry(stamp_sec, _pose(stamp_sec))
        smoother.add_range(_range(identity_config, sequence=index, stamp_sec=stamp_sec))
    snapshot = smoother.snapshot()
    pyfg_text, manifest = render_pyfg(snapshot)
    _assert(pyfg_text.startswith("VERTEX_SE3:QUAT "), "PyFG pose header missing")
    _assert(pyfg_text.count("EDGE_RANGE ") == 8, "PyFG range count mismatch")
    _assert(manifest["formal_certificate_claimed"] is False, "export overclaimed")
    with tempfile.TemporaryDirectory(prefix="range-aid-pyfg-") as directory:
        destination = Path(directory) / "audit.pyfg"
        report = export_snapshot(snapshot, destination)
        _assert(
            destination.read_text(encoding="utf-8") == pyfg_text,
            "PyFG write changed content",
        )
        manifest_path = Path(report["manifest_path"])
        _assert(manifest_path.is_file(), "PyFG manifest was not written")
        _assert(
            json.loads(manifest_path.read_text(encoding="utf-8"))["pyfg_sha256"]
            == manifest["pyfg_sha256"],
            "written PyFG manifest changed the content hash",
        )
    official_poses = {"O0": _solver_pose(_pose(0.0))}
    for index, entry in enumerate(snapshot["poses"]):
        official_poses["A{}".format(index)] = _solver_pose(
            _pose_payload(entry["estimate"])
        )
    official_landmarks = {
        "L{}".format(index): {"translation": list(entry[1]["position_m"])}
        for index, entry in enumerate(sorted(snapshot["landmarks"].items()))
    }
    parity = evaluate_objective_parity(
        snapshot,
        {
            "objective": 0.0,
            "poses": official_poses,
            "landmarks": official_landmarks,
        },
    )
    _assert(parity["passed"], "independent CORA objective evaluator drifted")
    gauge = gtsam.Pose3(gtsam.Rot3.RzRyRx(0.0, 0.0, 0.35), np.asarray([4.0, -2.0, 1.0]))
    gauged_poses = {
        name: _solver_pose(gauge.compose(_solver_payload_pose(payload)))
        for name, payload in official_poses.items()
    }
    gauged_landmarks = {
        name: {
            "translation": (
                gauge.rotation().rotate(np.asarray(payload["translation"]))
                + gauge.translation()
            ).tolist()
        }
        for name, payload in official_landmarks.items()
    }
    gauge_parity = evaluate_objective_parity(
        snapshot,
        {
            "objective": 0.0,
            "poses": gauged_poses,
            "landmarks": gauged_landmarks,
        },
    )
    _assert(gauge_parity["passed"], "origin-relative objective lost gauge invariance")
    perturbed_poses = json.loads(json.dumps(official_poses))
    perturbed = _solver_payload_pose(perturbed_poses["A3"])
    perturbed_poses["A3"] = _solver_pose(
        gtsam.Pose3(
            perturbed.rotation().compose(gtsam.Rot3.RzRyRx(0.0, 0.0, 0.12)),
            perturbed.translation(),
        )
    )
    rotation_probe = evaluate_objective_parity(
        snapshot,
        {
            "objective": 0.0,
            "poses": perturbed_poses,
            "landmarks": official_landmarks,
        },
    )
    _assert(
        rotation_probe["official_components"]["pose_rotation"] > 0.0,
        "chordal rotation residual was not evaluated",
    )
    _assert(not rotation_probe["passed"], "objective mismatch was not rejected")
    tampered = dict(snapshot)
    tampered["revision"] += 1
    try:
        render_pyfg(tampered)
    except PyfgExportError as exc:
        _assert("snapshot_id" in str(exc), "tamper rejection reason was ambiguous")
    else:
        raise AssertionError("tampered snapshot was exported")
    try:
        render_pyfg(_with_nonidentity_extrinsic(snapshot))
    except PyfgExportError as exc:
        _assert("lever-arm" in str(exc), "extrinsic rejection reason was ambiguous")
    else:
        raise AssertionError("nonidentity sensor extrinsic was silently exported")


def _validate_full_batch(config):
    records = []
    for index in range(20):
        stamp_sec = index * 0.1
        records.append(
            {
                "event_type": "odometry_factor",
                "payload": {
                    "graph_epoch": 3,
                    "pose_index": index,
                    "stamp_sec": stamp_sec,
                    "graph_pose": {
                        "position_m": [stamp_sec, 0.0, 0.0],
                        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    },
                },
            }
        )
        measurement = _range(config, sequence=index, stamp_sec=stamp_sec)
        records.append(
            {
                "event_type": "range_observation_raw",
                "payload": {
                    "graph_epoch": 3,
                    "observation_id": measurement.observation_id,
                    "stamp_sec": stamp_sec,
                    "landmark_id": measurement.landmark_id,
                    "range_m": measurement.range_m,
                    "variance_m2": measurement.sigma_m**2,
                    "valid": True,
                },
            }
        )
        if index % 2 == 0:
            records.append(
                {
                    "event_type": "range_factor_association",
                    "payload": {
                        "graph_epoch": 3,
                        "observation_id": measurement.observation_id,
                        "accepted": True,
                    },
                }
            )
    report = rebuild_full_batch(records, config, epoch=3)
    _assert(report["pose_count"] == 20, "full batch pose count mismatch")
    _assert(report["range_factor_count"] == 20, "full batch lost archived ranges")
    _assert(
        report["delayed_range_factor_count"] == 10,
        "full batch did not recover delayed ranges",
    )
    _assert(
        report["final_objective"] <= report["initial_objective"] + 1e-9,
        "full batch optimization increased objective",
    )


def _with_nonidentity_extrinsic(snapshot):
    changed = json.loads(json.dumps(snapshot))
    changed["sensor_extrinsic"]["translation_m"] = [0.1, 0.0, 0.0]
    payload = {
        key: value
        for key, value in changed.items()
        if key not in {"snapshot_id", "created_monotonic", "created_wall_sec"}
    }
    changed["snapshot_id"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return changed


def _pose_payload(payload):
    return pose3_from_components(
        tuple(payload["position_m"]), tuple(payload["quaternion_wxyz"])
    )


def _solver_pose(pose):
    return {
        "translation": np.asarray(pose.translation(), dtype=float).tolist(),
        "rotation": np.asarray(pose.rotation().matrix(), dtype=float).tolist(),
    }


def _solver_payload_pose(payload):
    return gtsam.Pose3(
        gtsam.Rot3(np.asarray(payload["rotation"], dtype=float)),
        np.asarray(payload["translation"], dtype=float),
    )


def _validate_json_writer():
    with tempfile.TemporaryDirectory(prefix="range-aid-json-") as directory:
        path = Path(directory) / "report.json"
        atomic_json_write(path, {"value": 1})
        raw = path.read_bytes()
        _assert(raw.endswith(b"\n") and b"\r" not in raw, "JSON must use LF")
        _assert(json.loads(raw.decode("utf-8")) == {"value": 1}, "invalid JSON")
        try:
            atomic_json_write(path, {"value": float("nan")})
        except ValueError:
            pass
        else:
            raise AssertionError("non-finite JSON must be rejected")
        _assert(path.read_bytes() == raw, "failed atomic write changed prior artifact")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "config" / "online.yaml",
    )
    args = parser.parse_args()
    config = load_online_config(args.config)
    snapshot = _validate_smoother(config)
    _validate_rtab(config)
    _validate_archive()
    _validate_certifier(snapshot)
    _validate_cora_export(config)
    _validate_full_batch(config)
    _validate_json_writer()
    report = {
        "status": "passed",
        "backend": "gtsam_unstable.IncrementalFixedLagSmoother",
        "checks": [
            "bounded exact fixed-lag state",
            "epoch versus revision semantics",
            "time rollback and explicit reset",
            "provider-invalid range rejection",
            "duplicate and changed observation identity rejection",
            "snapshot identity stability",
            "local uncalibrated covariance labeling",
            "RTAB closure filtering and information ordering",
            "hash-chained archive tamper detection",
            "stale diagnostic invalidation",
            "dense diagnostic cannot claim formal CORA certification",
            "content-addressed official-CORA PyFG export guardrails",
            "independent official-CORA objective convention evaluator",
            "archive-derived full-batch rebuild with delayed ranges",
            "Python 3.8-compatible deterministic JSON output",
        ],
        "snapshot_sha256": hashlib.sha256(
            snapshot["snapshot_id"].encode("utf-8")
        ).hexdigest(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
