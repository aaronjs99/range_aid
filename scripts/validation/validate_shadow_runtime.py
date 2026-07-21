#!/usr/bin/env python3
"""Validate the integrated synthetic range-aid shadow runtime in simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_srvs.srv import Trigger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from range_aid.msg import CorrectionProposal, RangeAidStatus, RangeObservation


class Observer:
    def __init__(self) -> None:
        self.status = None
        self.observation_count = 0
        self.shadow_count = 0
        self.proposal_count = 0
        rospy.Subscriber("/range_aid/status", RangeAidStatus, self._status_cb)
        rospy.Subscriber(
            "/range_aid/observations", RangeObservation, self._observation_cb
        )
        rospy.Subscriber("/range_aid/odometry_shadow", Odometry, self._shadow_cb)
        rospy.Subscriber(
            "/range_aid/correction_proposal", CorrectionProposal, self._proposal_cb
        )

    def _status_cb(self, message):
        self.status = message

    def _observation_cb(self, _message):
        self.observation_count += 1

    def _shadow_cb(self, _message):
        self.shadow_count += 1

    def _proposal_cb(self, _message):
        self.proposal_count += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-sec", type=float, default=45.0)
    args = parser.parse_args()
    rospy.init_node("validate_range_aid_shadow", anonymous=True)
    if str(rospy.get_param("/grande/runtime/mode", "")) != "sim":
        raise RuntimeError("range-aid command validation is simulation-only")
    observer = Observer()
    command = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    deadline = time.monotonic() + max(10.0, args.timeout_sec)
    rate = rospy.Rate(10.0)
    started = time.monotonic()
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        elapsed = time.monotonic() - started
        message = Twist()
        if elapsed < 5.0:
            message.linear.x = 0.25
        elif elapsed < 10.0:
            message.linear.x = 0.15
            message.angular.z = 0.18
        elif elapsed < 15.0:
            message.angular.z = -0.18
        command.publish(message)
        status = observer.status
        if (
            elapsed >= 15.0
            and status is not None
            and status.active_range_count >= 6
            and status.translational_observability_rank >= 2
            and observer.proposal_count >= 1
        ):
            break
        rate.sleep()
    command.publish(Twist())
    status = observer.status
    if status is None:
        raise RuntimeError("range-aid status was never published")
    epoch_before_reset = int(status.graph_epoch)
    archive_events_before_reset = int(status.archive_event_count)
    rospy.wait_for_service("/range_aid/reset", timeout=5.0)
    reset_response = rospy.ServiceProxy("/range_aid/reset", Trigger)()
    reset_deadline = time.monotonic() + 10.0
    reset_observed = False
    while not rospy.is_shutdown() and time.monotonic() < reset_deadline:
        status = observer.status
        reset_observed = bool(
            status is not None
            and int(status.graph_epoch) == epoch_before_reset + 1
            and int(status.archive_event_count) > archive_events_before_reset
            and status.estimate_available
        )
        if reset_observed:
            break
        rospy.sleep(0.1)
    status = observer.status
    checks = {
        "mode_shadow": status.mode == "shadow",
        "graph_frame_map": status.graph_frame == "map",
        "snapshot_addressed": len(status.snapshot_id) == 64,
        "revision_nonzero": status.graph_revision > 0,
        "archive_available": bool(status.archive_id) and bool(status.archive_path),
        "covariance_honest": (
            status.covariance_model == "local_linearized_robust_unvalidated"
            and not status.covariance_calibrated
        ),
        "dense_diagnostic_not_formal": not status.formal_full_graph_certificate,
        "reset_service": bool(reset_response.success) and reset_observed,
        "estimate_available": bool(status.estimate_available),
        "synthetic_evidence": bool(status.synthetic_evidence),
        "navigation_ineligible": not bool(status.navigation_eligible),
        "observations": observer.observation_count >= 6,
        "shadow_odometry": observer.shadow_count >= 5,
        "correction_proposal": observer.proposal_count >= 1,
        "bounded_pose_count": status.active_pose_count <= 160,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    report = {
        "result": "passed" if not failed else "failed",
        "checks": checks,
        "failed": failed,
        "active_pose_count": status.active_pose_count,
        "active_range_count": status.active_range_count,
        "translational_observability_rank": status.translational_observability_rank,
        "observability_condition": status.observability_condition,
        "range_residual_rms_m": status.range_residual_rms_m,
        "certification_tight": bool(status.certification_tight),
        "gate_reasons": list(status.gate_reasons),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if failed:
        raise RuntimeError(
            "range-aid shadow checks failed: {}".format(", ".join(failed))
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
