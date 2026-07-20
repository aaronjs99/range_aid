# Range-Aided Pose Estimation

`range_aid` is GRANDE's bounded, shadow-only research path for adding surveyed
acoustic range constraints to LiDAR-inertial odometry. It publishes an
inspectable alternate odometry stream and correction proposals, but it never
publishes TF, replaces the active navigation state, or promotes itself into
the controller.

The package also retains the earlier offline target-localization experiments.
The online path and offline experiments share the same sensor-frame geometry,
but they answer different questions and must not be treated as interchangeable.

## Runtime Contract

| Topic | Type | Meaning |
| --- | --- | --- |
| `/range_aid/observations` | `range_aid/RangeObservation` | Validated range to a named surveyed landmark, including source and synthetic provenance |
| `/range_aid/odometry_shadow` | `nav_msgs/Odometry` | Alternate bounded estimate for comparison only |
| `/range_aid/correction_proposal` | `range_aid/CorrectionProposal` | Gated, non-authoritative correction candidate |
| `/range_aid/status` | `range_aid/RangeAidStatus` | Backend, coverage, residual, observability, certification, and gate state |

The canonical names live in `config/runtime_surface.yaml`. GRANDE records these
topics in runtime, raw, and SLAM profiles and exposes the status as a diagnostic
domain. Range-aid availability does not gate mapping or water autonomy.

## Online Model

Let `X_k` be the map-frame boat pose, `T_BS` the fixed transform from boat to
the acoustic sensor, and `L_j` a surveyed landmark. For a scalar observation
`z_kj`, the factor residual is

```text
r_kj = || translation(X_k * T_BS) - L_j ||_2 - z_kj .
```

Odometry supplies relative factors

```text
Delta_k = inverse(X_(k-1)^odom) * X_k^odom
r_odom,k = Log(inverse(Delta_k) * inverse(X_(k-1)) * X_k).
```

The active objective is a robust nonlinear least-squares problem:

```text
min_X  ||r_prior||^2 + sum_k ||r_odom,k||^2
       + sum_(k,j) Huber(r_kj / sigma_kj).
```

Known landmarks are represented as tightly rotation-constrained `Pose3`
variables because the installed GTSAM Python binding's native
`RangeFactorWithTransformPose3` requires two `Pose3` keys. Only their
translation affects the range residual.

## Bounded Incremental Estimation

IG Handle's GTSAM Python binding exposes iSAM2 but not either fixed-lag
smoother class. `RebuildingFixedLagSmoother` therefore:

1. updates iSAM2 incrementally while states remain inside the configured lag;
2. retains at most `max_pose_count` states and `lag_sec` seconds of history;
3. rebuilds the active graph when old states leave the lag;
4. carries the prior estimate of the new boundary state with conservative
   configured uncertainty.

This keeps runtime and memory bounded. It is not exact Bayes-tree
marginalization, and the status names the backend `bounded_rebuilding_isam2`
so that limitation remains visible.

## Gates And Certification

Observations are rejected when their landmark is unknown, values are nonfinite
or outside bounds, or timestamp association exceeds the configured tolerance.
A correction candidate also requires minimum range coverage, translational
observability rank, bounded condition number, low residual, and bounded pose
correction.

An asynchronous CVXPY semidefinite diagnostic independently solves each
fixed-pose landmark snapshot and reports rank tightness and constraint
residual. It is intentionally called `cora_landmark_snapshot_diagnostic`:
it does not certify the full nonlinear pose graph. Certification runs outside
the odometry callback so solver latency cannot block the state stream.

Even when every gate passes:

- `navigation_eligible` remains false;
- no `map -> odom`, `odom -> base_link`, or sensor TF is published;
- no surveyed real landmark is inferred from imaging-sonar packets;
- no runtime correction is automatically approved.

## Imaging Sonar Is Separate

The Session 5 imaging sonar currently provides raw UDP datagrams on
`/sensors/sonar/raw`. Its model-specific decoder has not been established.
Those datagrams are not USBL ranges and cannot feed this estimator. A real
range-aid deployment requires a compatible ranging source, a surveyed landmark
position and uncertainty, and a physically measured sensor extrinsic.

The simulation source publishes explicitly synthetic observations and marks
them calibration-ineligible. Synthetic evidence validates contracts and
estimator behavior, not real localization accuracy.

## Running

Build the package and GRANDE dependency closure:

```bash
cd ~/catkin_ws/heron_ws
catkin build range_aid grande --no-status --summarize
```

Run the canonical isolated simulator regression:

```bash
cd ~/catkin_ws/heron_ws/src/grande
python3 grande/run.py --session s5.0 run 0.4 --yes
```

Direct launch remains available for diagnosis:

```bash
roslaunch grande bringup.launch \
  mode:=sim use_range_aid:=true range_aid_mode:=shadow \
  range_aid_synthetic_source:=true
```

## Configuration Ownership

- `config/online.yaml` owns lag, noise, robust-loss, gate, certification,
  extrinsic, and landmark values.
- `config/runtime_surface.yaml` owns ROS topic names.
- `launch/range_aid.launch` adapts those contracts to ROS.
- `grande/launch/bringup.launch` owns optional lifecycle integration.
- `grande/tests/s5.0/test_0_4.yaml` owns the integrated regression scenario.

Do not place physical calibration in a session file. Replace the placeholder
sensor transform and landmark only with measured values and preserve their
source and uncertainty.

## Remaining Real-World Evidence

Before range corrections can be considered for navigation, collect a surveyed
range dataset with synchronized 6-DoF odometry, exercise geometry that makes
3D translation observable, evaluate held-out residuals and multi-step drift,
verify timing and extrinsics, and compare against the unchanged odometry
baseline. Promotion needs explicit review; new data alone is never approval.
