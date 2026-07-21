# Range-Aided Pose Estimation

`range_aid` is GRANDE's standalone, shadow-only research back end for adding
provider-neutral acoustic observations and independently accepted RTAB-Map
closure links to local odometry. It publishes an inspectable map-frame estimate
and correction proposals. It never publishes TF, replaces navigation state, or
commands hardware.

The online graph and evidence archive have different jobs:

- `gtsam_unstable.IncrementalFixedLagSmoother` bounds online latency and memory.
- A hash-chained JSONL archive preserves raw observations, associations, resets,
  closures, and immutable certification snapshots for delayed full-batch work.

Marginal covariance is explicitly labeled
`local_linearized_robust_unvalidated`. Robust range factors reduce outlier
influence, but do not undo fixed-lag marginalization or establish statistical
consistency. Promotion requires held-out NEES/NIS and coverage evidence.

## Runtime contract

| Surface | Type | Meaning |
| --- | --- | --- |
| `/range_aid/observations` | `range_aid/RangeObservation` | Provider-neutral range/bearing observation with immutable identity, validity, uncertainty, provider provenance, and extrinsic revision |
| `/range_aid/odometry_shadow` | `nav_msgs/Odometry` | Non-authoritative estimate in `map` |
| `/range_aid/correction_proposal` | `range_aid/CorrectionProposal` | Snapshot-addressed candidate; always navigation-ineligible |
| `/range_aid/status` | `range_aid/RangeAidStatus` | Epoch, revision, snapshot, archive, covariance, observability, and assurance state |
| `/range_aid/reset` | `std_srvs/Trigger` | Clears only the active smoother, advances the epoch, preserves the archive, and invalidates pending diagnostics |

`graph_epoch` changes only at a discontinuity such as an explicit reset, time
rollback, source-frame change, or active-graph reconstruction.
`graph_revision` advances for accepted graph mutations within an epoch.

## Frame and factor policy

RTAB-Map owns `map -> odom` during shadow operation. At the beginning of each
epoch, range_aid acquires `map <- odom` once and transforms raw local odometry
poses into `map`. Subsequent local odometry enters once as relative
`BetweenFactorPose3` constraints. Ordinary odometry is never repeated as
absolute pose priors.

Known landmarks are `Point3` variables with `PriorFactorPoint3`; scalar ranges
use `RangeFactorWithTransform3D` and the configured body-to-ranging-sensor
extrinsic. RTAB-Map link types 1 through 4 may enter as closures after full
information-matrix validation. Neighbor, neighbor-merged, virtual, unknown,
non-SPD, duplicate, and out-of-lag links are rejected. RTAB odometry edges are
never imported.

Raw observations are immutable. Corrections change the optimized keyframe pose,
not the original sensor record. Every archive event carries sequence and hash
chain information; accepted certification snapshots are stored in full.

## Assurance boundaries

The asynchronous dense CVXPY path is named `snapshot_sdp_diagnostic`. Its rank
result is diagnostic only and cannot satisfy a formal CORA gate.

`scripts/export_cora_snapshot.py` exports an immutable bounded snapshot to the
official CORA PyFG format and writes a manifest containing the snapshot hash,
PyFG hash, and required official CORA commit. The adapter deliberately blocks
nonidentity sensor lever arms because the official `EDGE_RANGE` record has no
lever-arm field. The currently configured extrinsic is an unmeasured nonzero
placeholder, so official export remains blocked until a reviewed equivalent
reparameterization and objective-parity test exist. It never silently drops the
extrinsic.

`scripts/run_official_cora.py` verifies the official repository pin and clean
state, compiles the versioned machine-readable adapter against an external CORA
build, and runs a deterministic seeded solve. The adapter reports CORA's raw
state including its origin pose, objective, post-solve official-library
certificate check, and solver metadata. A separate Python evaluator recomputes
CORA's half-weighted chordal/scalar-precision objective and scores the GTSAM
state with that same convention. `official_certificate` and
`formal_gate_passed` remain separate: the latter also requires objective parity
and configured GTSAM/CORA state agreement. A certified disagreement is rejected
and never injected into the online estimator.

The official dependency is not vendored. The required audit pin is:

```text
repository: https://github.com/MarineRoboticsGroup/cora.git
commit:     015dc43340ca3aed07226bee1727ea929536fd01
```

A tight official CORA result would certify only the exact exported mathematical
instance. It would not validate sensors, covariances, outlier assumptions, or
future graphs. A non-tight result is initialization/lower-bound evidence only.
Any material GTSAM/CORA disagreement is a rejection and convention/model audit,
not an automatic correction.

SCORE is wired as an external, pinned offline initialization baseline. The
adapter requires the official SCORE repository at commit
`41626b49702d27a8fca03982533ff52f6306278d` and PyFactorGraph at commit
`87e18e9bab56b08dfe95e998c801226acba2439b`, then requests SCORE's SOCP
relaxation on the same hashed PyFG/manifest pair. SCORE also requires
`gurobipy` and a usable Gurobi license. Those dependencies are not vendored, and
the adapter writes an explicit unavailable result instead of inventing a SCORE
solution when they are absent. DCORA remains deferred until a genuine
multi-robot graph and measured communication requirement exist.

The pinned external dependencies can be installed outside Git under
`~/.local/share/range_aid`:

```bash
mkdir -p ~/.local/share/range_aid/deps ~/.local/share/range_aid/build

git clone https://github.com/MarineRoboticsGroup/cora.git \
  ~/.local/share/range_aid/deps/cora
git -C ~/.local/share/range_aid/deps/cora checkout --detach \
  015dc43340ca3aed07226bee1727ea929536fd01
git -C ~/.local/share/range_aid/deps/cora submodule update --init --recursive
cmake -S ~/.local/share/range_aid/deps/cora \
  -B ~/.local/share/range_aid/build/cora-015dc433 \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF \
  -DPERFORM_EXPERIMENTS=OFF -DENABLE_VISUALIZATION=OFF
cmake --build ~/.local/share/range_aid/build/cora-015dc433 \
  --target cora_example -j2

git clone https://github.com/MarineRoboticsGroup/score.git \
  ~/.local/share/range_aid/deps/score
git -C ~/.local/share/range_aid/deps/score checkout --detach \
  41626b49702d27a8fca03982533ff52f6306278d
git clone https://github.com/MarineRoboticsGroup/PyFactorGraph.git \
  ~/.local/share/range_aid/deps/PyFactorGraph
git -C ~/.local/share/range_aid/deps/PyFactorGraph checkout --detach \
  87e18e9bab56b08dfe95e998c801226acba2439b
```

SCORE additionally requires `gurobipy`, a valid Gurobi license, `attrs`, and
PyFactorGraph's runtime dependencies. On a headless host, configure the external
EVO dependency to use the `Agg` plotting backend.

## Archive and export tools

Generated archives, extracted snapshots, PyFG files, and solver outputs belong
outside Git.

```bash
rosrun range_aid range_aid_archive.py verify /path/to/session.jsonl
rosrun range_aid range_aid_archive.py extract-snapshot \
  /path/to/session.jsonl /tmp/snapshot.json --snapshot-id SHA256
rosrun range_aid range_aid_archive.py rebuild-full-batch \
  /path/to/session.jsonl config/online.yaml /tmp/full-batch.json --epoch 3
rosrun range_aid export_cora_snapshot.py \
  /tmp/snapshot.json /tmp/audit.pyfg
rosrun range_aid run_official_cora.py \
  /tmp/snapshot.json /tmp/audit.pyfg /tmp/audit.manifest.json \
  /tmp/cora.json \
  --cora-repo ~/.local/share/range_aid/deps/cora \
  --cora-build ~/.local/share/range_aid/build/cora-015dc433
rosrun range_aid run_score_baseline.py \
  /tmp/audit.pyfg /tmp/audit.manifest.json /tmp/score.json \
  --score-repo ~/.local/share/range_aid/deps/score \
  --pyfactorgraph-repo ~/.local/share/range_aid/deps/PyFactorGraph
```

The export produces `/tmp/audit.pyfg` and `/tmp/audit.manifest.json`.

## Validation

```bash
cd ~/catkin_ws/heron_ws/src/grande/range_aid
PYTHONPATH=scripts python3 scripts/validation/validate_estimator_consistency.py

cd ~/catkin_ws/heron_ws
catkin build range_aid grande --no-status --summarize
```

The deterministic validator covers exact lag bounding, reset and rollback
epochs, revision semantics, invalid observations, snapshot identity, covariance
labeling, RTAB type/information filtering, archive tamper detection, stale
diagnostic invalidation, and CORA-export guardrails. Integrated shadow replay is
validated separately with `validate_shadow_runtime.py`.

## Remaining promotion evidence

Navigation promotion is intentionally unavailable. Required work includes
deterministic archived full-batch reconstruction, delayed-loop and reset replay,
covariance coverage/NEES/NIS calibration against independent ground truth,
measured acoustic extrinsics, surveyed landmarks, multipath/NLOS tests, and
field trials. Official CORA ingestion, objective parity, disagreement handling,
and a licensed SCORE smoke solve now work on identity-extrinsic fixtures; they
are not physical validation. GTSAM can own `map -> odom` only after the remaining
gates pass and RTAB-Map TF publication is disabled first.
