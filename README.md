# Range-Aided Pose Estimation Demo

This project is a compact Python demo of **range-aided underwater target /
landmark position estimation**.

The simulated setup is:

- **Object A:** a surface boat / Heron-style USV.
- **Object B:** an underwater target, landmark, or diver-carried device.
- **Boat pose source:** a noisy DLiO-style estimate from a VLP-16 lidar and
  Xsens IMU.
- **Acoustic system:** a USBL transceiver mounted on the boat and a transponder
  mounted on the underwater target.
- **Estimated state:** the 3D position of B. The `full` backend estimates a
  moving B trajectory; the `cora` backend estimates a stationary B location
  only during explicit ping windows, then archives that event landmark.

This demo estimates **position only**. It does not estimate the target's full
6-DoF pose. B orientation would require an additional attitude source, scan
matching, multiple receivers, or other orientation-bearing constraints.

The demo is intentionally small and readable. It does not require ROS, GTSAM, or
Gazebo. It uses NumPy, SciPy, Matplotlib, and YAML.

## Why The Sensor Frame Matters

The USBL bearing measurement is not a world-frame bearing. The acoustic
transceiver is mounted on the boat, so range, azimuth, and elevation must be
computed in the USBL sensor frame.

At time step `k` the real simulation has a true boat pose and the estimator has
the boat pose estimate:

```text
A_k_true = true boat pose in map frame
A_k_est  = DLiO-like boat pose estimate used by the optimizer
B_k = unknown target/transponder position in map frame
T_A_S = fixed boat-to-USBL extrinsic transform
S_k = A_k * T_A_S = USBL pose in map frame
```

Measurements are generated from `A_k_true` and the true target position. The
optimizer predicts measurements using `A_k_est`, because in a real system the
SLAM/odometry estimate is what the backend has available.

Let the USBL pose used by the optimizer be:

```text
p_S_k = USBL position in map frame
R_S_k = USBL orientation in map frame
```

The target vector expressed in the USBL frame is:

```text
q_k = R_S_k.T @ (B_k - p_S_k)
```

The predicted USBL measurements are:

```text
range_k     = norm(q_k)
azimuth_k   = atan2(q_y, q_x)
elevation_k = atan2(q_z, hypot(q_x, q_y))
```

This is the physically correct model for a boat-mounted USBL transceiver. The
older shortcut `B_k - A_k` is only valid if the boat frame is always aligned
with the map and the USBL origin is exactly the boat origin.

## Why Range Alone Is Not Enough

One scalar range measurement gives:

```text
rho_k = norm(q_k) + noise
```

For a 3D target position, that is only **one scalar constraint** on **three
unknowns**. Geometrically, one range measurement places the target somewhere on
a sphere around the USBL sensor position:

```text
{ p in R^3 : norm(p - p_S_k) = rho_k }
```

For a moving target, estimating one independent `B_k` per timestamp from only
one range per timestamp gives:

```text
unknowns:      3N
measurements:  N
missing:       2N directions
```

So range-only moving-target estimation can match the measured ranges while
still choosing the wrong point on each range sphere.

## Factor Model Used Here

There are two backends:

- `full`: a local nonlinear moving-target baseline.
- `cora`: the default event-triggered CORA-style stationary-ping SDP backend.

### `full` Baseline

The `full` backend estimates moving target positions:

```text
B_0, B_1, ..., B_N
```

using estimated boat poses, range, optional USBL azimuth/elevation, optional
depth, and optional smoothness.

The USBL residuals are:

```text
r_range_k = (predicted_range_k - measured_range_k) / range_sigma_m

r_azimuth_k =
  wrap(predicted_azimuth_k - measured_azimuth_k) / angular_accuracy_rad

r_elevation_k =
  (predicted_elevation_k - measured_elevation_k) / angular_accuracy_rad
```

The moving-target smoothness residual is:

```text
r_smooth_k =
  (B_{k+1} - 2 * B_k + B_{k-1}) / dt^2 / smoothness_accel_sigma_mps2
```

The `full` objective is:

```text
min over B_0, ..., B_{N-1}:
  sum_k r_range_k^2
+ sum_k r_azimuth_k^2
+ sum_k r_elevation_k^2
+ sum_{k=1}^{N-2} ||r_smooth_k||^2
```

This is a reasonable smoothing problem for a moving target, but the smoothness
factor is still a modeling prior, not a direct target odometry measurement.

### `cora` Backend

The `cora` backend does **not** use target smoothness. The simulated diver /
handheld sonar target moves before the ping window, holds position during the
ping, and then moves again afterward. CORA only consumes measurements from the
stationary ping interval:

```text
before start ping: B moves, measurements are logged but not used by CORA
inside ping:       B_k = L_j, one stationary event transponder location
after end ping:    L_j is archived, while live B position becomes unavailable
```

The default scenario uses five stationary ping events with different durations
so the video and plots show repeated localize/archive cycles rather than a
single event:

```text
[20, 35)   3.0 s
[45, 65)   4.0 s
[113, 128) 3.0 s
[132, 156) 4.8 s
[157, 165) 1.6 s
```

For each solved ping event it estimates:

```text
a_0, ..., a_{W-1} in R^3 = boat positions in the map frame
l in R^3                 = stationary underwater transponder location
```

The boat orientation is fixed from the DLiO-like estimate, and the USBL
position is:

```text
s_k = a_k + Rhat_A_k * p_A_S_est
```

where `p_A_S_est` is the configured boat-to-USBL mount offset. Measurements are
generated with `T_AS_true` and optimized with `T_AS_est`, so fixed mount biases
can be simulated without estimating a separate extrinsic at every time step.

For an event with `W` valid ping measurements:

```text
x = [a_0, ..., a_{W-1}, l]
x_tilde = [1; x]
Z = x_tilde x_tilde.T
```

The dense SDP relaxation uses:

```text
Z >= 0
Z[0,0] = 1
trace(Q_range_k Z) - rho_k^2 = s_plus_k - s_minus_k
s_plus_k >= 0
s_minus_k >= 0
```

and minimizes:

```text
range_slack_weight * sum(s_plus + s_minus)
+ boat_position_prior_terms
+ boat_displacement_prior_terms
+ boat_surface_prior_terms
```

The boat priors are:

```text
||a_k - ahat_k||^2
||(a_{k+1} - a_k) - (ahat_{k+1} - ahat_k)||^2
a_{k,z} ~= 0 with a loose wave-band sigma
```

There is no `B_k` smoothness, no B acceleration prior, and no fake B odometry.
USBL azimuth/elevation are used only by the optional local nonlinear
refinement, not by the SDP relaxation or its rank diagnostic.

This includes the CORA ingredients: range-aided graph construction, lifted
QCQP, SDP relaxation, primal recovery, feasibility/rank diagnostics,
recovered-primal-vs-SDP objective gap reporting, and optional local nonlinear
refinement. It is still an educational dense-CVXPY
implementation, not the performant full CORA implementation with low-rank
Riemannian Staircase and full certificate machinery.
When the relaxation is non-tight or the solver returns `optimal_inaccurate`, the
report says so and records the recovery method. The objective gap is a useful
lower-bound diagnostic, but it is not a full certifiable CORA result.

The SDP diagnostic is intentionally fail-closed. An event is `sdp_valid` only
when the solver returns `optimal`, `Z[0,0]` is close to 1, the lifted matrix is
PSD within tolerance, range equality residuals are small, slack variables remain
nonnegative within tolerance, and the SDP objective is not negative beyond
tolerance. `optimal_inaccurate` and negative SDP lower-bound objectives are
reported as invalid/inconclusive diagnostics, even if local refinement still
finds a useful estimate.

The report uses three distinct terms:

```text
sdp_valid       = SDP feasibility/objective diagnostics pass
rank_tight      = rank_ratio < cora_rank_tightness_tol
certified_tight = sdp_valid and rank_tight
```

Rank ratio alone never makes an event certified.

The report keeps the two CORA stages separate:

```text
l_hat_sdp      = estimate recovered from the SDP relaxation
l_hat_refined  = optional local USBL range+bearing refinement result
l_hat_final    = published event estimate used by plots and scoring
```

SDP status, rank ratio, recovery method, and primal-vs-SDP gap apply to
`l_hat_sdp`. They do **not** certify `l_hat_refined` or `l_hat_final` when local
refinement is enabled. The refined estimate is reported with its own local USBL
objective.

Depth support remains implemented for experiments, but it is disabled by
default:

```text
r_depth_k = (B_{k,z} - measured_depth_k) / depth_sigma_m
```

## Sensor Assumptions

The config uses a physical sensor layout:

```text
sensors.boat.usbl
sensors.target.transponder
sensors.boat.imu
sensors.boat.lidar
```

The default boat USBL is:

```text
Sonardyne MRT Type 8243
role: boat-mounted submerged USBL transceiver
frequency band: 20-34 kHz
max range: 995 m
depth rating: 25 m
acoustic coverage: 200 deg
raw range sigma: 0.015 m
angular accuracy: 3 deg
update rate: 3 Hz
mount offset: [0.6, 0.0, -0.75] m in boat frame
mount RPY: [0.0, 0.0, 0.0] deg relative to boat frame
```

The default target transponder is:

```text
Sonardyne Nano Cabled Transponder
role: underwater tracked target transponder
frequency band: 19-34 kHz
max range: 995 m
depth rating: 500 m
beam shape: 260 deg
range precision: 0.015 m
```

Important caveat:

```text
range_sigma_m = 0.015
```

models raw range precision for the factor, not final USBL system position
accuracy. A full Micro-Ranger 2 system position result can be much looser, often
on the order of percent-of-slant-range depending on geometry, calibration,
sound-speed conditions, synchronization, and acoustic environment. This project
is a factor-level simulation, not a full acoustic navigation system model.

The boat lidar and IMU blocks document the assumed DLiO-style A pose source:

- `sensors.boat.lidar`: Velodyne VLP-16 Puck-style lidar specs
- `sensors.boat.imu`: Xsens MTi-30 AHRS-style IMU specs

The covariance values are intentionally demo-level assumptions, not a full
datasheet-derived DLiO covariance model.

In the simulation, those covariance values generate the A pose estimate used by
the optimizer. The plots show both true A and estimated A.

## Configuration

All tunable values live in:

```text
config/default.yaml
```

Important estimator switches:

```yaml
estimator:
  optimizer_backend: "cora"
  use_smoothness_factor: true
  smoothness_accel_sigma_mps2: 0.35
  use_usbl_angles: true
  use_depth_factor: false
  depth_sigma_m: 0.10
  cora_window_size: 25
  cora_solve_stride: 5
  cora_range_slack_weight: 1.0
  cora_boat_prior_weight: 100.0
  cora_boat_displacement_weight: 25.0
  cora_surface_prior_weight: 4.0
  cora_surface_sigma_m: 2.0
  cora_rank_tightness_tol: 1e-3
  cora_sdp_feasibility_tol: 1e-3
  cora_sdp_objective_tol: 1e-6
  cora_sdp_psd_tol: 1e-5
  cora_refine_with_full: true
  cora_solver: "SCS"
  usbl_mount_bias_m: [0.0, 0.0, 0.0]
  usbl_mount_bias_rpy_deg: [0.0, 0.0, 0.0]
```

Available optimizer backends:

- `full`: current SciPy least-squares optimization over the whole moving B path.
- `cora`: CORA-faithful educational backend using stationary ping windows,
  range-aided QCQP lifting, dense CVXPY SDP relaxation, SDP feasibility
  diagnostics, rank diagnostics, primal recovery, and optional local
  refinement.

Important backend difference:

- `full` is a local nonlinear baseline and may use moving-target smoothness.
- `cora` does **not** use artificial target smoothness. It estimates one
  stationary underwater B location per solved ping window from range
  measurements, boat position priors, boat displacement priors, and a loose
  surface prior on A.

The current `cora` backend includes CORA ingredients, but it is still
educational: it uses a dense CVXPY SDP, not the full performant C++ CORA stack
with low-rank Riemannian Staircase and stronger certificate machinery.

Useful experiments:

- Turn off `use_usbl_angles` to see the range-sphere ambiguity return.
- Turn off `use_smoothness_factor` to see each timestamp become independent.
- Turn on `use_depth_factor` to add depth measurements.
- Increase `smoothness_accel_sigma_mps2` to allow more aggressive target motion.
- Decrease `smoothness_accel_sigma_mps2` to force a smoother path.
- Change `sensors.boat.usbl.mount_offset_m` or `mount_rpy_deg` to test
  extrinsic sensitivity.

## Code Layout

```text
run.py                         command-line entrypoint
scripts/app/pipeline.py        end-to-end orchestration
scripts/configuration/config.py YAML parsing and typed config objects
scripts/math/geometry.py       RPY rotations and boat-to-USBL geometry
scripts/math/usbl.py           shared USBL prediction, angle wrap, covariance
scripts/sim/trajectories.py    synthetic A and B trajectory generation
scripts/optimization/common.py shared optimizer initialization
scripts/optimization/full.py   current full SciPy least-squares optimizer
scripts/optimization/cora.py   event CORA-style dense SDP backend
scripts/reports/summary.py     text summary output
scripts/viz/figures.py         static PNG plots
scripts/viz/video.py           four-view MP4 rendering
scripts/viz/plot_utils.py      shared plotting helpers
```

## Run

From the project root:

```bash
python3 run.py
```

For a faster terminal/PNG-only run:

```bash
python3 run.py --no-video
```

Use a custom config:

```bash
python3 run.py --config config/default.yaml
```

Select the nonlinear baseline by editing:

```yaml
estimator:
  optimizer_backend: "full"
```

Write outputs somewhere else:

```bash
python3 run.py --output-dir outputs/test_run
```

## Outputs

The default outputs are:

```text
outputs/results.txt
outputs/range_aided_pose_estimation.png
outputs/position_estimate.png
outputs/range_aided_pose_estimation_views.mp4
```

`results.txt` reports:

- active factor switches,
- active boat-mounted USBL and target transponder model,
- USBL mount offset and RPY,
- raw range sigma and angular sigma,
- stationary ping windows and USBL samples fed to CORA,
- A pose estimate RMSE,
- B estimation error max / mean / standard deviation,
- active-ping B estimation error max / mean / standard deviation,
- range / azimuth / elevation / depth consistency,
- event-available estimate RMSE,
- smoothness acceleration RMS,
- optimizer status.
- CORA SDP status, validity, rank-tight count, certified-tight count, invalid
  reasons, slack feasibility, lower-bound objective, recovered SDP primal
  objective, SDP gap, recovery method, final published event source, and
  separate refined local USBL objective when using `optimizer_backend: "cora"`.

`range_aided_pose_estimation.png` shows:

- true A trajectory,
- estimated A trajectory,
- true moving B trajectory,
- initial B trajectory,
- active event B estimate,
- archived event landmark marker,
- boat-mounted USBL range fit over time,
- B position error over time.

`position_estimate.png` has six rows:

```text
A x
A y
A z
B x
B y
B z
```

Each row has truth, estimate, shaded uncertainty, and dotted 2-sigma bounds.

The MP4 shows the estimate in four synchronized views:

- top,
- side,
- front,
- isometric 3D.

## Interpreting Error

With `optimizer_backend: "full"`, `use_usbl_angles`, and
`use_smoothness_factor` enabled, the moving target trajectory is much better
constrained than range-only. With `optimizer_backend: "cora"`, B is localized
only during configured stationary ping windows and each result is archived as
an event landmark. That means:

- active-ping B error is the localization quality during the stationary ping,
- moving-time measurements are logged but excluded from CORA,
- pre-ping and post-ping live B estimates are unavailable and are not scored,
- previous event factors leave the active graph unless a future config
  explicitly associates a new ping with the same physical landmark.

Remaining error comes from:

- raw range noise,
- USBL angular noise,
- boat pose uncertainty,
- USBL extrinsic assumptions,
- smoothness prior mismatch,
- geometry between the moving boat and moving target,
- the fact that this estimates target position only, not full 6-DoF pose.

For CORA specifically, the report includes SDP feasibility diagnostics,
rank-ratio diagnostics, slack/objective summaries, recovered-primal-vs-SDP gap,
recovery method, the SDP-recovered landmark, and the final published landmark.
If local refinement is enabled, the final landmark is the refined USBL
range+bearing result, and the SDP diagnostics remain attached only to the
SDP-recovered landmark. An invalid SDP diagnostic does not certify the final
estimate. Those are CORA-style diagnostics, but they are not a complete
certifiable optimality proof; full CORA would replace the dense SDP with
low-rank Riemannian Staircase and stronger certificate checks.

If only scalar range is enabled for a moving B, large position error is expected
even when range consistency is excellent. That is not a bug; it is the geometry
of the problem.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
