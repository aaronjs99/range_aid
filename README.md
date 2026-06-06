# Range-Aided Pose Estimation Demo

This project is a compact Python demo of **range-aided position estimation** for
a moving boat and a moving underwater target.

The simulated setup is:

- **Object A:** a surface boat / Heron-style USV.
- **Object B:** a moving underwater device, such as a handheld sonar carried by
  a diver.
- **Known information:** the boat pose over time, representing a DLiO-style
  estimate from a VLP-16 lidar and Xsens IMU.
- **Measurements:** acoustic range from B to A, with optional USBL
  azimuth/elevation and optional depth.
- **Estimated state:** the 3D position trajectory of B.

The demo is intentionally small and readable. It does not require ROS, GTSAM, or
Gazebo. It uses NumPy, SciPy, Matplotlib, and YAML.

## Why Range Alone Is Not Enough

At time step `k`, the unknown underwater position is:

```text
B_k = [x_k, y_k, z_k]^T in R^3
```

The boat position is known:

```text
A_k in R^3
```

A scalar range measurement is:

```text
rho_k = ||B_k - A_k|| + noise
```

The predicted range is:

```text
rho_hat_k = ||B_k - A_k||
```

The range residual is:

```text
r_range_k = (rho_hat_k - rho_k) / sigma_rho
```

This is only **one scalar constraint** on a **3D unknown**. Geometrically, one
range measurement places `B_k` somewhere on a sphere centered at `A_k`:

```text
{ p in R^3 : ||p - A_k|| = rho_k }
```

For a moving B, estimating one independent `B_k` per timestamp from only one
range per timestamp gives:

```text
unknowns:      3N
measurements:  N
missing:       2N directions
```

So pure range-only moving-target estimation can fit the range measurements while
still choosing the wrong point on each range sphere.

## Factor Graph Used Here

The default demo uses a more useful moving-target problem:

```text
A_k known from boat odometry/SLAM
B_k unknown underwater position
rho_k measured range
alpha_k measured azimuth
beta_k measured elevation
```

The objective is:

```text
min_{B_k}
  sum_k ((||B_k - A_k|| - rho_k) / sigma_rho)^2
+ sum_k ((azimuth(B_k - A_k) - alpha_k) / sigma_alpha)^2
+ sum_k ((elevation(B_k - A_k) - beta_k) / sigma_beta)^2
+ sum_k ||(B_{k+1} - 2B_k + B_{k-1}) / sigma_accel||^2
```

The terms mean:

- **Range factor:** keeps `B_k` at the measured acoustic distance from `A_k`.
- **USBL azimuth/elevation factors:** use the sensor angular accuracy to
  constrain the direction from A to B.
- **Smoothness factor:** penalizes large second differences in the B trajectory:

```text
B_{k+1} - 2B_k + B_{k-1} approx 0
```

This is a simple acceleration prior. It is not B odometry; it only encodes that
a diver-carried handheld sonar should not teleport between timestamps.

An optional depth factor is implemented but disabled by default:

```text
r_depth_k = (B_{k,z} - d_k) / sigma_depth
```

With depth enabled, each range sphere becomes a circle at the measured depth,
and the smoothness prior couples those circles over time.

## Sensor Assumptions

The config uses a nested sensor layout:

```text
sensors.diver.range
sensors.boat.imu
sensors.boat.lidar
```

The estimator reads the diver range block:

```text
sensors.diver.range.range_sigma_m
sensors.diver.range.angular_accuracy_deg
```

The current default models a Sonardyne Micro-Ranger 2 MRT USBL-style
transceiver:

- frequency band: `20-34 kHz`
- max range: `995 m`
- depth rating: `25 m`
- acoustic coverage: `>200 deg`
- range accuracy: better than `15 mm`, modeled as `sigma_rho = 0.015 m`
- angular accuracy: `3 deg`, used when `estimator.use_usbl_angles: true`

The boat sensor blocks document the assumed DLiO-style pose source:

- `sensors.boat.lidar`: Velodyne VLP-16 Puck-style lidar specs
- `sensors.boat.imu`: Xsens MTi-30 AHRS-style IMU specs

The demo covariance values are intentionally simple and are not a full
datasheet-derived DLiO covariance model.

## Configuration

All tunable values live in:

```text
config/default.yaml
```

Important estimator switches:

```yaml
estimator:
  use_smoothness_factor: true
  smoothness_accel_sigma_mps2: 0.35
  use_usbl_angles: true
  use_depth_factor: false
  depth_sigma_m: 0.10
```

Useful experiments:

- Turn off `use_usbl_angles` to see the range-sphere ambiguity return.
- Turn off `use_smoothness_factor` to see each timestamp become independent.
- Turn on `use_depth_factor` to add depth measurements.
- Increase `smoothness_accel_sigma_mps2` to allow more aggressive diver motion.
- Decrease `smoothness_accel_sigma_mps2` to force a smoother path.

## Code Layout

```text
run.py                  command-line entrypoint
scripts/config.py       YAML parsing and typed config objects
scripts/simulation.py   synthetic A and B trajectory generation
scripts/estimation.py   range/angle/depth/smoothness residuals and optimizer
scripts/figures.py      static PNG plots
scripts/video.py        four-view MP4 rendering
scripts/plot_utils.py   shared plotting helpers
scripts/reporting.py    text summary output
scripts/pipeline.py     end-to-end orchestration
```

## Run

From the project root:

```bash
python3 run.py
```

Use a custom config:

```bash
python3 run.py --config config/default.yaml
```

Skip MP4 generation for a faster check:

```bash
python3 run.py --no-video
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
- range / azimuth / elevation / depth consistency,
- trajectory RMSE,
- smoothness acceleration RMS,
- optimizer status.

`range_aided_pose_estimation.png` shows:

- A trajectory,
- true moving B trajectory,
- initial B trajectory,
- estimated B trajectory,
- range fit over time,
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

If `use_usbl_angles` and `use_smoothness_factor` are enabled, the trajectory
should be much better constrained than range-only. Remaining error comes from:

- range noise,
- USBL angular noise,
- smoothness prior mismatch,
- geometry between A and B,
- the fact that this estimates position only, not full 6-DoF pose.

If only scalar range is enabled for a moving B, large position error is expected
even when range consistency is excellent. That is not a bug; it is the geometry
of the problem.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
