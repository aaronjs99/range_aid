# Range-Aided Pose Estimation Demo

`range_aid` is a standalone Python research demo for range-aided underwater
target localization. It models a surface USV carrying a USBL transceiver and an
underwater target carrying a transponder.

This package is not part of the active GRANDE runtime. It is a compact,
readable experiment for studying acoustic range and bearing constraints. It
does not launch from `grande/run.py`, publish a production ROS interface, or
contribute to real-time navigation decisions.

## Problem

The simulated system has:

- a surface boat with a noisy LiDAR-inertial pose estimate,
- a fixed USBL sensor extrinsic on the boat,
- an underwater target or landmark,
- acoustic range and optional azimuth/elevation observations, and
- optional depth or smoothness priors depending on the backend.

The estimator solves for target position. It does not estimate the target's
full 6-DoF pose.

## Measurement Model

USBL measurements are sensor-frame measurements, not world-frame measurements.
At time `k`:

```text
A_k_true = true boat pose in map
A_k_est  = estimated boat pose used by the optimizer
T_A_S    = fixed boat-to-USBL transform
B_k      = target position in map
S_k      = A_k * T_A_S
```

The target vector in the USBL frame is:

```text
q_k = R_S_k.T @ (B_k - p_S_k)
```

Predicted acoustic observations are:

```text
range     = norm(q_k)
azimuth   = atan2(q_y, q_x)
elevation = atan2(q_z, hypot(q_x, q_y))
```

This sensor-frame formulation is required whenever the USBL is offset from the
boat origin or the boat frame is not aligned with the map frame.

## Backends

| Backend | Purpose |
| --- | --- |
| `full` | Local nonlinear moving-target baseline |
| `cora` | Event-triggered stationary-target CORA-style SDP backend |

### `full`

The `full` backend estimates a target position at every time step and can use
range, optional bearing, optional depth, and optional smoothness. Smoothness is a
modeling prior, not a real target odometry measurement.

### `cora`

The `cora` backend estimates stationary event landmarks during explicit ping
windows. It does not use target smoothness and does not invent target odometry.
After a ping event, the recovered landmark is archived.

The default CORA mode fixes the estimated boat/USBL poses as anchors and solves
for the landmark. The larger boat-variable mode also estimates boat positions
inside the event window and reports diagnostics separately.

## Why Range Alone Is Ambiguous

One scalar range measurement constrains a 3D target to a sphere around the USBL
sensor. A moving target with one range per time step has three position
unknowns and one scalar measurement per step, so additional information or
priors are required for a meaningful estimate.

## Typical Use

Install Python dependencies in the environment used for the demo:

```bash
pip install numpy scipy matplotlib pyyaml
```

Run a backend from the package scripts or notebooks configured for the active
experiment. Generated plots, videos, and dense trial output should stay outside
source control unless reduced to a compact reviewed summary.

## Workspace Role

`range_aid` is useful for acoustic-localization experiments and future graph
factor design. Production GRANDE localization currently uses LiDAR-inertial
odometry plus RTAB-Map global mapping; acoustic range factors are future graph
evidence, not a replacement for the current runtime stack.

## Development Notes

- Keep the sensor-frame measurement model explicit.
- Separate measurement generation from estimator inputs.
- Report rank, residual, and physical-bounds diagnostics for SDP experiments.
- Do not interpret a feasible range-only solution as a unique ground truth.
