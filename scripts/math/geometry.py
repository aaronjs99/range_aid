"""Rigid-body geometry helpers for the boat-mounted USBL model."""

from __future__ import annotations

import numpy as np


def rotation_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return the body-to-parent rotation matrix for roll, pitch, yaw."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    rot_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rot_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rot_z @ rot_y @ rot_x


def sensor_pose_from_boat_pose(
    a_pose: np.ndarray,
    mount_offset: np.ndarray,
    mount_rpy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return USBL position and orientation in map frame from a boat pose."""
    boat_position = a_pose[:3]
    boat_rotation = rotation_from_rpy(*a_pose[3:6])
    mount_rotation = rotation_from_rpy(*mount_rpy)
    sensor_position = boat_position + boat_rotation @ mount_offset
    sensor_rotation = boat_rotation @ mount_rotation
    return sensor_position, sensor_rotation


def relative_vector_in_sensor_frame(
    a_pose: np.ndarray,
    b_xyz: np.ndarray,
    mount_offset: np.ndarray,
    mount_rpy: np.ndarray,
) -> np.ndarray:
    """Return target vector expressed in the boat-mounted USBL frame."""
    sensor_position, sensor_rotation = sensor_pose_from_boat_pose(
        a_pose, mount_offset, mount_rpy
    )
    return sensor_rotation.T @ (b_xyz - sensor_position)


def relative_vectors_in_sensor_frame(
    a_poses: np.ndarray,
    b_xyz: np.ndarray,
    mount_offset: np.ndarray,
    mount_rpy: np.ndarray,
) -> np.ndarray:
    """Vectorized target vectors expressed in each timestep's USBL frame."""
    return np.array(
        [
            relative_vector_in_sensor_frame(a_pose, target, mount_offset, mount_rpy)
            for a_pose, target in zip(a_poses, b_xyz)
        ]
    )


def spherical_from_sensor_vectors(
    q_vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return range, azimuth, and elevation for vectors in sensor frame."""
    horizontal = np.linalg.norm(q_vectors[:, :2], axis=1)
    ranges = np.linalg.norm(q_vectors, axis=1)
    azimuths = np.arctan2(q_vectors[:, 1], q_vectors[:, 0])
    elevations = np.arctan2(q_vectors[:, 2], horizontal)
    return ranges, azimuths, elevations


def sensor_positions_from_boat_poses(
    a_poses: np.ndarray,
    mount_offset: np.ndarray,
    mount_rpy: np.ndarray,
) -> np.ndarray:
    """Return USBL map-frame positions for each boat pose."""
    return np.array(
        [
            sensor_pose_from_boat_pose(a_pose, mount_offset, mount_rpy)[0]
            for a_pose in a_poses
        ]
    )
