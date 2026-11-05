"""Quaternion and frame-transformation utilities.

Convention: quaternions are [w, x, y, z] (Hamilton, scalar-first).
"""

import numpy as np


def quat_to_rot(q):
    """[qw, qx, qy, qz] -> 3x3 rotation matrix (world <- body)."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def quat_mul(q1, q2):
    """Hamilton product: q = q1 * q2 (both wxyz)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)


def safe_quat(q):
    """Normalize quaternion, guard against NaN/zero-norm."""
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12 or not np.isfinite(n):
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def obs_to_world_pva(obs):
    """Convert Flightmare observation (body-frame vel/acc) to world-frame PVA.

    Args:
        obs: ndarray [num_envs, obs_dim], uses env 0.
             Layout: [0:3]=pos_w, [3:6]=vel_b, [6:9]=acc_b, [9:13]=quat_wxyz

    Returns:
        pos, vel, acc: each ndarray [3], world frame.
    """
    pos = obs[0, 0:3].copy()
    quat = obs[0, 9:13]
    R = quat_to_rot(quat)
    vel = R @ obs[0, 3:6]
    acc = R @ obs[0, 6:9]
    return pos, vel, acc
