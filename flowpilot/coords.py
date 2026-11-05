"""Body frame coordinate transforms for FlowPilot.

Body frame is defined per-sample using differential flatness (mirrors flatness.hpp):
  - Origin: current position p0
  - Body Z axis: thrust direction = normalize(acc + drag_correction + g_world_z)
  - Yaw: atan2(vy, vx)  (fallback: identity quat when speed < 1e-3)
  - Full quaternion from tilt (acc-derived) × yaw rotation

Physical parameters match the data-collection / simulation vehicle model.
"""

import numpy as np

# Physical parameters — PX4 Gazebo iris model
# Drag from rotor H-force: 4 * hover_omega(~794) * rotorDragCoeff(0.000175) * v
# See PX4-Autopilot/.../models/iris/iris.sdf
MASS = 1.5     # kg
GRAV = 9.81    # m/s²
DH   = 0.556   # effective horizontal drag (from rotor H-force)
CP   = 0.0     # no parasitic drag in PX4 Gazebo
VEPS = 1e-6    # speed smoothing factor (prevents divide-by-zero)


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """[qw, qx, qy, qz] → 3×3 rotation matrix R (world ← body).

    Matches the quaternion convention used by the Flightmare data collector.
    v_world = R @ v_body
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),    2*(x*z + w*y)],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def compute_body_quat(
    vel: np.ndarray,
    acc: np.ndarray,
    jer: np.ndarray,
    dh: float = DH,
    cp: float = CP,
    mass: float = MASS,
    grav: float = GRAV,
    veps: float = VEPS,
    yaw_bias: float = 0.0,
) -> np.ndarray:
    """Compute body frame quaternion from vel/acc/jerk via differential flatness.

    Mirrors the quaternion computation in flatness.hpp::forward().
    Fallback: identity quaternion when speed < 1e-3 (matches collect_gcopter_trajs.py).

    Args:
        yaw_bias: additional yaw offset (rad) added on top of the velocity-derived
                  yaw.  Pass the per-trajectory value from yaw_bias.npy so the body
                  frame matches the camera orientation used during depth rendering.

    Returns:
        quat: [qw, qx, qy, qz] float64, world ← body convention
    """
    speed = float(np.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2))
    if speed < 1e-3:
        return np.array([1.0, 0.0, 0.0, 0.0])

    psi = np.arctan2(vel[1], vel[0]) + yaw_bias

    # Drag-weighted velocity (parasitic drag term)
    cp_term = np.sqrt(vel @ vel + veps)
    w_term = 1.0 + cp * cp_term
    w = w_term * vel

    # Unnormalized thrust direction: acc + horizontal_drag_accel + gravity
    zu = acc + (dh / mass) * w
    zu = zu.copy()
    zu[2] += grav

    zu_norm = np.linalg.norm(zu)
    z = zu / zu_norm  # body Z axis in world frame

    # Tilt quaternion (aligns world Z to body Z)
    tilt_den = np.sqrt(2.0 * (1.0 + z[2]))
    tilt0 = 0.5 * tilt_den
    tilt1 = -z[1] / tilt_den
    tilt2 =  z[0] / tilt_den

    # Combine tilt with yaw rotation
    c_half = np.cos(0.5 * psi)
    s_half = np.sin(0.5 * psi)
    qw = tilt0 * c_half
    qx = tilt1 * c_half + tilt2 * s_half
    qy = tilt2 * c_half - tilt1 * s_half
    qz = tilt0 * s_half

    return np.array([qw, qx, qy, qz])


def world_to_body(
    wp_world: np.ndarray,
    p0: np.ndarray,
    R_wb: np.ndarray,
) -> np.ndarray:
    """Transform [N, 12] waypoints from world frame to body frame.

    Args:
        wp_world: [N, 12] float32 — [px,py,pz, vx,vy,vz, ax,ay,az, jx,jy,jz] world frame
        p0:       [3] — origin of body frame in world coordinates
        R_wb:     [3, 3] — rotation matrix world ← body (from quat_to_rot)

    Returns:
        wp_body: [N, 12] float32
    """
    R_bw = R_wb.T  # body ← world (transpose of world←body)
    wp_body = np.empty_like(wp_world)
    # Position: translate then rotate
    wp_body[:, 0:3] = (wp_world[:, 0:3] - p0[None]) @ R_bw.T
    # Velocity, acceleration, jerk: rotate only (vectors, not points)
    wp_body[:, 3:6]  = wp_world[:, 3:6]  @ R_bw.T
    wp_body[:, 6:9]  = wp_world[:, 6:9]  @ R_bw.T
    wp_body[:, 9:12] = wp_world[:, 9:12] @ R_bw.T
    return wp_body


def body_to_world(
    wp_body: np.ndarray,
    p0: np.ndarray,
    R_wb: np.ndarray,
) -> np.ndarray:
    """Inverse of world_to_body. Transform [N, 12] body-frame waypoints back to world.

    Args:
        wp_body: [N, 12] — body frame waypoints
        p0:      [3] — body frame origin in world coordinates
        R_wb:    [3, 3] — rotation matrix world ← body

    Returns:
        wp_world: [N, 12]
    """
    wp_world = np.empty_like(wp_body)
    # Position: rotate then translate
    wp_world[:, 0:3] = wp_body[:, 0:3] @ R_wb.T + p0[None]
    # Velocity, acceleration, jerk: rotate only
    wp_world[:, 3:6]  = wp_body[:, 3:6]  @ R_wb.T
    wp_world[:, 6:9]  = wp_body[:, 6:9]  @ R_wb.T
    wp_world[:, 9:12] = wp_body[:, 9:12] @ R_wb.T
    return wp_world


def get_body_frame(
    vel: np.ndarray,
    acc: np.ndarray,
    jer: np.ndarray,
    yaw_bias: float = 0.0,
) -> tuple:
    """Compute body frame transform parameters from current state vel/acc/jerk.

    Args:
        yaw_bias: additional yaw offset (rad); pass the per-trajectory value from
                  yaw_bias.npy so the frame matches the camera orientation used
                  during depth image rendering.

    Returns:
        quat: [4] float64 — [qw, qx, qy, qz] body quaternion
        R_wb: [3, 3] float64 — rotation matrix world ← body
    """
    quat = compute_body_quat(vel, acc, jer, yaw_bias=yaw_bias)
    R_wb = quat_to_rot(quat)
    return quat, R_wb
