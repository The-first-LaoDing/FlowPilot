"""WorldState dataclass + body-frame state_goal construction.

This module is a direct port of the identical code originally duplicated across
the runtime entry scripts. Pure numpy; no torch, no flowpilot imports.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np


@dataclass
class WorldState:
    pos:  np.ndarray   # [3] world frame
    vel:  np.ndarray   # [3] world frame
    acc:  np.ndarray   # [3] world frame
    jerk: np.ndarray   # [3] world frame


def build_state_goal(
    cur: WorldState,
    goal_pos_world: np.ndarray,
    get_body_frame_fn: Callable,
    quat_wb_override: Optional[np.ndarray] = None,
    r_wb_override:    Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build FlowPilot state_goal [25] in body frame.

    Body frame is derived from differential flatness via get_body_frame_fn
    (data-collection convention) unless the caller overrides with
    quat_wb_override + r_wb_override (px4 uses measured odometry attitude;
    rviz uses a yaw-toward-goal quaternion on the first frame).

    Args:
        cur:            WorldState with pos, vel, acc, jerk (all [3], world frame).
        goal_pos_world: [3] goal position, world frame.
        get_body_frame_fn: Typically flowpilot.coords.get_body_frame; signature
                           (vel, acc, jerk) -> (quat_wb [4] wxyz, r_wb [3,3]).
        quat_wb_override: [4] wxyz quaternion, optional.
        r_wb_override:    [3,3] rotation (world <- body), optional.

    Returns:
        state_goal: [25] float32 — [pos_b, vel_b, acc_b, jerk_b, quat_wb,
                                    goal_pos_b, goal_vel_b, goal_acc_b]
        quat_wb:    [4]   float64
        r_wb:       [3,3] float64
    """
    if quat_wb_override is not None:
        quat_wb = quat_wb_override.astype(np.float64)
        r_wb    = r_wb_override.astype(np.float64)
    else:
        quat_wb, r_wb = get_body_frame_fn(cur.vel, cur.acc, cur.jerk)
    # q and -q encode the same rotation; keep the scalar part non-negative so
    # the 25-D model input matches the sign convention used during collection.
    if quat_wb[0] < 0:
        quat_wb = -quat_wb
    r_bw       = r_wb.T
    state_pos  = np.zeros(3, dtype=np.float32)
    state_vel  = (r_bw @ cur.vel ).astype(np.float32)
    state_acc  = (r_bw @ cur.acc ).astype(np.float32)
    state_jerk = (r_bw @ cur.jerk).astype(np.float32)
    goal_pos_b = (r_bw @ (goal_pos_world - cur.pos)).astype(np.float32)
    goal_vel_b = np.zeros(3, dtype=np.float32)
    goal_acc_b = np.zeros(3, dtype=np.float32)
    state_goal = np.concatenate([state_pos, state_vel, state_acc, state_jerk,
                                  quat_wb.astype(np.float32),
                                  goal_pos_b, goal_vel_b, goal_acc_b])
    return state_goal, quat_wb, r_wb
