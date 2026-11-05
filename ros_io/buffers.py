"""Thread-safe ROS-subscriber state container (duck-typed, rospy-free).

GoalBuffer encapsulates:
  * A lock guarding the latest goal.
  * A `callback(msg)` that accepts a duck-typed ROS PoseStamped.
  * A `snapshot()` that returns an immutable snapshot (or None if no goal
    has been received yet).

Snapshots use `.copy()` to decouple from subsequent callbacks so stashed
snapshots remain valid even if a new goal arrives.

Wire in the script's `main()` with a standard ROS subscriber, e.g.:

    goals = GoalBuffer(goal_z=1.5)
    rospy.Subscriber('/move_base_simple/goal', PoseStamped, goals.callback, queue_size=1)
"""

import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GoalSnapshot:
    pos: np.ndarray    # [3] float64, world frame
    seq: int           # monotonically increasing message index


# ---------------------------------------------------------------------------
# GoalBuffer
# ---------------------------------------------------------------------------

class GoalBuffer:
    """Thread-safe buffer for RViz 2D Nav Goal clicks (geometry_msgs/PoseStamped).

    Duck-typed on msg.pose.position.{x, y}. The z value is supplied at
    construction time (RViz 2D Nav Goal clicks only carry 2D position, and
    different scripts use different hover altitudes).
    """

    def __init__(self, goal_z: float = 1.5):
        self._lock   = threading.Lock()
        self._pos    = None          # Optional[np.ndarray[3]]
        self._seq    = 0
        self._goal_z = goal_z

    def callback(self, msg) -> None:
        """Wire via rospy.Subscriber('/move_base_simple/goal', PoseStamped, buf.callback)."""
        with self._lock:
            self._pos = np.array([msg.pose.position.x,
                                  msg.pose.position.y,
                                  self._goal_z], dtype=np.float64)
            self._seq += 1

    def snapshot(self) -> Optional[GoalSnapshot]:
        with self._lock:
            if self._pos is None:
                return None
            return GoalSnapshot(pos=self._pos.copy(), seq=self._seq)
