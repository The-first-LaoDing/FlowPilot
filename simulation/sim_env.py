"""Lightweight Flightmare simulation environment wrapper.

Replaces flightpolicy.envs.vec_env_wrapper.FlightEnvVec:
- No dependency on gym, stable-baselines3, or torch.
- Only exposes methods used by data collection and testing scripts.
"""

import numpy as np


class SimEnv:
    """Thin wrapper around flightgym.QuadrotorEnv_v1 C++ binding."""

    def __init__(self, impl):
        """
        Args:
            impl: A flightgym.QuadrotorEnv_v1 instance.
        """
        self._impl = impl
        self._num_envs = self._impl.getNumOfEnvs()
        self._obs_dim = self._impl.getObsDim()
        self._rew_dim = self._impl.getRewDim()
        self._img_w = self._impl.getImgWidth()
        self._img_h = self._impl.getImgHeight()

        # Pre-allocate buffers
        self._obs = np.zeros((self._num_envs, self._obs_dim), dtype=np.float32)
        self._rew = np.zeros((self._num_envs, self._rew_dim), dtype=np.float32)
        self._done = np.zeros(self._num_envs, dtype=bool)
        self._depth_buf = np.zeros(
            (self._num_envs, self._img_w * self._img_h), dtype=np.float32
        )
        self._rgb_buf = np.zeros(
            (self._num_envs, self._img_w * self._img_h * 3), dtype=np.uint8
        )
        self._world_box = np.zeros(6, dtype=np.float32)
        self._impl.getWorldBox(self._world_box)

    # -- Properties --

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def img_width(self) -> int:
        return self._img_w

    @property
    def img_height(self) -> int:
        return self._img_h

    @property
    def world_box(self) -> np.ndarray:
        """[x_min, y_min, z_min, x_max, y_max, z_max]."""
        return self._world_box.copy()

    # -- Connection --

    def connect_unity(self):
        self._impl.connectUnity()

    def disconnect_unity(self):
        self._impl.disconnectUnity()

    # -- Simulation --

    def reset(self) -> np.ndarray:
        self._rew[:] = 0
        self._impl.reset(self._obs)
        return self._obs.copy()

    def step(self, action: np.ndarray):
        if action.ndim <= 1:
            action = action.reshape((self._num_envs, -1))
        if action.dtype == np.dtype("int"):
            action = action.astype(np.float32)
        self._impl.step(action, self._obs, self._rew, self._done)
        return self._obs.copy(), self._rew.copy(), self._done.copy()

    def set_state(self, pos, vel, acc, quat):
        """Set drone state (world frame).

        Args:
            pos: [3] or [num_envs, 3]
            vel: [3] or [num_envs, 3]
            acc: [3] or [num_envs, 3]
            quat: [4] or [num_envs, 4] (wxyz)
        """
        pos = np.atleast_2d(pos).astype(np.float32)
        vel = np.atleast_2d(vel).astype(np.float32)
        acc = np.atleast_2d(acc).astype(np.float32)
        quat = np.atleast_2d(quat).astype(np.float32)
        state = np.hstack((pos, vel, acc, quat))
        self._impl.setState(state)

    def set_goal(self, goal):
        if np.ndim(goal) <= 1:
            goal = np.reshape(goal, (self._num_envs, -1))
        self._impl.setGoal(goal)

    def set_map_id(self, map_id):
        if np.ndim(map_id) <= 1:
            map_id = np.reshape(map_id, (self._num_envs, -1))
        self._impl.setMapID(map_id)

    def render(self) -> bool:
        return self._impl.render()

    def seed(self, s: int):
        self._impl.setSeed(s)

    # -- Scene --

    def spawn_trees_and_save_ply(self, ply_idx: int = -1, spacing: float = -1):
        self._impl.spawnTreesAndSavePointcloud(ply_idx, spacing)

    # -- Sensors --

    def get_depth(self, normalize: bool = True) -> np.ndarray:
        """Get depth images.

        Args:
            normalize: If True, clip to 20m and scale to [0, 1]. NaN -> 1.0.

        Returns:
            depth: [num_envs, H, W] float32
        """
        self._impl.getDepthImage(self._depth_buf)
        depth = self._depth_buf.copy()
        if normalize:
            depth = depth * 1000.0  # Flightmare returns km
            depth = np.minimum(depth, 20.0)
            depth = depth / 20.0
            depth[np.isnan(depth)] = 1.0
        depth = depth.reshape((self._num_envs, self._img_h, self._img_w))
        return depth

    def get_rgb(self) -> np.ndarray:
        """Get RGB images.

        Returns:
            rgb: [num_envs, H, W, 3] uint8
        """
        self._impl.getRGBImage(self._rgb_buf, True)
        return self._rgb_buf.copy().reshape(
            (self._num_envs, self._img_h, self._img_w, 3)
        )
