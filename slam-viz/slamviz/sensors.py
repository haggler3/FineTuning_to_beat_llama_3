"""Simulated range sensors built on mujoco.mj_multiRay."""
from __future__ import annotations

import mujoco
import numpy as np

ROVER_BEAMS = 360
ROVER_MAX_RANGE = 10.0
ROVER_ANGLES = np.linspace(-np.pi, np.pi, ROVER_BEAMS, endpoint=False)

DRONE_AZ = np.linspace(-np.pi, np.pi, 96, endpoint=False)
DRONE_EL = np.radians(np.linspace(-50, 30, 9))
DRONE_MAX_RANGE = 12.0
_az, _el = np.meshgrid(DRONE_AZ, DRONE_EL)
DRONE_DIRS = np.stack([np.cos(_el) * np.cos(_az), np.cos(_el) * np.sin(_az), np.sin(_el)], -1).reshape(-1, 3)


class Lidar:
    def __init__(self, model, data, body_name: str, dirs_body: np.ndarray, max_range: float, sigma: float, rng):
        self.m, self.d = model, data
        self.body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        self.dirs = np.ascontiguousarray(dirs_body, dtype=np.float64)
        self.max_range, self.sigma, self.rng = max_range, sigma, rng
        n = len(self.dirs)
        self._geom = np.zeros(n, dtype=np.int32)
        self._dist = np.zeros(n, dtype=np.float64)

    def scan(self, origin_world: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Return ranges (0 where no return). R rotates sensor frame -> world."""
        vec = np.ascontiguousarray((self.dirs @ R.T).reshape(-1))
        mujoco.mj_multiRay(self.m, self.d, origin_world.astype(np.float64), vec, None, 1, self.body,
                           self._geom, self._dist, None, len(self.dirs), self.max_range)
        r = self._dist.copy()
        hit = (self._geom >= 0) & (r > 0.05) & (r < self.max_range)
        r = np.where(hit, r + self.rng.normal(0, self.sigma, r.shape), 0.0)
        return np.where(hit, r, 0.0)


def rover_lidar(model, data, rng) -> Lidar:
    dirs = np.stack([np.cos(ROVER_ANGLES), np.sin(ROVER_ANGLES), np.zeros_like(ROVER_ANGLES)], -1)
    return Lidar(model, data, "rover", dirs, ROVER_MAX_RANGE, 0.015, rng)


def drone_lidar(model, data, rng) -> Lidar:
    return Lidar(model, data, "drone", DRONE_DIRS, DRONE_MAX_RANGE, 0.02, rng)
