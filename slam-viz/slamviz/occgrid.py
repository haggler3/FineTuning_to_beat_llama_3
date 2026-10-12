"""Log-odds occupancy grid with ray-cast updates (optionally decaying, so it acts as a local sub-map)."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

L_OCC, L_FREE, L_MIN, L_MAX = 0.85, -0.45, -4.0, 5.0


class OccupancyGrid:
    def __init__(self, half: float, res: float = 0.1, max_range: float = 10.0, decay: float = 1.0, with_dist: bool = False):
        self.half, self.res, self.decay, self.with_dist = half, res, decay, with_dist
        self.n = int(2 * half / res)
        self.lo = np.zeros((self.n, self.n), np.float32)
        self.touched = np.zeros((self.n, self.n), bool)
        self.dist = np.full((self.n, self.n), 5.0, np.float32)
        self._s = np.arange(0.05, max_range, 0.05)

    def reset(self):
        self.lo[:] = 0
        self.touched[:] = False

    def cell(self, xy):
        return np.floor((xy + self.half) / self.res).astype(np.int64)

    def integrate(self, pose, ranges, angles):
        n = self.n
        hit = ranges > 0
        if not hit.any():
            return
        ang = pose[2] + angles[hit]
        r = ranges[hit]
        dirs = np.stack([np.cos(ang), np.sin(ang)], 1)
        s = self._s[None, :]
        mask = s < (r[:, None] - 0.12)
        pts = np.stack([pose[0] + dirs[:, 0:1] * s, pose[1] + dirs[:, 1:2] * s], -1)[mask]
        c = self.cell(pts)
        ok = (c >= 0).all(1) & (c < n).all(1)
        lin = np.unique(c[ok, 0] * n + c[ok, 1])
        fi, fj = lin // n, lin % n
        if self.decay < 1.0:
            self.lo *= self.decay
        self.lo[fi, fj] = np.maximum(self.lo[fi, fj] + L_FREE, L_MIN)
        self.touched[fi, fj] = True
        ep = self.cell(np.stack([pose[0] + dirs[:, 0] * r, pose[1] + dirs[:, 1] * r], 1))
        ok = (ep >= 0).all(1) & (ep < n).all(1)
        oi, oj = ep[ok, 0], ep[ok, 1]
        self.lo[oi, oj] = np.minimum(self.lo[oi, oj] + L_OCC - L_FREE, L_MAX)  # cancels the free update from neighbouring rays
        self.touched[oi, oj] = True
        if self.with_dist:
            self.dist = (distance_transform_edt(~(self.lo > 1.0)) * self.res).astype(np.float32)

    def to_bytes(self) -> bytes:
        """uint8 image, 0 = unknown, else 1..255 occupancy probability. Row-major [y][x]."""
        p = 1.0 / (1.0 + np.exp(-self.lo))
        g = np.where(self.touched, 1 + np.round(p * 254), 0).astype(np.uint8)
        return np.ascontiguousarray(g.T).tobytes()
