"""2D occupancy-grid SLAM for the rover.

Particle filter over wheel odometry (weights from a likelihood field of the current map) followed by a
correlative scan-match refinement; the map is a log-odds grid that is updated by ray casting, so moving
obstacles are written *and erased* as they travel through the scene.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

from .sensors import ROVER_ANGLES, ROVER_MAX_RANGE

RES = 0.1
HALF = 13.0
N = int(2 * HALF / RES)
L_OCC, L_FREE, L_MIN, L_MAX = 0.85, -0.45, -4.0, 5.0


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class RoverSLAM:
    # scan-match window / step, beam subsampling, likelihood-field sigma, odometry prior sigmas
    W_XY, S_XY, W_TH, S_TH, SUB, SIG, PRI_XY, PRI_TH = 0.15, 0.025, 0.06, 0.01, 2, 0.15, 0.05, 0.02
    def __init__(self, start_pose, n_particles=80, seed=1):
        self.rng = np.random.default_rng(seed)
        self.lo = np.zeros((N, N), np.float32)
        self.touched = np.zeros((N, N), bool)
        self.dist = np.full((N, N), 5.0, np.float32)
        self.P = np.tile(np.asarray(start_pose, float), (n_particles, 1))
        self.w = np.full(n_particles, 1.0 / n_particles)
        self.est = np.asarray(start_pose, float).copy()
        self.dr = self.est.copy()
        self.frame = 0
        self.neff = float(n_particles)
        self._ray_s = np.arange(0.05, ROVER_MAX_RANGE, 0.05)

    # ---- helpers -----------------------------------------------------------------------------
    @staticmethod
    def _cell(xy):
        return np.floor((xy + HALF) / RES).astype(np.int64)

    def _endpoints(self, poses, ranges, ang):
        """(P,B,2) world endpoints for each pose."""
        a = poses[:, 2:3] + ang[None, :]
        return np.stack([poses[:, 0:1] + ranges[None, :] * np.cos(a), poses[:, 1:2] + ranges[None, :] * np.sin(a)], -1)

    def _loglik(self, poses, ranges, ang):
        ep = self._endpoints(poses, ranges, ang)
        c = self._cell(ep)
        inb = (c >= 0).all(-1) & (c < N).all(-1)
        d = np.where(inb, self.dist[np.clip(c[..., 0], 0, N - 1), np.clip(c[..., 1], 0, N - 1)], 1.0)
        d = np.minimum(d, 1.0)
        return (-(d ** 2) / (2 * self.SIG ** 2)).sum(-1) * 0.35

    def _refine(self, pose, ranges, ang, pred, sxy=None, sth=None):
        sxy, sth = sxy or self.PRI_XY, sth or self.PRI_TH
        ds = np.arange(-self.W_XY, self.W_XY + 1e-6, self.S_XY)
        dth = np.arange(-self.W_TH, self.W_TH + 1e-6, self.S_TH)
        gx, gy, gt = np.meshgrid(ds, ds, dth, indexing="ij")
        cand = np.stack([pose[0] + gx.ravel(), pose[1] + gy.ravel(), pose[2] + gt.ravel()], 1)
        ll = self._loglik(cand, ranges, ang)
        # Gaussian prior on the odometry prediction keeps the match from sliding along featureless walls
        ll -= 0.5 * (((cand[:, 0] - pred[0]) ** 2 + (cand[:, 1] - pred[1]) ** 2) / sxy ** 2 + wrap(cand[:, 2] - pred[2]) ** 2 / sth ** 2)
        return cand[np.argmax(ll)]

    # ---- main step ---------------------------------------------------------------------------
    def step(self, ranges, odom, dt):
        v, w = odom
        dist, dth = v * dt, w * dt
        pred = np.array([self.est[0] + dist * np.cos(self.est[2] + dth / 2), self.est[1] + dist * np.sin(self.est[2] + dth / 2), self.est[2] + dth])
        # dead reckoning (odometry only)
        self.dr[0] += dist * np.cos(self.dr[2] + dth / 2)
        self.dr[1] += dist * np.sin(self.dr[2] + dth / 2)
        self.dr[2] = wrap(self.dr[2] + dth)
        # particle motion model
        n = len(self.P)
        s_xy, s_th = 0.015 + 0.08 * abs(dist), 0.008 + 0.1 * abs(dth)
        d_ = dist + self.rng.normal(0, s_xy, n)
        t_ = dth + self.rng.normal(0, s_th, n)
        ym = self.P[:, 2] + t_ / 2
        self.P[:, 0] += d_ * np.cos(ym)
        self.P[:, 1] += d_ * np.sin(ym)
        self.P[:, 2] = wrap(self.P[:, 2] + t_)

        valid = ranges > 0
        idx = np.where(valid)[0][::self.SUB]
        if self.frame >= 3 and len(idx) > 20:
            r, ang = ranges[idx], ROVER_ANGLES[idx]
            ll = self._loglik(self.P, r, ang)
            self.w = np.exp(ll - ll.max()); self.w /= self.w.sum()
            self.neff = 1.0 / np.sum(self.w ** 2)
            mean = np.array([np.sum(self.w * self.P[:, 0]), np.sum(self.w * self.P[:, 1]),
                             np.arctan2(np.sum(self.w * np.sin(self.P[:, 2])), np.sum(self.w * np.cos(self.P[:, 2])))])
            refined = self._refine(mean, r, ang, pred)
            if self.neff < 0.5 * n:  # systematic resampling
                u = (self.rng.random() + np.arange(n)) / n
                self.P = self.P[np.minimum(np.searchsorted(np.cumsum(self.w), u), n - 1)]
                self.w[:] = 1.0 / n
            shift = refined - np.array([self.P[:, 0].mean(), self.P[:, 1].mean(), mean[2]])
            shift[2] = wrap(shift[2])
            self.P += shift
            self.est = refined
        else:
            self.est = np.array([self.P[:, 0].mean(), self.P[:, 1].mean(), self.P[:, 2].mean()])
        self.est[2] = wrap(self.est[2])
        self._update_map(ranges)
        self.frame += 1

    def _update_map(self, ranges):
        x, y, th = self.est
        hit = ranges > 0
        ang = th + ROVER_ANGLES[hit]
        r = ranges[hit]
        dirs = np.stack([np.cos(ang), np.sin(ang)], 1)
        # free space samples along each ray
        s = self._ray_s[None, :]
        mask = s < (r[:, None] - 0.12)
        pts = np.stack([x + dirs[:, 0:1] * s, y + dirs[:, 1:2] * s], -1)[mask]
        c = self._cell(pts)
        ok = (c >= 0).all(1) & (c < N).all(1)
        lin = np.unique(c[ok, 0] * N + c[ok, 1])
        fi, fj = lin // N, lin % N
        self.lo[fi, fj] = np.maximum(self.lo[fi, fj] + L_FREE, L_MIN)
        self.touched[fi, fj] = True
        ep = self._cell(np.stack([x + dirs[:, 0] * r, y + dirs[:, 1] * r], 1))
        ok = (ep >= 0).all(1) & (ep < N).all(1)
        oi, oj = ep[ok, 0], ep[ok, 1]
        self.lo[oi, oj] = np.minimum(self.lo[oi, oj] + L_OCC - L_FREE, L_MAX)  # cancel free update from neighbours
        self.touched[oi, oj] = True
        occ = self.lo > 1.0
        self.dist = (distance_transform_edt(~occ) * RES).astype(np.float32)

    # ---- export ------------------------------------------------------------------------------
    def grid_bytes(self) -> bytes:
        """uint8 image, 0 = unknown, else 1..255 occupancy probability. Row-major [y][x]."""
        p = 1.0 / (1.0 + np.exp(-self.lo))
        g = np.where(self.touched, 1 + np.round(p * 254), 0).astype(np.uint8)
        return np.ascontiguousarray(g.T).tobytes()

    def particles_xy(self, k=40):
        return self.P[:k, :2]
