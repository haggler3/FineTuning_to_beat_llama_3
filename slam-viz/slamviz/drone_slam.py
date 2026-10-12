"""3D LiDAR-inertial-style SLAM for the drone.

Velocity + gyro dead reckoning (biased and drifting) gives the prediction; each scan is registered with
point-to-plane ICP against a voxel map. Voxels become part of the map only after repeated observation and
are *cleared* when later rays pass through them, so moving obstacles do not leave permanent ghosts.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .sensors import DRONE_AZ, DRONE_DIRS, DRONE_EL, DRONE_MAX_RANGE

VOX = 0.2
PERSIST, HIDE = 5.0, 2.0


def rodrigues(w):
    th = np.linalg.norm(w)
    if th < 1e-9:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def to_quat(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w > 1e-6:
        return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])
    # fallback for 180 deg rotations
    i = np.argmax(np.diag(R)); j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1) / 2
    q = np.zeros(4); q[1 + i] = s
    q[1 + j] = (R[j, i] + R[i, j]) / (4 * s); q[1 + k] = (R[k, i] + R[i, k]) / (4 * s)
    q[0] = (R[k, j] - R[j, k]) / (4 * s)
    return q


class DroneSLAM:
    def __init__(self, p0, R0):
        self.p, self.R = np.array(p0, float), np.array(R0, float)
        self.p_dr, self.R_dr = self.p.copy(), self.R.copy()
        cap = 4096
        self.keys: dict[int, int] = {}
        self.ctr = np.zeros((cap, 3), np.float32)
        self.cnt = np.zeros(cap, np.float32)
        self.shown = np.zeros(cap, bool)
        self.n = 0
        self.tree = None; self.tree_pts = None; self.tree_nrm = None
        self.frame = 0
        self.icp_ok = True

    # ---- registration ------------------------------------------------------------------------
    def _rebuild_tree(self):
        idx = np.where(self.shown[: self.n])[0]
        if len(idx) < 150:
            self.tree = None
            return
        pts = self.ctr[idx].astype(np.float64)
        tree = cKDTree(pts)
        _, nb = tree.query(pts, k=min(10, len(pts)))
        q = pts[nb] - pts[nb].mean(1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", q, q) / q.shape[1]
        ev, evec = np.linalg.eigh(cov)
        good = ev[:, 0] < 0.25 * ev[:, 1] + 1e-9  # locally planar
        self.tree, self.tree_pts, self.tree_nrm = cKDTree(pts[good]), pts[good], evec[good, :, 0]
        if good.sum() < 100:
            self.tree = None

    def _icp(self, src_b, R, p):
        gates = [1.0, 0.8, 0.6, 0.5, 0.4, 0.4, 0.35, 0.3]
        R0, p0 = R.copy(), p.copy()
        for g in gates:
            w = src_b @ R.T + p
            d, j = self.tree.query(w, distance_upper_bound=g)
            m = np.isfinite(d)
            if m.sum() < 60:
                return R0, p0, 0.0
            pw, q, n = w[m], self.tree_pts[j[m]], self.tree_nrm[j[m]]
            r = np.einsum("ij,ij->i", n, pw - q)
            keep = np.abs(r) < np.quantile(np.abs(r), 0.85) + 1e-3
            pw, n, r = pw[keep], n[keep], r[keep]
            A = np.hstack([np.cross(pw, n), n])
            x = np.linalg.solve(A.T @ A + 1e-2 * np.eye(6), -A.T @ r)
            dR = rodrigues(x[:3])
            R, p = dR @ R, dR @ p + x[3:]
        frac = m.sum() / len(src_b)
        return R, p, frac

    # ---- main step ---------------------------------------------------------------------------
    def step(self, vel_body, gyro, ranges, dt):
        ok = ranges > 0
        pts_body = DRONE_DIRS[ok] * ranges[ok, None]
        self._range_img = np.where(ranges > 0, ranges, DRONE_MAX_RANGE).reshape(len(DRONE_EL), len(DRONE_AZ))
        # dead reckoning of both the raw estimate and the SLAM prediction
        self.p_dr = self.p_dr + self.R_dr @ vel_body * dt
        self.R_dr = self.R_dr @ rodrigues(gyro * dt)
        Rp = self.R @ rodrigues(gyro * dt)
        pp = self.p + self.R @ vel_body * dt

        # downsample scan to ~0.3 m voxels
        v = np.floor(pts_body / 0.3).astype(np.int64)
        _, u = np.unique(v, axis=0, return_index=True)
        src = pts_body[u]
        self.icp_ok = False
        if self.tree is not None and len(src) > 80:
            R, p, frac = self._icp(src, Rp, pp)
            dp, ang = np.linalg.norm(p - pp), np.arccos(np.clip((np.trace(R @ Rp.T) - 1) / 2, -1, 1))
            if frac > 0.35 and dp < 0.6 and ang < 0.2:
                Rp, pp, self.icp_ok = R, p, True
        self.R, self.p = Rp, pp
        added, removed = self._update_map(pts_body)
        if self.frame % 3 == 0 or self.tree is None:
            self._rebuild_tree()
        self.frame += 1
        return added, removed

    def _update_map(self, pts_body):
        w = pts_body @ self.R.T + self.p
        vox = np.unique(np.floor(w / VOX).astype(np.int64), axis=0)
        # --- clear voxels contradicted by this scan (range-image test) ---
        removed = []
        if self.n:
            alive = np.where(self.cnt[: self.n] > 0)[0]
            rel = (self.ctr[alive].astype(np.float64) - self.p) @ self.R
            rng = np.linalg.norm(rel, axis=1)
            az = np.arctan2(rel[:, 1], rel[:, 0])
            el = np.arcsin(np.clip(rel[:, 2] / np.maximum(rng, 1e-6), -1, 1))
            ia = np.round((az - DRONE_AZ[0]) / (DRONE_AZ[1] - DRONE_AZ[0])).astype(int)
            ie = np.round((el - DRONE_EL[0]) / (DRONE_EL[1] - DRONE_EL[0])).astype(int)
            da = np.abs(az - (DRONE_AZ[0] + ia * (DRONE_AZ[1] - DRONE_AZ[0])))
            de = np.abs(el - (DRONE_EL[0] + ie * (DRONE_EL[1] - DRONE_EL[0])))
            inb = (ie >= 0) & (ie < len(DRONE_EL)) & (da < 0.2 * (DRONE_AZ[1] - DRONE_AZ[0])) & (de < 0.15 * (DRONE_EL[1] - DRONE_EL[0])) & (rng < 9)
            if self._range_img is not None and inb.any():
                ia_ = ia[inb] % len(DRONE_AZ)
                seen = self._range_img[ie[inb], ia_]
                contra = seen > rng[inb] * 1.12 + 0.8  # margin grows with range: grazing surfaces must survive
                ids = alive[inb][contra]
                self.cnt[ids] -= 1.0
                gone = ids[self.shown[ids] & (self.cnt[ids] < HIDE)]
                self.shown[gone] = False
                removed = gone.tolist()
        # --- integrate observed voxels ---
        added = []
        for k in map(tuple, vox):
            key = hash((int(k[0]), int(k[1]), int(k[2])))
            i = self.keys.get(key)
            if i is None:
                if self.n == len(self.cnt):
                    self._grow()
                i = self.n; self.n += 1
                self.keys[key] = i
                self.ctr[i] = (np.array(k) + 0.5) * VOX
            self.cnt[i] = min(self.cnt[i] + 1.0, 30.0)
            if not self.shown[i] and self.cnt[i] >= PERSIST:
                self.shown[i] = True
                added.append(i)
        return added, removed

    _range_img = None

    def _grow(self):
        for name in ("ctr", "cnt", "shown"):
            a = getattr(self, name)
            setattr(self, name, np.concatenate([a, np.zeros_like(a)]))
