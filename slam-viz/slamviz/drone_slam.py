"""Graph SLAM for the 3D drone.

Front-end   LiDAR-inertial dead reckoning refined by point-to-plane ICP against a *local* voxel map (only
            voxels seen in the last ``LOCAL_AGE`` frames). Locally accurate, drifts over minutes.
Graph       keyframe nodes every ``KF`` frames tied by odometry edges.
Loop closure  revisits are detected by distance in the current graph estimate and verified with multi-start
            ICP between the new scan and a short sub-map around the old keyframe.
Back-end    the SE(3) pose graph is re-optimised (``posegraph.optimize_se3``) and the global voxel map is
            rebuilt from the corrected poses.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

from .posegraph import optimize_se3
from .sensors import DRONE_AZ, DRONE_DIRS, DRONE_EL, DRONE_MAX_RANGE

VOX = 0.2
KF = 3
LOCAL_AGE = 120            # frames a voxel stays in the front-end's local map
LOOP_MIN_GAP = 20          # keyframes between a loop-closure pair ...
LOOP_MIN_PATH = 20.0       # ... and metres of travelled path (a hovering drone is not 'revisiting')
LOOP_RADIUS = 4.5
SIG_ODOM = (0.05, 0.015)   # translation (m), rotation (rad)
SIG_LOOP = (0.08, 0.03)


def rodrigues(w):
    return Rot.from_rotvec(w).as_matrix()


def to_quat(R):
    x, y, z, w = Rot.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


class VoxelMap:
    """Voxels appear after repeated observation and are cleared when later rays pass through them."""

    def __init__(self, persist=5.0, hide=2.0):
        self.persist, self.hide = persist, hide
        self.keys: dict = {}
        cap = 4096
        self.ctr = np.zeros((cap, 3), np.float32)
        self.cnt = np.zeros(cap, np.float32)
        self.last = np.zeros(cap, np.int32)
        self.shown = np.zeros(cap, bool)
        self.n = 0

    def _grow(self):
        for name in ("ctr", "cnt", "last", "shown"):
            a = getattr(self, name)
            setattr(self, name, np.concatenate([a, np.zeros_like(a)]))

    def update(self, R, p, ranges, frame):
        """Integrate one scan taken from pose (R, p). Returns (added ids, removed ids)."""
        ok = ranges > 0
        pts = DRONE_DIRS[ok] * ranges[ok, None]
        range_img = np.where(ok, ranges, DRONE_MAX_RANGE).reshape(len(DRONE_EL), len(DRONE_AZ))
        vox = np.unique(np.floor((pts @ R.T + p) / VOX).astype(np.int64), axis=0)
        removed = []
        if self.n:
            alive = np.where(self.cnt[: self.n] > 0)[0]
            rel = (self.ctr[alive].astype(np.float64) - p) @ R
            rng = np.linalg.norm(rel, axis=1)
            az = np.arctan2(rel[:, 1], rel[:, 0])
            el = np.arcsin(np.clip(rel[:, 2] / np.maximum(rng, 1e-6), -1, 1))
            daz, de_ = DRONE_AZ[1] - DRONE_AZ[0], DRONE_EL[1] - DRONE_EL[0]
            ia = np.round((az - DRONE_AZ[0]) / daz).astype(int)
            ie = np.round((el - DRONE_EL[0]) / de_).astype(int)
            inb = ((ie >= 0) & (ie < len(DRONE_EL)) & (np.abs(az - (DRONE_AZ[0] + ia * daz)) < 0.2 * daz)
                   & (np.abs(el - (DRONE_EL[0] + ie * de_)) < 0.15 * de_) & (rng < 9))
            if inb.any():
                seen = range_img[ie[inb], ia[inb] % len(DRONE_AZ)]
                # margin grows with range: grazing surfaces (the floor!) must survive
                ids = alive[inb][seen > rng[inb] * 1.12 + 0.8]
                self.cnt[ids] -= 1.0
                gone = ids[self.shown[ids] & (self.cnt[ids] < self.hide)]
                self.shown[gone] = False
                removed = gone.tolist()
        added = []
        for k in map(tuple, vox):
            key = (int(k[0]), int(k[1]), int(k[2]))
            i = self.keys.get(key)
            if i is None:
                if self.n == len(self.cnt):
                    self._grow()
                i = self.n
                self.n += 1
                self.keys[key] = i
                self.ctr[i] = (np.array(k) + 0.5) * VOX
            self.cnt[i] = min(self.cnt[i] + 1.0, 30.0)
            self.last[i] = frame
            if not self.shown[i] and self.cnt[i] >= self.persist:
                self.shown[i] = True
                added.append(i)
        return added, removed

    def registration_tree(self, frame, max_age, min_count=2.0):
        """KD-tree + normals over recent, locally planar voxels (point-to-plane ICP target). Uses a lower hit
        count than the display threshold so sparsely-seen surfaces like the floor still constrain height."""
        idx = np.where((self.cnt[: self.n] >= min_count) & (frame - self.last[: self.n] <= max_age))[0]
        if len(idx) < 150:
            return None
        pts = self.ctr[idx].astype(np.float64)
        tree = cKDTree(pts)
        _, nb = tree.query(pts, k=min(10, len(pts)))
        q = pts[nb] - pts[nb].mean(1, keepdims=True)
        ev, evec = np.linalg.eigh(np.einsum("nki,nkj->nij", q, q) / q.shape[1])
        good = ev[:, 0] < 0.25 * ev[:, 1] + 1e-9
        if good.sum() < 100:
            return None
        return cKDTree(pts[good]), pts[good], evec[good, :, 0]

    def shown_voxels(self):
        ids = np.where(self.shown[: self.n])[0]
        return ids, np.floor(self.ctr[ids] / VOX).astype(int)


MU_T, MU_R = 1.0, 10.0   # light odometry prior on the ICP correction (a stiffer one fights real corrections)


def _icp_p2plane(src_b, R, p, target):
    tree, tpts, tnrm = target
    R0, p0 = R.copy(), p.copy()
    m = None
    for g in (1.0, 0.8, 0.6, 0.5, 0.4, 0.4, 0.35, 0.3):
        w = src_b @ R.T + p
        d, j = tree.query(w, distance_upper_bound=g)
        m = np.isfinite(d)
        if m.sum() < 60:
            return R0, p0, 0.0
        pw, q, n = w[m], tpts[j[m]], tnrm[j[m]]
        r = np.einsum("ij,ij->i", n, pw - q)
        keep = np.abs(r) < np.quantile(np.abs(r), 0.85) + 1e-3
        pw, n, r = pw[keep], n[keep], r[keep]
        A = np.hstack([np.cross(pw, n), n])
        H, b = A.T @ A + 1e-2 * np.eye(6), -A.T @ r
        # Gaussian prior on the *total* correction relative to the odometry prediction
        H[:3, :3] += MU_R * np.eye(3); H[3:, 3:] += MU_T * np.eye(3)
        b[:3] -= MU_R * Rot.from_matrix(R @ R0.T).as_rotvec(); b[3:] -= MU_T * (p - p0)
        x = np.linalg.solve(H, b)
        dR = rodrigues(x[:3])
        R, p = dR @ R, dR @ p + x[3:]
    return R, p, m.sum() / len(src_b)


class DroneFrontend:
    """Dead reckoning + scan-to-local-map ICP."""

    def __init__(self, p0, R0):
        self.p, self.R = np.array(p0, float), np.array(R0, float)
        self.local = VoxelMap()
        self.target = None
        self.icp_ok = False
        self.frame = 0

    def step(self, vel_body, gyro, ranges, dt):
        Rp = self.R @ rodrigues(gyro * dt)
        pp = self.p + self.R @ vel_body * dt
        ok = ranges > 0
        pts = DRONE_DIRS[ok] * ranges[ok, None]
        _, u = np.unique(np.floor(pts / 0.3).astype(np.int64), axis=0, return_index=True)
        src = pts[u]
        self.icp_ok = False
        if self.target is not None and len(src) > 80:
            R, p, frac = _icp_p2plane(src, Rp, pp, self.target)
            ang = np.arccos(np.clip((np.trace(R @ Rp.T) - 1) / 2, -1, 1))
            if frac > 0.35 and np.linalg.norm(p - pp) < 0.6 and ang < 0.2:
                Rp, pp, self.icp_ok = R, p, True
        self.R, self.p = Rp, pp
        self.local.update(self.R, self.p, ranges, self.frame)
        if self.frame % 3 == 0 or self.target is None:
            self.target = self.local.registration_tree(self.frame, LOCAL_AGE)
        self.frame += 1


# ------------------------------------------------------------------------------------------ loop closure
def _icp3d(src, tree, dst, R, t, gates=(0.8, 0.5, 0.35, 0.25, 0.2, 0.15, 0.15, 0.12)):
    for g in gates:
        w = src @ R.T + t
        d, k = tree.query(w, distance_upper_bound=g)
        m = np.isfinite(d)
        if m.sum() < 40:
            return R, t, 0.0, 1.0
        a, b = w[m], dst[k[m]]
        keep = d[m] <= np.quantile(d[m], 0.85)
        a, b = a[keep], b[keep]
        ca, cb = a.mean(0), b.mean(0)
        U, _, Vt = np.linalg.svd((a - ca).T @ (b - cb))
        D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
        Rm = Vt.T @ D @ U.T
        R, t = Rm @ R, Rm @ t + (cb - Rm @ ca)
    w = src @ R.T + t
    d, _ = tree.query(w, distance_upper_bound=0.15)
    inl = np.isfinite(d)
    return R, t, float(inl.mean()), float(np.sqrt(np.mean(d[inl] ** 2))) if inl.any() else 1.0


def match_scans3d(src, dst, tree, R_init, t_init):
    """Multi-start ICP. Starts cover +-2 m in x/y and +-0.2 rad in yaw at the predicted height, plus a coarser grid at
    +-1.2 m in z (LiDAR odometry can lose height in corridors). Returns the start with the best inlier fraction."""
    starts = [(dx, dy, 0.0, dyaw) for dx in (-2.0, -1.0, 0.0, 1.0, 2.0) for dy in (-2.0, -1.0, 0.0, 1.0, 2.0)
              for dyaw in (-0.2, -0.1, 0.0, 0.1, 0.2)]
    starts += [(dx, dy, dz, dyaw) for dz in (-1.2, 1.2) for dx in (-1.0, 0.0, 1.0) for dy in (-1.0, 0.0, 1.0)
               for dyaw in (-0.1, 0.0, 0.1)]
    best = None
    for dx, dy, dz, dyaw in starts:
        R0 = rodrigues(np.array([0, 0, dyaw])) @ R_init
        R, t, frac, rms = _icp3d(src, tree, dst, R0, t_init + np.array([dx, dy, dz]))
        if best is None or frac > best[2]:
            best = (R, t, frac, rms)
    R, t, frac, rms = best
    d, _ = cKDTree(src @ R.T + t).query(dst, distance_upper_bound=0.15)
    return R, t, frac, rms, float(np.isfinite(d).mean())


# ------------------------------------------------------------------------------------------ the SLAM system
class DroneGraphSLAM:
    def __init__(self, p0, R0):
        self.front = DroneFrontend(p0, R0)
        self.gmap = VoxelMap()                        # global (display) map
        self.p_dr, self.R_dr = np.array(p0, float), np.array(R0, float)
        self.R, self.p = self.front.R.copy(), self.front.p.copy()
        self.frame = 0
        self.fe: list[tuple] = []
        self.ranges: list[np.ndarray] = []
        self.kf_fe: list[tuple] = []
        self.kf_s: list[float] = []                   # front-end path length at each keyframe
        self.kf_pts: list[np.ndarray] = []
        self.nodes: list[tuple] = []                  # optimised (R, t)
        self.edges: list[tuple] = []                  # (i, j, Rz, tz, sig_t, sig_r)
        self.loops: list[tuple] = []
        self.events: list[dict] = []
        self.last_loop_node = -10
        self.last_rebuild_frame = -1000
        self.diag: list[dict] = []
        self._pending_rebuild = False                 # rebuild the display map on the next step

    @property
    def icp_ok(self):
        return self.front.icp_ok

    # -- helpers ---------------------------------------------------------------------------------
    def _corr(self, k):
        Rn, tn = self.nodes[k]
        Rf, tf = self.kf_fe[k]
        Rc = Rn @ Rf.T
        return Rc, tn - Rc @ tf

    def frame_poses(self):
        Rs, ts = [], []
        for g, (Rf, tf) in enumerate(self.fe):
            Rc, tc = self._corr(min(g // KF, len(self.nodes) - 1))
            Rs.append(Rc @ Rf)
            ts.append(Rc @ tf + tc)
        return np.array(Rs), np.array(ts)

    def _optimise(self):
        i = [e[0] for e in self.edges]; j = [e[1] for e in self.edges]
        Rz = np.array([e[2] for e in self.edges]); tz = np.array([e[3] for e in self.edges])
        st = np.array([e[4] for e in self.edges]); sr = np.array([e[5] for e in self.edges])
        R0 = np.array([n[0] for n in self.nodes]); t0 = np.array([n[1] for n in self.nodes])
        R, t = optimize_se3(R0, t0, i, j, Rz, tz, st, sr)
        self.nodes = list(zip(R, t))
        return float(np.max(np.linalg.norm(t - t0, axis=1)))

    def _rebuild_map(self):
        """Re-integrate every scan with the corrected poses into a fresh voxel map."""
        self.gmap = VoxelMap()
        Rs, ts = self.frame_poses()
        for g in range(len(self.fe)):
            self.gmap.update(Rs[g], ts[g], self.ranges[g], g)
        self.last_rebuild_frame = self.frame
        ids, vox = self.gmap.shown_voxels()
        return np.concatenate([ids[:, None], vox], 1).ravel().astype(int).tolist()

    # -- main step -------------------------------------------------------------------------------
    def step(self, vel_body, gyro, ranges, dt):
        """Returns (added, removed, reset): voxel ids to show/hide in the display map; reset is the full voxel
        list if the map was rebuilt this frame (then added/removed are void)."""
        self.p_dr = self.p_dr + self.R_dr @ vel_body * dt
        self.R_dr = self.R_dr @ rodrigues(gyro * dt)
        self.front.step(vel_body, gyro, ranges, dt)
        Rf, tf = self.front.R.copy(), self.front.p.copy()
        self.fe.append((Rf, tf))
        self.ranges.append(ranges.copy())
        if self.frame % KF == 0:
            self._add_keyframe(ranges, Rf, tf)
        k = min(self.frame // KF, len(self.nodes) - 1)
        Rc, tc = self._corr(k)
        self.R, self.p = Rc @ Rf, Rc @ tf + tc
        reset = None
        if self._pending_rebuild:
            self._pending_rebuild = False
            reset = self._rebuild_map()
            added, removed = [], []
        else:
            added, removed = self.gmap.update(self.R, self.p, ranges, self.frame)
        self.frame += 1
        return added, removed, reset

    def _add_keyframe(self, ranges, Rf, tf):
        n = len(self.nodes)
        ok = ranges > 0
        pts = DRONE_DIRS[ok] * ranges[ok, None]
        _, u = np.unique(np.floor(pts / 0.3).astype(np.int64), axis=0, return_index=True)
        self.kf_s.append((self.kf_s[-1] + float(np.linalg.norm(tf - self.kf_fe[-1][1]))) if n else 0.0)
        self.kf_fe.append((Rf, tf)); self.kf_pts.append(pts[u])
        Rc, tc = (np.eye(3), np.zeros(3)) if not n else self._corr(n - 1)  # new node inherits the latest correction
        self.nodes.append((Rc @ Rf, Rc @ tf + tc))
        if n:
            Rp, tp = self.kf_fe[n - 1]
            self.edges.append((n - 1, n, Rp.T @ Rf, Rp.T @ (tf - tp), SIG_ODOM[0], SIG_ODOM[1]))
        found = self._detect_loops(n) if (n - self.last_loop_node >= 3 and n % 2 == 0) else []
        if not found:
            return
        for (j, R, t, frac, rms) in found:
            self.edges.append((j, n, R, t, SIG_LOOP[0], SIG_LOOP[1]))
            self.loops.append((j, n))
        self.last_loop_node = n
        shift = self._optimise()
        if shift > 0.12 and self.frame - self.last_rebuild_frame >= 150:
            self._pending_rebuild = True
        self.events.append(dict(f=self.frame, nodes=np.round(np.array([nd[1] for nd in self.nodes]), 3).tolist(),
                                loops=[list(l) for l in self.loops], rebuilt=self._pending_rebuild, shift=shift))

    def _submap(self, j):
        """Scans of nodes j-2..j+2 expressed in node j's front-end frame."""
        Rj, tj = self.kf_fe[j]
        out = []
        for m in range(max(0, j - 2), min(len(self.kf_fe), j + 3)):
            Rm, tm = self.kf_fe[m]
            out.append(self.kf_pts[m] @ (Rj.T @ Rm).T + Rj.T @ (tm - tj))
        return np.concatenate(out)

    def _detect_loops(self, n):
        Rn, tn = self.nodes[n]
        cand = [j for j in range(n - LOOP_MIN_GAP) if self.kf_s[n] - self.kf_s[j] > LOOP_MIN_PATH
                and np.linalg.norm(self.nodes[j][1] - tn) < LOOP_RADIUS]
        cand.sort(key=lambda j: np.linalg.norm(self.nodes[j][1] - tn))
        out = []
        for j in cand[:1]:
            Rj, tj = self.nodes[j]
            R0, t0 = Rj.T @ Rn, Rj.T @ (tn - tj)
            dst = self._submap(j)
            tree = cKDTree(dst)
            R, t, frac, rms, rev = match_scans3d(self.kf_pts[n], dst, tree, R0, t0)
            ang = np.arccos(np.clip((np.trace(R @ R0.T) - 1) / 2, -1, 1))
            ok = frac > 0.6 and rev > 0.22 and rms < 0.11 and np.linalg.norm(t - t0) < 3.5 and ang < 0.35
            self.diag.append(dict(i=j, j=n, frac=frac, rev=rev, rms=rms, ok=bool(ok), R=R, t=t))
            if ok:
                out.append((j, R, t, frac, rms))
        return out

    # -- finishing -------------------------------------------------------------------------------
    def finish(self):
        """Final optimisation; returns smoothed (R, t) for every frame."""
        if self.loops:
            self._optimise()
        Rs, ts = self.frame_poses()
        self.events.append(dict(f=self.frame, nodes=np.round(np.array([nd[1] for nd in self.nodes]), 3).tolist(),
                                loops=[list(l) for l in self.loops], rebuilt=False, shift=0.0))
        return Rs, ts
