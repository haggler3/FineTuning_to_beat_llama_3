"""Graph SLAM for the 2D rover.

Front-end   particle filter + correlative scan matching against a *local, decaying* occupancy sub-map. This is
            scan-to-sub-map odometry: accurate over seconds, but it drifts over minutes.
Graph       every ``KF`` frames a keyframe node is added; consecutive keyframes are tied by odometry edges.
Loop closure  when the rover comes back near an old keyframe, ICP between the two LiDAR scans yields a
            relative-pose constraint (verified by inlier ratio and mutual overlap).
Back-end    the pose graph is re-optimised (``posegraph.optimize_se2``) and the global occupancy map is rebuilt
            from the corrected poses, which is what visibly "snaps" the map shut.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .occgrid import OccupancyGrid
from .posegraph import optimize_se2, se2_between, se2_compose, se2_inverse, wrap
from .sensors import ROVER_ANGLES

KF = 4                     # frames per keyframe
LOOP_MIN_GAP = 20          # keyframes that must separate a loop-closure pair ...
LOOP_MIN_PATH = 15.0       # ... together with this much travelled path
LOOP_RADIUS = 3.5          # m, candidate search radius in the current graph estimate
SIG_ODOM = (0.04, 0.04, 0.012)
SIG_LOOP = (0.05, 0.05, 0.02)


class RoverFrontend:
    """Particle filter + correlative scan matching against a decaying local grid."""
    # scan-match window / step, beam subsampling, likelihood-field sigma, odometry prior sigmas
    W_XY, S_XY, W_TH, S_TH, SUB, SIG, PRI_XY, PRI_TH = 0.15, 0.025, 0.06, 0.01, 2, 0.15, 0.05, 0.02

    def __init__(self, start_pose, half, n_particles=80, seed=1, local_decay=0.985):
        self.rng = np.random.default_rng(seed)
        self.grid = OccupancyGrid(half, decay=local_decay, with_dist=True)
        self.P = np.tile(np.asarray(start_pose, float), (n_particles, 1))
        self.w = np.full(n_particles, 1.0 / n_particles)
        self.est = np.asarray(start_pose, float).copy()
        self.frame = 0

    def _loglik(self, poses, ranges, ang):
        g = self.grid
        a = poses[:, 2:3] + ang[None, :]
        ep = np.stack([poses[:, 0:1] + ranges[None, :] * np.cos(a), poses[:, 1:2] + ranges[None, :] * np.sin(a)], -1)
        c = g.cell(ep)
        inb = (c >= 0).all(-1) & (c < g.n).all(-1)
        d = np.where(inb, g.dist[np.clip(c[..., 0], 0, g.n - 1), np.clip(c[..., 1], 0, g.n - 1)], 1.0)
        return (-(np.minimum(d, 1.0) ** 2) / (2 * self.SIG ** 2)).sum(-1) * 0.35

    def _refine(self, pose, ranges, ang, pred):
        ds = np.arange(-self.W_XY, self.W_XY + 1e-6, self.S_XY)
        dth = np.arange(-self.W_TH, self.W_TH + 1e-6, self.S_TH)
        gx, gy, gt = np.meshgrid(ds, ds, dth, indexing="ij")
        cand = np.stack([pose[0] + gx.ravel(), pose[1] + gy.ravel(), pose[2] + gt.ravel()], 1)
        ll = self._loglik(cand, ranges, ang)
        # Gaussian prior on the odometry prediction keeps the match from sliding along featureless walls
        ll -= 0.5 * (((cand[:, 0] - pred[0]) ** 2 + (cand[:, 1] - pred[1]) ** 2) / self.PRI_XY ** 2
                     + wrap(cand[:, 2] - pred[2]) ** 2 / self.PRI_TH ** 2)
        return cand[np.argmax(ll)]

    def step(self, ranges, odom, dt):
        v, w = odom
        dist, dth = v * dt, w * dt
        pred = np.array([self.est[0] + dist * np.cos(self.est[2] + dth / 2), self.est[1] + dist * np.sin(self.est[2] + dth / 2),
                         self.est[2] + dth])
        n = len(self.P)
        d_ = dist + self.rng.normal(0, 0.015 + 0.08 * abs(dist), n)
        t_ = dth + self.rng.normal(0, 0.008 + 0.1 * abs(dth), n)
        ym = self.P[:, 2] + t_ / 2
        self.P[:, 0] += d_ * np.cos(ym)
        self.P[:, 1] += d_ * np.sin(ym)
        self.P[:, 2] = wrap(self.P[:, 2] + t_)
        idx = np.where(ranges > 0)[0][::self.SUB]
        if self.frame >= 3 and len(idx) > 20:
            r, ang = ranges[idx], ROVER_ANGLES[idx]
            ll = self._loglik(self.P, r, ang)
            self.w = np.exp(ll - ll.max()); self.w /= self.w.sum()
            mean = np.array([np.sum(self.w * self.P[:, 0]), np.sum(self.w * self.P[:, 1]),
                             np.arctan2(np.sum(self.w * np.sin(self.P[:, 2])), np.sum(self.w * np.cos(self.P[:, 2])))])
            refined = self._refine(mean, r, ang, pred)
            if 1.0 / np.sum(self.w ** 2) < 0.5 * n:  # systematic resampling
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
        self.grid.integrate(self.est, ranges, ROVER_ANGLES)
        self.frame += 1


# ------------------------------------------------------------------------------------------ loop closure
def icp2d(src, tree, dst, T0, gates=(0.8, 0.5, 0.35, 0.25, 0.2, 0.15, 0.15, 0.12)):
    """Register src (N,2, own frame) to dst (target frame). T0 = initial pose of src frame in the target frame."""
    T = np.array(T0, float)
    for g in gates:
        c, s = np.cos(T[2]), np.sin(T[2])
        R = np.array([[c, -s], [s, c]])
        w = src @ R.T + T[:2]
        d, k = tree.query(w, distance_upper_bound=g)
        m = np.isfinite(d)
        if m.sum() < 25:
            return T, 0.0, 1.0
        a, b = w[m], dst[k[m]]
        keep = d[m] <= np.quantile(d[m], 0.85)
        a, b = a[keep], b[keep]
        ca, cb = a.mean(0), b.mean(0)
        U, _, Vt = np.linalg.svd((a - ca).T @ (b - cb))
        Rm = Vt.T @ U.T
        if np.linalg.det(Rm) < 0:
            Vt[-1] *= -1
            Rm = Vt.T @ U.T
        tm = cb - Rm @ ca
        T = se2_compose(np.array([tm[0], tm[1], np.arctan2(Rm[1, 0], Rm[0, 0])]), T)
    c, s = np.cos(T[2]), np.sin(T[2])
    w = src @ np.array([[c, -s], [s, c]]).T + T[:2]
    d, _ = tree.query(w, distance_upper_bound=0.12)
    inl = np.isfinite(d)
    return T, float(inl.mean()), float(np.sqrt(np.mean(d[inl] ** 2))) if inl.any() else 1.0


def match_scans(src, dst, tree_dst, T_init):
    """Multi-start ICP; returns (T, inlier fraction, rms, reverse overlap) of the best start."""
    best = None
    for dx in (-1.0, -0.5, 0.0, 0.5, 1.0):
        for dy in (-1.0, -0.5, 0.0, 0.5, 1.0):
            for dth in (-0.12, 0.0, 0.12):
                T, frac, rms = icp2d(src, tree_dst, dst, np.array([T_init[0] + dx, T_init[1] + dy, T_init[2] + dth]))
                if best is None or frac > best[1]:
                    best = (T, frac, rms)
    T, frac, rms = best
    c, s = np.cos(T[2]), np.sin(T[2])
    w = src @ np.array([[c, -s], [s, c]]).T + T[:2]
    d, _ = cKDTree(w).query(dst, distance_upper_bound=0.12)
    return T, frac, rms, float(np.isfinite(d).mean())


# ------------------------------------------------------------------------------------------ the SLAM system
class RoverGraphSLAM:
    def __init__(self, start_pose, arena, seed=1):
        self.half = arena + 1.0
        self.front = RoverFrontend(start_pose, self.half, seed=seed)
        self.gmap = OccupancyGrid(self.half)           # global (display) map, rebuilt after each optimisation
        self.start = np.asarray(start_pose, float)
        self.dr = self.start.copy()                    # pure odometry, for comparison
        self.est = self.start.copy()                   # online estimate = T_corr o front-end pose
        self.T_corr = np.zeros(3)
        self.frame = 0
        self.fe: list[np.ndarray] = []                 # front-end pose per frame
        self.ranges: list[np.ndarray] = []
        self.kf_frame: list[int] = []
        self.kf_fe: list[np.ndarray] = []
        self.kf_s: list[float] = []                    # front-end path length at each keyframe
        self.kf_pts: list[np.ndarray] = []
        self.kf_trees: list[cKDTree] = []
        self.nodes: list[np.ndarray] = []              # current optimised node poses
        self.edges: list[tuple] = []                   # (i, j, z, sigma, kind)
        self.loops: list[tuple] = []                   # (i, j) accepted loop closures
        self.events: list[dict] = []
        self.last_loop_node = -10
        self.last_rebuild_frame = -100
        self.diag: list[dict] = []                     # every loop-closure verification, for tests / debugging

    # -- helpers ---------------------------------------------------------------------------------
    def _frame_poses(self):
        """Corrected pose for every frame so far (each frame hangs off its keyframe's optimised pose)."""
        out = []
        for g, p in enumerate(self.fe):
            k = min(g // KF, len(self.nodes) - 1)
            out.append(se2_compose(se2_compose(self.nodes[k], se2_inverse(self.kf_fe[k])), p))
        return out

    def _optimise(self):
        i = [e[0] for e in self.edges]; j = [e[1] for e in self.edges]
        z = np.array([e[2] for e in self.edges]); sg = np.array([e[3] for e in self.edges])
        before = np.array(self.nodes)
        opt = optimize_se2(before, i, j, z, sg)
        self.nodes = [p for p in opt]
        return float(np.max(np.linalg.norm(opt[:, :2] - before[:, :2], axis=1)))

    def _rebuild_map(self):
        self.gmap.reset()
        for g, p in enumerate(self._frame_poses()):
            self.gmap.integrate(p, self.ranges[g], ROVER_ANGLES)
        self.last_rebuild_frame = self.frame

    def _snapshot(self, rebuilt, shift):
        self.events.append(dict(f=self.frame, nodes=np.round(np.array(self.nodes)[:, :2], 3).tolist(),
                                loops=[list(l) for l in self.loops], rebuilt=bool(rebuilt), shift=round(shift, 3)))

    # -- main step -------------------------------------------------------------------------------
    def step(self, ranges, odom, dt):
        v, w = odom
        self.dr[0] += v * dt * np.cos(self.dr[2] + w * dt / 2)
        self.dr[1] += v * dt * np.sin(self.dr[2] + w * dt / 2)
        self.dr[2] = wrap(self.dr[2] + w * dt)
        self.front.step(ranges, odom, dt)
        fe = self.front.est.copy()
        self.fe.append(fe)
        self.ranges.append(ranges.copy())
        self.est = se2_compose(self.T_corr, fe)
        event = None
        if self.frame % KF == 0:
            event = self._add_keyframe(ranges, fe)
        if not (event and event["rebuilt"]):  # a rebuild already re-integrated this frame's scan
            self.gmap.integrate(self.est, ranges, ROVER_ANGLES)
        self.frame += 1
        return event

    def _add_keyframe(self, ranges, fe):
        n = len(self.nodes)
        valid = ranges > 0
        pts = np.stack([ranges[valid] * np.cos(ROVER_ANGLES[valid]), ranges[valid] * np.sin(ROVER_ANGLES[valid])], 1)[::2]
        self.kf_s.append((self.kf_s[-1] + float(np.hypot(*(fe[:2] - self.kf_fe[-1][:2])))) if n else 0.0)
        self.kf_frame.append(self.frame); self.kf_fe.append(fe); self.kf_pts.append(pts); self.kf_trees.append(cKDTree(pts))
        self.nodes.append(self.est.copy())
        if n:
            self.edges.append((n - 1, n, se2_between(self.kf_fe[n - 1], fe), SIG_ODOM, "odom"))
        found = self._detect_loops(n) if n - self.last_loop_node >= 3 else []
        if not found:
            return None
        for (j, T, frac, rms) in found:
            self.edges.append((j, n, T, SIG_LOOP, "loop"))
            self.loops.append((j, n))
        self.last_loop_node = n
        shift = self._optimise()
        self.T_corr = se2_compose(self.nodes[n], se2_inverse(fe))
        self.est = se2_compose(self.T_corr, fe)
        rebuilt = shift > 0.08 and self.frame - self.last_rebuild_frame >= 20
        if rebuilt:
            self._rebuild_map()
        self._snapshot(rebuilt, shift)
        return dict(kind="loop", frame=self.frame, shift=shift, n_loops=len(found), rebuilt=rebuilt)

    def _detect_loops(self, n):
        cur = self.nodes[n]
        cand = [j for j in range(n - LOOP_MIN_GAP) if self.kf_s[n] - self.kf_s[j] > LOOP_MIN_PATH
                and np.hypot(*(self.nodes[j][:2] - cur[:2])) < LOOP_RADIUS]
        cand.sort(key=lambda j: np.hypot(*(self.nodes[j][:2] - cur[:2])))
        out = []
        for j in cand[:2]:
            if any(abs(j - jj) < 6 for jj in [l[0] for l in self.loops[-4:]]) and n - self.last_loop_node < 10:
                continue
            T0 = se2_between(self.nodes[j], cur)
            T, frac, rms, rev = match_scans(self.kf_pts[n], self.kf_pts[j], self.kf_trees[j], T0)
            ok = frac > 0.6 and rev > 0.45 and rms < 0.075 and np.hypot(*(T[:2] - T0[:2])) < 2.0
            self.diag.append(dict(i=j, j=n, frac=frac, rev=rev, rms=rms, ok=bool(ok), T=T, T0=T0))
            if ok:
                out.append((j, T, frac, rms))
        return out

    # -- finishing -------------------------------------------------------------------------------
    def finish(self):
        """Final global optimisation; returns the smoothed (post-optimisation) pose of every frame."""
        if self.loops:
            self._optimise()
        smooth = self._frame_poses()
        self._rebuild_map()
        self._snapshot(True, 0.0)
        return np.array(smooth)
