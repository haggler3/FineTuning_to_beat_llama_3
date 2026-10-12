"""Pose-graph optimisation (the "graph" in graph SLAM) for SE(2) and SE(3).

Nodes are keyframe poses, edges are relative-pose measurements (odometry edges between consecutive keyframes,
loop-closure edges between revisits). We minimise the sum of squared, sigma-whitened edge residuals with a
sparse Levenberg-Marquardt solver (numerical edge Jacobians, Huber IRLS weights) so that a bad loop
closure cannot drag the whole trajectory.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation as Rot


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _gather(state, idx):
    return tuple(a[idx] for a in state)


def _lm(n, d, i, j, state, res_fn, retract, iters=30, f_scale=3.0, eps=1e-6, tol=1e-7):
    """Generic sparse LM over a pose graph. ``state`` is a tuple of per-node arrays; ``res_fn(gi, gj)`` gives
    whitened residuals (m,k) for gathered node states; ``retract(state, dx)`` applies a local update."""
    i, j = np.asarray(i), np.asarray(j)
    m = len(i)

    def edge_stuff(st):
        gi, gj = _gather(st, i), _gather(st, j)
        r = res_fn(gi, gj)
        k = r.shape[1]
        Ji = np.zeros((m, k, d)); Jj = np.zeros((m, k, d))
        for c in range(d):
            dx = np.zeros((m, d)); dx[:, c] = eps
            Ji[:, :, c] = (res_fn(retract(gi, dx), gj) - r) / eps
            Jj[:, :, c] = (res_fn(gi, retract(gj, dx)) - r) / eps
        return r, Ji, Jj

    def cost(r):
        nr = np.linalg.norm(r, axis=1)
        return np.sum(np.where(nr <= f_scale, 0.5 * nr ** 2, f_scale * (nr - 0.5 * f_scale)))

    lam = 1e-3
    r, Ji, Jj = edge_stuff(state)
    c0 = cost(r)
    for _ in range(iters):
        nr = np.linalg.norm(r, axis=1)
        w = np.where(nr <= f_scale, 1.0, f_scale / np.maximum(nr, 1e-12))  # Huber IRLS weight per edge
        rows, cols, vals = [], [], []
        b = np.zeros(n * d)
        for (idx, J) in ((i, Ji), (j, Jj)):
            np.add.at(b, (idx[:, None] * d + np.arange(d)[None, :]).ravel(), np.einsum("mkd,mk->md", J, w[:, None] * r).ravel())
        for (a, Ja) in ((i, Ji), (j, Jj)):
            for (bb, Jb) in ((i, Ji), (j, Jj)):
                blk = np.einsum("mkd,mke->mde", Ja, w[:, None, None] * Jb)
                rr = (a[:, None, None] * d + np.arange(d)[None, :, None]) + 0 * blk
                cc = (bb[:, None, None] * d + np.arange(d)[None, None, :]) + 0 * blk
                rows.append(rr.ravel()); cols.append(cc.ravel()); vals.append(blk.ravel())
        H = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n * d, n * d)).tocsr()
        H = H + coo_matrix((np.full(d, 1e8), (np.arange(d), np.arange(d))), shape=(n * d, n * d)).tocsr()  # gauge: pin node 0
        diag = H.diagonal()
        while True:
            dx = spsolve(H + coo_matrix((lam * diag, (np.arange(n * d), np.arange(n * d))), shape=(n * d, n * d)).tocsr(), -b)
            new = retract(state, dx.reshape(n, d))
            r2, Ji2, Jj2 = edge_stuff(new)
            c1 = cost(r2)
            if c1 < c0 or lam > 1e6:
                break
            lam *= 10
        if c1 >= c0:
            break
        state, r, Ji, Jj = new, r2, Ji2, Jj2
        lam = max(lam / 5, 1e-9)
        done = (c0 - c1) < tol * max(c0, 1e-9)
        c0 = c1
        if done:
            break
    return state


# ------------------------------------------------------------------------------------------ SE(2)
def se2_compose(a, b):
    c, s = np.cos(a[2]), np.sin(a[2])
    return np.array([a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2])])


def se2_inverse(a):
    c, s = np.cos(a[2]), np.sin(a[2])
    return np.array([-(c * a[0] + s * a[1]), s * a[0] - c * a[1], wrap(-a[2])])


def se2_between(a, b):
    """Pose of b expressed in the frame of a."""
    return se2_compose(se2_inverse(a), b)


def optimize_se2(poses, i, j, z, sigma, f_scale=3.0, iters=30):
    """poses (n,3); edges i,j (m,), measurements z (m,3) expressed in frame i, sigma (m,3). Node 0 is held fixed."""
    poses = np.asarray(poses, float).copy()
    z, sigma = np.asarray(z, float), np.asarray(sigma, float)

    def res(gi, gj):
        (pi,), (pj,) = gi, gj
        d = pj[:, :2] - pi[:, :2]
        c, s = np.cos(pi[:, 2]), np.sin(pi[:, 2])
        return np.stack([c * d[:, 0] + s * d[:, 1] - z[:, 0], -s * d[:, 0] + c * d[:, 1] - z[:, 1],
                         wrap(pj[:, 2] - pi[:, 2] - z[:, 2])], 1) / sigma

    def retract(st, dx):
        p = st[0] + dx
        p[:, 2] = wrap(p[:, 2])
        return (p,)

    return _lm(len(poses), 3, i, j, (poses,), res, retract, iters, f_scale)[0]


# ------------------------------------------------------------------------------------------ SE(3)
def optimize_se3(R0, t0, i, j, Rz, tz, sig_t, sig_r, f_scale=3.0, iters=30):
    """Absolute poses R0 (n,3,3), t0 (n,3); edges i,j with measured relative rotation Rz (m,3,3) and translation
    tz (m,3) in frame i; sig_t, sig_r are per-edge scalar standard deviations (m,). Rotations are updated as R <- R * Exp(delta) (no angle-wrapping singularities).
    Node 0 is held fixed."""
    Rz_inv = Rot.from_matrix(np.asarray(Rz, float)).inv()
    tz, sig_t, sig_r = np.asarray(tz, float), np.asarray(sig_t, float), np.asarray(sig_r, float)

    def res(gi, gj):
        (Ri, ti), (Rj, tj) = gi, gj
        Rii = Rot.from_matrix(Ri).inv()
        dt = Rii.apply(tj - ti) - tz
        dr = (Rz_inv * Rii * Rot.from_matrix(Rj)).as_rotvec()
        return np.concatenate([dt / sig_t[:, None], dr / sig_r[:, None]], 1)

    def retract(st, dx):
        R, t = st
        return ((Rot.from_matrix(R) * Rot.from_rotvec(dx[:, 3:])).as_matrix(), t + dx[:, :3])

    R, t = _lm(len(t0), 6, i, j, (np.array(R0, float), np.array(t0, float)), res, retract, iters, f_scale)
    return R, t
