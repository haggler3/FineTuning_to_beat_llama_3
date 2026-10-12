"""Tests: pose-graph solvers, loop-closure ICP, custom-map loading, and an end-to-end graph-SLAM run (~1 min)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scipy.spatial import cKDTree  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

from slamviz import world as W  # noqa: E402
from slamviz.drone_slam import DroneGraphSLAM  # noqa: E402
from slamviz.posegraph import optimize_se2, optimize_se3, se2_between, se2_compose  # noqa: E402
from slamviz.rover_slam import RoverGraphSLAM, match_scans  # noqa: E402
from slamviz.sim import Simulation  # noqa: E402

MINI_MAP = dict(
    name="mini", arena=8.0,
    statics=[dict(kind="wall", p1=[-8, -8], p2=[8, -8], h=3, th=0.1), dict(kind="wall", p1=[-8, 8], p2=[8, 8], h=3, th=0.1),
             dict(kind="wall", p1=[-8, -8], p2=[-8, 8], h=3, th=0.1), dict(kind="wall", p1=[8, -8], p2=[8, 8], h=3, th=0.1),
             dict(kind="box", pos=[0, 0, 1], size=[1.5, 0.7, 1], yaw=25, color="#8a6a40"),
             dict(kind="cyl", pos=[-4.5, 3.5, 1.5], size=[0.4, 1.5]), dict(kind="box", pos=[4, -3, 0.6], size=[0.6, 0.6, 0.6], yaw=0),
             dict(kind="wall", p1=[2, 4], p2=[5.5, 6.5], h=2.5, th=0.1)],
    movers=[dict(name="m", kind="box", size=[0.3, 0.3, 0.4], color="#ff7a59",
                 motion=dict(type="line", c=[0, 4.5], dir=0, amp=2.0, w=0.4, phase=0))],
    zones=[],
    rover=dict(start=[-5, -5, 0], route=[[5, -5], [5, 5], [-5, 5], [-5, -5]]),
    drone=dict(start=[-5, -5, 0.15], route=[[-5, -5, 1.5], [5, -5, 1.6], [5, 5, 1.6], [-5, 5, 1.6], [-5, -5, 1.6]]))


@pytest.fixture(scope="module")
def mini_world(tmp_path_factory):
    p = tmp_path_factory.mktemp("maps") / "mini.json"
    p.write_text(json.dumps(MINI_MAP))
    return W.load_world(p)


def test_default_map_loads_and_compiles():
    w = W.load_world()
    assert len(w.statics) > 20 and len(w.movers) == 4
    import mujoco
    mujoco.MjModel.from_xml_string(W.mjcf(w))


def test_custom_map_sensors(mini_world):
    sim = Simulation(0, mini_world)
    f = sim.step_frame()
    assert (f.rover_ranges > 0).mean() > 0.5 and (f.drone_ranges > 0).mean() > 0.5


def test_pose_graph_se2_closes_a_drifted_loop():
    n = 50
    gt = np.array([[3 * np.cos(a), 3 * np.sin(a), a + np.pi / 2] for a in np.linspace(0, 2 * np.pi, n, endpoint=False)])
    i, j = list(range(n - 1)) + [n - 1], list(range(1, n)) + [0]
    z = [se2_between(gt[a], gt[b]) for a, b in zip(i, j)]
    p = [gt[0]]
    for k in range(n - 1):
        p.append(se2_compose(p[-1], z[k] + np.array([0.02, 0.0, 0.01])))
    out = optimize_se2(np.array(p), i, j, z, np.tile([0.05, 0.05, 0.02], (n, 1)))
    assert np.abs(out[:, :2] - gt[:, :2]).max() < 0.02


def test_pose_graph_se3_closes_a_drifted_loop():
    n = 40
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    R0 = Rot.from_rotvec(np.c_[np.zeros(n), np.zeros(n), ang + np.pi / 2])
    t0 = np.c_[3 * np.cos(ang), 3 * np.sin(ang), 0.3 * np.sin(2 * ang)]
    i, j = list(range(n - 1)) + [n - 1], list(range(1, n)) + [0]
    Rz = np.array([(R0[a].inv() * R0[b]).as_matrix() for a, b in zip(i, j)])
    tz = np.array([R0[a].inv().apply(t0[b] - t0[a]) for a, b in zip(i, j)])
    Rp, tp = [R0[0].as_matrix()], [t0[0]]
    for k in range(n - 1):
        Rk = Rot.from_matrix(Rp[-1])
        tp.append(tp[-1] + Rk.apply(tz[k] + [0.02, 0.01, 0.01]))
        Rp.append((Rk * Rot.from_matrix(Rz[k]) * Rot.from_rotvec([0, 0, 0.01])).as_matrix())
    R, t = optimize_se3(np.array(Rp), np.array(tp), i, j, Rz, tz, np.full(n, 0.05), np.full(n, 0.02))
    assert np.abs(t - t0).max() < 0.02


def test_scan_matching_recovers_relative_pose():
    rng = np.random.default_rng(0)
    room = np.concatenate([np.c_[np.linspace(-4, 4, 80), np.full(80, -3)], np.c_[np.linspace(-4, 4, 80), np.full(80, 3)],
                           np.c_[np.full(60, -4), np.linspace(-3, 3, 60)], np.c_[np.full(30, 1.0), np.linspace(-3, 0, 30)]])
    T = np.array([0.35, -0.2, 0.1])  # pose of scan B's frame in scan A's frame
    c, s = np.cos(T[2]), np.sin(T[2])
    Rm = np.array([[c, -s], [s, c]])
    b = (room - T[:2]) @ Rm  # the same room seen from B
    got, frac, rms, rev = match_scans(b + rng.normal(0, 0.01, b.shape), room, cKDTree(room), np.zeros(3))
    assert frac > 0.9 and rev > 0.9 and np.abs(got - T).max() < 0.08  # target is sampled every 10 cm, so ~5 cm is the floor


def test_end_to_end_graph_slam_on_custom_map(mini_world):
    """A short loop in a custom map: the rover must close the loop, and graph SLAM must beat raw odometry."""
    sim = Simulation(0, mini_world)
    rover = RoverGraphSLAM(mini_world.rover_start, mini_world.arena, seed=1)
    f = sim.step_frame()
    drone = DroneGraphSLAM(f.drone_pos, f.drone_R)
    e_r, e_rdr, e_d, e_ddr, gt_r = [], [], [], [], []
    for _ in range(780):  # 78 s: one lap takes ~62 s (avoidance slows the rover), then a revisit
        gt_r.append(f.rover_pose.copy())
        rover.step(f.rover_ranges, f.rover_odom, 0.1)
        drone.step(f.drone_vel_body, f.drone_gyro, f.drone_ranges, 0.1)
        e_r.append(np.linalg.norm(rover.est[:2] - f.rover_pose[:2]))
        e_rdr.append(np.linalg.norm(rover.dr[:2] - f.rover_pose[:2]))
        e_d.append(np.linalg.norm(drone.p - f.drone_pos))
        e_ddr.append(np.linalg.norm(drone.p_dr - f.drone_pos))
        f = sim.step_frame()
    rms = lambda e: float(np.sqrt(np.mean(np.square(e))))
    assert len(rover.loops) >= 1, "rover found no loop closure"
    assert rms(e_r) < 0.4 * rms(e_rdr)
    assert rms(e_d) < 0.4 * rms(e_ddr)
    # every accepted loop closure must agree with ground truth (no false positives)
    for d in rover.diag:
        if d["ok"]:
            truth = se2_between(gt_r[rover.kf_frame[d["i"]]], gt_r[rover.kf_frame[d["j"]]])
            assert np.linalg.norm(d["T"][:2] - truth[:2]) < 0.25
    smooth = rover.finish()
    assert smooth.shape == (len(e_r), 3)
