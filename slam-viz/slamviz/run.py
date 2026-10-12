"""Run the full simulation + SLAM and export everything the web viewer needs to one JSON file."""
from __future__ import annotations

import argparse
import base64
import json
import time
import zlib
from pathlib import Path

import numpy as np

from . import world as W
from .drone_slam import VOX, DroneSLAM, to_quat
from .rover_slam import HALF, N, RES, RoverSLAM
from .sensors import DRONE_AZ, DRONE_EL, DRONE_MAX_RANGE, ROVER_BEAMS, ROVER_MAX_RANGE
from .sim import LIDAR_HZ, Simulation


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def run(duration: float, seed: int, out: Path, quiet: bool = False):
    sim = Simulation(seed)
    dt = 1.0 / LIDAR_HZ
    nframes = int(duration * LIDAR_HZ)
    rover = RoverSLAM(W.ROVER_START, seed=seed + 1)
    f0 = None
    rec = dict(t=[], movers=[], r_gt=[], r_est=[], r_dr=[], r_rng=[], r_part=[], d_gt=[], d_est=[], d_dr=[],
               d_scan=[], d_add=[], d_rem=[], d_ok=[], grids=[])
    drone = None
    t0 = time.time()
    for k in range(nframes):
        f = sim.step_frame()
        if drone is None:
            drone = DroneSLAM(f.drone_pos, f.drone_R)
        rover.step(f.rover_ranges, f.rover_odom, dt)
        added, removed = drone.step(f.drone_vel_body, f.drone_gyro, f.drone_ranges, dt)

        rec["t"].append(round(f.t, 3))
        rec["movers"].append([round(float(v), 3) for m in f.movers for v in m])
        wy = lambda y: float(wrap(y))
        rec["r_gt"].append([round(f.rover_pose[0], 3), round(f.rover_pose[1], 3), round(wy(f.rover_pose[2]), 3)])
        rec["r_est"].append([round(float(v), 3) for v in rover.est])
        rec["r_dr"].append([round(float(v), 3) for v in rover.dr])
        rec["r_rng"].append(np.round(f.rover_ranges * 1000).astype(np.uint16))
        rec["r_part"].append(np.round(rover.particles_xy(40) * 100).astype(np.int16))
        q_est = to_quat(drone.R)
        q_dr = to_quat(drone.R_dr)
        rec["d_gt"].append([round(float(v), 3) for v in (*f.drone_pos, *f.drone_quat)])
        rec["d_est"].append([round(float(v), 3) for v in (*drone.p, *q_est)])
        rec["d_dr"].append([round(float(v), 3) for v in (*drone.p_dr, *q_dr)])
        ok = f.drone_ranges > 0
        from .sensors import DRONE_DIRS
        pts = (DRONE_DIRS[ok] * f.drone_ranges[ok, None])[::3]
        rec["d_scan"].append(np.round(pts * 100).astype(np.int16))
        a = []
        for i in added:
            c = np.floor(drone.ctr[i] / VOX).astype(int)
            a += [int(i), int(c[0]), int(c[1]), int(c[2])]
        rec["d_add"].append(a)
        rec["d_rem"].append([int(i) for i in removed])
        rec["d_ok"].append(int(drone.icp_ok))
        if k % 5 == 0 or k == nframes - 1:
            rec["grids"].append(dict(f=k, z=base64.b64encode(zlib.compress(rover.grid_bytes(), 6)).decode()))
        if not quiet and k % 100 == 0:
            e = np.linalg.norm(np.array(rec["r_est"][-1][:2]) - np.array(rec["r_gt"][-1][:2]))
            de = np.linalg.norm(np.array(rec["d_est"][-1][:3]) - np.array(rec["d_gt"][-1][:3]))
            print(f"  frame {k:4d}/{nframes}  rover err {e:.2f} m  drone err {de:.2f} m  ({time.time() - t0:.0f}s)")

    def err(est, gt, n):
        return np.linalg.norm(np.array(est)[:, :n] - np.array(gt)[:, :n], axis=1)

    e_r, e_rdr = err(rec["r_est"], rec["r_gt"], 2), err(rec["r_dr"], rec["r_gt"], 2)
    e_d, e_ddr = err(rec["d_est"], rec["d_gt"], 3), err(rec["d_dr"], rec["d_gt"], 3)
    rms = lambda e: float(np.sqrt(np.mean(e ** 2)))
    metrics = dict(rover_rmse=rms(e_r), rover_dr_rmse=rms(e_rdr), drone_rmse=rms(e_d), drone_dr_rmse=rms(e_ddr),
                   rover_err=np.round(e_r, 3).tolist(), rover_dr_err=np.round(e_rdr, 3).tolist(),
                   drone_err=np.round(e_d, 3).tolist(), drone_dr_err=np.round(e_ddr, 3).tolist(),
                   drone_icp_ok=float(np.mean(rec["d_ok"])), voxels=int(drone.shown.sum()))
    data = dict(
        meta=dict(n=nframes, dt=dt, rover_beams=ROVER_BEAMS, rover_max=ROVER_MAX_RANGE, res=RES, half=HALF, grid=N,
                  drone_az=len(DRONE_AZ), drone_el=np.round(np.degrees(DRONE_EL), 1).tolist(), drone_max=DRONE_MAX_RANGE,
                  vox=VOX, seed=seed),
        world=dict(statics=[dict(kind=s.kind, pos=s.pos, size=s.size, color=s.color) for s in sim.statics],
                   movers=[dict(name=m.name, kind=m.kind, size=m.size, color=m.color, z=m.z) for m in sim.movers],
                   rover_route=W.ROVER_ROUTE, drone_route=W.DRONE_ROUTE, arena=W.ARENA),
        metrics=metrics,
        t=rec["t"], movers=rec["movers"],
        rover=dict(gt=rec["r_gt"], est=rec["r_est"], dr=rec["r_dr"], ranges=b64(np.stack(rec["r_rng"])),
                   particles=b64(np.stack(rec["r_part"]))),
        drone=dict(gt=rec["d_gt"], est=rec["d_est"], dr=rec["d_dr"], icp_ok=rec["d_ok"], add=rec["d_add"], rem=rec["d_rem"],
                   scan_len=[len(s) for s in rec["d_scan"]], scan=b64(np.concatenate(rec["d_scan"]))),
        grids=rec["grids"],
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, separators=(",", ":")))
    if not quiet:
        print(f"rover  ATE  SLAM {metrics['rover_rmse']:.3f} m   dead-reckoning {metrics['rover_dr_rmse']:.3f} m")
        print(f"drone  ATE  SLAM {metrics['drone_rmse']:.3f} m   dead-reckoning {metrics['drone_dr_rmse']:.3f} m"
              f"   (ICP accepted {100 * metrics['drone_icp_ok']:.0f}% of frames, {metrics['voxels']} map voxels)")
        print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duration", type=float, default=80.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "viewer" / "data" / "run.json")
    a = ap.parse_args()
    run(a.duration, a.seed, a.out)


if __name__ == "__main__":
    main()
