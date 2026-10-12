"""Run the full simulation + graph SLAM and export everything the web viewer needs to one JSON file."""
from __future__ import annotations

import argparse
import base64
import json
import time
import zlib
from pathlib import Path

import numpy as np

from . import world as W
from .drone_slam import KF as D_KF
from .drone_slam import VOX, DroneGraphSLAM, to_quat
from .rover_slam import KF as R_KF
from .rover_slam import RoverGraphSLAM
from .sensors import DRONE_AZ, DRONE_DIRS, DRONE_EL, DRONE_MAX_RANGE, ROVER_BEAMS, ROVER_MAX_RANGE
from .sim import LIDAR_HZ, Simulation

ROOT = Path(__file__).resolve().parents[1]


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def r3(a):
    return [round(float(v), 3) for v in a]


def run(duration: float, seed: int, out: Path, map_path: Path | None = None, quiet: bool = False):
    world = W.load_world(map_path)
    sim = Simulation(seed, world)
    dt = 1.0 / LIDAR_HZ
    nframes = int(duration * LIDAR_HZ)
    rover = RoverGraphSLAM(world.rover_start, world.arena, seed=seed + 1)
    drone = None
    rec = dict(t=[], movers=[], r_gt=[], r_est=[], r_dr=[], r_fe=[], r_rng=[], r_part=[],
               d_gt=[], d_est=[], d_dr=[], d_fe=[], d_scan=[], d_add=[], d_rem=[], d_ok=[], d_resets=[], grids=[])
    t0 = time.time()
    for k in range(nframes):
        f = sim.step_frame()
        if drone is None:
            drone = DroneGraphSLAM(f.drone_pos, f.drone_R)
        ev = rover.step(f.rover_ranges, f.rover_odom, dt)
        added, removed, reset = drone.step(f.drone_vel_body, f.drone_gyro, f.drone_ranges, dt)

        rec["t"].append(round(f.t, 3))
        rec["movers"].append(r3([v for m in f.movers for v in m]))
        rec["r_gt"].append(r3([f.rover_pose[0], f.rover_pose[1], wrap(f.rover_pose[2])]))
        rec["r_est"].append(r3(rover.est))
        rec["r_dr"].append(r3(rover.dr))
        rec["r_fe"].append(r3(rover.fe[-1]))
        rec["r_rng"].append(np.round(f.rover_ranges * 1000).astype(np.uint16))
        rec["r_part"].append(np.round(rover.front.P[:40, :2] * 100).astype(np.int16))
        rec["d_gt"].append(r3([*f.drone_pos, *f.drone_quat]))
        rec["d_est"].append(r3([*drone.p, *to_quat(drone.R)]))
        rec["d_dr"].append(r3([*drone.p_dr, *to_quat(drone.R_dr)]))
        rec["d_fe"].append(r3([*drone.front.p, *to_quat(drone.front.R)]))
        ok = f.drone_ranges > 0
        rec["d_scan"].append(np.round((DRONE_DIRS[ok] * f.drone_ranges[ok, None])[::3] * 100).astype(np.int16))
        rec["d_add"].append([int(v) for i in added for v in (i, *np.floor(drone.gmap.ctr[i] / VOX).astype(int))])
        rec["d_rem"].append([int(i) for i in removed])
        rec["d_ok"].append(int(drone.icp_ok))
        if reset is not None:
            rec["d_resets"].append(dict(f=k, v=reset))
        if k % 5 == 0 or (ev and ev["rebuilt"]):
            rec["grids"].append(dict(f=k, z=base64.b64encode(zlib.compress(rover.gmap.to_bytes(), 6)).decode()))
        if not quiet and k % 100 == 0:
            e = np.linalg.norm(rover.est[:2] - f.rover_pose[:2])
            de = np.linalg.norm(drone.p - f.drone_pos)
            print(f"  frame {k:4d}/{nframes}  rover err {e:.2f} m  drone err {de:.2f} m  "
                  f"loops r/d {len(rover.loops)}/{len(drone.loops)}  ({time.time() - t0:.0f}s)")

    # ---- final optimisation: smoothed trajectories + final maps ------------------------------------------
    if not quiet:
        print("  final optimisation + map rebuild ...")
    r_smooth = rover.finish()
    d_R, d_t = drone.finish()
    rover.events[-1]["f"] = drone.events[-1]["f"] = nframes - 1  # final optimisation shown on the last frame
    rec["grids"].append(dict(f=nframes - 1, z=base64.b64encode(zlib.compress(rover.gmap.to_bytes(), 6)).decode()))
    final_vox = drone._rebuild_map()
    rec["d_resets"].append(dict(f=nframes - 1, v=final_vox))
    rec["d_add"][-1], rec["d_rem"][-1] = [], []
    d_q = [to_quat(R) for R in d_R]

    gt_r = np.array(rec["r_gt"])[:, :2]
    gt_d = np.array(rec["d_gt"])[:, :3]
    e = lambda est, gt, n: np.linalg.norm(np.array(est)[:, :n] - gt, axis=1)
    rms = lambda x: float(np.sqrt(np.mean(np.square(x))))
    err = dict(
        rover_odom=e(rec["r_dr"], gt_r, 2), rover_front=e(rec["r_fe"], gt_r, 2), rover_online=e(rec["r_est"], gt_r, 2),
        rover_opt=e(r_smooth, gt_r, 2),
        drone_odom=e(rec["d_dr"], gt_d, 3), drone_front=e(rec["d_fe"], gt_d, 3), drone_online=e(rec["d_est"], gt_d, 3),
        drone_opt=e(d_t, gt_d, 3))
    metrics = {f"{k}_rmse": rms(v) for k, v in err.items()}
    metrics.update({f"{k}_err": np.round(v, 3).tolist() for k, v in err.items()})
    metrics.update(rover_loops=len(rover.loops), drone_loops=len(drone.loops), drone_icp_ok=float(np.mean(rec["d_ok"])),
                   voxels=int(drone.gmap.shown.sum()))

    data = dict(
        meta=dict(n=nframes, dt=dt, rover_beams=ROVER_BEAMS, rover_max=ROVER_MAX_RANGE, res=0.1, half=rover.half,
                  grid=rover.gmap.n, drone_az=len(DRONE_AZ), drone_el=np.round(np.degrees(DRONE_EL), 1).tolist(),
                  drone_max=DRONE_MAX_RANGE, vox=VOX, seed=seed, rover_kf=R_KF, drone_kf=D_KF),
        world=dict(name=world.name, arena=world.arena, zones=world.zones, rover_route=world.rover_route,
                   drone_route=world.drone_route,
                   statics=[dict(kind=s.kind, pos=s.pos, size=s.size, yaw=s.yaw, color=s.color) for s in world.statics],
                   movers=[dict(name=m.name, kind=m.kind, size=m.size, color=m.color, z=m.z) for m in world.movers]),
        metrics=metrics, t=rec["t"], movers=rec["movers"],
        rover=dict(gt=rec["r_gt"], est=rec["r_est"], dr=rec["r_dr"], fe=rec["r_fe"], opt=[r3(p) for p in r_smooth],
                   ranges=b64(np.stack(rec["r_rng"])), particles=b64(np.stack(rec["r_part"])),
                   graph=rover.events),
        drone=dict(gt=rec["d_gt"], est=rec["d_est"], dr=rec["d_dr"], fe=rec["d_fe"],
                   opt=[r3([*p, *q]) for p, q in zip(d_t, d_q)], icp_ok=rec["d_ok"], add=rec["d_add"], rem=rec["d_rem"],
                   resets=rec["d_resets"], scan_len=[len(s) for s in rec["d_scan"]], scan=b64(np.concatenate(rec["d_scan"])),
                   graph=drone.events),
        grids=rec["grids"],
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, separators=(",", ":")))
    if not quiet:
        m = metrics
        print(f"rover  RMSE  odometry {m['rover_odom_rmse']:.2f} m | front-end {m['rover_front_rmse']:.2f} m | "
              f"graph SLAM online {m['rover_online_rmse']:.2f} m, optimised {m['rover_opt_rmse']:.2f} m  ({m['rover_loops']} loop closures)")
        print(f"drone  RMSE  odometry {m['drone_odom_rmse']:.2f} m | front-end {m['drone_front_rmse']:.2f} m | "
              f"graph SLAM online {m['drone_online_rmse']:.2f} m, optimised {m['drone_opt_rmse']:.2f} m  ({m['drone_loops']} loop closures)")
        print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s")
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duration", type=float, default=115.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--map", type=Path, default=None, help="JSON map (default: maps/foundry.json)")
    ap.add_argument("--out", type=Path, default=ROOT / "viewer" / "data" / "run.json")
    a = ap.parse_args()
    run(a.duration, a.seed, a.out, a.map)


if __name__ == "__main__":
    main()
