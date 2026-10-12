"""Fast end-to-end checks (~15 s): world builds, sensors see geometry, SLAM beats dead reckoning."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slamviz.drone_slam import DroneSLAM  # noqa: E402
from slamviz.rover_slam import RoverSLAM  # noqa: E402
from slamviz.sim import Simulation  # noqa: E402
from slamviz import world as W  # noqa: E402


def test_world_and_sensors():
    sim = Simulation(0)
    f = sim.step_frame()
    assert (f.rover_ranges > 0).mean() > 0.5          # rover lidar returns
    assert (f.drone_ranges > 0).mean() > 0.5          # drone lidar returns
    assert len(f.movers) == len(W.build_movers())


def test_slam_beats_dead_reckoning():
    sim = Simulation(0)
    rover = RoverSLAM(W.ROVER_START)
    f = sim.step_frame()
    drone = DroneSLAM(f.drone_pos, f.drone_R)
    e_r, e_rdr, e_d, e_ddr = [], [], [], []
    for _ in range(300):  # 30 s
        rover.step(f.rover_ranges, f.rover_odom, 0.1)
        drone.step(f.drone_vel_body, f.drone_gyro, f.drone_ranges, 0.1)
        e_r.append(np.linalg.norm(rover.est[:2] - f.rover_pose[:2]))
        e_rdr.append(np.linalg.norm(rover.dr[:2] - f.rover_pose[:2]))
        e_d.append(np.linalg.norm(drone.p - f.drone_pos))
        e_ddr.append(np.linalg.norm(drone.p_dr - f.drone_pos))
        f = sim.step_frame()
    rms = lambda e: float(np.sqrt(np.mean(np.square(e))))
    assert rms(e_r) < 0.5 * rms(e_rdr)
    assert rms(e_d) < 0.5 * rms(e_ddr)
    assert rms(e_r) < 1.0 and rms(e_d) < 1.0
