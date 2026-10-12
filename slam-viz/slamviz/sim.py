"""Physics stepping: MuJoCo world with a rover and a quadrotor, yielding one record per lidar frame."""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from . import world as W
from .control import DroneController, RoverNav, wrap
from .sensors import drone_lidar, rover_lidar

LIDAR_HZ = 10


@dataclass
class Frame:
    t: float
    rover_pose: np.ndarray       # gt x, y, yaw
    rover_ranges: np.ndarray     # (360,) metres, 0 = no return
    rover_odom: tuple            # measured (v, w) over the frame interval (noisy)
    drone_pos: np.ndarray
    drone_R: np.ndarray
    drone_quat: np.ndarray       # w x y z
    drone_ranges: np.ndarray     # (864,) lidar ranges, 0 = no return
    movers: list                 # [(x, y, yaw)]
    drone_vel_body: np.ndarray   # noisy body-frame velocity (VIO-ish), m/s
    drone_gyro: np.ndarray       # noisy angular rate, rad/s (body)


class Simulation:
    def __init__(self, seed: int = 0, world: W.World | None = None):
        self.rng = np.random.default_rng(seed)
        self.world = world or W.load_world()
        self.statics, self.movers = self.world.statics, self.world.movers
        self.m = mujoco.MjModel.from_xml_string(W.mjcf(self.world))
        self.d = mujoco.MjData(self.m)
        self.drone_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "drone")
        self.rl, self.dl = rover_lidar(self.m, self.d, self.rng), drone_lidar(self.m, self.d, self.rng)
        self.nav, self.ctl = RoverNav(self.world.rover_route), DroneController(self.world.drone_route)
        self.d.qpos[:3] = self.world.rover_start
        mujoco.mj_forward(self.m, self.d)
        self.ranges = np.full(360, 10.0)  # last rover scan; the reactive navigator reads it between lidar frames

    @property
    def rover_pose(self):
        return np.array(self.d.qpos[:3])

    def _drone_R(self):
        return self.d.xmat[self.drone_id].reshape(3, 3).copy()

    def _set_movers(self, t):
        for i, mo in enumerate(self.movers):
            x, y, yaw = mo.pose(t)
            self.d.mocap_pos[i] = [x, y, mo.z]
            self.d.mocap_quat[i] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

    def step_frame(self) -> Frame:
        m, d = self.m, self.d
        dt = m.opt.timestep
        n = int(round(1 / LIDAR_HZ / dt))
        x0, y0, yaw0 = self.rover_pose
        for _ in range(n):
            x, y, yaw = d.qpos[:3]
            v, w = self.nav.command(x, y, yaw, self.ranges)
            d.ctrl[:] = [v * np.cos(yaw), v * np.sin(yaw), w]
            R = self._drone_R()
            f, tq = self.ctl.step(d.qpos[3:6], d.qvel[3:6], R, d.qvel[6:9], dt)
            d.xfrc_applied[self.drone_id, :3] = f
            d.xfrc_applied[self.drone_id, 3:] = tq
            self._set_movers(d.time + dt)
            mujoco.mj_step(m, d)
        t = d.time
        # --- sensors at frame time
        x, y, yaw = self.rover_pose
        c, s = np.cos(yaw), np.sin(yaw)
        self.ranges = self.rl.scan(np.array([x, y, 0.25]), np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]))
        # wheel odometry: forward speed + yaw rate measured over the interval, with scale error, bias and noise
        dx, dy, dyaw = x - x0, y - y0, wrap(yaw - yaw0)
        v_true = (dx * np.cos(yaw0 + dyaw / 2) + dy * np.sin(yaw0 + dyaw / 2)) / (1 / LIDAR_HZ)
        w_true = dyaw / (1 / LIDAR_HZ)
        odom = (v_true * 1.035 + self.rng.normal(0, 0.03), w_true * 1.02 + 0.018 + self.rng.normal(0, 0.03))
        R = self._drone_R()
        pos = np.array(d.qpos[3:6])
        ranges = self.dl.scan(pos, R)
        vel_b = R.T @ d.qvel[3:6]
        vel_meas = vel_b * 1.04 + self.rng.normal(0, 0.03, 3) + np.array([0.02, -0.015, 0.0])
        gyro = d.qvel[6:9] + self.rng.normal(0, 0.01, 3) + np.array([0.0, 0.0, 0.012])
        return Frame(t, self.rover_pose.copy(), self.ranges.copy(), odom, pos, R, np.array(d.qpos[6:10]),
                     ranges, [mo.pose(t) for mo in self.movers], vel_meas, gyro)

