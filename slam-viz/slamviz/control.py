"""Navigation + low level control for the rover (reactive) and the quadrotor (geometric)."""
from __future__ import annotations

import numpy as np

from .sensors import ROVER_ANGLES


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class RoverNav:
    def __init__(self, route, vmax=0.95):
        self.route, self.vmax, self.i = list(route), vmax, 0
        self.laps = 0

    def command(self, x, y, yaw, ranges):
        wp = self.route[self.i]
        if np.hypot(wp[0] - x, wp[1] - y) < 0.8:
            self.i = (self.i + 1) % len(self.route)
            self.laps += self.i == 0
            wp = self.route[self.i]
        err = wrap(np.arctan2(wp[1] - y, wp[0] - x) - yaw)
        r = np.where(ranges > 0, ranges, 10.0)
        # repulsion: obstacles close on one side steer the other way
        near = np.clip((1.5 - r) / 1.5, 0, 1)
        steer = -np.sum(near * np.sin(ROVER_ANGLES) * np.cos(ROVER_ANGLES / 2).clip(0)) / 20.0
        front = r[np.abs(ROVER_ANGLES) < np.radians(30)].min()
        v = self.vmax * np.clip((front - 0.5) / 1.0, 0.08, 1.0) * max(0.15, 1 - abs(err) / 1.3)
        w = np.clip(2.0 * err + 2.2 * steer, -1.6, 1.6)
        if front < 0.55:  # boxed in: spin toward the more open side
            left, right = r[(ROVER_ANGLES > 0.3) & (ROVER_ANGLES < 1.5)].mean(), r[(ROVER_ANGLES < -0.3) & (ROVER_ANGLES > -1.5)].mean()
            w, v = (1.4 if left > right else -1.4), 0.0
        return v, w


class DroneController:
    """Waypoint follower: velocity loop -> thrust vector -> geometric attitude control."""
    mass, inertia, g = 1.2, np.array([0.012, 0.012, 0.022]), 9.81

    def __init__(self, route, vmax=1.5):
        self.route, self.vmax, self.i, self.yaw = list(route), vmax, 0, 0.0
        self.done = False

    def step(self, pos, vel, R, wb, dt):
        wp = np.array(self.route[self.i])
        to = wp - pos
        if np.linalg.norm(to) < 0.6 and self.i < len(self.route) - 1:
            self.i += 1
            wp = np.array(self.route[self.i])
            to = wp - pos
        dist = np.linalg.norm(to)
        speed = min(self.vmax, 1.2 * dist)
        v_des = to / max(dist, 1e-6) * speed
        a = np.clip(2.6 * (v_des - vel), -4.5, 4.5)
        a_cmd = a + np.array([0, 0, self.g])
        # tilt limit
        zdes = a_cmd / np.linalg.norm(a_cmd)
        tilt = np.arccos(np.clip(zdes[2], -1, 1))
        if tilt > np.radians(30):
            h = np.array([zdes[0], zdes[1], 0]); h /= np.linalg.norm(h) + 1e-9
            zdes = np.array([h[0] * np.sin(np.radians(30)), h[1] * np.sin(np.radians(30)), np.cos(np.radians(30))])
        zb = R[:, 2]
        thrust = float(np.clip(self.mass * (a_cmd @ zb), 0, 2.2 * self.mass * self.g))
        # yaw slowly follows travel direction (visually pleasing, lidar is 360 deg anyway)
        if np.hypot(vel[0], vel[1]) > 0.4:
            tgt = np.arctan2(vel[1], vel[0])
            self.yaw += np.clip(wrap(tgt - self.yaw), -1, 1) * 1.5 * dt
        xc = np.array([np.cos(self.yaw), np.sin(self.yaw), 0.0])
        yd = np.cross(zdes, xc); yd /= np.linalg.norm(yd)
        xd = np.cross(yd, zdes)
        Rd = np.stack([xd, yd, zdes], 1)
        E = Rd.T @ R - R.T @ Rd
        e = 0.5 * np.array([E[2, 1], E[0, 2], E[1, 0]])
        tau = self.inertia * (-90.0 * e - 16.0 * wb)
        return thrust * zb, R @ tau
