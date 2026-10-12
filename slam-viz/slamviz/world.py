"""Custom dynamic environment: one description feeds both the MuJoCo model and the web viewer."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

ARENA = 12.0  # half-extent of the arena (m)


@dataclass
class Static:
    kind: str  # "box" | "cyl"
    pos: tuple
    size: tuple  # box: half extents; cyl: (radius, half-height)
    color: str = "#3a4256"


@dataclass
class Mover:
    """A kinematic (mocap) obstacle whose pose is a closed-form function of time."""
    name: str
    kind: str
    size: tuple
    color: str
    fn: callable = field(repr=False)

    def pose(self, t: float):
        """Return (x, y, yaw) at time t; z is size-derived."""
        return self.fn(t)

    @property
    def z(self):
        return self.size[2] if self.kind == "box" else self.size[1]


def _wall(x1, y1, x2, y2, h=3.0, th=0.1, color="#2c3446"):
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return Static("box", (cx, cy, h / 2), (abs(x2 - x1) / 2 + th, abs(y2 - y1) / 2 + th, h / 2), color)


def build_statics() -> list[Static]:
    A = ARENA
    s: list[Static] = []
    # outer walls
    s += [_wall(-A, -A, A, -A), _wall(-A, A, A, A), _wall(-A, -A, -A, A), _wall(A, -A, A, A)]
    # central room (4m half-size) with 2.4 m doorways in the middle of each face
    r, g = 4.0, 1.2
    for sx in (-1, 1):
        s += [_wall(sx * r, -r, sx * r, -g, color="#34405a"), _wall(sx * r, g, sx * r, r, color="#34405a")]
    for sy in (-1, 1):
        s += [_wall(-r, sy * r, -g, sy * r, color="#34405a"), _wall(g, sy * r, r, sy * r, color="#34405a")]
    # crates on the west side
    for (x, y, w, h) in [(-9, 6, 0.55, 0.9), (-10, -3, 0.7, 0.6), (-9.5, -8.5, 0.5, 1.2), (-8.6, 1.6, 0.45, 0.5)]:
        s.append(Static("box", (x, y, h / 2), (w, w, h / 2), "#7a5c3a"))
    # east side: baffle walls + pillars
    s += [_wall(8, -9, 8, -4.5, h=2.4, color="#2f4a4d"), _wall(5, 7, 10, 7, h=2.4, color="#2f4a4d")]
    s += [Static("cyl", (10.5, 3.5, 1.5), (0.35, 1.5), "#5d6b86"), Static("cyl", (10.0, -7.5, 1.5), (0.35, 1.5), "#5d6b86")]
    # overhead gantry the drone can fly under
    s += [Static("box", (8.2, 0, 2.4), (3.4, 1.1, 0.1), "#b0553a")]
    for px in (5.0, 11.4):
        for py in (-0.95, 0.95):
            s.append(Static("cyl", (px, py, 1.2), (0.15, 1.2), "#b0553a"))
    return s


def build_movers() -> list[Mover]:
    return [
        # crate shuttling east-west across the rover's north-east leg
        Mover("shuttle", "box", (0.3, 0.3, 0.45), "#ff7a59",
              lambda t: (6.5 + 3.4 * math.sin(0.55 * t + 0.4), 5.2, 0.0)),
        # post orbiting on the west leg
        Mover("orbiter", "cyl", (0.28, 0.5), "#ffd166",
              lambda t: (-6.5 + 2.2 * math.cos(0.42 * t), -4.0 + 2.2 * math.sin(0.42 * t), 0.0)),
        # windmill bar inside the central room (the drone flies over it)
        Mover("windmill", "box", (1.6, 0.08, 0.45), "#ef476f",
              lambda t: (0.0, 0.0, 0.7 * t)),
        # slider in the south corridor
        Mover("slider", "box", (0.35, 0.35, 0.4), "#06d6a0",
              lambda t: (-1.0 + 5.0 * math.sin(0.35 * t), -10.2, 0.0)),
    ]


# --- vehicles: spawn + routes (shared by controllers and viewer) -------------------------------

ROVER_START = (-2.0, -8.0, 0.0)
ROVER_ROUTE = [(6.0, -8.0), (6.5, -1.0), (6.5, 6.0), (1.0, 8.6), (-6.0, 8.6), (-6.0, 0.0),
               (-6.0, -8.0), (-2.0, -8.0)]

DRONE_START = (0.0, -6.5, 0.15)
DRONE_ROUTE = [(0.0, -6.5, 1.5), (0.0, -2.5, 1.6), (0.0, 0.0, 1.8), (2.2, 0.0, 1.6), (6.0, 0.0, 1.4),
               (8.0, 0.0, 1.3), (11.0, 0.0, 1.4), (11.0, 5.0, 1.7), (11.0, 9.5, 1.7), (4.0, 10.0, 1.8),
               (-4.0, 10.0, 1.9), (-10.5, 9.0, 1.9), (-11.0, 2.0, 1.8), (-11.0, -6.0, 1.9),
               (-5.0, -10.0, 1.7), (2.0, -9.0, 1.6), (0.0, -8.0, 1.6), (0.0, -6.0, 1.6), (0.0, -3.0, 1.6), (0.0, 0.0, 1.7),
               (0.0, 2.5, 1.6), (0.0, 6.0, 1.7), (-6.0, 6.5, 1.7)]


def mjcf(statics: list[Static], movers: list[Mover]) -> str:
    def geom_static(i, s: Static):
        rgba = _rgba(s.color)
        if s.kind == "box":
            return (f'<geom name="s{i}" type="box" pos="{s.pos[0]} {s.pos[1]} {s.pos[2]}" '
                    f'size="{s.size[0]} {s.size[1]} {s.size[2]}" rgba="{rgba}"/>')
        return (f'<geom name="s{i}" type="cylinder" pos="{s.pos[0]} {s.pos[1]} {s.pos[2]}" '
                f'size="{s.size[0]} {s.size[1]}" rgba="{rgba}"/>')

    sgeoms = "\n    ".join(geom_static(i, s) for i, s in enumerate(statics))
    mbodies = []
    for m in movers:
        x, y, yaw = m.pose(0.0)
        gs = (f'<geom type="box" size="{m.size[0]} {m.size[1]} {m.size[2]}" rgba="{_rgba(m.color)}"/>'
              if m.kind == "box" else f'<geom type="cylinder" size="{m.size[0]} {m.size[1]}" rgba="{_rgba(m.color)}"/>')
        mbodies.append(f'<body name="{m.name}" mocap="true" pos="{x} {y} {m.z}">{gs}</body>')
    mb = "\n    ".join(mbodies)
    rx, ry, ryaw = ROVER_START
    dx, dy, dz = DRONE_START
    return f"""
<mujoco model="slam_world">
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <default><geom contype="1" conaffinity="1" friction="0.8 0.01 0.01"/></default>
  <worldbody>
    <geom name="floor" type="plane" size="{ARENA} {ARENA} 0.1" rgba="0.1 0.12 0.16 1"/>
    {sgeoms}
    {mb}
    <body name="rover" pos="0 0 0">
      <joint name="rx" type="slide" axis="1 0 0" damping="2"/>
      <joint name="ry" type="slide" axis="0 1 0" damping="2"/>
      <joint name="rz" type="hinge" axis="0 0 1" damping="1"/>
      <geom type="box" size="0.30 0.20 0.07" pos="0 0 0.16" mass="8" contype="2" conaffinity="1" rgba="0.2 0.8 1 1"/>
      <geom type="cylinder" size="0.06 0.05" pos="0.05 0 0.28" mass="0.5" contype="2" conaffinity="1" rgba="0.9 0.9 1 1"/>
      <geom type="cylinder" euler="90 0 0" size="0.1 0.03" pos=" 0.18  0.22 0.1" mass="0.2" contype="0" conaffinity="0"/>
      <geom type="cylinder" euler="90 0 0" size="0.1 0.03" pos=" 0.18 -0.22 0.1" mass="0.2" contype="0" conaffinity="0"/>
      <geom type="cylinder" euler="90 0 0" size="0.1 0.03" pos="-0.18  0.22 0.1" mass="0.2" contype="0" conaffinity="0"/>
      <geom type="cylinder" euler="90 0 0" size="0.1 0.03" pos="-0.18 -0.22 0.1" mass="0.2" contype="0" conaffinity="0"/>
    </body>
    <body name="drone" pos="{dx} {dy} {dz}">
      <freejoint name="drone_free"/>
      <inertial pos="0 0 0" mass="1.2" diaginertia="0.012 0.012 0.022"/>
      <geom type="sphere" size="0.14" contype="4" conaffinity="1" mass="0" rgba="1 0.5 0.2 1"/>
    </body>
  </worldbody>
  <actuator>
    <velocity name="vx" joint="rx" kv="300" forcerange="-250 250" ctrlrange="-3 3"/>
    <velocity name="vy" joint="ry" kv="300" forcerange="-250 250" ctrlrange="-3 3"/>
    <velocity name="wz" joint="rz" kv="60" forcerange="-80 80" ctrlrange="-4 4"/>
  </actuator>
</mujoco>
"""


def _rgba(hexcolor: str) -> str:
    h = hexcolor.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return f"{r:.3f} {g:.3f} {b:.3f} 1"
