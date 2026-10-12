"""World description loaded from a JSON map; one description feeds both the MuJoCo model and the viewer.

See ``maps/make_foundry.py`` for the schema and the default map.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MAP = Path(__file__).resolve().parents[1] / "maps" / "foundry.json"


@dataclass
class Static:
    kind: str  # "box" | "cyl"
    pos: tuple
    size: tuple  # box: half extents; cyl: (radius, half-height)
    yaw: float = 0.0  # degrees (boxes only)
    color: str = "#3a4256"


@dataclass
class Mover:
    """A kinematic (mocap) obstacle whose pose is a closed-form function of time."""
    name: str
    kind: str
    size: tuple
    color: str
    motion: dict = field(repr=False)

    def pose(self, t: float):
        """Return (x, y, yaw) at time t; z is size-derived."""
        m = self.motion
        kind = m["type"]
        if kind == "line":
            d = math.radians(m["dir"])
            s = m["amp"] * math.sin(m["w"] * t + m.get("phase", 0.0))
            return (m["c"][0] + s * math.cos(d), m["c"][1] + s * math.sin(d), d)
        if kind == "circle":
            a = m["w"] * t + m.get("phase", 0.0)
            return (m["c"][0] + m["r"] * math.cos(a), m["c"][1] + m["r"] * math.sin(a), 0.0)
        if kind == "spin":
            return (m["c"][0], m["c"][1], m["w"] * t)
        raise ValueError(f"unknown motion type {kind!r}")

    @property
    def z(self) -> float:
        return self.size[2] if self.kind == "box" else self.size[1]


@dataclass
class World:
    name: str
    arena: float
    statics: list
    movers: list
    zones: list
    rover_start: tuple
    rover_route: list
    drone_start: tuple
    drone_route: list


def load_world(path: str | Path | None = None) -> World:
    d = json.loads(Path(path or DEFAULT_MAP).read_text())
    statics = []
    for s in d["statics"]:
        if s["kind"] == "wall":  # arbitrary-angle wall -> rotated box
            (x1, y1), (x2, y2) = s["p1"], s["p2"]
            length = math.hypot(x2 - x1, y2 - y1)
            statics.append(Static("box", ((x1 + x2) / 2, (y1 + y2) / 2, s["h"] / 2),
                                  (length / 2 + s["th"], s["th"], s["h"] / 2),
                                  math.degrees(math.atan2(y2 - y1, x2 - x1)), s.get("color", "#2c3446")))
        else:
            statics.append(Static(s["kind"], tuple(s["pos"]), tuple(s["size"]), s.get("yaw", 0.0), s.get("color", "#3a4256")))
    movers = [Mover(m["name"], m["kind"], tuple(m["size"]), m["color"], m["motion"]) for m in d["movers"]]
    return World(d.get("name", "map"), float(d["arena"]), statics, movers, d.get("zones", []),
                 tuple(d["rover"]["start"]), [tuple(p) for p in d["rover"]["route"]],
                 tuple(d["drone"]["start"]), [tuple(p) for p in d["drone"]["route"]])


def _rgba(hexcolor: str) -> str:
    h = hexcolor.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return f"{r:.3f} {g:.3f} {b:.3f} 1"


def mjcf(w: World) -> str:
    geoms = []
    for i, s in enumerate(w.statics):
        if s.kind == "box":
            geoms.append(f'<geom name="s{i}" type="box" pos="{s.pos[0]} {s.pos[1]} {s.pos[2]}" euler="0 0 {s.yaw}" '
                         f'size="{s.size[0]} {s.size[1]} {s.size[2]}" rgba="{_rgba(s.color)}"/>')
        else:
            geoms.append(f'<geom name="s{i}" type="cylinder" pos="{s.pos[0]} {s.pos[1]} {s.pos[2]}" '
                         f'size="{s.size[0]} {s.size[1]}" rgba="{_rgba(s.color)}"/>')
    bodies = []
    for m in w.movers:
        x, y, _ = m.pose(0.0)
        gs = (f'<geom type="box" size="{m.size[0]} {m.size[1]} {m.size[2]}" rgba="{_rgba(m.color)}"/>' if m.kind == "box"
              else f'<geom type="cylinder" size="{m.size[0]} {m.size[1]}" rgba="{_rgba(m.color)}"/>')
        bodies.append(f'<body name="{m.name}" mocap="true" pos="{x} {y} {m.z}">{gs}</body>')
    dx, dy, dz = w.drone_start
    sg, mb = "\n    ".join(geoms), "\n    ".join(bodies)
    return f"""
<mujoco model="slam_world">
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <default><geom contype="1" conaffinity="1" friction="0.8 0.01 0.01"/></default>
  <worldbody>
    <geom name="floor" type="plane" size="{w.arena} {w.arena} 0.1" rgba="0.1 0.12 0.16 1"/>
    {sg}
    {mb}
    <body name="rover" pos="0 0 0">
      <joint name="rx" type="slide" axis="1 0 0" damping="2"/>
      <joint name="ry" type="slide" axis="0 1 0" damping="2"/>
      <joint name="rz" type="hinge" axis="0 0 1" damping="1"/>
      <geom type="box" size="0.30 0.20 0.07" pos="0 0 0.16" mass="8" contype="2" conaffinity="1" rgba="0.2 0.8 1 1"/>
      <geom type="cylinder" size="0.06 0.05" pos="0.05 0 0.28" mass="0.5" contype="2" conaffinity="1" rgba="0.9 0.9 1 1"/>
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
