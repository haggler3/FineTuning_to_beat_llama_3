"""Generates maps/foundry.json -- "The Foundry": a custom 28 x 28 m industrial hall.

Edit this script (or the JSON it writes) to build your own map. Run:  python maps/make_foundry.py
Map schema (all lengths in metres, z up):
  statics: {"kind":"box","pos":[x,y,z],"size":[hx,hy,hz],"yaw":deg,"color":hex}
           {"kind":"cyl","pos":[x,y,z],"size":[r,half_height],"color":hex}
           {"kind":"wall","p1":[x,y],"p2":[x,y],"h":height,"th":half_thickness,"color":hex}   (any angle)
  movers : {"name","kind":"box|cyl","size":[...],"color",
            "motion":{"type":"line","c":[x,y],"dir":deg,"amp":m,"w":rad/s,"phase":rad}
                    |{"type":"circle","c":[x,y],"r":m,"w":rad/s,"phase":rad}
                    |{"type":"spin","c":[x,y],"w":rad/s}}
  rover  : {"start":[x,y,yaw],"route":[[x,y],...]}   (route repeats)
  drone  : {"start":[x,y,z],"route":[[x,y,z],...]}
"""
import json
import math
from pathlib import Path

A = 14.0
H = 3.2
S, M, Z = [], [], []


def wall(p1, p2, h=H, th=0.12, color="#2c3446"):
    S.append(dict(kind="wall", p1=list(map(float, p1)), p2=list(map(float, p2)), h=h, th=th, color=color))


def box(x, y, hx, hy, hz, yaw=0.0, color="#7a5c3a"):
    S.append(dict(kind="box", pos=[x, y, hz], size=[hx, hy, hz], yaw=yaw, color=color))


def cyl(x, y, r, hh, color="#5d6b86"):
    S.append(dict(kind="cyl", pos=[x, y, hh], size=[r, hh], color=color))


# --- shell: outer walls and chamfered corners -------------------------------------------------
for p1, p2 in [((-A, -A), (A, -A)), ((-A, A), (A, A)), ((-A, -A), (-A, A)), ((A, -A), (A, A))]:
    wall(p1, p2)
for sx in (-1, 1):
    for sy in (-1, 1):
        wall((sx * A, sy * (A - 2.2)), (sx * (A - 2.2), sy * A), color="#3a4560")

# --- the rotunda: an octagon with doorways on the four cardinal faces -------------------------
R, DOOR = 5.2, 1.2
V = [(R * math.cos(math.radians(22.5 + 45 * k)), R * math.sin(math.radians(22.5 + 45 * k))) for k in range(8)]
for k in range(8):
    a, b = V[k], V[(k + 1) % 8]
    mid_angle = 45 * (k + 1)
    if mid_angle % 90 == 0:  # cardinal face: leave a doorway
        m = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        d = ((b[0] - a[0]), (b[1] - a[1]))
        L = math.hypot(*d)
        u = (d[0] / L, d[1] / L)
        wall(a, (m[0] - u[0] * DOOR, m[1] - u[1] * DOOR), h=3.0, color="#34405a")
        wall((m[0] + u[0] * DOOR, m[1] + u[1] * DOOR), b, h=3.0, color="#34405a")
    else:
        wall(a, b, h=3.0, color="#34405a")

# --- west: shelving aisles (the drone can fly down the aisle at 1.3 m or over it at 2.5 m) ------
for x in (-12.6, -9.8):
    box(x, -6.0, 0.4, 4.0, 0.9, color="#9a6a3a")
    box(x, 6.0, 0.4, 4.0, 0.9, color="#9a6a3a")
for (x, y, w, h) in [(-12.7, -12.2, 0.6, 0.4), (-8.7, 12.4, 0.5, 0.5), (-13.0, 0.4, 0.5, 0.45)]:
    box(x, y, w, w, h, color="#6b5a45")

# --- east: container yard ---------------------------------------------------------------------
for (x, y, c) in [(10.5, -10.5, "#b3473a"), (10.5, -6.0, "#2f6f8f"), (10.5, 6.0, "#3c8f6a"), (10.5, 10.5, "#c98a2b")]:
    box(x, y, 3.0, 1.2, 1.3, color=c)
box(11.4, -0.2, 1.8, 1.0, 1.3, yaw=20.0, color="#7b4fa0")
# clutter in the two container lanes (the drone flies over it at 1.4 m). Featureless parallel walls are a classic
# degenerate case for LiDAR odometry: without these the front-end can slide along the lane.
for (x, y, w, h, yaw) in [(8.6, -8.85, 0.4, 0.35, 15.0), (11.0, -7.65, 0.35, 0.3, 0.0), (9.9, -8.7, 0.25, 0.25, 40.0),
                          (8.8, 8.85, 0.4, 0.35, -10.0), (11.2, 7.65, 0.35, 0.3, 25.0), (10.0, 8.7, 0.25, 0.25, 0.0)]:
    box(x, y, w, w, h, yaw=yaw, color="#8a6a40")

# --- south: gantry crane the drone flies under -----------------------------------------------
box(0.0, -10.6, 4.2, 1.3, 0.06, color="#d0663a")
S[-1]["pos"][2] = 2.7
for px in (-3.8, 3.8):
    for py in (-9.5, -11.7):
        cyl(px, py, 0.15, 1.3, "#d0663a")

# --- north: irregular pillar row (irregular on purpose: repeated structure fools loop closure) --
for x in (-4.5, -0.8, 2.9):
    cyl(x, 10.6, 0.4, 1.6, "#6b7a99")

# --- moving things -----------------------------------------------------------------------------
M.append(dict(name="forklift", kind="box", size=[0.6, 0.4, 0.5], color="#ff7a59",
              motion=dict(type="line", c=[8.4, 0.0], dir=90, amp=3.6, w=0.55, phase=0.4)))
M.append(dict(name="walker", kind="cyl", size=[0.28, 0.5], color="#ffd166",
              motion=dict(type="circle", c=[-7.2, -0.5], r=2.0, w=0.5, phase=0.0)))
M.append(dict(name="windmill", kind="box", size=[1.7, 0.08, 0.45], color="#ef476f",
              motion=dict(type="spin", c=[0.0, 0.0], w=0.7)))
M.append(dict(name="sweeper", kind="box", size=[0.35, 0.35, 0.4], color="#06d6a0",
              motion=dict(type="line", c=[-2.0, 7.7], dir=90, amp=1.7, w=0.45, phase=1.0)))

# --- zone labels for the viewer ---------------------------------------------------------------
Z += [dict(text="ROTUNDA", pos=[0, 0, 3.7]), dict(text="SHELVING", pos=[-11.2, 0, 3.0]),
      dict(text="CONTAINER YARD", pos=[11.0, 0, 3.0]), dict(text="GANTRY", pos=[0, -10.6, 3.5]),
      dict(text="PILLAR ROW", pos=[-1, 10.6, 3.7])]

ring = [(6.2, -7.5), (6.2, 0.0), (6.2, 7.6), (0.0, 7.6), (-7.2, 7.6), (-7.2, 0.0), (-7.2, -7.5), (-2.0, -7.5)]
excursion = [(-7.2, -7.5), (-7.2, -1.0), (-11.2, -1.0), (-11.2, -11.3), (-7.2, -11.3), (-7.2, -7.5), (-2.0, -7.5)]
rover = dict(start=[-2.0, -7.5, 0.0], route=[list(p) for p in ring + excursion])

drone_route = [(0, -7.5, 1.6), (0, -2, 1.7), (0, 0, 1.9), (0, 2, 1.7), (0, 7.6, 2.0), (-7.2, 7.6, 2.3),
               (-7.2, 11.0, 2.3), (-11.2, 11.0, 1.5), (-11.2, 6, 1.3), (-11.2, 0, 1.3), (-11.2, -6, 1.3),
               (-11.2, -11.3, 1.5), (-6, -10.6, 1.6), (6.2, -10.6, 1.6), (6.2, -8.25, 1.4), (12.6, -8.25, 1.4),
               (6.2, -8.25, 1.4), (6.2, 0, 1.7), (6.2, 8.25, 1.7), (12.6, 8.25, 1.4), (6.2, 8.25, 1.5),
               (0, 7.6, 2.0), (0, 2, 1.7), (0, 0, 1.9), (0, -2, 1.7), (0, -7.5, 1.6)]
drone = dict(start=[0.0, -7.5, 0.15], route=[list(p) for p in drone_route])

out = Path(__file__).with_name("foundry.json")
out.write_text(json.dumps(dict(name="The Foundry", arena=A, statics=S, movers=M, zones=Z, rover=rover, drone=drone), indent=1))
print(f"wrote {out} ({len(S)} statics, {len(M)} movers)")
