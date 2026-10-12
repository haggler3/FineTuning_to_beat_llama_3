"""Render the viewer to an MP4: scripted camera shots + captions, frame-exact, headless Chromium + ffmpeg.

    python tools/make_video.py --out docs/slam_lab.mp4            # ~100 s film
    python tools/make_video.py --out /tmp/test.mp4 --only 5-8     # preview shots 5..8 only

Needs:  pip install playwright  (+ a Chromium; set CHROMIUM=/path/to/chrome if playwright's own isn't installed)
        ffmpeg on PATH.  Software WebGL (SwiftShader) is used, so no GPU is required -- but it is slow.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import math
import os
import shutil
import subprocess
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / "viewer"


def smooth(u):
    return u * u * (3 - 2 * u)


def lerp(a, b, u):
    return a + (b - a) * u


def vlerp(a, b, u):
    return [lerp(x, y, u) for x, y in zip(a, b)]


def build_shots(D):
    """Shots are (name, duration s, frame range, camera fn(u, ft) -> (pos, target), captions, layers, title card)."""
    N = D["meta"]["n"]
    r1 = D["rover"]["graph"][0]["f"]
    d1 = D["drone"]["graph"][0]["f"]
    rebuilt = [e["f"] for e in D["rover"]["graph"] if e["rebuilt"] and e["f"] < N - 2]
    m = D["metrics"]
    rg, dg = D["rover"]["gt"], D["drone"]["gt"]

    def pos_at(arr, ft):
        i = max(0, min(len(arr) - 1, int(ft)))
        return arr[i]

    def heading(arr, ft, w=6):
        a, b = pos_at(arr, ft - w), pos_at(arr, ft + w)
        return math.atan2(b[1] - a[1], b[0] - a[0])

    def orbit(c, r, h, a0, a1, tz=0.8):
        def cam(u, ft):
            a = lerp(a0, a1, u)
            return [c[0] + r * math.cos(a), c[1] + r * math.sin(a), h], [c[0], c[1], tz]
        return cam

    def chase(arr, back, up, look=2.0, zlook=None, side=0.0):
        def cam(u, ft):
            p = pos_at(arr, ft)
            hd = heading(arr, ft)
            z = p[2] if len(p) > 3 else 0.0
            pos = [p[0] - back * math.cos(hd) - side * math.sin(hd), p[1] - back * math.sin(hd) + side * math.cos(hd), z + up]
            tgt = [p[0] + look * math.cos(hd), p[1] + look * math.sin(hd), z if zlook is None else zlook]
            return pos, tgt
        return cam

    def top(c, h, tilt=0.5):
        return lambda u, ft: ([c[0], c[1] - tilt, h], [c[0], c[1], 0.0])

    def follow_top(arr, h, tilt=2.0):
        def cam(u, ft):
            p = pos_at(arr, ft)
            return [p[0], p[1] - tilt, h], [p[0], p[1], 0.0]
        return cam

    L_ALL = dict(world=True, rmap=True, dmap=True, graph=True, rscan=True, dscan=True, parts=True, trails=True,
                 fe=False, dr=False, opt=False, ghost=True, zones=True)
    shots = [
        dict(name="title", dur=5.0, f=(0, 30), cam=orbit((0, 0), 42, 24, -2.5, -2.1), layers=dict(L_ALL, graph=False),
             title=("SLAM LAB", "Graph SLAM with a MuJoCo rover and drone in a dynamic, custom world")),
        dict(name="world", dur=12.0, f=(30, 170), cam=orbit((0, 0), 36, 16, -2.1, -1.2),
             caps=[(0, "The Foundry", "A custom 28 × 28 m hall: rotunda, shelving aisles, container yard, gantry — and four moving obstacles")]),
        dict(name="rover", dur=9.0, f=(170, 280), cam=chase(rg, 6.0, 3.6), layers=dict(L_ALL, dr=True, graph=False, zones=False),
             caps=[(0, "Rover: 2D LiDAR → scan matching on an occupancy grid", "Wheel odometry alone (red) wanders off by metres")]),
        dict(name="drone", dur=9.0, f=(280, 380), cam=chase(dg, 7.0, 3.6, look=3.0), layers=dict(L_ALL, dr=True, graph=False, zones=False),
             caps=[(0, "Drone: 3D LiDAR → ICP against a local voxel map", "Moving obstacles never make it into the map: voxels need repeat hits and are cleared when rays pass through")]),
        dict(name="drift", dur=9.0, f=(380, r1 - 45), cam=top((0, 0), 35), layers=dict(L_ALL, dr=True, fe=True),
             caps=[(0, "Front-end odometry is accurate for seconds…", "…but small errors accumulate: this is drift (purple) against ground truth (white)")]),
        dict(name="loop_rover", dur=14.0, f=(r1 - 45, r1 + 55), cam=orbit((-1, -4), 20, 12, -2.0, -1.2), layers=dict(L_ALL, fe=True, graph=True, zones=False),
             caps=[(0, "Back at the start…", "Every 0.4 s the rover drops a keyframe (violet nodes) joined by odometry edges"),
                   (r1 - 6, "Loop closure!", "ICP between today's LiDAR scan and an old one gives a constraint that links the present to the past"),
                   (r1 + 18, "Pose-graph optimisation", "The whole trajectory is pulled taut to satisfy every constraint at once")]),
        dict(name="map_snap", dur=10.0, f=(r1 + 55, d1 - 70), cam=follow_top(rg, 34, 3.0), layers=dict(L_ALL, fe=True),
             caps=[(0, "The map snaps into place", "The global occupancy grid is rebuilt from the corrected poses — walls that were doubled become one"),
                   ] + [(f - 2, "Map rebuilt", "re-integrating every scan with the optimised poses") for f in rebuilt if r1 + 55 < f < d1 - 70][:1]),
        dict(name="loop_drone", dur=14.0, f=(d1 - 70, d1 + 90), cam=chase(dg, 9.0, 6.0, look=2.0), layers=dict(L_ALL, fe=True, zones=False),
             caps=[(0, "The drone returns to the rotunda…", "3D pose graph: keyframe nodes in violet"),
                   (d1 - 5, "Loop closure — in 3D", "Multi-start ICP against a local sub-map verifies the revisit; the SE(3) graph is re-optimised and the voxel map rebuilt")]),
        dict(name="final", dur=14.0, f=(d1 + 90, N - 1), cam=orbit((0, 0), 38, 20, -1.2, -0.2), layers=dict(L_ALL, dr=True, opt=True, fe=False),
             caps=[(0, "Optimised trajectories (green) vs raw odometry (red)", "Every pose in the history is refined — not just the latest one"),
                   (N - 90, f"Rover {m['rover_odom_rmse']:.1f} m → {m['rover_online_rmse']:.2f} m  ·  Drone {m['drone_odom_rmse']:.1f} m → {m['drone_online_rmse']:.2f} m",
                    "position RMSE: dead-reckoning → graph SLAM (online)")]),
        dict(name="end", dur=4.0, f=(N - 1, N - 1), cam=orbit((0, 0), 38, 20, -0.2, 0.1), layers=dict(L_ALL, dr=False, opt=False),
             title=("SLAM LAB", f"rover {m['rover_odom_rmse']:.1f} → {m['rover_opt_rmse']:.2f} m   ·   drone {m['drone_odom_rmse']:.1f} → {m['drone_opt_rmse']:.2f} m   (RMSE after graph optimisation)")),
    ]
    return shots


def frames(shots, fps):
    """Yield per-video-frame state dicts."""
    prev_cam = None
    last_layers = None
    for si, s in enumerate(shots):
        n = max(1, round(s["dur"] * fps))
        for k in range(n):
            u = k / max(1, n - 1)
            ft = lerp(s["f"][0], s["f"][1], u)
            pos, tgt = s["cam"](u, ft)
            if prev_cam is not None and k < 1.5 * fps:  # glide in from the previous shot's camera
                b = smooth(k / (1.5 * fps))
                pos, tgt = vlerp(prev_cam[0], pos, b), vlerp(prev_cam[1], tgt, b)
            caption = None
            for (cf, title, sub) in s.get("caps", []):
                if ft >= cf if cf else True:
                    caption = (title, sub)
            title = s.get("title")
            if title and "caps" not in s and s["name"] == "title" and k > 0.8 * n:
                title = None
            layers = s.get("layers") if s.get("layers") != last_layers else None
            last_layers = s.get("layers")
            yield dict(shot=si, name=s["name"], k=k, n=n, ft=ft, pos=pos, tgt=tgt, caption=caption, title=title, layers=layers)
            last = (pos, tgt)
        prev_cam = last


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "slam_lab.mp4")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--only", default=None, help="shot range to render, e.g. 5-8 (1-based) for previews")
    ap.add_argument("--frames-dir", type=Path, default=None, help="keep rendered JPEGs here (default: temp dir)")
    ap.add_argument("--data", type=Path, default=VIEWER / "data" / "run.json")
    ap.add_argument("--preview", action="store_true", help="render only two frames per shot (framing check)")
    ap.add_argument("--ftwin", default=None, help="render only video frames whose sim frame is in A,B (consecutive)")
    ap.add_argument("--limit", type=int, default=None, help="render only the first N video frames (timing tests)")
    a = ap.parse_args()

    from playwright.sync_api import sync_playwright

    D = json.loads(a.data.read_text())
    shots = build_shots(D)
    if a.only:
        lo, hi = (int(x) for x in a.only.split("-"))
        shots = shots[lo - 1:hi]
    states = list(frames(shots, a.fps))
    if a.preview:
        states = [st for st in states if st["k"] in (st["n"] // 4, 3 * st["n"] // 4)]
    if a.ftwin:
        lo, hi = (float(x) for x in a.ftwin.split(","))
        states = [st for st in states if lo <= st["ft"] <= hi]
    if a.limit:
        states = states[:a.limit]
    print(f"{len(shots)} shots, {len(states)} frames @ {a.fps} fps = {len(states) / a.fps:.1f} s")

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    handler = functools.partial(Quiet, directory=str(VIEWER))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/index.html?video"

    fdir = a.frames_dir or (ROOT / ".video_frames")
    if fdir.exists():
        shutil.rmtree(fdir)
    fdir.mkdir(parents=True)
    exe = os.environ.get("CHROMIUM")
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=exe, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                                                                "--ignore-gpu-blocklist", "--enable-webgl"])
        page = browser.new_page(viewport={"width": a.width, "height": a.height})
        logs = []
        page.on("pageerror", lambda e: logs.append(str(e)))
        page.goto(url)
        page.wait_for_function("window.__slam && window.__slam.ready", timeout=120000)
        dt = 1.0 / a.fps
        for i, st in enumerate(states):
            if st["layers"]:
                for name, on in st["layers"].items():
                    page.evaluate("([n,o]) => __slam.setLayer(n,o)", [name, on])
            c = st["caption"]
            page.evaluate("([t,s]) => __slam.caption(t,s)", [c[0], c[1]] if c else ["", ""])
            t = st["title"]
            page.evaluate("([t,s]) => __slam.titleCard(t,s)", [t[0], t[1]] if t else ["", ""])
            page.evaluate("([ft,p,t,dt]) => __slam.renderAt(ft,p,t,dt)", [st["ft"], st["pos"], st["tgt"], dt])
            page.screenshot(path=str(fdir / f"{i:05d}.jpg"), type="jpeg", quality=93)
            if i % 50 == 0:
                print(f"  frame {i}/{len(states)}  shot '{st['name']}'", flush=True)
        browser.close()
    if logs:
        print("page errors:", logs[:5])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", str(fdir / "%05d.jpg"),
                    "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(a.out)], check=True)
    if not a.frames_dir:
        shutil.rmtree(fdir)
    print(f"wrote {a.out} ({a.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
