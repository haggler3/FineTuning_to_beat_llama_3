# SLAM Lab

**Graph SLAM** for a MuJoCo ground rover and quadrotor exploring a custom, dynamic world — with an interactive
Three.js viewer and a scripted film of the whole run.

![preview](docs/preview.png)

▶ **Film:** [`docs/slam_lab.mp4`](docs/slam_lab.mp4) — 100 s, 1280×720, 27 MB (rendered by `tools/make_video.py`)

Both robots fly/drive through **The Foundry**, a 28 × 28 m industrial hall (octagonal rotunda, shelving aisles,
container yard with cluttered lanes, a gantry crane, a pillar row) shared with four moving obstacles. Each builds its
own map from simulated LiDAR and noisy odometry; the viewer shows the maps, the pose graphs, every loop closure, and
how the error changes when the back-end optimises.

## How the SLAM works

| | Rover (2D, SE(2)) | Drone (3D, SE(3)) |
|---|---|---|
| **Sensors** | 360° LiDAR (`mj_multiRay`) + wheel odometry (3.5 % scale error, yaw bias, noise) | 96×9-beam 3D LiDAR + body velocity / gyro (4 % scale error, biases, noise) |
| **Front-end** | particle filter + correlative scan matching against a *local, decaying* occupancy sub-map | point-to-plane ICP against a *local* voxel map (voxels seen in the last 12 s) |
| **Graph** | keyframe every 0.4 s, odometry edges | keyframe every 0.3 s, odometry edges |
| **Loop closure** | revisit candidates by distance in the current graph + ≥ 15 m of travel; multi-start 2D ICP between the two scans; accepted on inlier ratio, mutual overlap, RMS | revisit candidates by distance + ≥ 20 m of travel; multi-start 3D ICP (x/y/z/yaw offsets) against a sub-map around the old keyframe |
| **Back-end** | sparse Levenberg–Marquardt on the pose graph (Huber loss) | same, with SE(3) and a local rotation parametrisation |
| **Map** | log-odds occupancy grid, **rebuilt from the optimised poses** after a big correction | voxel map (voxels need repeat hits, are cleared when later rays pass through), rebuilt likewise |

The front-end is deliberately a *local* odometry: it is accurate over seconds and drifts over minutes. That is what
makes the graph back-end matter — when the rover returns to a place it has seen (~60 s in), ICP between the two scans
adds a constraint, the whole trajectory is pulled taut at once, and the map visibly snaps shut. The solver
(`slamviz/posegraph.py`) is implemented from scratch; its Jacobians are numerical, so it is slow-ish but short.

## The mathematics

Notation: $\mathbf{x}_i$ is the pose of keyframe $i$ (SE(2): $(x,y,\theta)$; SE(3): rotation $R_i\in SO(3)$ and translation $\mathbf t_i\in\mathbb R^3$),
$\operatorname{wrap}(a)=a-2\pi\lfloor (a+\pi)/2\pi\rfloor$, and $\operatorname{Exp},\operatorname{Log}$ are the exponential / logarithm maps between rotation
vectors and rotations.

### 1. The pose graph (back-end)

Each edge $(i,j)$ carries a measurement $\mathbf z_{ij}$ of the pose of node $j$ *in the frame of node $i$*:
odometry edges (consecutive keyframes) take $\mathbf z_{ij}$ from the front-end, loop-closure edges from scan matching.
With per-component standard deviations $\boldsymbol\sigma_{ij}$ the whitened residuals are

$$
\mathbf r_{ij}^{\text{SE(2)}}=\begin{bmatrix}
R(\theta_i)^{\top}(\mathbf t_j-\mathbf t_i)-\mathbf z^{t}_{ij}\\[2pt]
\operatorname{wrap}\!\big(\theta_j-\theta_i-z^{\theta}_{ij}\big)\end{bmatrix}\oslash\boldsymbol\sigma_{ij},
\qquad
\mathbf r_{ij}^{\text{SE(3)}}=\begin{bmatrix}
R_i^{\top}(\mathbf t_j-\mathbf t_i)-\mathbf t_{ij}\\[2pt]
\operatorname{Log}\!\big(R_{ij}^{\top}R_i^{\top}R_j\big)\end{bmatrix}\oslash\boldsymbol\sigma_{ij},
$$

and the maximum-a-posteriori trajectory solves the robust nonlinear least-squares problem

$$
\mathbf X^{\star}=\arg\min_{\mathbf X}\sum_{(i,j)\in\mathcal E}\rho_H\!\big(\lVert\mathbf r_{ij}(\mathbf X)\rVert\big),
\qquad
\rho_H(u)=\begin{cases}\tfrac12u^2 & u\le k\\ k\,(u-\tfrac12k) & u>k\end{cases}\quad(k=3),
$$

with node 0 held fixed (gauge freedom). The Huber loss $\rho_H$ limits the damage a wrong loop closure can do.
Each iteration linearises around the current estimate, $\mathbf r_{ij}(\mathbf X\boxplus\boldsymbol\delta)\approx\mathbf r_{ij}+J_i\boldsymbol\delta_i+J_j\boldsymbol\delta_j$,
and solves the damped, IRLS-weighted normal equations (Levenberg–Marquardt)

$$
\Big(\sum_{(i,j)} w_{ij}\,J_{ij}^{\top}J_{ij}+\lambda\operatorname{diag}(\cdot)\Big)\boldsymbol\delta=-\sum_{(i,j)} w_{ij}\,J_{ij}^{\top}\mathbf r_{ij},
\qquad
w_{ij}=\min\!\Big(1,\frac{k}{\lVert\mathbf r_{ij}\rVert}\Big),
$$

a sparse system (each edge touches two nodes) solved with `scipy.sparse.linalg.spsolve`. $\lambda$ is divided by 5 after a successful step
and multiplied by 10 after a rejected one. Updates use the retraction $t\leftarrow t+\delta_t$, $R\leftarrow R\operatorname{Exp}(\boldsymbol\delta_\omega)$, so there is no
angle-wrapping singularity. Jacobians are formed by forward differences ($\varepsilon=10^{-6}$) per edge, vectorised over all edges.

### 2. Rover front-end (2D)

**Occupancy grid.** Every cell holds a log-odds value $l=\log\frac{p}{1-p}$. A ray from the sensor to a return updates the cells it crosses
(free, $\Delta l=-0.45$) and the end cell (occupied, net $\Delta l=+0.85$), clamped to $[-4,5]$; occupancy is recovered as $p=\sigma(l)=1/(1+e^{-l})$.
The *local* sub-map used for odometry additionally decays, $l\leftarrow 0.985\,l$ per scan, so it forgets after $\approx 10$ s — that is what makes the
front-end an odometry that drifts, rather than a SLAM system that cannot.

**Likelihood field.** With $D(\mathbf q)$ the Euclidean distance from $\mathbf q$ to the nearest occupied cell (an exact distance transform), the log-likelihood of a pose
$\mathbf x=(x,y,\theta)$ given beam ranges $r_b$ at body angles $\alpha_b$ is

$$
\log\mathcal L(\mathbf x)=c\sum_b -\frac{\min\!\big(D(\mathbf q_b(\mathbf x)),d_{\max}\big)^2}{2\sigma_s^2},
\qquad
\mathbf q_b=\begin{bmatrix}x+r_b\cos(\theta+\alpha_b)\\ y+r_b\sin(\theta+\alpha_b)\end{bmatrix},
$$

($\sigma_s=0.15$ m, $d_{\max}=1$ m, $c=0.35$ tempers over-confidence from correlated beams).

**Particle filter.** Particles are propagated by the odometry increment $(d,\Delta\theta)$ plus noise
$\sigma_d=0.015+0.08|d|$, $\sigma_\theta=0.008+0.1|\Delta\theta|$, weighted $w^{(k)}\propto\exp\log\mathcal L(\mathbf x^{(k)})$ and resampled
(systematic) when $N_{\text{eff}}=1/\sum_k (w^{(k)})^2<N/2$.

**Correlative scan matching.** The weighted mean is refined by exhaustive search over a window ($\pm0.15$ m in $x,y$; $\pm0.06$ rad in $\theta$) maximising

$$
\log\mathcal L(\mathbf x)\;-\;\tfrac12\Big(\frac{\lVert\mathbf t-\hat{\mathbf t}\rVert^2}{0.05^2}+\frac{\operatorname{wrap}(\theta-\hat\theta)^2}{0.02^2}\Big),
$$

where $(\hat{\mathbf t},\hat\theta)$ is the pure-odometry prediction. The Gaussian prior is what stops the match sliding along featureless walls.

### 3. Drone front-end (3D)

**Point-to-plane ICP.** Voxel centres $\mathbf q_i$ of the local map carry a surface normal $\mathbf n_i$: the eigenvector of the smallest eigenvalue of the
covariance of the 10 nearest neighbours, kept only if the neighbourhood is planar ($\lambda_0<0.25\lambda_1$). With scan points $\mathbf p_i$ and the unknown
increment $(R,\mathbf t)\approx(I+[\boldsymbol\omega]_\times,\ \mathbf t)$, the cost

$$
E(\boldsymbol\omega,\mathbf t)=\sum_i\big(\mathbf n_i^{\top}(R\mathbf p_i+\mathbf t-\mathbf q_i)\big)^2
\;\Rightarrow\;
A_i=\big[(\mathbf p_i\times\mathbf n_i)^{\top},\ \mathbf n_i^{\top}\big],\quad r_i=\mathbf n_i^{\top}(\mathbf p_i-\mathbf q_i)
$$

is minimised by $\big(A^{\top}A+\Lambda\big)\mathbf x=-A^{\top}\mathbf r-\Lambda\,\mathbf c$ where $\mathbf x=(\boldsymbol\omega,\delta\mathbf t)$ and
$\mathbf c$ is the *total* correction so far relative to the inertial prediction. $\Lambda=\operatorname{diag}(\mu_R I_3,\mu_t I_3)$ ($\mu_R=10$, $\mu_t=1$)
is a light prior that anchors ICP to the odometry. Eight iterations with a shrinking correspondence gate ($1.0\to0.3$ m) and 15 % residual trimming are run.

**Inertial prediction.** Body velocity $\mathbf v_b$ and gyro $\boldsymbol\omega_b$ give $R_{k+1}=R_k\operatorname{Exp}(\boldsymbol\omega_b\Delta t)$ and
$\mathbf t_{k+1}=\mathbf t_k+R_k\mathbf v_b\Delta t$. The simulated measurements are biased, $\tilde{\mathbf v}=1.04\,\mathbf v+\mathbf b+\mathcal N(0,0.03^2)$,
which is why pure dead reckoning ends metres away.

**Voxel map with clearing.** A voxel's count rises by one for every scan that hits it and it is displayed once the count reaches 5. For every stored voxel
at range $\rho$ and bearing $(\mathrm{az},\mathrm{el})$ the nearest beam's measured range $\rho_b$ is looked up; if $\rho_b>1.12\,\rho+0.8$ the ray passed straight
through the voxel, so its count is decremented. The range-proportional margin matters: a fixed margin deletes the floor, which is seen at grazing angles.

### 4. Loop closure

*Candidates:* nodes $j$ with at least $G$ keyframes and $S$ metres of travelled path ($S=15$ m rover, $20$ m drone) between them and
$\lVert\mathbf t_j-\mathbf t_n\rVert<$ 3.5 / 4.5 m in the *current optimised* graph. *Registration:* from many initial offsets (rover: a $5\times5\times3$ grid
over $\pm1$ m and $\pm0.12$ rad; drone: $\pm2$ m in $x,y$, $\pm0.2$ rad in yaw, plus $\pm1.2$ m in $z$) run point-to-point ICP, whose best rigid motion for matched
sets $\{\mathbf a_i\},\{\mathbf b_i\}$ is the Kabsch solution

$$
H=\sum_i(\mathbf a_i-\bar{\mathbf a})(\mathbf b_i-\bar{\mathbf b})^{\top}=U\Sigma V^{\top},\quad
R=V\operatorname{diag}\!\big(1,\dots,1,\det(VU^{\top})\big)U^{\top},\quad
\mathbf t=\bar{\mathbf b}-R\bar{\mathbf a}.
$$

The start with the highest inlier fraction wins. *Verification:* accept only if the inlier fraction (points within 0.12 m rover / 0.15 m drone) exceeds 0.6, the
reverse overlap exceeds 0.45 (rover) / 0.22 (drone), the inlier RMS is below 0.075 / 0.11 m, and the correction to the predicted relative pose is bounded. The accepted
$(R,\mathbf t)$ becomes an edge $\mathbf z_{jn}$ with $\boldsymbol\sigma=(0.05\text{ m},0.02\text{ rad})$ (rover) or $(0.08,0.03)$ (drone).

### 5. Online estimate and map rebuild

Let $T_{\text{corr}}=\mathbf x_n^{\text{opt}}\circ(\mathbf x_n^{\text{front}})^{-1}$ be the correction at the latest keyframe. The online pose of any later frame is
$T_{\text{corr}}\circ\mathbf x^{\text{front}}$, and when a large correction occurs the global map is re-integrated from scratch: every stored scan $k$ is replayed with
pose $\mathbf x^{\text{opt}}_{\kappa(k)}\circ(\mathbf x^{\text{front}}_{\kappa(k)})^{-1}\circ\mathbf x^{\text{front}}_k$, where $\kappa(k)$ is the keyframe that frame $k$ hangs off.

### 6. Metric

Because both robots start at a known pose, no trajectory alignment is needed and

$$
\operatorname{RMSE}=\sqrt{\frac1N\sum_{k=1}^{N}\big\lVert\hat{\mathbf p}_k-\mathbf p_k\big\rVert^2}.
$$

## Run it

```bash
pip install -r requirements.txt
python -m slamviz.run                  # simulate + graph SLAM -> viewer/data/run.json  (~2 min)
python serve.py                        # open http://localhost:8000
python -m pytest tests                 # ~1 min
```

A pre-computed run (`--seed 0`) is committed, so `python serve.py` works straight away.
Options: `--duration 115 --seed N --map maps/your_map.json --out path.json`.

### Make the film

```bash
pip install -r requirements-video.txt && playwright install chromium     # or: export CHROMIUM=/path/to/chrome
python tools/make_video.py --out docs/slam_lab.mp4         # ~100 s film, 1280x720 @ 20 fps (needs ffmpeg on PATH)
python tools/make_video.py --preview --out /tmp/p.mp4      # two frames per shot, to check the framing
```

Frames are rendered one by one (software WebGL, ~1.2 s per frame, so ~40 min for the full film) with scripted camera
paths and captions; the shot timing is derived from the actual loop-closure events in `run.json`.

## Your own map

Maps are plain JSON (`maps/foundry.json` is generated by `maps/make_foundry.py`, which also documents the schema):
boxes, cylinders and arbitrary-angle walls; movers following line / circle / spin motions; a rover route and a drone
route. Drop in a file and run `python -m slamviz.run --map maps/mine.json --out viewer/data/mine.json`, then open
`http://localhost:8000/?data=data/mine.json` — the viewer picks up everything (statics, movers, zone labels, arena
size) from the exported data. `tests/test_slam.py` builds a small map inline as an
example and runs the whole pipeline on it.

Design tip learned the hard way: long featureless parallel walls (corridors) make LiDAR odometry slide. The Foundry's
container lanes have crates in them for that reason; without them the drone front-end drifted by 2 m or more.

## Viewer

* **Space** play/pause · **← →** step · drag the timeline to scrub (maps and graphs rewind correctly)
* **1–5** camera: orbit / rover chase / drone chase / top-down / cinematic
* Layers: ground truth, occupancy map, voxel map (coloured by height), pose graph + loop closures (violet / pink; a ring
  flashes where a loop is closed), LiDAR rays, particles, truth + online estimate, front-end odometry, raw dead
  reckoning, the final optimised trajectory, zone labels.
* The panel plots pose error (log scale) of raw odometry, front-end, online graph SLAM and the optimised trajectory.

Everything drawn is the **estimated** map, so a mismatch with the translucent ground-truth world is real SLAM error.

## Results

Position RMSE in metres over 115 s (`python -m slamviz.run --seed N`):

| seed | rover: odometry → front-end → graph (online) → optimised | loops | drone: odometry → front-end → graph (online) → optimised | loops |
|---|---|---|---|---|
| 0 | 8.21 → 0.89 → 0.66 → 0.30 | 20 | 7.31 → 0.26 → 0.21 → 0.22 | 7 |
| 1 | 9.11 → 0.95 → 0.54 → 0.23 | 20 | 6.84 → 0.68 → 0.35 → 0.51 | 7 |
| 2 | 8.27 → 0.94 → 0.65 → 0.24 | 16 | 6.99 → 0.26 → 0.21 → 0.21 | 7 |

*Online* is what a robot would know at each moment; *optimised* is every pose in the history after the final
optimisation. Accepted loop closures were compared with ground truth for seeds 0–2: none was wrong (rover: worst
0.12 m / 0.9°; drone: all within 0.3 m / 3°).

### Honest limitations

* The rover benefits a lot from graph optimisation. The drone's front-end is already good, so the gain is small;
  on seed 1 the *optimised* history is **worse** than the online estimate (0.51 vs 0.35 m): a height slip happened in
  one lane, and the optimiser spreads the correction evenly over all odometry edges (they share one covariance), which
  slightly damages the accurate early part. Per-edge covariances from scan-match quality would fix this; not done.
* The drone finds loop closures only late in the run (~83 s) because its route revisits few places.
* Loop-closure verification is geometric only (no place-recognition descriptor), so very repetitive scenes could fool it.
  The pillar row is irregular on purpose.
* "Wheel odometry" and the drone's velocity are derived from simulator ground truth plus an error model; there is no
  wheel/IMU physics model.
* Not tested on a real GPU or in a non-headless browser; the film is software-rendered.

## Layout

```
maps/                  foundry.json (+ the script that writes it and documents the schema)
slamviz/world.py       map loader, MJCF generation
slamviz/sim.py         MuJoCo stepping, sensors, noisy odometry
slamviz/control.py     rover reactive navigation, drone geometric controller
slamviz/posegraph.py   SE(2)/SE(3) pose-graph optimiser
slamviz/occgrid.py     log-odds occupancy grid
slamviz/rover_slam.py  front-end, keyframes, loop closure, graph back-end (2D)
slamviz/drone_slam.py  same in 3D, voxel map
slamviz/run.py         runs everything, exports JSON
viewer/                static Three.js app (three.js vendored, no build step)
tools/make_video.py    frame-exact video renderer
tests/                 solvers, scan matching, custom map, end-to-end graph SLAM
```
