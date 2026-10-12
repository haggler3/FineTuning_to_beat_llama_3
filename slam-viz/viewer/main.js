import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

THREE.Object3D.DEFAULT_UP.set(0, 0, 1); // MuJoCo convention: z up

const COL = { rover: 0x3dd9ff, drone: 0xffb347, dr: 0xff4d6d, fe: 0x9a86c9, gt: 0xffffff, est: 0x9dffb0, graph: 0xd77bff, loop: 0xff4fd8 };
const $ = (id) => document.getElementById(id);
const msg = (t) => ($('loadmsg').textContent = t);
const VIDEO = new URLSearchParams(location.search).has('video');
if (VIDEO) document.body.classList.add('video');

// ───────────────────────────── data ─────────────────────────────
const b64buf = (s) => { const bin = atob(s), u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u.buffer; };
const inflate = async (s) => new Uint8Array(await new Response(new Blob([b64buf(s)]).stream().pipeThrough(new DecompressionStream('deflate'))).arrayBuffer());

msg('Loading simulation…');
const D = await (await fetch('data/run.json')).json();
const M = D.meta, NF = M.n, GN = M.grid, HALF = M.half, ARENA = D.world.arena, E = D.metrics;
const rRanges = new Uint16Array(b64buf(D.rover.ranges));
const rParts = new Int16Array(b64buf(D.rover.particles));
const dScanAll = new Int16Array(b64buf(D.drone.scan));
const dScanOff = new Uint32Array(NF + 1);
for (let i = 0; i < NF; i++) dScanOff[i + 1] = dScanOff[i] + D.drone.scan_len[i];
msg('Decoding maps…');
const grids = [];
for (const g of D.grids) grids.push({ f: g.f, data: await inflate(g.z) });
const gridAt = (f) => { let lo = 0; for (let i = 0; i < grids.length; i++) if (grids[i].f <= f) lo = i; return lo; };
$('mapname').textContent = `${D.world.name} · graph SLAM · MuJoCo rover + drone`;

// ───────────────────────────── renderer ─────────────────────────────
const canvas = $('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance', preserveDrawingBuffer: VIDEO });
renderer.setPixelRatio(VIDEO ? 1 : Math.min(devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.95;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x070a12);
scene.fog = new THREE.FogExp2(0x070a12, 0.010);
const camera = new THREE.PerspectiveCamera(48, 1, 0.1, 400);
camera.up.set(0, 0, 1);
camera.position.set(-24, -26, 20);
const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 0, 0.5);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.495;

scene.add(new THREE.HemisphereLight(0xaec6ff, 0x1a2036, 1.5));
const sun = new THREE.DirectionalLight(0xffffff, 2.2); sun.position.set(-12, -8, 22); scene.add(sun);

const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
composer.addPass(new UnrealBloomPass(new THREE.Vector2(512, 512), 0.32, 0.55, 0.62));
composer.addPass(new OutputPass());
function resize() {
  const w = innerWidth, h = innerHeight;
  renderer.setSize(w, h, false); composer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix();
}
addEventListener('resize', resize); resize();

const floor = new THREE.Mesh(new THREE.CircleGeometry(70, 64), new THREE.MeshBasicMaterial({ color: 0x0b101c }));
floor.position.z = -0.01; scene.add(floor);
const grid = new THREE.GridHelper(2 * ARENA, 2 * ARENA, 0x2a3a66, 0x182036);
grid.rotation.x = Math.PI / 2; grid.position.z = 0.001; grid.material.transparent = true; grid.material.opacity = 0.7; scene.add(grid);

const circleTex = (() => { const c = document.createElement('canvas'); c.width = c.height = 64; const x = c.getContext('2d');
  const g = x.createRadialGradient(32, 32, 0, 32, 32, 32); g.addColorStop(0, '#fff'); g.addColorStop(0.45, '#fff'); g.addColorStop(1, 'rgba(255,255,255,0)');
  x.fillStyle = g; x.fillRect(0, 0, 64, 64); return new THREE.CanvasTexture(c); })();

// ───────────────────────────── ground-truth world ─────────────────────────────
const layers = {};
const worldG = (layers.world = new THREE.Group()); scene.add(worldG);
function solid(kind, size, color, opacity, emissive = 0) {
  const geo = kind === 'box' ? new THREE.BoxGeometry(2 * size[0], 2 * size[1], 2 * size[2])
    : new THREE.CylinderGeometry(size[0], size[0], 2 * size[1], 28).rotateX(Math.PI / 2);
  const g = new THREE.Group();
  g.add(new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color, transparent: true, opacity, roughness: 0.55, metalness: 0.1, emissive: color, emissiveIntensity: emissive })));
  g.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo, 30), new THREE.LineBasicMaterial({ color: new THREE.Color(color).multiplyScalar(2.2), transparent: true, opacity: 0.55 })));
  return g;
}
for (const s of D.world.statics) { const o = solid(s.kind, s.size, s.color, 0.42); o.position.set(...s.pos); o.rotation.z = THREE.MathUtils.degToRad(s.yaw || 0); worldG.add(o); }
const moverObjs = D.world.movers.map((m) => { const o = solid(m.kind, m.size, m.color, 0.85, 0.35); o.position.z = m.z; worldG.add(o); return o; });

// zone labels
function label(text) {
  const c = document.createElement('canvas'); c.width = 512; c.height = 96; const x = c.getContext('2d');
  x.font = '600 44px system-ui, sans-serif'; x.textAlign = 'center'; x.textBaseline = 'middle'; x.fillStyle = 'rgba(230,236,255,.92)';
  x.shadowColor = 'rgba(61,217,255,.9)'; x.shadowBlur = 14; x.fillText(text, 256, 48);
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthWrite: false, opacity: 0.8 }));
  sp.scale.set(5.2, 0.98, 1); return sp;
}
const zonesG = (layers.zones = new THREE.Group()); scene.add(zonesG);
for (const z of D.world.zones) { const s = label(z.text); s.position.set(...z.pos); zonesG.add(s); }

// ───────────────────────────── vehicles ─────────────────────────────
function makeRover() {
  const g = new THREE.Group();
  const body = new THREE.MeshStandardMaterial({ color: COL.rover, roughness: 0.35, metalness: 0.5, emissive: COL.rover, emissiveIntensity: 0.25 });
  const chassis = new THREE.Mesh(new THREE.BoxGeometry(0.6, 0.4, 0.14), body); chassis.position.z = 0.16; g.add(chassis);
  const dome = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, 0.1, 20).rotateX(Math.PI / 2), new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xffffff, emissiveIntensity: 0.6 }));
  dome.position.set(0.05, 0, 0.28); g.add(dome);
  const wheels = [];
  for (const [x, y] of [[0.18, 0.22], [0.18, -0.22], [-0.18, 0.22], [-0.18, -0.22]]) {
    const w = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.1, 0.06, 18), new THREE.MeshStandardMaterial({ color: 0x1a2030, roughness: 0.8 }));
    w.position.set(x, y, 0.1); g.add(w); wheels.push(w);
  }
  g.userData.wheels = wheels; return g;
}
function makeDrone() {
  const g = new THREE.Group();
  const mat = new THREE.MeshStandardMaterial({ color: COL.drone, roughness: 0.35, metalness: 0.5, emissive: COL.drone, emissiveIntensity: 0.3 });
  g.add(new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.22, 0.08), mat));
  const rotors = [];
  for (let i = 0; i < 4; i++) {
    const a = Math.PI / 4 + (i * Math.PI) / 2, arm = new THREE.Group(); arm.rotation.z = a;
    const bar = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.025, 0.02), mat); bar.position.x = 0.17; arm.add(bar);
    const r = new THREE.Mesh(new THREE.CircleGeometry(0.13, 24), new THREE.MeshBasicMaterial({ color: 0xffe2b0, transparent: true, opacity: 0.35, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide }));
    r.position.set(0.34, 0, 0.03); arm.add(r);
    const blade = new THREE.Mesh(new THREE.BoxGeometry(0.26, 0.015, 0.004), new THREE.MeshBasicMaterial({ color: 0xffffff })); blade.position.set(0.34, 0, 0.032); arm.add(blade);
    rotors.push(blade); g.add(arm);
  }
  g.userData.rotors = rotors; return g;
}
const roverGT = makeRover(), droneGT = makeDrone(); scene.add(roverGT, droneGT);
roverGT.scale.setScalar(1.5); droneGT.scale.setScalar(1.8);
roverGT.add(new THREE.PointLight(COL.rover, 6, 5)); droneGT.add(new THREE.PointLight(COL.drone, 7, 6));
function beacon(color) {
  const m = new THREE.Mesh(new THREE.RingGeometry(0.55, 0.68, 48), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.85, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide }));
  m.position.z = 0.03; scene.add(m); return m;
}
const roverRing = beacon(COL.rover), droneRing = beacon(COL.drone);
const dropGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3(0, 0, 1)]);
const dropLine = new THREE.Line(dropGeo, new THREE.LineBasicMaterial({ color: COL.drone, transparent: true, opacity: 0.6, blending: THREE.AdditiveBlending, depthWrite: false })); dropLine.frustumCulled = false; scene.add(dropLine);
const ghostMat = new THREE.LineBasicMaterial({ color: COL.est, transparent: true, opacity: 0.9 });
const roverGhostG = new THREE.Group();
const rg = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(0.66, 0.46, 0.34)), ghostMat); rg.position.z = 0.2; roverGhostG.add(rg);
const droneGhostG = new THREE.Group();
droneGhostG.add(new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.OctahedronGeometry(0.42)), ghostMat));
scene.add(roverGhostG, droneGhostG);
layers.ghost = new THREE.Group(); layers.ghost.add(roverGhostG, droneGhostG);

// ───────────────────────────── trails ─────────────────────────────
function trail(arr, color, zFn, opacity) {
  const pos = new Float32Array(arr.length * 3);
  arr.forEach((p, i) => { pos[3 * i] = p[0]; pos[3 * i + 1] = p[1]; pos[3 * i + 2] = zFn(p); });
  const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(pos, 3)); geo.setDrawRange(0, 0);
  const l = new THREE.Line(geo, new THREE.LineBasicMaterial({ color, transparent: true, opacity })); l.frustumCulled = false; scene.add(l); return l;
}
const T = {
  rgt: trail(D.rover.gt, COL.gt, () => 0.03, 0.5), rest: trail(D.rover.est, COL.rover, () => 0.07, 1),
  rfe: trail(D.rover.fe, COL.fe, () => 0.10, 0.9), rdr: trail(D.rover.dr, COL.dr, () => 0.13, 0.85), ropt: trail(D.rover.opt, COL.est, () => 0.16, 0.95),
  dgt: trail(D.drone.gt, COL.gt, (p) => p[2], 0.5), dest: trail(D.drone.est, COL.drone, (p) => p[2], 1),
  dfe: trail(D.drone.fe, COL.fe, (p) => p[2], 0.9), ddr: trail(D.drone.dr, COL.dr, (p) => p[2], 0.85), dopt: trail(D.drone.opt, COL.est, (p) => p[2], 0.95),
};
const group = (...ks) => ({ set visible(v) { ks.forEach((k) => (T[k].visible = v)); } });
layers.trails = group('rgt', 'rest', 'dgt', 'dest'); layers.fe = group('rfe', 'dfe'); layers.dr = group('rdr', 'ddr'); layers.opt = group('ropt', 'dopt');
['rfe', 'dfe', 'rdr', 'ddr', 'ropt', 'dopt'].forEach((k) => (T[k].visible = false));

// ───────────────────────────── rover map (floor texture + extruded cells) ─────────────────────────────
const tex = new THREE.DataTexture(new Uint8Array(GN * GN * 4), GN, GN, THREE.RGBAFormat);
tex.colorSpace = THREE.SRGBColorSpace; tex.magFilter = THREE.NearestFilter; tex.minFilter = THREE.NearestFilter; tex.needsUpdate = true;
const mapPlane = new THREE.Mesh(new THREE.PlaneGeometry(2 * HALF, 2 * HALF), new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false }));
mapPlane.position.z = 0.006;
const OCC_CAP = 16000;
const occMesh = new THREE.InstancedMesh(new THREE.BoxGeometry(M.res * 0.95, M.res * 0.95, 0.3),
  new THREE.MeshStandardMaterial({ color: COL.rover, emissive: COL.rover, emissiveIntensity: 0.3, roughness: 0.5, transparent: true, opacity: 0.8 }), OCC_CAP);
occMesh.frustumCulled = false; occMesh.count = 0;
const rmapG = (layers.rmap = new THREE.Group()); rmapG.add(mapPlane, occMesh); scene.add(rmapG);
let shownGrid = -1;
const miniOff = document.createElement('canvas'); miniOff.width = miniOff.height = GN;
const miniOffCtx = miniOff.getContext('2d');
function showGrid(gi) {
  if (gi === shownGrid) return; shownGrid = gi;
  const g = grids[gi].data, px = tex.image.data, img = miniOffCtx.createImageData(GN, GN), m = new THREE.Matrix4();
  let n = 0;
  for (let i = 0; i < GN * GN; i++) {
    const v = g[i]; let r = 0, gg = 0, b = 0, a = 0;
    if (v) { const p = (v - 1) / 254;
      if (p < 0.4) { r = 14; gg = 54; b = 74; a = 150; } else if (p < 0.62) { r = 30; gg = 90; b = 120; a = 150; } else { r = 90; gg = 235; b = 255; a = 255; } }
    px[4 * i] = r; px[4 * i + 1] = gg; px[4 * i + 2] = b; px[4 * i + 3] = a;
    img.data[4 * i] = r; img.data[4 * i + 1] = gg; img.data[4 * i + 2] = b; img.data[4 * i + 3] = a ? Math.max(a, 120) : 0;
    if (v && (v - 1) / 254 > 0.65 && n < OCC_CAP) {
      const cx = i % GN, cy = (i / GN) | 0;
      m.makeTranslation(-HALF + (cx + 0.5) * M.res, -HALF + (cy + 0.5) * M.res, 0.15); occMesh.setMatrixAt(n++, m);
    }
  }
  occMesh.count = n; occMesh.instanceMatrix.needsUpdate = true; tex.needsUpdate = true; miniOffCtx.putImageData(img, 0, 0);
  $('rmapN').textContent = `${n.toLocaleString()} occupied cells`;
}

// ───────────────────────────── drone voxel map (instanced cubes) ─────────────────────────────
const resetAt = new Map(D.drone.resets.map((r) => [r.f, r.v]));
let maxId = 0;
for (const a of D.drone.add) for (let i = 0; i < a.length; i += 4) maxId = Math.max(maxId, a[i]);
for (const v of resetAt.values()) for (let i = 0; i < v.length; i += 4) maxId = Math.max(maxId, v[i]);
const VCAP = maxId + 1;
const voxMesh = new THREE.InstancedMesh(new THREE.BoxGeometry(M.vox * 0.86, M.vox * 0.86, M.vox * 0.86),
  new THREE.MeshStandardMaterial({ roughness: 0.6, metalness: 0.0, color: 0xffffff }), VCAP);
voxMesh.frustumCulled = false; voxMesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(VCAP * 3), 3);
const dmapG = (layers.dmap = new THREE.Group()); dmapG.add(voxMesh); scene.add(dmapG);
const HIDDEN = new THREE.Matrix4().makeScale(0, 0, 0);
const stops = ['#4b2bb8', '#2563ff', '#12c8d8', '#6dff9a', '#ffe44d', '#ff7b3d'].map((c) => new THREE.Color(c));
function heat(z) { const t = Math.min(Math.max(z / 3, 0), 0.999) * (stops.length - 1), i = t | 0; return stops[i].clone().lerp(stops[i + 1], t - i); }
let voxFrame = -1, voxCount = 0;
const mTmp = new THREE.Matrix4();
function voxPlace(v) {
  for (let i = 0; i < v.length; i += 4) {
    const x = (v[i + 1] + 0.5) * M.vox, y = (v[i + 2] + 0.5) * M.vox, z = (v[i + 3] + 0.5) * M.vox;
    mTmp.makeTranslation(x, y, z); voxMesh.setMatrixAt(v[i], z < 0.45 ? HIDDEN : mTmp);  // floor voxels stay in the map but are not drawn
    voxMesh.setColorAt(v[i], heat(z));
  }
}
function voxReset() { for (let i = 0; i < VCAP; i++) voxMesh.setMatrixAt(i, HIDDEN); voxFrame = -1; voxCount = 0; }
function voxApply(f) {
  if (resetAt.has(f)) { // the back-end rebuilt the whole map from optimised poses
    for (let i = 0; i < VCAP; i++) voxMesh.setMatrixAt(i, HIDDEN);
    const v = resetAt.get(f); voxPlace(v); voxCount = v.length / 4; return;
  }
  for (const id of D.drone.rem[f]) { voxMesh.setMatrixAt(id, HIDDEN); voxCount--; }
  voxPlace(D.drone.add[f]); voxCount += D.drone.add[f].length / 4;
}
function voxSeek(f) {
  if (f < voxFrame) voxReset();
  while (voxFrame < f) voxApply(++voxFrame);
  voxMesh.instanceMatrix.needsUpdate = true; voxMesh.instanceColor.needsUpdate = true;
}
voxReset();

// ───────────────────────────── pose graphs ─────────────────────────────
class GraphViz {
  constructor(events, kf, targetFn, dim) {
    this.events = events; this.kf = kf; this.target = targetFn; this.dim = dim;
    this.cap = Math.ceil(NF / kf) + 4;
    this.pos = new Float32Array(this.cap * 3); this.n = 0; this.evIdx = -1; this.pulse = 0; this.lastF = 0;
    this.nodeGeo = new THREE.BufferGeometry(); this.nodeGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(this.cap * 3), 3));
    this.nodes = new THREE.Points(this.nodeGeo, new THREE.PointsMaterial({ color: COL.graph, size: 0.28, map: circleTex, transparent: true, alphaTest: 0.01, depthWrite: false, blending: THREE.AdditiveBlending }));
    this.odGeo = new THREE.BufferGeometry(); this.odGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(this.cap * 6), 3));
    this.od = new THREE.LineSegments(this.odGeo, new THREE.LineBasicMaterial({ color: COL.graph, transparent: true, opacity: 0.5, blending: THREE.AdditiveBlending, depthWrite: false }));
    this.lpGeo = new THREE.BufferGeometry(); this.lpGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(this.cap * 12), 3));
    this.lpMat = new THREE.LineBasicMaterial({ color: COL.loop, transparent: true, opacity: 0.8, blending: THREE.AdditiveBlending, depthWrite: false });
    this.lp = new THREE.LineSegments(this.lpGeo, this.lpMat);
    // flash on a new loop closure: an expanding ring and a light beam at the place that was recognised
    this.ring = new THREE.Mesh(new THREE.RingGeometry(0.9, 1.0, 64), new THREE.MeshBasicMaterial({ color: COL.loop, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide }));
    this.beamGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3(0, 0, 3.4)]);
    this.beam = new THREE.Line(this.beamGeo, new THREE.LineBasicMaterial({ color: COL.loop, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false }));
    this.flashNode = -1;
    this.group = new THREE.Group();
    for (const o of [this.nodes, this.od, this.lp, this.ring, this.beam]) { o.frustumCulled = false; this.group.add(o); }
    scene.add(this.group);
  }
  update(f, dt) {
    const nn = Math.min(Math.floor(f / this.kf) + 1, this.cap);
    let ei = -1; for (let i = 0; i < this.events.length; i++) if (this.events[i].f <= f) ei = i;
    const ev = ei >= 0 ? this.events[ei] : null;
    if (ei !== this.evIdx) { if (ei > this.evIdx && Math.abs(f - this.lastF) <= 3 && ev && ev.loops.length) { this.pulse = 1; this.flashNode = ev.loops[ev.loops.length - 1][1]; } this.evIdx = ei; }
    const snap = Math.abs(f - this.lastF) > 3; this.lastF = f;
    const k = 1 - Math.exp(-dt * 7), t = [0, 0, 0];
    for (let i = 0; i < nn; i++) {
      this.target(i, ev, t);
      if (i >= this.n || snap) { this.pos.set(t, 3 * i); }
      else for (let c = 0; c < 3; c++) this.pos[3 * i + c] += (t[c] - this.pos[3 * i + c]) * k;
    }
    this.n = nn;
    const np = this.nodeGeo.attributes.position.array, od = this.odGeo.attributes.position.array, lp = this.lpGeo.attributes.position.array;
    np.set(this.pos.subarray(0, 3 * nn)); this.nodeGeo.setDrawRange(0, nn); this.nodeGeo.attributes.position.needsUpdate = true;
    for (let i = 0; i + 1 < nn; i++) { od.set(this.pos.subarray(3 * i, 3 * i + 3), 6 * i); od.set(this.pos.subarray(3 * i + 3, 3 * i + 6), 6 * i + 3); }
    this.odGeo.setDrawRange(0, Math.max(0, nn - 1) * 2); this.odGeo.attributes.position.needsUpdate = true;
    let nl = 0;
    if (ev) for (const [a, b] of ev.loops) if (a < nn && b < nn && nl < this.cap) { lp.set(this.pos.subarray(3 * a, 3 * a + 3), 6 * nl); lp.set(this.pos.subarray(3 * b, 3 * b + 3), 6 * nl + 3); nl++; }
    this.lpGeo.setDrawRange(0, nl * 2); this.lpGeo.attributes.position.needsUpdate = true;
    this.pulse = Math.max(0, this.pulse - dt * 0.6);
    this.lpMat.opacity = 0.55 + 0.45 * this.pulse; this.nodes.material.size = 0.28 + 0.25 * this.pulse;
    const showFlash = this.pulse > 0 && this.flashNode >= 0 && this.flashNode < nn;
    this.ring.visible = this.beam.visible = showFlash;
    if (showFlash) {
      const x = this.pos[3 * this.flashNode], y = this.pos[3 * this.flashNode + 1], z = this.dim === 3 ? this.pos[3 * this.flashNode + 2] : 0.08;
      const sc = 1 + (1 - this.pulse) * 5; this.ring.position.set(x, y, z); this.ring.scale.set(sc, sc, 1);
      this.ring.material.opacity = this.pulse * 0.95; this.beam.position.set(x, y, 0); this.beam.material.opacity = this.pulse;
    }
    this.loopCount = ev ? ev.loops.length : 0; this.optCount = ei + 1;
  }
}
const roverGraph = new GraphViz(D.rover.graph, M.rover_kf, (i, ev, t) => {
  const p = ev && i < ev.nodes.length ? ev.nodes[i] : D.rover.est[Math.min(i * M.rover_kf, NF - 1)];
  t[0] = p[0]; t[1] = p[1]; t[2] = 0.2;
}, 2);
const droneGraph = new GraphViz(D.drone.graph, M.drone_kf, (i, ev, t) => {
  const p = ev && i < ev.nodes.length ? ev.nodes[i] : D.drone.est[Math.min(i * M.drone_kf, NF - 1)];
  t[0] = p[0]; t[1] = p[1]; t[2] = p[2];
}, 3);
layers.graph = { set visible(v) { roverGraph.group.visible = droneGraph.group.visible = v; } };

// ───────────────────────────── live scans ─────────────────────────────
function rayLines(max, color) {
  const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(max * 6), 3)); geo.setDrawRange(0, 0);
  const l = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.16, blending: THREE.AdditiveBlending, depthWrite: false }));
  l.frustumCulled = false; scene.add(l); return l;
}
function hitPoints(max, color, size) {
  const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(max * 3), 3)); geo.setDrawRange(0, 0);
  const p = new THREE.Points(geo, new THREE.PointsMaterial({ color, size, map: circleTex, transparent: true, alphaTest: 0.01, depthWrite: false, blending: THREE.AdditiveBlending }));
  p.frustumCulled = false; scene.add(p); return p;
}
const rRays = rayLines(M.rover_beams, COL.rover), rHits = hitPoints(M.rover_beams, 0xbff4ff, 0.14);
const dRays = rayLines(500, COL.drone), dHits = hitPoints(500, 0xffe2b0, 0.12);
layers.rscan = { set visible(v) { rRays.visible = rHits.visible = v; } };
layers.dscan = { set visible(v) { dRays.visible = dHits.visible = v; } };
const partGeo = new THREE.BufferGeometry(); partGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(40 * 3), 3));
const parts = new THREE.Points(partGeo, new THREE.PointsMaterial({ color: 0xb794ff, size: 0.2, map: circleTex, transparent: true, alphaTest: 0.01, depthWrite: false, blending: THREE.AdditiveBlending }));
parts.frustumCulled = false; scene.add(parts); layers.parts = parts;

// ───────────────────────────── per-frame update ─────────────────────────────
const wrapA = (a) => Math.atan2(Math.sin(a), Math.cos(a));
const lerp = (a, b, t) => a + (b - a) * t;
const qa = new THREE.Quaternion(), qb = new THREE.Quaternion(), qe = new THREE.Quaternion(), v3 = new THREE.Vector3();
let wheelPhase = 0, rotorPhase = 0;

function update(f, a, dtReal, dtGraph = dtReal) {
  const g = Math.min(f + 1, NF - 1);
  D.world.movers.forEach((m, i) => {
    const p = D.movers[f], q = D.movers[g];
    moverObjs[i].position.set(lerp(p[3 * i], q[3 * i], a), lerp(p[3 * i + 1], q[3 * i + 1], a), m.z);
    moverObjs[i].rotation.z = p[3 * i + 2] + wrapA(q[3 * i + 2] - p[3 * i + 2]) * a;
  });
  const rp = D.rover.gt[f], rq = D.rover.gt[g];
  roverGT.position.set(lerp(rp[0], rq[0], a), lerp(rp[1], rq[1], a), 0);
  roverGT.rotation.z = rp[2] + wrapA(rq[2] - rp[2]) * a;
  wheelPhase += dtReal * 6; roverGT.userData.wheels.forEach((w) => (w.rotation.x = wheelPhase));
  const re = D.rover.est[f], re2 = D.rover.est[g];
  roverGhostG.position.set(lerp(re[0], re2[0], a), lerp(re[1], re2[1], a), 0); roverGhostG.rotation.z = re[2] + wrapA(re2[2] - re[2]) * a;
  const dgp = D.drone.gt[f], dgq = D.drone.gt[g];
  droneGT.position.set(lerp(dgp[0], dgq[0], a), lerp(dgp[1], dgq[1], a), lerp(dgp[2], dgq[2], a));
  droneGT.quaternion.copy(qa.set(dgp[4], dgp[5], dgp[6], dgp[3]).slerp(qb.set(dgq[4], dgq[5], dgq[6], dgq[3]), a));
  rotorPhase += dtReal * 55; droneGT.userData.rotors.forEach((r, i) => (r.rotation.z = rotorPhase * (i % 2 ? 1 : -1)));
  const dep = D.drone.est[f], deq = D.drone.est[g];
  droneGhostG.position.set(lerp(dep[0], deq[0], a), lerp(dep[1], deq[1], a), lerp(dep[2], deq[2], a));
  droneGhostG.quaternion.copy(qa.set(dep[4], dep[5], dep[6], dep[3]).slerp(qb.set(deq[4], deq[5], deq[6], deq[3]), a));
  roverRing.position.set(roverGT.position.x, roverGT.position.y, 0.03);
  droneRing.position.set(droneGT.position.x, droneGT.position.y, 0.03); droneRing.scale.setScalar(1 + 0.12 * Math.sin(rotorPhase * 0.15));
  const dp = dropGeo.attributes.position; dp.setXYZ(0, droneGT.position.x, droneGT.position.y, 0.03); dp.setXYZ(1, droneGT.position.x, droneGT.position.y, droneGT.position.z); dp.needsUpdate = true;
  for (const k in T) T[k].geometry.setDrawRange(0, f + 1);
  showGrid(gridAt(f)); voxSeek(f);
  roverGraph.update(f, dtGraph); droneGraph.update(f, dtGraph);
  // rover scan rays (from the SLAM estimate)
  const rr = rRays.geometry.attributes.position.array, rh = rHits.geometry.attributes.position.array;
  let n = 0; const base = f * M.rover_beams;
  for (let i = 0; i < M.rover_beams; i++) {
    const r = rRanges[base + i] / 1000; if (r <= 0) continue;
    const ang = re[2] + (-Math.PI + (i * 2 * Math.PI) / M.rover_beams), ox = re[0], oy = re[1], hx = ox + r * Math.cos(ang), hy = oy + r * Math.sin(ang);
    rr.set([ox, oy, 0.25, hx, hy, 0.25], 6 * n); rh.set([hx, hy, 0.25], 3 * n); n++;
  }
  rRays.geometry.setDrawRange(0, n * 2); rHits.geometry.setDrawRange(0, n);
  rRays.geometry.attributes.position.needsUpdate = rHits.geometry.attributes.position.needsUpdate = true;
  const pp = partGeo.attributes.position.array;
  for (let i = 0; i < 40; i++) { pp[3 * i] = rParts[(f * 40 + i) * 2] / 100; pp[3 * i + 1] = rParts[(f * 40 + i) * 2 + 1] / 100; pp[3 * i + 2] = 0.07; }
  partGeo.attributes.position.needsUpdate = true;
  qe.set(dep[4], dep[5], dep[6], dep[3]);
  const dr = dRays.geometry.attributes.position.array, dh = dHits.geometry.attributes.position.array;
  const s0 = 3 * dScanOff[f], cnt = Math.min(D.drone.scan_len[f], 500); // scan_len / dScanOff count points; 3 int16 per point
  for (let i = 0; i < cnt; i++) {
    v3.set(dScanAll[s0 + 3 * i] / 100, dScanAll[s0 + 3 * i + 1] / 100, dScanAll[s0 + 3 * i + 2] / 100).applyQuaternion(qe);
    const hx = dep[0] + v3.x, hy = dep[1] + v3.y, hz = dep[2] + v3.z;
    dr.set([dep[0], dep[1], dep[2], hx, hy, hz], 6 * i); dh.set([hx, hy, hz], 3 * i);
  }
  dRays.geometry.setDrawRange(0, cnt * 2); dHits.geometry.setDrawRange(0, cnt);
  dRays.geometry.attributes.position.needsUpdate = dHits.geometry.attributes.position.needsUpdate = true;
  updateHud(f);
}

// ───────────────────────────── HUD / charts ─────────────────────────────
const fmt = (v) => v.toFixed(2) + ' m';
// symlog y-axis so that 0.05 m and 10 m errors are both readable
const ymax = Math.max(...E.rover_odom_err, ...E.drone_odom_err), sl = (v) => Math.log10(1 + v * 12) / Math.log10(1 + ymax * 12);
function drawChart(cv, f, series, title) {
  const cx = cv.getContext('2d'), W = cv.width, H = cv.height, pad = 6;
  cx.clearRect(0, 0, W, H);
  cx.font = '10px system-ui'; cx.lineWidth = 1;
  for (const v of [0.1, 0.3, 1, 3, 10]) if (v < ymax) { const y = H - pad - sl(v) * (H - 2 * pad); cx.strokeStyle = 'rgba(255,255,255,.07)'; cx.beginPath(); cx.moveTo(0, y); cx.lineTo(W, y); cx.stroke(); cx.fillStyle = 'rgba(255,255,255,.35)'; cx.fillText(v + ' m', W - 34, y - 2); }
  cx.fillStyle = 'rgba(255,255,255,.55)'; cx.fillText(title, 6, 11);
  for (const [arr, color, w] of series) {
    cx.strokeStyle = color; cx.lineWidth = 1; cx.globalAlpha = 0.2; cx.beginPath();
    arr.forEach((v, i) => { const x = (i / (NF - 1)) * W, y = H - pad - sl(v) * (H - 2 * pad); i ? cx.lineTo(x, y) : cx.moveTo(x, y); }); cx.stroke();
    cx.globalAlpha = 1; cx.lineWidth = w; cx.beginPath();
    for (let i = 0; i <= f; i++) { const x = (i / (NF - 1)) * W, y = H - pad - sl(arr[i]) * (H - 2 * pad); i ? cx.lineTo(x, y) : cx.moveTo(x, y); } cx.stroke();
  }
  const x = (f / (NF - 1)) * W; cx.strokeStyle = 'rgba(255,255,255,.6)'; cx.lineWidth = 1; cx.beginPath(); cx.moveTo(x, 0); cx.lineTo(x, H); cx.stroke();
}
const mini = $('mini'), mx = mini.getContext('2d');
function drawMini(f) {
  const W = mini.width, S = W / (2 * HALF), X = (x) => (x + HALF) * S, Y = (y) => W - (y + HALF) * S;
  mx.fillStyle = '#05070d'; mx.fillRect(0, 0, W, W);
  mx.save(); mx.imageSmoothingEnabled = false; mx.translate(0, W); mx.scale(1, -1); mx.drawImage(miniOff, 0, 0, W, W); mx.restore();
  const path = (arr, col, w) => { mx.strokeStyle = col; mx.lineWidth = w; mx.beginPath(); for (let i = 0; i <= f; i += 2) { const p = arr[i]; i ? mx.lineTo(X(p[0]), Y(p[1])) : mx.moveTo(X(p[0]), Y(p[1])); } mx.stroke(); };
  path(D.rover.dr, 'rgba(255,77,109,.55)', 1); path(D.rover.gt, 'rgba(255,255,255,.5)', 1); path(D.rover.est, '#3dd9ff', 1.6);
  const rp = roverGraph.pos; mx.strokeStyle = 'rgba(255,79,216,.9)'; mx.lineWidth = 1.2;
  const ev = roverGraph.events[roverGraph.evIdx];
  if (ev) for (const [a, b] of ev.loops) if (a < roverGraph.n && b < roverGraph.n) { mx.beginPath(); mx.moveTo(X(rp[3 * a]), Y(rp[3 * a + 1])); mx.lineTo(X(rp[3 * b]), Y(rp[3 * b + 1])); mx.stroke(); }
  mx.fillStyle = 'rgba(215,123,255,.9)'; for (let i = 0; i < roverGraph.n; i++) mx.fillRect(X(rp[3 * i]) - 1, Y(rp[3 * i + 1]) - 1, 2, 2);
  const e = D.rover.est[f], base = f * M.rover_beams;
  mx.strokeStyle = 'rgba(61,217,255,.14)'; mx.beginPath();
  for (let i = 0; i < M.rover_beams; i += 3) { const r = rRanges[base + i] / 1000; if (r <= 0) continue; const a = e[2] + (-Math.PI + (i * 2 * Math.PI) / M.rover_beams); mx.moveTo(X(e[0]), Y(e[1])); mx.lineTo(X(e[0] + r * Math.cos(a)), Y(e[1] + r * Math.sin(a))); }
  mx.stroke();
  D.world.movers.forEach((m, i) => { mx.fillStyle = m.color; mx.beginPath(); mx.arc(X(D.movers[f][3 * i]), Y(D.movers[f][3 * i + 1]), 3, 0, 6.3); mx.fill(); });
  mx.save(); mx.translate(X(e[0]), Y(e[1])); mx.rotate(-e[2]); mx.fillStyle = '#fff'; mx.beginPath(); mx.moveTo(8, 0); mx.lineTo(-5, 5); mx.lineTo(-5, -5); mx.fill(); mx.restore();
}
const rowv = (l, a, b, c) => `<div class="v"><span>${l}</span><span>${a} → <b>${b}</b>${c ? ` → ${c}` : ''}</span></div>`;
$('final').innerHTML = `<h2>Position RMSE</h2>` +
  `<div class="v"><span></span><span>odom → front-end → <b>graph</b></span></div>` +
  rowv('Rover', E.rover_odom_rmse.toFixed(2), E.rover_online_rmse.toFixed(2), '') .replace('→ <b>', `→ ${E.rover_front_rmse.toFixed(2)} → <b>`) +
  rowv('Drone', E.drone_odom_rmse.toFixed(2), E.drone_online_rmse.toFixed(2), '').replace('→ <b>', `→ ${E.drone_front_rmse.toFixed(2)} → <b>`) +
  `<div class="v"><span></span><span>optimised: <b>${E.rover_opt_rmse.toFixed(2)}</b> · <b>${E.drone_opt_rmse.toFixed(2)}</b> m</span></div>`;
function updateHud(f) {
  $('rErr').textContent = fmt(E.rover_online_err[f]); $('rFe').textContent = fmt(E.rover_front_err[f]); $('rDr').textContent = fmt(E.rover_odom_err[f]);
  $('dErr').textContent = fmt(E.drone_online_err[f]); $('dFe').textContent = fmt(E.drone_front_err[f]); $('dDr').textContent = fmt(E.drone_odom_err[f]);
  $('loops').textContent = `r ${roverGraph.loopCount} · d ${droneGraph.loopCount}`; $('opts').textContent = `r ${roverGraph.optCount} · d ${droneGraph.optCount}`;
  $('icp').textContent = D.drone.icp_ok[f] ? 'locked' : 'coasting'; $('icp').style.color = D.drone.icp_ok[f] ? '#9dffb0' : '#ff7a8a';
  $('vox').textContent = Math.max(voxCount, 0).toLocaleString();
  $('time').textContent = D.t[f].toFixed(1) + ' s';
  const s = $('scrub'); s.value = f; s.style.setProperty('--p', (100 * f / (NF - 1)) + '%');
  drawChart($('chartR'), f, [[E.rover_odom_err, '#ff4d6d', 1.4], [E.rover_front_err, '#9a86c9', 1.4], [E.rover_online_err, '#3dd9ff', 1.8], [E.rover_opt_err, '#9dffb0', 1.2]], 'ROVER error (log scale)');
  drawChart($('chartD'), f, [[E.drone_odom_err, '#ff4d6d', 1.4], [E.drone_front_err, '#9a86c9', 1.4], [E.drone_online_err, '#ffb347', 1.8], [E.drone_opt_err, '#9dffb0', 1.2]], 'DRONE error (log scale)');
  drawMini(f);
}

// events -> toast
const allEvents = [...D.rover.graph.map((e) => ({ f: e.f, who: 'rover', shift: e.shift, rebuilt: e.rebuilt })), ...D.drone.graph.map((e) => ({ f: e.f, who: 'drone', shift: e.shift, rebuilt: e.rebuilt }))]
  .filter((e) => e.f < NF - 1).sort((a, b) => a.f - b.f);
let toastTimer = 0, lastToastF = 0;
function toast(text) { if (VIDEO) return; const t = $('toast'); t.textContent = text; t.classList.add('on'); clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove('on'), 2600); }
function checkEvents(f) {
  if (f > lastToastF && f - lastToastF <= 3) {
    const e = allEvents.filter((x) => x.f > lastToastF && x.f <= f).pop();
    if (e) toast(`↺  LOOP CLOSURE · ${e.who} · pose graph optimised${e.shift ? ` (Δ ${e.shift.toFixed(2)} m)` : ''}${e.rebuilt ? ' · map rebuilt' : ''}`);
  }
  lastToastF = f;
}

// ───────────────────────────── controls ─────────────────────────────
let playing = !VIDEO, speed = 1, tpos = 0, cam = 'orbit';
$('scrub').max = NF - 1;
const setPlaying = (p) => { playing = p; $('play').textContent = p ? '❚❚' : '▶'; };
$('play').onclick = () => setPlaying(!playing);
$('scrub').oninput = (e) => { tpos = +e.target.value; };
$('step-').onclick = () => { setPlaying(false); tpos = Math.max(0, Math.floor(tpos) - 1); };
$('step+').onclick = () => { setPlaying(false); tpos = Math.min(NF - 1, Math.floor(tpos) + 1); };
document.querySelectorAll('#speeds button').forEach((b) => (b.onclick = () => { speed = +b.dataset.s; document.querySelectorAll('#speeds button').forEach((x) => x.classList.toggle('on', x === b)); }));
document.querySelectorAll('[data-layer]').forEach((c) => {
  const apply = () => { const l = layers[c.dataset.layer]; if (l) l.visible = c.checked; };
  c.onchange = apply; apply();
});
const setCam = (c) => { cam = c; document.querySelectorAll('#cams button').forEach((b) => b.classList.toggle('on', b.dataset.cam === c)); controls.enabled = c === 'orbit' || c === 'cine' || c === 'top'; };
document.querySelectorAll('#cams button').forEach((b) => (b.onclick = () => setCam(b.dataset.cam)));
addEventListener('keydown', (e) => {
  if (e.code === 'Space') { e.preventDefault(); setPlaying(!playing); }
  else if (e.code === 'ArrowRight') $('step+').click(); else if (e.code === 'ArrowLeft') $('step-').click();
  else if ('12345'.includes(e.key)) setCam(['orbit', 'rover', 'drone', 'top', 'cine'][+e.key - 1]);
});

const camGoal = new THREE.Vector3(), lookGoal = new THREE.Vector3();
function updateCamera(dt, tNow) {
  const k = 1 - Math.pow(0.002, dt);
  if (cam === 'rover') {
    const y = roverGT.rotation.z, p = roverGT.position;
    camGoal.set(p.x - 5.5 * Math.cos(y), p.y - 5.5 * Math.sin(y), 3.4); lookGoal.set(p.x + 2 * Math.cos(y), p.y + 2 * Math.sin(y), 0.3);
  } else if (cam === 'drone') {
    const p = droneGT.position, e = new THREE.Euler().setFromQuaternion(droneGT.quaternion, 'ZYX');
    camGoal.set(p.x - 6.5 * Math.cos(e.z), p.y - 6.5 * Math.sin(e.z), p.z + 3.2); lookGoal.copy(p);
  } else if (cam === 'top') {
    camGoal.set(0, -0.01, 42); lookGoal.set(0, 0, 0);
  } else if (cam === 'cine') {
    const a = tNow * 0.12; camGoal.set(30 * Math.cos(a), 30 * Math.sin(a), 13 + 3 * Math.sin(a * 1.7)); lookGoal.set(0, 0, 0.5);
  } else { controls.update(); return; }
  camera.position.lerp(camGoal, k); controls.target.lerp(lookGoal, k); controls.update();
}

// ───────────────────────────── main loop ─────────────────────────────
let last = performance.now();
function frame(now) {
  const dt = Math.min(Math.max((now - last) / 1000, 0), 0.1); last = now;
  if (playing) { tpos += dt * speed * (1 / M.dt); if (tpos >= NF - 1) tpos = 0; }
  const f = Math.floor(tpos), a = playing ? tpos - f : 0;
  update(f, a, playing ? dt * speed : 0, dt);
  checkEvents(f);
  updateCamera(dt, now / 1000);
  composer.render();
  requestAnimationFrame(frame);
}
$('loading').classList.add('done');

// Deterministic single-frame rendering for the video pipeline (see director.js / tools/make_video.py).
const api = {
  D, NF, camera, controls, setCam, setPlaying, layers, COL, scene,
  seek: (f) => { tpos = f; },
  get frame() { return Math.floor(tpos); },
  /** Render sim time `ft` (fractional frame) with an explicit camera; dt is the wall-clock step of the video. */
  renderAt(ft, camPos, camTarget, dt) {
    const f = Math.min(Math.floor(ft), NF - 1), a = ft - Math.floor(ft);
    update(f, a, dt * 2, dt);
    checkEvents(f);
    camera.position.set(...camPos); controls.target.set(...camTarget); camera.lookAt(...camTarget);
    composer.render();
  },
  setLayer(name, on) { const c = document.querySelector(`[data-layer="${name}"]`); if (c) { c.checked = on; c.onchange(); } },
  caption(title, sub = '') { const c = $('caption'); c.innerHTML = title ? `${title}${sub ? `<small>${sub}</small>` : ''}` : ''; c.classList.toggle('on', !!title); },
  titleCard(title, sub) { $('tcTitle').textContent = title || ''; $('tcSub').textContent = sub || ''; $('titlecard').classList.toggle('on', !!title); },
  ready: true,
};
window.__slam = api;
if (!VIDEO) requestAnimationFrame(frame);
