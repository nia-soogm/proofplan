/* 3D evidence view for IFC answers. three.js is vendored under /static/vendor/three (MIT). */
import * as THREE from 'three';
import { OrbitControls } from './vendor/three/OrbitControls.js';

const COL_CTX = 0x8b939e;
const COL_HL = 0xff9a1f;    // bright amber: cited, unchecked
const COL_OK = 0x22f06a;    // saturated green: plan agrees
const COL_BAD = 0xff3b30;
const COL_SEL = 0x19d0ff;
const CTX_OPACITY = 0.10;
const HL_SCALE = 1.18;
// Classic 3/4 view: from the front-right corner, slightly elevated.
const VIEW_DIR = new THREE.Vector3(0.92, 0.70, 0.80).normalize();

const QTY_UNIT = {
  Width: 'm', Height: 'm', Length: 'm', Depth: 'm', Thickness: 'm', Perimeter: 'm',
  OverallWidth: 'm', OverallHeight: 'm', OverallDepth: 'm',
  Area: 'm²', NetFloorArea: 'm²', GrossFloorArea: 'm²', NetSideArea: 'm²', GrossSideArea: 'm²',
  NetVolume: 'm³', GrossVolume: 'm³', Volume: 'm³',
};
const QTY_ORDER = [
  'Width', 'Height', 'Length', 'Depth', 'Thickness',
  'OverallWidth', 'OverallHeight', 'OverallDepth',
  'Area', 'NetFloorArea', 'GrossFloorArea', 'NetSideArea', 'GrossSideArea',
  'Volume', 'NetVolume', 'GrossVolume', 'Perimeter',
];

let host = null, canvasHost = null, infoEl = null, statusEl = null;
let renderer = null, scene = null, camera = null, controls = null, root = null;
let raf = 0, visible = false;
let meshesByGuid = new Map();
let loadedPath = null, loadedData = null;
let citations = [];
let selected = null;
let hooks = {};
let storeyOnly = true;

function esc(s) {
  return String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');
}

function hlColor(guid) {
  const c = citations.find(x => x.guid === guid);
  if (!c) return COL_CTX;
  if (c.verified === true) return COL_OK;
  if (c.verified === false) return COL_BAD;
  return COL_HL;
}

function setStatus(msg) {
  if (statusEl) statusEl.textContent = msg || '';
}

function citedStoreys() {
  const s = new Set();
  for (const c of citations) {
    if (!c.guid || !meshesByGuid.has(c.guid)) continue;
    const st = meshesByGuid.get(c.guid).userData.storey;
    if (st) s.add(st);
  }
  return s;
}

function inCitedStorey(mesh) {
  const ss = citedStoreys();
  if (!ss.size) return true;
  const st = mesh.userData && mesh.userData.storey;
  return !st || ss.has(st);
}

export function mount(el, h = {}) {
  host = el;
  hooks = h || {};
  canvasHost = el.querySelector('#v3d-canvas') || el;
  infoEl = el.querySelector('#v3d-info');
  statusEl = el.querySelector('#v3d-status');
  const fitHl = el.querySelector('#v3d-fit-hl');
  const fitAllBtn = el.querySelector('#v3d-fit-all');
  const storeyCk = el.querySelector('#v3d-storey');
  if (fitHl) fitHl.onclick = () => fitHighlighted();
  if (fitAllBtn) fitAllBtn.onclick = () => fitAll();
  if (storeyCk) {
    storeyCk.checked = storeyOnly;
    storeyCk.onchange = () => { storeyOnly = storeyCk.checked; applyMaterials(); };
  }
  _ensureRenderer();
  window.addEventListener('resize', resize);
  if (typeof ResizeObserver !== 'undefined') {
    new ResizeObserver(() => resize()).observe(canvasHost);
  }
  canvasHost.addEventListener('pointerdown', onPointer);
}

function _ensureRenderer() {
  if (renderer) return;
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x2c313a);
  camera = new THREE.PerspectiveCamera(45, 1, 0.05, 2000);
  camera.position.set(20, 16, 20);
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  canvasHost.appendChild(renderer.domElement);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.target.set(0, 0, 0);
  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  scene.add(new THREE.HemisphereLight(0xe8eef7, 0x3d4338, 0.5));
  const sun = new THREE.DirectionalLight(0xffffff, 0.85);
  sun.position.set(0.45, 1, 0.35);
  scene.add(sun);
  const fill = new THREE.DirectionalLight(0xb9c6d8, 0.28);
  fill.position.set(-0.6, 0.2, -0.5);
  scene.add(fill);
  root = new THREE.Group();
  root.rotation.x = -Math.PI / 2; // IFC Z-up -> three.js Y-up
  scene.add(root);
  _loop();
}

function _loop() {
  if (raf) cancelAnimationFrame(raf);
  const tick = () => {
    raf = requestAnimationFrame(tick);
    if (!visible) return;
    controls && controls.update();
    renderer && renderer.render(scene, camera);
  };
  tick();
}

export function setVisible(v) {
  visible = !!v;
  if (visible) {
    resize();
    _loop();
  }
}

export function resize() {
  if (!renderer || !canvasHost) return;
  const w = Math.max(1, canvasHost.clientWidth || host.clientWidth || 800);
  const h = Math.max(1, canvasHost.clientHeight || host.clientHeight || 600);
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

function disposeRoot() {
  root.traverse(obj => {
    if (obj.geometry) obj.geometry.dispose();
    if (obj.material) {
      if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
      else obj.material.dispose();
    }
  });
  while (root.children.length) root.remove(root.children[0]);
  meshesByGuid = new Map();
  loadedPath = null;
  loadedData = null;
  selected = null;
}

function ensureEdges(mesh, color) {
  let line = mesh.getObjectByName('hl-edge');
  if (!line) {
    const eg = new THREE.EdgesGeometry(mesh.geometry, 40);
    line = new THREE.LineSegments(eg, new THREE.LineBasicMaterial({
      color, depthTest: false, depthWrite: false, transparent: false,
    }));
    line.name = 'hl-edge';
    line.renderOrder = 12;
    line.raycast = () => {};
    mesh.add(line);
  } else {
    line.material.color.setHex(color);
    line.visible = true;
  }
}

function removeEdges(mesh) {
  const line = mesh.getObjectByName('hl-edge');
  if (!line) return;
  line.geometry.dispose();
  line.material.dispose();
  mesh.remove(line);
}

function styleMesh(mesh, isHl) {
  const guid = mesh.userData.guid;
  const otherStorey = storeyOnly && citedStoreys().size && !inCitedStorey(mesh) && !isHl;
  mesh.visible = !otherStorey;
  if (mesh.material) mesh.material.dispose();
  if (isHl) {
    const color = guid === selected ? COL_SEL : hlColor(guid);
    mesh.material = new THREE.MeshLambertMaterial({
      color,
      emissive: new THREE.Color(color),
      emissiveIntensity: guid === selected ? 0.85 : 0.65,
      depthTest: false,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    mesh.renderOrder = 10;
    mesh.scale.setScalar(HL_SCALE);
    ensureEdges(mesh, color);
  } else {
    mesh.material = new THREE.MeshLambertMaterial({
      color: COL_CTX,
      transparent: true,
      opacity: CTX_OPACITY,
      depthWrite: false,
      depthTest: true,
      side: THREE.DoubleSide,
      polygonOffset: true,
      polygonOffsetFactor: 1,
      polygonOffsetUnits: 1,
    });
    mesh.renderOrder = 0;
    mesh.scale.setScalar(1);
    removeEdges(mesh);
  }
}

function applyMaterials() {
  const hl = new Set(citations.map(c => c.guid).filter(Boolean));
  for (const [guid, mesh] of meshesByGuid) styleMesh(mesh, hl.has(guid));
  const wrap = host && host.querySelector('#v3d-storey-wrap');
  if (wrap) wrap.style.display = citedStoreys().size ? '' : 'none';
}

function buildMeshes(data) {
  disposeRoot();
  const scale = data.scale || 0.001;
  const hl = new Set((data.highlight || []).concat(citations.map(c => c.guid).filter(Boolean)));
  for (const e of data.elements) {
    if (!e.v || !e.f || e.v.length < 9 || e.f.length < 3) continue;
    const pos = new Float32Array(e.v.length);
    for (let i = 0; i < e.v.length; i++) pos[i] = e.v[i] * scale;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setIndex(e.f);
    geo.computeBoundingBox();
    const mid = geo.boundingBox.getCenter(new THREE.Vector3());
    geo.translate(-mid.x, -mid.y, -mid.z);
    geo.computeVertexNormals();
    const isHl = hl.has(e.guid);
    const mesh = new THREE.Mesh(geo, new THREE.MeshLambertMaterial());
    mesh.position.copy(mid);
    mesh.userData = e;
    styleMesh(mesh, isHl);
    root.add(mesh);
    meshesByGuid.set(e.guid, mesh);
  }
  loadedData = data;
  const wrap = host && host.querySelector('#v3d-storey-wrap');
  if (wrap) wrap.style.display = citedStoreys().size ? '' : 'none';
}

/* Fit the box in the current viewport from a 3/4 camera angle (not a bounding-sphere estimate). */
function frame(objects, pad = 1.12) {
  if (!objects.length) return;
  const box = new THREE.Box3();
  for (const o of objects) box.expandByObject(o);
  if (box.isEmpty()) return;
  const center = box.getCenter(new THREE.Vector3());
  const dir = VIEW_DIR.clone();
  const up = new THREE.Vector3(0, 1, 0);
  const viewRight = new THREE.Vector3().crossVectors(dir, up);
  if (viewRight.lengthSq() < 1e-8) viewRight.set(1, 0, 0);
  else viewRight.normalize();
  const viewUp = new THREE.Vector3().crossVectors(viewRight, dir).normalize();
  const min = box.min, max = box.max;
  const corners = [
    new THREE.Vector3(min.x, min.y, min.z), new THREE.Vector3(min.x, min.y, max.z),
    new THREE.Vector3(min.x, max.y, min.z), new THREE.Vector3(min.x, max.y, max.z),
    new THREE.Vector3(max.x, min.y, min.z), new THREE.Vector3(max.x, min.y, max.z),
    new THREE.Vector3(max.x, max.y, min.z), new THREE.Vector3(max.x, max.y, max.z),
  ];
  let maxR = 0, maxU = 0;
  const v = new THREE.Vector3();
  for (const c of corners) {
    v.copy(c).sub(center);
    maxR = Math.max(maxR, Math.abs(v.dot(viewRight)));
    maxU = Math.max(maxU, Math.abs(v.dot(viewUp)));
  }
  const vFov = camera.fov * Math.PI / 180;
  const aspect = Math.max(0.2, camera.aspect || 1);
  const hFov = 2 * Math.atan(Math.tan(vFov / 2) * aspect);
  const dist = pad * Math.max(
    maxU / Math.max(0.05, Math.tan(vFov / 2)),
    maxR / Math.max(0.05, Math.tan(hFov / 2))
  );
  const size = box.getSize(new THREE.Vector3());
  const radius = Math.max(size.x, size.y, size.z, 1);
  camera.position.copy(center).add(dir.multiplyScalar(Math.max(dist, radius * 0.6)));
  camera.near = Math.max(0.05, radius / 200);
  camera.far = Math.max(200, radius * 30 + dist);
  camera.updateProjectionMatrix();
  controls.target.copy(center);
  controls.update();
}

export function fitAll() {
  const objs = [...meshesByGuid.values()].filter(m => m.visible);
  frame(objs, 1.12);
}

export function fitHighlighted() {
  const hl = new Set(citations.map(c => c.guid).filter(Boolean));
  const objs = [...meshesByGuid.entries()].filter(([g]) => hl.has(g)).map(([, m]) => m);
  if (objs.length) frame(objs, 1.14);
  else fitAll();
}

function fmtQty(k, v) {
  const u = QTY_UNIT[k] || '';
  const n = typeof v === 'number' ? (Number.isInteger(v) ? String(v) : v.toFixed(Math.abs(v) >= 10 ? 2 : 3)) : String(v);
  return `${n}${u ? ' ' + u : ''}`;
}

function usedFromCitation(c, el) {
  if (!c) return null;
  const t = c.text || '';
  const m = t.match(/\b(area|length|side|vol|volume|width|height)=([0-9.]+)/i);
  if (m) return `${m[1]} = ${m[2]}`;
  if (c.kind && /door|window|column|beam|stair|wall|slab|space/i.test(c.kind)) return 'counted in this answer';
  if (el && el.qty) {
    for (const k of ['Width', 'Height', 'Area', 'Length', 'NetFloorArea', 'Volume']) {
      if (el.qty[k] != null) return `${k} = ${fmtQty(k, el.qty[k])}`;
    }
  }
  return null;
}

function showInfo(guid) {
  if (!infoEl) return;
  if (!guid) { infoEl.style.display = 'none'; infoEl.innerHTML = ''; return; }
  const mesh = meshesByGuid.get(guid);
  const el = mesh ? mesh.userData : (loadedData && loadedData.elements.find(e => e.guid === guid));
  if (!el) { infoEl.style.display = 'none'; return; }
  const c = citations.find(x => x.guid === guid);
  const qty = el.qty || {};
  const skip = new Set();
  for (const short of ['Width', 'Height', 'Depth', 'Area', 'Volume', 'Length']) {
    if (qty[short] != null) skip.add('Overall' + short).add('Net' + short).add('Gross' + short)
      .add('NetFloor' + short).add('GrossFloor' + short).add('NetSide' + short).add('GrossSide' + short);
  }
  const keys = QTY_ORDER.filter(k => qty[k] != null && !skip.has(k)).slice(0, 8);
  const used = usedFromCitation(c, el);
  const rows = keys.map(k => `<div class="kv"><span>${esc(k)}</span><b>${esc(fmtQty(k, qty[k]))}</b></div>`).join('');
  const ver = c && c.verified === true ? 'green — second source agrees' : c && c.verified === false ? 'red — disagrees / not on the plan' : c ? 'cited in the answer' : 'not cited';
  infoEl.innerHTML = `
    <div class="cls">${esc(el.cls || '')}</div>
    <div class="nm">${esc(el.name || '(unnamed)')}</div>
    <div class="kv"><span>GlobalId</span><b class="gid">${esc(el.guid)}</b></div>
    ${el.storey ? `<div class="kv"><span>Storey</span><b>${esc(el.storey)}</b></div>` : ''}
    ${used ? `<div class="kv used"><span>Quantity used</span><b>${esc(used)}</b></div>` : ''}
    ${rows}
    <div class="note">${esc(ver)}${c ? ' · citation [' + c.ref + ']' : ''}</div>`;
  infoEl.style.display = 'block';
}

export function select(guid, {frameSel = false} = {}) {
  selected = guid || null;
  applyMaterials();
  showInfo(guid);
  if (frameSel && guid && meshesByGuid.has(guid)) frame([meshesByGuid.get(guid)], 2.0);
}

function onPointer(ev) {
  if (!renderer || ev.button !== 0) return;
  const rect = renderer.domElement.getBoundingClientRect();
  const mouse = new THREE.Vector2(
    ((ev.clientX - rect.left) / rect.width) * 2 - 1,
    -((ev.clientY - rect.top) / rect.height) * 2 + 1
  );
  const ray = new THREE.Raycaster();
  ray.setFromCamera(mouse, camera);
  const hits = ray.intersectObjects([...meshesByGuid.values()].filter(m => m.visible), false);
  if (!hits.length) { select(null); return; }
  const guid = hits[0].object.userData.guid;
  select(guid);
  if (hooks.onSelect) hooks.onSelect(guid, hits[0].object.userData);
}

export async function show({path, citations: cites, highlight} = {}) {
  citations = Array.isArray(cites) ? cites : [];
  const hl = highlight || citations.map(c => c.guid).filter(Boolean);
  if (!path) { setStatus('No IFC path'); return; }
  setVisible(true);
  if (loadedPath === path && loadedData) {
    applyMaterials();
    const n = hl.length;
    setStatus(`${loadedData.n_elements} elements · ${n} highlighted`);
    if (n) fitHighlighted();
    return;
  }
  setStatus('Loading 3D model… first time can take several seconds');
  if (infoEl) infoEl.style.display = 'none';
  const qs = new URLSearchParams();
  qs.set('path', path);
  for (const g of hl) qs.append('highlight', g);
  const t0 = performance.now();
  let r, data;
  try {
    r = await fetch('/api/ifc3d?' + qs.toString());
    data = await r.json();
  } catch (e) {
    setStatus('Could not load 3D model');
    return;
  }
  if (!r.ok || !data.ok) {
    setStatus((data && (data.detail || data.message)) || ('3D view failed (' + r.status + ')'));
    return;
  }
  buildMeshes(data);
  loadedPath = path;
  const dt = Math.round(performance.now() - t0);
  const n = (data.highlight || hl).length;
  setStatus(`${data.n_elements} elements · ${data.n_triangles.toLocaleString()} triangles · ${n} highlighted · ${dt} ms`
    + (data.cached ? ' (cached)' : ` (tessellated in ${data.tessellate_ms} ms)`));
  resize();
  if (n) fitHighlighted();
  else fitAll();
}

export function current() { return {path: loadedPath, selected, n: meshesByGuid.size}; }
