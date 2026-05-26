// Phase 3b/3c — reciprocal cell, Brillouin zone & k-path selection.
// 3b: render the BZ (edges + translucent faces) with labeled, clickable
//     high-symmetry points.
// 3c: interactive k-path building. The path is an ordered list of segments
//     {from, to, npoints}; consecutive segments may be discontinuous (a "break")
//     when seg[i].to !== seg[i+1].from. It is pre-populated from seekpath's
//     suggested_path and editable by clicking points (extend the active tip) or
//     via the segment list (edit npoints / remove). The assembled path is stored
//     in state.kpath for the calculation step. from/to carry frac_conv (the
//     conventional reciprocal coords src_gpu consumes).
'use strict';

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { api } from '../api.js';
import { state } from '../state.js';

const GAMMA = 'Γ';
const PRIMES = ['′', '″', '‴'];
// seekpath disambiguates symmetry-distinct points of the same type with a
// numeric suffix (W, W_2, W_3, …). Render those as primes (W, W′, W″) and map
// GAMMA → Γ.
function showLabel(raw) {
    let base = raw, suffix = '';
    const m = /^(.*)_(\d+)$/.exec(raw);
    if (m) {
        base = m[1];
        const n = parseInt(m[2], 10);
        suffix = PRIMES[n - 2] || "'".repeat(Math.max(1, n - 1));
    }
    if (base === 'GM' || base === 'GAMMA' || base === 'G') base = GAMMA;
    return base + suffix;
}

const COLOR_FREE = 0x16a34a;   // green  — not on path
const COLOR_PATH = 0xf59e0b;   // orange — on path
const COLOR_TIP  = 0xdc2626;   // red    — active tip (next click extends from here)
const COLOR_SEG  = 0xea580c;   // orange — path segments (tubes + arrowheads)

const BASE_NPTS = 50;          // npoints for the longest segment; others scale by length

export async function mountBZView(root, _opts = {}) {
    root.innerHTML = `
      <section class="panel">
        <h2><span class="step-badge">3</span>Brillouin zone &amp; k-path</h2>
        <div id="bz-info"></div>
        <div id="bz-canvas" class="canvas3d"></div>
        <div id="bz-readout">Click a high-symmetry point to extend the path.</div>
        <div id="bz-path"></div>
      </section>
    `;
    const info = root.querySelector('#bz-info');
    try {
        const data = await api.getReciprocal();
        state.set('reciprocal', data);
        info.innerHTML =
            `<span class="chip">${data.spacegroup}</span> `
            + `<span class="chip">${data.crystal_system}</span> `
            + `<span class="chip">${data.high_sym_points.length} high-symmetry points</span>`;
        renderBZ(root.querySelector('#bz-canvas'), data, root);
    } catch (err) {
        info.innerHTML = `<span class="err">✗ ${err.message}</span>`;
    }
}

function renderBZ(container, data, root) {
    const W = container.clientWidth || 600, H = container.clientHeight || 380;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf4f4f5);

    const camera = new THREE.PerspectiveCamera(45, W / H, 0.001, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(W, H);
    container.innerHTML = '';
    container.appendChild(renderer.domElement);

    scene.add(new THREE.AmbientLight(0xffffff, 0.75));
    const dl = new THREE.DirectionalLight(0xffffff, 0.6);
    dl.position.set(1, 1, 2); scene.add(dl);

    // BZ extent → scale for camera, point radius, label size
    const allV = [];
    for (const f of data.bz_facets) for (const v of f) allV.push(v);
    const maxR = Math.max(...allV.map(v => Math.hypot(v[0], v[1], v[2]))) || 1;

    // ── BZ translucent faces (fan-triangulated) + edges ─────────────────
    const facePts = [];
    const edgePts = [];
    for (const facet of data.bz_facets) {
        const n = facet.length;
        for (let i = 1; i < n - 1; i++) {
            facePts.push(...facet[0], ...facet[i], ...facet[i + 1]);
        }
        for (let i = 0; i < n; i++) {
            const a = facet[i], b = facet[(i + 1) % n];
            edgePts.push(...a, ...b);
        }
    }
    const faceGeom = new THREE.BufferGeometry();
    faceGeom.setAttribute('position', new THREE.Float32BufferAttribute(facePts, 3));
    scene.add(new THREE.Mesh(faceGeom, new THREE.MeshBasicMaterial({
        color: 0x2563eb, transparent: true, opacity: 0.07, side: THREE.DoubleSide, depthWrite: false,
    })));
    const edgeGeom = new THREE.BufferGeometry();
    edgeGeom.setAttribute('position', new THREE.Float32BufferAttribute(edgePts, 3));
    scene.add(new THREE.LineSegments(edgeGeom, new THREE.LineBasicMaterial({ color: 0x5b6470 })));

    // ── Reciprocal axes (b1,b2,b3) ──────────────────────────────────────
    const axLen = maxR * 1.25;
    for (let i = 0; i < 3; i++) {
        const v = data.recip_lattice[i];
        const n = Math.hypot(v[0], v[1], v[2]) || 1;
        const dir = new THREE.Vector3(v[0] / n, v[1] / n, v[2] / n);
        scene.add(new THREE.ArrowHelper(dir, new THREE.Vector3(0, 0, 0), axLen, 0xb0b0b0, axLen * 0.08, axLen * 0.05));
        const lab = makeLabel('b' + (i + 1), '#888', maxR * 0.22);
        lab.position.copy(dir.multiplyScalar(axLen * 1.08));
        scene.add(lab);
    }

    // ── High-symmetry points (clickable spheres + labels) ───────────────
    const ptRadius = maxR * 0.045;
    const sphere = new THREE.SphereGeometry(ptRadius, 20, 16);
    const pointMeshes = [];
    const pointByLabel = new Map();
    for (const p of data.high_sym_points) {
        const mat = new THREE.MeshStandardMaterial({ color: COLOR_FREE, roughness: 0.4 });
        const m = new THREE.Mesh(sphere, mat);
        m.position.set(p.cart[0], p.cart[1], p.cart[2]);
        m.userData = { point: p };
        scene.add(m);
        pointMeshes.push(m);
        pointByLabel.set(p.label, { point: p, mesh: m });

        const lab = makeLabel(showLabel(p.label), '#111', maxR * 0.26);
        lab.position.set(p.cart[0] * 1.12, p.cart[1] * 1.12 + ptRadius, p.cart[2] * 1.12);
        scene.add(lab);
    }

    // Camera framing
    camera.position.set(maxR * 2.0, maxR * 1.5, maxR * 2.4);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, 0, 0);
    controls.update();

    // ── Path controller (3c) ────────────────────────────────────────────
    const pathGroup = new THREE.Group();
    scene.add(pathGroup);
    const pathCtl = makePathController({
        data, pointByLabel, pointMeshes, pathGroup, maxR,
        readout: root.querySelector('#bz-readout'),
        listRoot: root.querySelector('#bz-path'),
    });
    pathCtl.loadDefault();   // pre-populate from seekpath suggested_path

    // ── Click selection (raycast) → extend path ─────────────────────────
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    let downXY = null;
    renderer.domElement.addEventListener('pointerdown', (ev) => { downXY = [ev.clientX, ev.clientY]; });
    renderer.domElement.addEventListener('pointerup', (ev) => {
        if (!downXY) return;
        const moved = Math.hypot(ev.clientX - downXY[0], ev.clientY - downXY[1]);
        downXY = null;
        if (moved > 5) return;   // ignore orbit-drags; only treat as a click if (nearly) stationary
        const rect = renderer.domElement.getBoundingClientRect();
        mouse.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
        raycaster.setFromCamera(mouse, camera);
        const hit = raycaster.intersectObjects(pointMeshes, false)[0];
        if (!hit) return;
        pathCtl.clickPoint(hit.object.userData.point);
    });

    // ── Loop + teardown ─────────────────────────────────────────────────
    let running = true;
    (function animate() {
        if (!running) return;
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    })();
    const onResize = () => {
        const w = container.clientWidth || W, h = container.clientHeight || H;
        camera.aspect = w / h; camera.updateProjectionMatrix(); renderer.setSize(w, h);
    };
    window.addEventListener('resize', onResize);
    const obs = new MutationObserver(() => {
        if (!document.body.contains(container)) {
            running = false;
            window.removeEventListener('resize', onResize);
            renderer.dispose();
            obs.disconnect();
        }
    });
    obs.observe(document.body, { childList: true, subtree: true });

    root._bz = { scene, camera, pointMeshes, maxR, THREE, pathCtl };
}

// ── Path controller ─────────────────────────────────────────────────────
// Owns the ordered segment list, the active "tip" we extend from, the 3D
// segment lines, point colors, the segment-list DOM, and state.kpath.
function makePathController({ data, pointByLabel, pointMeshes, pathGroup, maxR, readout, listRoot }) {
    let segments = [];        // [{ from, to, npoints, manual }]
    let tip = null;           // <pt> the next click extends from; null = start fresh
    let density = BASE_NPTS;  // target k-points for the LONGEST segment; shorter ones scale by length

    const cartLen = (a, b) =>
        Math.hypot(a.cart[0] - b.cart[0], a.cart[1] - b.cart[1], a.cart[2] - b.cart[2]);

    // Allocate npoints to every non-manual segment proportionally to its length,
    // so the longest segment gets `density` points. Manual (user-edited) segments
    // keep their value.
    function recompute() {
        const maxLen = Math.max(0, ...segments.map(s => cartLen(s.from, s.to)));
        for (const s of segments) {
            if (s.manual) continue;
            s.npoints = maxLen > 0
                ? Math.max(2, Math.round(density * cartLen(s.from, s.to) / maxLen))
                : 2;
        }
    }

    function loadDefault() {
        segments = [];
        for (const [a, b] of (data.suggested_path || [])) {
            const pa = pointByLabel.get(a), pb = pointByLabel.get(b);
            if (!pa || !pb) continue;
            segments.push({ from: pa.point, to: pb.point, npoints: 2, manual: false });
        }
        recompute();
        tip = segments.length ? segments[segments.length - 1].to : null;
        refresh();
    }

    function clickPoint(p) {
        if (!tip) { tip = p; refresh(); return; }       // anchor a fresh start
        if (p.label === tip.label) return;               // ignore re-click on the tip
        segments.push({ from: tip, to: p, npoints: 2, manual: false });
        recompute();
        tip = p;
        refresh();
    }

    function removeSegment(i) {
        segments.splice(i, 1);
        recompute();
        tip = segments.length ? segments[segments.length - 1].to : null;
        refresh();
    }

    function setNpoints(i, n) {
        segments[i].npoints = Math.max(2, Math.round(n) || 2);
        segments[i].manual = true;   // sticky: survives global density changes
        refresh();
    }

    function setDensity(n) {
        density = Math.max(2, Math.round(n) || 2);
        recompute();
        refresh();
    }

    function newBranch() { tip = null; refresh(); }      // next click starts a disconnected branch
    function clearPath() { segments = []; tip = null; refresh(); }

    // Redraw 3D lines, recolor points, rebuild the DOM list, push to state.
    function refresh() {
        // 3D path: bold tubes + directional arrowheads so traversal order is clear
        for (const c of [...pathGroup.children]) { c.geometry?.dispose(); pathGroup.remove(c); }
        const tubeR = maxR * 0.012;
        const segMat = new THREE.MeshStandardMaterial({ color: COLOR_SEG, roughness: 0.45 });
        const up = new THREE.Vector3(0, 1, 0);
        for (const s of segments) {
            const A = new THREE.Vector3(...s.from.cart);
            const B = new THREE.Vector3(...s.to.cart);
            const dir = new THREE.Vector3().subVectors(B, A);
            const len = dir.length();
            if (len < 1e-9) continue;
            dir.normalize();
            const quat = new THREE.Quaternion().setFromUnitVectors(up, dir);
            const shaft = new THREE.Mesh(new THREE.CylinderGeometry(tubeR, tubeR, len, 10), segMat);
            shaft.position.copy(A).addScaledVector(dir, len / 2);
            shaft.quaternion.copy(quat);
            pathGroup.add(shaft);
            // arrowhead at ~60% toward B (kept clear of the destination sphere)
            const head = new THREE.Mesh(new THREE.ConeGeometry(tubeR * 2.4, tubeR * 6, 12), segMat);
            head.position.copy(A).addScaledVector(dir, len * 0.6);
            head.quaternion.copy(quat);
            pathGroup.add(head);
        }

        // point colors
        const onPath = new Set();
        for (const s of segments) { onPath.add(s.from.label); onPath.add(s.to.label); }
        for (const m of pointMeshes) {
            const lbl = m.userData.point.label;
            const c = (tip && lbl === tip.label) ? COLOR_TIP : onPath.has(lbl) ? COLOR_PATH : COLOR_FREE;
            m.material.color.set(c);
        }

        renderList();
        const total = segments.reduce((a, s) => a + s.npoints, 0);
        if (readout) {
            readout.textContent = tip
                ? `Active tip: ${showLabel(tip.label)} — click another point to add a segment.`
                : 'Click a high-symmetry point to start a new branch.';
        }
        state.set('kpath', {
            density,
            segments: segments.map(s => ({
                from: s.from.label, to: s.to.label,
                from_frac_conv: s.from.frac_conv, to_frac_conv: s.to.frac_conv,
                npoints: s.npoints,
            })),
            totalPoints: total,
        });
    }

    function renderList() {
        const total = segments.reduce((a, s) => a + s.npoints, 0);
        const rows = segments.map((s, i) => {
            const isBreak = i > 0 && segments[i - 1].to.label !== s.from.label;
            return `
              ${isBreak ? '<div class="kp-break">— break —</div>' : ''}
              <div class="kp-row" data-i="${i}">
                <span class="kp-seg">${showLabel(s.from.label)} → ${showLabel(s.to.label)}</span>
                <label class="kp-npts">pts
                  <input type="number" min="2" step="1" value="${s.npoints}" data-i="${i}">
                </label>
                <button class="kp-del" data-i="${i}" title="Remove segment">×</button>
              </div>`;
        }).join('');

        listRoot.innerHTML = `
          <h3>k-path</h3>
          <div class="kp-toolbar">
            <button class="kp-default">Load seekpath default</button>
            <button class="kp-branch">New branch</button>
            <button class="kp-clear">Clear</button>
            <span class="kp-density" title="Scale k-points across the whole path proportionally to segment length. Lower = coarser, higher = finer. Per-segment edits are kept.">
              k-point density
              <button class="kp-dens-dn" title="Fewer k-points">−</button>
              <button class="kp-dens-up" title="More k-points">+</button>
            </span>
          </div>
          <div class="kp-list">${rows || '<div class="muted">No segments. Click a point or load the default path.</div>'}</div>
          <div class="kp-total muted">${segments.length} segment(s) · <strong>${total}</strong> k-points total</div>
        `;

        listRoot.querySelector('.kp-default').onclick = loadDefault;
        listRoot.querySelector('.kp-branch').onclick = newBranch;
        listRoot.querySelector('.kp-clear').onclick = clearPath;
        const DSTEP = 5;
        listRoot.querySelector('.kp-dens-dn').onclick = () => setDensity(density - DSTEP);
        listRoot.querySelector('.kp-dens-up').onclick = () => setDensity(density + DSTEP);
        listRoot.querySelectorAll('.kp-del').forEach(b => {
            b.onclick = () => removeSegment(Number(b.dataset.i));
        });
        listRoot.querySelectorAll('.kp-npts input').forEach(inp => {
            inp.onchange = () => setNpoints(Number(inp.dataset.i), Number(inp.value));
        });
    }

    return { loadDefault, clickPoint, removeSegment, setNpoints, setDensity, newBranch, clearPath, refresh };
}

function makeLabel(text, color, worldSize) {
    const px = 128;
    const c = document.createElement('canvas');
    c.width = px; c.height = px;
    const ctx = c.getContext('2d');
    ctx.font = 'bold 80px Inter, system-ui, sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillStyle = color;
    ctx.fillText(text, px / 2, px / 2);
    const tex = new THREE.CanvasTexture(c);
    tex.needsUpdate = true;
    const spr = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
    spr.scale.set(worldSize, worldSize, 1);
    return spr;
}
