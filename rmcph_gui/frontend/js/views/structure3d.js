// Phase 2 — crystal structure 3D view.
// Renders the unit cell (lattice box + element-colored atom spheres) with
// three.js + OrbitControls. Geometry comes from GET /api/structure.
'use strict';

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { api } from '../api.js';

// Minimal CPK-ish element styling; unknowns fall back to grey.
const ELEMENTS = {
    Ga: { color: 0xc28f8f, radius: 0.62 },
    Ta: { color: 0x4da6ff, radius: 0.70 },
    Se: { color: 0xffa100, radius: 0.55 },
    O:  { color: 0xff3030, radius: 0.45 },
    Ti: { color: 0xbfc2c7, radius: 0.62 },
    Ba: { color: 0x00c900, radius: 0.80 },
    Sr: { color: 0x00ff00, radius: 0.78 },
    Pb: { color: 0x575961, radius: 0.78 },
};
const DEFAULT_EL = { color: 0xb0b0b0, radius: 0.55 };

export async function mountStructureView(root, _dataset, opts = {}) {
    root.innerHTML = `
      <section class="panel">
        <h2><span class="step-badge">2</span>Crystal structure</h2>
        <div class="controls">
          <label class="chk"><input type="checkbox" id="st-bonds" checked> Bonds</label>
          <label class="chk"><input type="checkbox" id="st-cell" checked> Cell</label>
        </div>
        <div id="st-canvas" class="canvas3d"></div>
        <div id="st-legend" class="legend"></div>
        <div id="st-msg" class="muted"></div>
        <div class="next">
          <button id="st-continue" class="primary">Continue to k-path →</button>
        </div>
      </section>
    `;
    const contBtn = root.querySelector('#st-continue');
    if (contBtn) contBtn.addEventListener('click', () => { if (opts.onContinue) opts.onContinue(); });
    const msg = root.querySelector('#st-msg');
    msg.textContent = 'loading structure…';

    let data;
    try {
        data = await api.getStructure();
    } catch (err) {
        msg.innerHTML = `<span class="err">✗ ${err.message}</span>`;
        return;
    }

    msg.textContent = `${data.natom} atoms · ${data.nbond ?? 0} bonds / unit cell · source: ${data.source}`;
    renderLegend(root.querySelector('#st-legend'), data.atoms);

    try {
        renderScene(root.querySelector('#st-canvas'), data, root);
    } catch (err) {
        msg.innerHTML = `<span class="err">✗ 3D render failed: ${err.message}</span>`;
        console.error(err);
    }
}

function elStyle(sym) { return ELEMENTS[sym] || DEFAULT_EL; }

function fracToCart(L, f) {
    return new THREE.Vector3(
        f[0] * L[0][0] + f[1] * L[1][0] + f[2] * L[2][0],
        f[0] * L[0][1] + f[1] * L[1][1] + f[2] * L[2][1],
        f[0] * L[0][2] + f[1] * L[1][2] + f[2] * L[2][2],
    );
}

function renderLegend(el, atoms) {
    const syms = [...new Set(atoms.map(a => a.symbol))];
    el.innerHTML = syms.map(s => {
        const c = '#' + elStyle(s).color.toString(16).padStart(6, '0');
        return `<span class="leg"><i style="background:${c}"></i>${s}</span>`;
    }).join('');
}

function renderScene(container, data, root) {
    const L = data.lattice;
    const W = container.clientWidth || 600;
    const H = container.clientHeight || 380;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf4f4f5);

    const camera = new THREE.PerspectiveCamera(45, W / H, 0.01, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(W, H);
    container.innerHTML = '';
    container.appendChild(renderer.domElement);

    // Cell center (frac 0.5,0.5,0.5) for camera target + framing
    const center = fracToCart(L, [0.5, 0.5, 0.5]);
    const aLen = fracToCart(L, [1, 0, 0]).length();
    const bLen = fracToCart(L, [0, 1, 0]).length();
    const cLen = fracToCart(L, [0, 0, 1]).length();
    const maxDim = Math.max(aLen, bLen, cLen);

    camera.position.copy(center).add(new THREE.Vector3(maxDim * 1.6, maxDim * 1.1, maxDim * 1.8));

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.copy(center);
    controls.enableDamping = true;
    controls.update();

    scene.add(new THREE.AmbientLight(0xffffff, 0.65));
    const dir = new THREE.DirectionalLight(0xffffff, 0.8);
    dir.position.set(1, 1.5, 2);
    scene.add(dir);

    // ── Unit cell box (12 edges) ────────────────────────────────────────
    const corners = [];
    for (const i of [0, 1]) for (const j of [0, 1]) for (const k of [0, 1])
        corners.push(fracToCart(L, [i, j, k]));
    const idx = (i, j, k) => (i * 4 + j * 2 + k);
    const edges = [
        [0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1],
        [1,1,1,0,1,1],[1,1,1,1,0,1],[1,1,1,1,1,0],
        [1,0,0,1,1,0],[1,0,0,1,0,1],[0,1,0,1,1,0],
        [0,1,0,0,1,1],[0,0,1,1,0,1],[0,0,1,0,1,1],
    ];
    const lineGeom = new THREE.BufferGeometry();
    const pts = [];
    for (const e of edges) {
        const p1 = corners[idx(e[0], e[1], e[2])];
        const p2 = corners[idx(e[3], e[4], e[5])];
        pts.push(p1.x, p1.y, p1.z, p2.x, p2.y, p2.z);
    }
    lineGeom.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
    const cellBox = new THREE.LineSegments(lineGeom,
        new THREE.LineBasicMaterial({ color: 0x888888 }));
    scene.add(cellBox);

    // ── Bonds (cylinders) ───────────────────────────────────────────────
    const bondGroup = new THREE.Group();
    const bondMat = new THREE.MeshStandardMaterial({ color: 0x9aa0a6, roughness: 0.6 });
    const bondGeom = new THREE.CylinderGeometry(0.09, 0.09, 1, 8);
    const Y = new THREE.Vector3(0, 1, 0);
    for (const b of (data.bonds || [])) {
        const p1 = fracToCart(L, b.a);
        const p2 = fracToCart(L, b.b);
        const dir = new THREE.Vector3().subVectors(p2, p1);
        const len = dir.length();
        if (len < 1e-6) continue;
        const m = new THREE.Mesh(bondGeom, bondMat);
        m.position.copy(p1).add(p2).multiplyScalar(0.5);
        m.quaternion.setFromUnitVectors(Y, dir.clone().normalize());
        m.scale.set(1, len, 1);
        bondGroup.add(m);
    }
    scene.add(bondGroup);

    // ── Atoms ───────────────────────────────────────────────────────────
    const sphere = new THREE.SphereGeometry(1, 24, 18);
    for (const atom of data.atoms) {
        const st = elStyle(atom.symbol);
        const mesh = new THREE.Mesh(
            sphere,
            new THREE.MeshStandardMaterial({ color: st.color, roughness: 0.45, metalness: 0.1 }),
        );
        mesh.scale.setScalar(st.radius);
        mesh.position.copy(fracToCart(L, atom.frac));
        scene.add(mesh);
    }

    // ── Toggles ─────────────────────────────────────────────────────────
    const bondsChk = root.querySelector('#st-bonds');
    const cellChk  = root.querySelector('#st-cell');
    if (bondsChk) bondsChk.addEventListener('change', () => { bondGroup.visible = bondsChk.checked; });
    if (cellChk)  cellChk.addEventListener('change',  () => { cellBox.visible   = cellChk.checked; });

    // ── Loop + resize ───────────────────────────────────────────────────
    let running = true;
    (function animate() {
        if (!running) return;
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    })();

    const onResize = () => {
        const w = container.clientWidth || W, h = container.clientHeight || H;
        camera.aspect = w / h; camera.updateProjectionMatrix();
        renderer.setSize(w, h);
    };
    window.addEventListener('resize', onResize);

    // Stop if the container leaves the DOM (view swapped out).
    const obs = new MutationObserver(() => {
        if (!document.body.contains(container)) {
            running = false;
            window.removeEventListener('resize', onResize);
            renderer.dispose();
            obs.disconnect();
        }
    });
    obs.observe(document.body, { childList: true, subtree: true });
}
