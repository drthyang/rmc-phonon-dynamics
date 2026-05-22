// Phase 3b/3c — reciprocal cell, Brillouin zone & k-path selection.
// 3b: render the BZ (edges + translucent faces) with labeled, clickable
//     high-symmetry points. 3c (later) adds path assembly + per-segment counts.
'use strict';

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { api } from '../api.js';
import { state } from '../state.js';

const GAMMA = 'Γ';
const showLabel = (l) => (l === 'GM' || l === 'GAMMA' || l === 'G') ? GAMMA : l;

export async function mountBZView(root, _opts = {}) {
    root.innerHTML = `
      <section class="panel">
        <h2>3 · Reciprocal cell & k-path</h2>
        <div id="bz-info" class="muted"></div>
        <div id="bz-canvas" class="canvas3d"></div>
        <div id="bz-readout" class="muted">Click a high-symmetry point to select it.</div>
      </section>
    `;
    const info = root.querySelector('#bz-info');
    try {
        const data = await api.getReciprocal();
        state.set('reciprocal', data);
        info.textContent =
            `${data.crystal_system} · ${data.spacegroup} · ${data.high_sym_points.length} high-symmetry points`;
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
    for (const p of data.high_sym_points) {
        const mat = new THREE.MeshStandardMaterial({ color: 0x16a34a, roughness: 0.4 });
        const m = new THREE.Mesh(sphere, mat);
        m.position.set(p.cart[0], p.cart[1], p.cart[2]);
        m.userData = { point: p, selected: false, baseColor: 0x16a34a };
        scene.add(m);
        pointMeshes.push(m);

        const lab = makeLabel(showLabel(p.label), '#111', maxR * 0.26);
        lab.position.set(p.cart[0] * 1.12, p.cart[1] * 1.12 + ptRadius, p.cart[2] * 1.12);
        scene.add(lab);
    }

    // Camera framing
    const cam0 = new THREE.Vector3(maxR * 2.0, maxR * 1.5, maxR * 2.4);
    camera.position.copy(cam0);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, 0, 0);
    controls.update();

    // ── Click selection (raycast) ───────────────────────────────────────
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    const readout = root.querySelector('#bz-readout');

    renderer.domElement.addEventListener('pointerdown', (ev) => {
        const rect = renderer.domElement.getBoundingClientRect();
        mouse.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
        raycaster.setFromCamera(mouse, camera);
        const hit = raycaster.intersectObjects(pointMeshes, false)[0];
        if (!hit) return;
        const m = hit.object;
        // 3b behavior: highlight the clicked point + show its info.
        // 3c will turn this into ordered path selection.
        pointMeshes.forEach(pm => { pm.material.color.set(pm.userData.baseColor); pm.userData.selected = false; });
        m.material.color.set(0xf59e0b);
        m.userData.selected = true;
        const p = m.userData.point;
        readout.textContent =
            `Selected ${showLabel(p.label)}  ·  frac (${p.frac.map(x => x.toFixed(3)).join(', ')})`;
        if (typeof root._onPointClick === 'function') root._onPointClick(p);
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

    // Expose handles so Phase 3c can drive path drawing on this scene.
    root._bz = { scene, camera, pointMeshes, maxR, THREE };
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
