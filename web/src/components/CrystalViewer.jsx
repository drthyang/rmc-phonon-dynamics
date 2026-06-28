import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import GIF from 'gif.js';
import gifWorkerUrl from 'gif.js/dist/gif.worker.js?url';
import { conventionalLattice } from '../math/reciprocal';
import { TWO_PI_PHASE, DEFAULT_COLORS, COVALENT_R } from '../constants';

function downloadBlob(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = name;
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  URL.revokeObjectURL(a.href);
}

const defColor = (el) => DEFAULT_COLORS[el] || '#cccccc';
const defRadius = (el) => COVALENT_R[el] || 1.0;

/**
 * 3D phonon-mode viewer (unit-cell model, tiled by supercell). Full appearance
 * controls: per-element color/radius, display style (ball-and-stick / spacefill /
 * wireframe), bonds (covalent-radius cutoff × scale), shading, displacement
 * vectors, unit-cell wireframe, camera presets. WebM capture via `recording`.
 */
export default function CrystalViewer({
  baseStructure, eigenvector, qPoint,
  isPlaying = true, amplitude = 2.0, speed = 0.08,
  supercell = [2, 2, 1], showVectors = false, showCell = true, atomScale = 1.0, cameraAxis = null,
  elementColors = {}, elementRadii = {}, displayStyle = 'ballstick',
  showBonds = true, bondScale = 1.15, bondRules = {}, shading = true, recording = false, gifSignal = 0,
  vectorScale = 1.5,
}) {
  const mountRef = useRef(null);
  const objs = useRef(null);
  const params = useRef({ isPlaying, amplitude, speed, eigenvector, qPoint });
  const recRef = useRef(null);
  const gifRef = useRef(null);   // { active, frames, gif } during GIF capture
  const view = useRef(null);     // saved camera {pos,target} preserved across rebuilds

  useEffect(() => { params.current = { isPlaying, amplitude, speed, eigenvector, qPoint, vectorScale }; },
    [isPlaying, amplitude, speed, eigenvector, qPoint, vectorScale]);

  // Camera preset.
  useEffect(() => {
    if (!objs.current || !cameraAxis) return;
    const { camera, controls, span, center } = objs.current;
    const d = span * 1.2 + 6;
    if (cameraAxis === 'x') camera.position.set(center.x + d, center.y, center.z);
    if (cameraAxis === 'y') camera.position.set(center.x, center.y + d, center.z);
    if (cameraAxis === 'z') camera.position.set(center.x, center.y, center.z + d);
    camera.lookAt(center); controls.target.copy(center); controls.update();
    view.current = { pos: camera.position.clone(), target: controls.target.clone() };
  }, [cameraAxis]);

  // WebM recording of the canvas.
  useEffect(() => {
    if (!objs.current) return;
    if (recording && !recRef.current) {
      const canvas = objs.current.renderer.domElement;
      const stream = canvas.captureStream(30);
      const chunks = [];
      let mr;
      try { mr = new MediaRecorder(stream, { mimeType: 'video/webm' }); }
      catch { return; }
      mr.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
      mr.onstop = () => downloadBlob(new Blob(chunks, { type: 'video/webm' }), 'mode.webm');
      mr.start();
      recRef.current = mr;
    } else if (!recording && recRef.current) {
      recRef.current.stop();
      recRef.current = null;
    }
  }, [recording]);

  // GIF capture: start a ~50-frame grab; the animation loop adds frames.
  useEffect(() => {
    if (!gifSignal || !objs.current || gifRef.current?.active) return;
    const canvas = objs.current.renderer.domElement;
    const gif = new GIF({ workers: 2, quality: 10, workerScript: gifWorkerUrl, width: canvas.width, height: canvas.height });
    gif.on('finished', (blob) => { downloadBlob(blob, 'mode.gif'); gifRef.current = null; });
    gifRef.current = { active: true, frames: 0, gif };
  }, [gifSignal]);

  useEffect(() => {
    if (!mountRef.current || !baseStructure || !baseStructure.hsym_xyz) return;
    const { v1, v2, v3, dim, hsym_xyz, atomType, uniqueRN, atomDic } = baseStructure;
    const [nx, ny, nz] = supercell;
    const A = conventionalLattice(v1, v2, v3, dim);
    const rnToRow = new Map((uniqueRN || []).map((rn, r) => [rn, r]));
    const reverseAtomDic = {};
    for (const [el, idxs] of Object.entries(atomDic)) idxs.forEach(i => { reverseAtomDic[i] = el; });
    const colorOf = (el) => new THREE.Color(elementColors[el] || defColor(el));
    const radiusOf = (el) => (elementRadii[el] || defRadius(el));

    const scene = new THREE.Scene();
    const width = mountRef.current.clientWidth, height = mountRef.current.clientHeight;
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 8000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    renderer.setSize(width, height); renderer.setPixelRatio(window.devicePixelRatio);
    mountRef.current.innerHTML = '';
    mountRef.current.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    scene.add(new THREE.AmbientLight(0xffffff, shading ? 0.6 : 1.0));
    if (shading) { const dl = new THREE.DirectionalLight(0xffffff, 0.8); dl.position.set(10, 20, 10); scene.add(dl); }

    const matvec = (f) => [
      f[0] * A[0][0] + f[1] * A[1][0] + f[2] * A[2][0],
      f[0] * A[0][1] + f[1] * A[1][1] + f[2] * A[2][1],
      f[0] * A[0][2] + f[1] * A[1][2] + f[2] * A[2][2],
    ];

    // Per-display-style sphere radius factor.
    const styleFactor = displayStyle === 'spacefill' ? 1.0 : displayStyle === 'wireframe' ? 0.10 : 0.32;
    const bondsOn = (showBonds || displayStyle === 'wireframe') && displayStyle !== 'spacefill';

    const nSites = hsym_xyz.length / 3;
    const stride = Math.max(1, Math.floor((nSites * nx * ny * nz) / 4000));
    const atoms = [];
    const center = new THREE.Vector3();

    // Shared geometries per element radius (cache by rounded radius).
    const geoCache = new Map();
    const getGeo = (rad) => {
      const key = rad.toFixed(2);
      if (!geoCache.has(key)) geoCache.set(key, new THREE.SphereGeometry(rad, 18, 18));
      return geoCache.get(key);
    };

    // Displacement-vector arrows: solid shaft (cylinder) + cone head, sized in
    // the animation loop. Unit geometry along +Y; base of the cylinder at origin.
    const showArrows = showVectors && nSites * nx * ny * nz <= 1500;
    let shaftGeo, headGeo, vecMat;
    if (showArrows) {
      shaftGeo = new THREE.CylinderGeometry(1, 1, 1, 10); shaftGeo.translate(0, 0.5, 0);
      headGeo = new THREE.ConeGeometry(1, 1, 14);
      vecMat = new THREE.MeshBasicMaterial({ color: 0xfde047 });
    }

    for (let cx = 0; cx < nx; cx++) for (let cy = 0; cy < ny; cy++) for (let cz = 0; cz < nz; cz++) {
      for (let s = 0; s < nSites; s += stride) {
        const rn = atomType[s];
        const el = reverseAtomDic[rn] || 'H';
        const r0 = matvec([hsym_xyz[s * 3] + cx, hsym_xyz[s * 3 + 1] + cy, hsym_xyz[s * 3 + 2] + cz]);
        const rad = radiusOf(el) * styleFactor * atomScale;
        const mat = shading
          ? new THREE.MeshPhongMaterial({ color: colorOf(el), shininess: 70 })
          : new THREE.MeshBasicMaterial({ color: colorOf(el) });
        const mesh = new THREE.Mesh(getGeo(rad), mat);
        mesh.position.set(...r0);
        scene.add(mesh);
        let arrow = null;
        if (showArrows) {
          const g = new THREE.Group();
          const shaft = new THREE.Mesh(shaftGeo, vecMat);
          const head = new THREE.Mesh(headGeo, vecMat);
          g.add(shaft); g.add(head); g.visible = false;
          scene.add(g);
          arrow = { g, shaft, head };
        }
        atoms.push({ mesh, arrow, r0, el, row: rnToRow.get(rn) ?? 0, cell: [cx, cy, cz], vd: [0, 0, 0], vlen: 0 });
        center.add(mesh.position);
      }
    }
    if (atoms.length) center.divideScalar(atoms.length);

    // Bonds (line segments, covalent-radius cutoff × bondScale). Updated per frame.
    let bondLines = null, bondPairs = [];
    if (bondsOn && atoms.length <= 1600) {
      for (let i = 0; i < atoms.length; i++) for (let j = i + 1; j < atoms.length; j++) {
        const key = [atoms[i].el, atoms[j].el].sort().join('-');
        const cut = (bondRules[key] != null) ? bondRules[key] : bondScale * (defRadius(atoms[i].el) + defRadius(atoms[j].el));
        const dx = atoms[i].r0[0] - atoms[j].r0[0], dy = atoms[i].r0[1] - atoms[j].r0[1], dz = atoms[i].r0[2] - atoms[j].r0[2];
        if (dx * dx + dy * dy + dz * dz <= cut * cut) bondPairs.push([i, j]);
      }
      const pos = new Float32Array(bondPairs.length * 6);
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      bondLines = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0x9aa4b2 }));
      scene.add(bondLines);
    }

    if (showCell) {
      const corners = [];
      for (const i of [0, 1]) for (const j of [0, 1]) for (const k of [0, 1]) corners.push(matvec([i * nx, j * ny, k * nz]));
      const e = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]];
      const pts = [];
      for (const [a, b] of e) pts.push(new THREE.Vector3(...corners[a]), new THREE.Vector3(...corners[b]));
      scene.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: 0x6366f1, transparent: true, opacity: 0.5 })));
    }

    const span = Math.hypot(...matvec([nx, ny, nz]));
    // Arrow visual sizes (world units, scaled to the cell extent).
    const aShaftR = span * 0.006, aHeadR = span * 0.018, aHeadH = span * 0.05;
    controls.target.copy(center);
    camera.position.copy(center).add(new THREE.Vector3(span * 0.4, span * 0.3, span * 1.1 + 5));
    // Restore the previous view so appearance tweaks don't reset the camera.
    if (view.current) { camera.position.copy(view.current.pos); controls.target.copy(view.current.target); controls.update(); }
    controls.addEventListener('change', () => { view.current = { pos: camera.position.clone(), target: controls.target.clone() }; });
    objs.current = { camera, controls, span, center, renderer };

    let t = 0, animId;
    const up = new THREE.Vector3();
    const Y0 = new THREE.Vector3(0, 1, 0);
    const animate = () => {
      animId = requestAnimationFrame(animate);
      const P = params.current;
      const ev = P.eigenvector;
      const k = P.qPoint ? [P.qPoint[0] * TWO_PI_PHASE, P.qPoint[1] * TWO_PI_PHASE, P.qPoint[2] * TWO_PI_PHASE] : [0, 0, 0];
      // Advance the animation phase only while playing (paused = frozen frame,
      // but atoms/arrows are still placed at the current phase below).
      if (P.isPlaying) t += P.speed;

      if (ev && ev.real) {
        for (const at of atoms) {
          const r = at.row;
          if (r * 3 + 2 >= ev.real.length) continue;
          // INSTANTANEOUS displacement u(t) = Re(e · e^{i(k·n + t)}) — exactly the
          // vector the atom is moving along right now (phononwebsite convention).
          const kn = k[0] * at.cell[0] + k[1] * at.cell[1] + k[2] * at.cell[2];
          const cp = Math.cos(kn + t), sp = Math.sin(kn + t);
          const ux = ev.real[r * 3] * cp - ev.imag[r * 3] * sp;
          const uy = ev.real[r * 3 + 1] * cp - ev.imag[r * 3 + 1] * sp;
          const uz = ev.real[r * 3 + 2] * cp - ev.imag[r * 3 + 2] * sp;
          const dx = ux * P.amplitude, dy = uy * P.amplitude, dz = uz * P.amplitude;
          at.mesh.position.set(at.r0[0] + dx, at.r0[1] + dy, at.r0[2] + dz);

          if (at.arrow) {
            // Arrow = the SAME instantaneous displacement, so every moving atom
            // has a matching vector that oscillates with it. Anchored at rest.
            const len = Math.hypot(dx, dy, dz);
            const L = len * P.vectorScale;
            if (L > 1e-4 * span) {
              at.arrow.g.visible = true;
              const hH = Math.min(aHeadH, L * 0.45);
              const hR = aHeadR * (hH / aHeadH);
              const shaftLen = Math.max(L - hH, L * 0.02);
              at.arrow.shaft.scale.set(aShaftR, shaftLen, aShaftR);
              at.arrow.head.scale.set(hR, hH, hR);
              at.arrow.head.position.set(0, shaftLen + hH / 2, 0);
              up.set(dx / len, dy / len, dz / len);
              at.arrow.g.quaternion.setFromUnitVectors(Y0, up);
              at.arrow.g.position.set(at.r0[0], at.r0[1], at.r0[2]);
            } else { at.arrow.g.visible = false; }
          }
        }
        if (bondLines) {
          const pos = bondLines.geometry.attributes.position.array;
          for (let b = 0; b < bondPairs.length; b++) {
            const [i, j] = bondPairs[b];
            const pi = atoms[i].mesh.position, pj = atoms[j].mesh.position;
            pos[b * 6] = pi.x; pos[b * 6 + 1] = pi.y; pos[b * 6 + 2] = pi.z;
            pos[b * 6 + 3] = pj.x; pos[b * 6 + 4] = pj.y; pos[b * 6 + 5] = pj.z;
          }
          bondLines.geometry.attributes.position.needsUpdate = true;
        }
      }

      controls.update();
      renderer.render(scene, camera);
      const G = gifRef.current;
      if (G && G.active) {
        G.gif.addFrame(renderer.domElement, { copy: true, delay: 40 });
        if (++G.frames >= 50) { G.active = false; G.gif.render(); }
      }
    };
    animate();

    const onResize = () => {
      if (!mountRef.current) return;
      const w = mountRef.current.clientWidth, h = mountRef.current.clientHeight;
      camera.aspect = w / h; camera.updateProjectionMatrix(); renderer.setSize(w, h);
    };
    window.addEventListener('resize', onResize);
    return () => { window.removeEventListener('resize', onResize); cancelAnimationFrame(animId); renderer.dispose(); };
  }, [baseStructure, supercell, showVectors, showCell, atomScale, elementColors, elementRadii, displayStyle, showBonds, bondScale, bondRules, shading]);

  return <div ref={mountRef} className="w-full h-full min-h-[360px] cursor-move rounded-xl overflow-hidden" />;
}
