import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { conventionalLattice } from '../math/reciprocal';
import { TWO_PI_PHASE, DEFAULT_COLORS, COVALENT_R } from '../constants';

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
  showBonds = true, bondScale = 1.15, shading = true, recording = false,
}) {
  const mountRef = useRef(null);
  const objs = useRef(null);
  const params = useRef({ isPlaying, amplitude, speed, eigenvector, qPoint });
  const recRef = useRef(null);
  const view = useRef(null);     // saved camera {pos,target} preserved across rebuilds

  useEffect(() => { params.current = { isPlaying, amplitude, speed, eigenvector, qPoint }; },
    [isPlaying, amplitude, speed, eigenvector, qPoint]);

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
      mr.onstop = () => {
        const blob = new Blob(chunks, { type: 'video/webm' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob); a.download = 'mode.webm';
        document.body.appendChild(a); a.click(); document.body.removeChild(a);
        URL.revokeObjectURL(a.href);
      };
      mr.start();
      recRef.current = mr;
    } else if (!recording && recRef.current) {
      recRef.current.stop();
      recRef.current = null;
    }
  }, [recording]);

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
        if (showVectors && nSites * nx * ny * nz <= 2000) {
          arrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(...r0), 0.001, 0x22d3ee, undefined, 0.4);
          scene.add(arrow);
        }
        atoms.push({ mesh, arrow, r0, el, row: rnToRow.get(rn) ?? 0, cell: [cx, cy, cz] });
        center.add(mesh.position);
      }
    }
    if (atoms.length) center.divideScalar(atoms.length);

    // Bonds (line segments, covalent-radius cutoff × bondScale). Updated per frame.
    let bondLines = null, bondPairs = [];
    if (bondsOn && atoms.length <= 1600) {
      for (let i = 0; i < atoms.length; i++) for (let j = i + 1; j < atoms.length; j++) {
        const cut = bondScale * (defRadius(atoms[i].el) + defRadius(atoms[j].el));
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
    controls.target.copy(center);
    camera.position.copy(center).add(new THREE.Vector3(span * 0.4, span * 0.3, span * 1.1 + 5));
    // Restore the previous view so appearance tweaks don't reset the camera.
    if (view.current) { camera.position.copy(view.current.pos); controls.target.copy(view.current.target); controls.update(); }
    controls.addEventListener('change', () => { view.current = { pos: camera.position.clone(), target: controls.target.clone() }; });
    objs.current = { camera, controls, span, center, renderer };

    let t = 0, animId;
    const up = new THREE.Vector3();
    const animate = () => {
      animId = requestAnimationFrame(animate);
      const P = params.current;
      const ev = P.eigenvector;
      if (P.isPlaying && ev && ev.real) {
        t += P.speed;
        const k = P.qPoint ? [P.qPoint[0] * TWO_PI_PHASE, P.qPoint[1] * TWO_PI_PHASE, P.qPoint[2] * TWO_PI_PHASE] : [0, 0, 0];
        for (const at of atoms) {
          const r = at.row;
          if (r * 3 + 2 >= ev.real.length) continue;
          const kn = k[0] * at.cell[0] + k[1] * at.cell[1] + k[2] * at.cell[2];
          const cp = Math.cos(kn + t), sp = Math.sin(kn + t);
          const dx = (ev.real[r * 3] * cp - ev.imag[r * 3] * sp) * P.amplitude;
          const dy = (ev.real[r * 3 + 1] * cp - ev.imag[r * 3 + 1] * sp) * P.amplitude;
          const dz = (ev.real[r * 3 + 2] * cp - ev.imag[r * 3 + 2] * sp) * P.amplitude;
          at.mesh.position.set(at.r0[0] + dx, at.r0[1] + dy, at.r0[2] + dz);
          if (at.arrow) {
            const len = Math.hypot(dx, dy, dz);
            if (len > 1e-4) { up.set(dx / len, dy / len, dz / len); at.arrow.setDirection(up); at.arrow.setLength(len, Math.min(0.3, len * 0.4), Math.min(0.2, len * 0.25)); }
            at.arrow.position.set(...at.r0);
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
    };
    animate();

    const onResize = () => {
      if (!mountRef.current) return;
      const w = mountRef.current.clientWidth, h = mountRef.current.clientHeight;
      camera.aspect = w / h; camera.updateProjectionMatrix(); renderer.setSize(w, h);
    };
    window.addEventListener('resize', onResize);
    return () => { window.removeEventListener('resize', onResize); cancelAnimationFrame(animId); renderer.dispose(); };
  }, [baseStructure, supercell, showVectors, showCell, atomScale, elementColors, elementRadii, displayStyle, showBonds, bondScale, shading]);

  return <div ref={mountRef} className="w-full h-full min-h-[360px] cursor-move rounded-xl overflow-hidden" />;
}
