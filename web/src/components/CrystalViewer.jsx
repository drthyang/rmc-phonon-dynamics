import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { conventionalLattice } from '../math/reciprocal';
import { TWO_PI_PHASE } from '../constants';

const COLOR_MAP = {
  H: 0xffffff, O: 0xff0000, C: 0x444444, N: 0x0000ff,
  Pb: 0x333333, Te: 0xd4aa00, Se: 0xff9900, S: 0xffff00,
  Ga: 0xa67e5b, Ta: 0x4da6ff,
};
const DEFAULT_COLOR = 0xcccccc;

/**
 * 3D phonon-mode viewer (unit-cell model, tiled by supercell nx,ny,nz).
 *
 * Atom positions: r = (cell + within-cell-frac) @ A. The selected mode's
 * eigenvector (rows by reference number) animates each atom with the per-cell
 * Bloch phase: u(t) = Re(e_site · exp(i(k·n + ω t))) · amplitude. Optional
 * displacement arrows, unit-cell wireframe, camera presets, and atom scaling.
 */
export default function CrystalViewer({
  baseStructure, eigenvector, qPoint,
  isPlaying = true, amplitude = 2.0, speed = 0.08,
  supercell = [2, 2, 1], showVectors = false, showCell = true,
  atomScale = 1.0, cameraAxis = null,
}) {
  const mountRef = useRef(null);
  const objs = useRef(null);
  const params = useRef({ isPlaying, amplitude, speed, eigenvector, qPoint });

  // Live params for the animation loop (no scene rebuild).
  useEffect(() => { params.current = { isPlaying, amplitude, speed, eigenvector, qPoint }; },
    [isPlaying, amplitude, speed, eigenvector, qPoint]);

  // Camera preset.
  useEffect(() => {
    if (!objs.current || !cameraAxis) return;
    const { camera, controls, span, center } = objs.current;
    const d = span * 1.2 + 6;
    const t = center;
    if (cameraAxis === 'x') camera.position.set(t.x + d, t.y, t.z);
    if (cameraAxis === 'y') camera.position.set(t.x, t.y + d, t.z);
    if (cameraAxis === 'z') camera.position.set(t.x, t.y, t.z + d);
    camera.lookAt(t); controls.target.copy(t); controls.update();
  }, [cameraAxis]);

  useEffect(() => {
    if (!mountRef.current || !baseStructure || !baseStructure.hsym_xyz) return;
    const { v1, v2, v3, dim, hsym_xyz, atomType, uniqueRN, atomDic } = baseStructure;
    const [nx, ny, nz] = supercell;
    const A = conventionalLattice(v1, v2, v3, dim);

    const rnToRow = new Map((uniqueRN || []).map((rn, r) => [rn, r]));
    const reverseAtomDic = {};
    for (const [el, idxs] of Object.entries(atomDic)) idxs.forEach(i => { reverseAtomDic[i] = el; });

    const scene = new THREE.Scene();
    const width = mountRef.current.clientWidth, height = mountRef.current.clientHeight;
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 8000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height); renderer.setPixelRatio(window.devicePixelRatio);
    mountRef.current.innerHTML = '';
    mountRef.current.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    scene.add(new THREE.AmbientLight(0xffffff, 0.65));
    const dl = new THREE.DirectionalLight(0xffffff, 0.8); dl.position.set(10, 20, 10); scene.add(dl);

    const matvec = (f) => [
      f[0] * A[0][0] + f[1] * A[1][0] + f[2] * A[2][0],
      f[0] * A[0][1] + f[1] * A[1][1] + f[2] * A[2][1],
      f[0] * A[0][2] + f[1] * A[1][2] + f[2] * A[2][2],
    ];

    const nSites = hsym_xyz.length / 3;
    const sphereGeo = new THREE.SphereGeometry(0.4 * atomScale, 18, 18);
    const atoms = [];
    const center = new THREE.Vector3();
    const showArrows = showVectors && nSites * nx * ny * nz <= 2000;

    for (let cx = 0; cx < nx; cx++) for (let cy = 0; cy < ny; cy++) for (let cz = 0; cz < nz; cz++) {
      for (let s = 0; s < nSites; s++) {
        const rn = atomType[s];
        const el = reverseAtomDic[rn] || 'H';
        const fr = [hsym_xyz[s * 3] + cx, hsym_xyz[s * 3 + 1] + cy, hsym_xyz[s * 3 + 2] + cz];
        const r0 = matvec(fr);
        const mesh = new THREE.Mesh(sphereGeo, new THREE.MeshPhongMaterial({ color: COLOR_MAP[el] || DEFAULT_COLOR, shininess: 80 }));
        mesh.position.set(r0[0], r0[1], r0[2]);
        scene.add(mesh);
        let arrow = null;
        if (showArrows) {
          arrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(r0[0], r0[1], r0[2]), 0.001, 0x22d3ee, undefined, 0.4);
          scene.add(arrow);
        }
        atoms.push({ mesh, arrow, r0, row: rnToRow.get(rn) ?? 0, cell: [cx, cy, cz] });
        center.add(mesh.position);
      }
    }
    if (atoms.length) center.divideScalar(atoms.length);

    // Unit cell / supercell wireframe.
    if (showCell) {
      const corners = [];
      for (const i of [0, 1]) for (const j of [0, 1]) for (const k of [0, 1]) corners.push(matvec([i * nx, j * ny, k * nz]));
      const e = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]];
      const pts = [];
      for (const [a, b] of e) { pts.push(new THREE.Vector3(...corners[a]), new THREE.Vector3(...corners[b])); }
      const geo = new THREE.BufferGeometry().setFromPoints(pts);
      scene.add(new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: 0x6366f1, transparent: true, opacity: 0.5 })));
    }

    const span = Math.hypot(...matvec([nx, ny, nz]));
    controls.target.copy(center);
    camera.position.copy(center).add(new THREE.Vector3(span * 0.4, span * 0.3, span * 1.1 + 5));
    objs.current = { camera, controls, span, center, renderer };

    let t = 0;
    let animId;
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
            at.arrow.position.set(at.r0[0], at.r0[1], at.r0[2]);
          }
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
    return () => {
      window.removeEventListener('resize', onResize);
      cancelAnimationFrame(animId);
      renderer.dispose();
    };
  }, [baseStructure, supercell, showVectors, showCell, atomScale]);

  return <div ref={mountRef} className="w-full h-full min-h-[360px] cursor-move rounded-xl overflow-hidden" />;
}
