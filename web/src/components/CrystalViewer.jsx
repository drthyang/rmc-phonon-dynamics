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
 * 3D phonon-mode viewer.
 *
 * Each supercell atom is placed at its true Cartesian position
 *   r = (cell_idx + within_cell_frac) @ A_conv
 * and animated by the selected mode's eigenvector. Eigenvector rows are indexed
 * by BASIS SITE (sorted reference number, `uniqueRN`), matching the diagonalizer
 * output. For finite k the per-cell Bloch phase exp(i k·n) (k = 2π·q_frac) is
 * applied so the spatial modulation of the mode is visible:
 *   u_atom(t) = Re( eig_site · exp(i (k·n + ω t)) )
 */
export default function CrystalViewer({ baseStructure, eigenvector, qPoint, isPlaying = true, amplitude = 2.0 }) {
  const mountRef = useRef(null);
  const animRef = useRef(null);

  useEffect(() => {
    if (!mountRef.current || !baseStructure || !baseStructure.hsym_xyz) return;
    const { v1, v2, v3, dim, hsym_xyz, atomType, cellIdx, uniqueRN, atomDic } = baseStructure;

    const A = conventionalLattice(v1, v2, v3, dim); // rows a,b,c (Å)
    const rnToRow = new Map();
    (uniqueRN || []).forEach((rn, r) => rnToRow.set(rn, r));
    const reverseAtomDic = {};
    for (const [el, idxs] of Object.entries(atomDic)) idxs.forEach(idx => { reverseAtomDic[idx] = el; });

    const scene = new THREE.Scene();
    const width = mountRef.current.clientWidth;
    const height = mountRef.current.clientHeight;
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 5000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    mountRef.current.innerHTML = '';
    mountRef.current.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dir = new THREE.DirectionalLight(0xffffff, 0.8); dir.position.set(10, 20, 10); scene.add(dir);

    const numAtoms = hsym_xyz.length / 3;
    const matvec = (f) => [
      f[0] * A[0][0] + f[1] * A[1][0] + f[2] * A[2][0],
      f[0] * A[0][1] + f[1] * A[1][1] + f[2] * A[2][1],
      f[0] * A[0][2] + f[1] * A[1][2] + f[2] * A[2][2],
    ];

    const sphereGeo = new THREE.SphereGeometry(0.45, 16, 16);
    const atoms = [];
    const center = new THREE.Vector3();
    // Cap rendered atoms for responsiveness (validation viewer); supercells can
    // be tens of thousands of atoms.
    const stride = Math.max(1, Math.floor(numAtoms / 4000));

    for (let i = 0; i < numAtoms; i += stride) {
      const rn = atomType[i];
      const el = reverseAtomDic[rn] || 'H';
      const frac = [
        ((hsym_xyz[i * 3] % 1) + 1) % 1,
        ((hsym_xyz[i * 3 + 1] % 1) + 1) % 1,
        ((hsym_xyz[i * 3 + 2] % 1) + 1) % 1,
      ];
      const cell = cellIdx ? [cellIdx[i * 3], cellIdx[i * 3 + 1], cellIdx[i * 3 + 2]] : [0, 0, 0];
      const r0 = matvec([cell[0] + frac[0], cell[1] + frac[1], cell[2] + frac[2]]);
      const mesh = new THREE.Mesh(sphereGeo, new THREE.MeshPhongMaterial({ color: COLOR_MAP[el] || DEFAULT_COLOR, shininess: 80 }));
      mesh.position.set(r0[0], r0[1], r0[2]);
      scene.add(mesh);
      atoms.push({ mesh, r0, row: rnToRow.get(rn) ?? 0, cell });
      center.add(mesh.position);
    }
    if (atoms.length) {
      center.divideScalar(atoms.length);
      controls.target.copy(center);
      const span = norm(matvec([dim[0], dim[1], dim[2]]));
      camera.position.copy(center).add(new THREE.Vector3(0, 0, span * 0.8 + 5));
    }

    const k = qPoint ? [qPoint[0] * TWO_PI_PHASE, qPoint[1] * TWO_PI_PHASE, qPoint[2] * TWO_PI_PHASE] : [0, 0, 0];

    let t = 0;
    const animate = () => {
      animRef.current = requestAnimationFrame(animate);
      if (isPlaying && eigenvector && eigenvector.real) {
        t += 0.08;
        for (const at of atoms) {
          const r = at.row;
          if (r * 3 + 2 >= eigenvector.real.length) continue;
          const kn = k[0] * at.cell[0] + k[1] * at.cell[1] + k[2] * at.cell[2];
          const ph = kn + t;
          const cp = Math.cos(ph), sp = Math.sin(ph);
          const dx = eigenvector.real[r * 3] * cp - eigenvector.imag[r * 3] * sp;
          const dy = eigenvector.real[r * 3 + 1] * cp - eigenvector.imag[r * 3 + 1] * sp;
          const dz = eigenvector.real[r * 3 + 2] * cp - eigenvector.imag[r * 3 + 2] * sp;
          at.mesh.position.set(at.r0[0] + dx * amplitude, at.r0[1] + dy * amplitude, at.r0[2] + dz * amplitude);
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
      if (animRef.current) cancelAnimationFrame(animRef.current);
      renderer.dispose();
    };
  }, [baseStructure, eigenvector, qPoint, isPlaying, amplitude]);

  return <div ref={mountRef} className="w-full h-full min-h-[400px] cursor-move rounded-xl overflow-hidden" />;
}

function norm(v) { return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }
