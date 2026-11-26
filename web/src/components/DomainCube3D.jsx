import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

const DEFAULT_PALETTE = [
  '#c95d3a', '#2b6cb0', '#2f855a', '#b7791f',
  '#7b4b94', '#1d8a8a', '#b83280', '#5a67d8',
];

/**
 * Interactive 3D Supercell Domain Cube (Three.js InstancedMesh).
 * Renders each unit cell (x, y, z) colored by its assigned twin-domain variant.
 * Supports:
 *   - Drag to orbit, scroll to zoom
 *   - `activeDomain`: solo/isolate a single domain variant (null = show all)
 *   - `sliceMode`: 'all' | 'slice' (highlight z = sliceZ) | 'cutaway' (show z <= sliceZ)
 */
export default function DomainCube3D({
  grid,
  dim = [8, 8, 8],
  palette = DEFAULT_PALETTE,
  activeDomain = null,
  sliceZ = 0,
  sliceMode = 'all',
  gap = 0.14,
}) {
  const mountRef = useRef(null);
  const stateRef = useRef(null);

  // Initialize Three.js scene once
  useEffect(() => {
    const el = mountRef.current;
    if (!el) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#ffffff');

    const w = Math.max(100, el.clientWidth);
    const h = Math.max(100, el.clientHeight);
    const calcFov = (aspect) => (
      aspect < 1.08
        ? Math.min(68, 2 * Math.atan(Math.tan((34 * Math.PI) / 360) / Math.max(0.42, aspect / 1.08)) * (180 / Math.PI))
        : 34
    );
    const camera = new THREE.PerspectiveCamera(calcFov(w / h), w / h, 0.1, 200);
    camera.position.set(1.65, 1.35, 1.85);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(w, h);
    el.innerHTML = '';
    el.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 0, 0);

    // Clean studio lighting (MOTIF editorial look)
    const amb = new THREE.AmbientLight(0xffffff, 0.72);
    const dir1 = new THREE.DirectionalLight(0xffffff, 0.65);
    dir1.position.set(4, 6, 5);
    const dir2 = new THREE.DirectionalLight(0xffffff, 0.30);
    dir2.position.set(-5, -3, -4);
    scene.add(amb, dir1, dir2);

    const root = new THREE.Group();
    scene.add(root);

    let raf = 0;
    const animate = () => {
      raf = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const ro = new ResizeObserver(() => {
      if (!el) return;
      const nw = Math.max(60, el.clientWidth);
      const nh = Math.max(60, el.clientHeight);
      const aspect = nw / nh;
      camera.aspect = aspect;
      camera.fov = calcFov(aspect);
      camera.updateProjectionMatrix();
      renderer.setSize(nw, nh);
    });
    ro.observe(el);

    stateRef.current = { scene, camera, renderer, controls, root };

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      renderer.dispose();
      stateRef.current = null;
    };
  }, []);

  // Rebuild instanced voxels whenever grid / dim / slice / activeDomain changes
  useEffect(() => {
    const st = stateRef.current;
    if (!st || !grid?.length || !dim?.length) return;
    const { root } = st;

    // Clear previous children & dispose geometries/materials
    while (root.children.length) {
      const child = root.children[0];
      root.remove(child);
      child.geometry?.dispose();
      if (Array.isArray(child.material)) child.material.forEach(m => m.dispose());
      else child.material?.dispose();
    }

    const [N1, N2, N3] = dim;
    const maxN = Math.max(N1, N2, N3, 1);
    const scale = 1.18 / maxN; // normalize cube to ~1.18 world units across
    const boxSize = Math.max(0.25, 1 - gap) * scale;

    const ox = (N1 - 1) * 0.5;
    const oy = (N2 - 1) * 0.5;
    const oz = (N3 - 1) * 0.5;

    const solidItems = [];
    const ghostItems = [];

    for (let x = 0; x < N1; x++) {
      for (let y = 0; y < N2; y++) {
        for (let z = 0; z < N3; z++) {
          const c = (x * N2 + y) * N3 + z;
          const dom = grid[c] ?? 0;

          if (sliceMode === 'cutaway' && z > sliceZ) continue;

          const domainMatch = activeDomain === null || dom === activeDomain;
          const sliceMatch = sliceMode !== 'slice' || z === sliceZ;

          const pos = [(x - ox) * scale, (z - oz) * scale, (y - oy) * scale]; // z is vertical up
          if (domainMatch && sliceMatch) {
            solidItems.push({ pos, dom });
          } else if (sliceMode === 'slice' && domainMatch) {
            ghostItems.push({ pos, dom });
          }
        }
      }
    }

    const dummy = new THREE.Object3D();
    const color = new THREE.Color();

    if (solidItems.length > 0) {
      const geom = new THREE.BoxGeometry(boxSize, boxSize, boxSize);
      const mat = new THREE.MeshStandardMaterial({ roughness: 0.42, metalness: 0.04 });
      const mesh = new THREE.InstancedMesh(geom, mat, solidItems.length);
      for (let i = 0; i < solidItems.length; i++) {
        const it = solidItems[i];
        dummy.position.set(it.pos[0], it.pos[1], it.pos[2]);
        dummy.updateMatrix();
        mesh.setMatrixAt(i, dummy.matrix);
        color.set(palette[it.dom % palette.length]);
        mesh.setColorAt(i, color);
      }
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
      root.add(mesh);
    }

    if (ghostItems.length > 0) {
      const geom = new THREE.BoxGeometry(boxSize * 0.85, boxSize * 0.85, boxSize * 0.85);
      const mat = new THREE.MeshStandardMaterial({
        roughness: 0.6,
        metalness: 0,
        transparent: true,
        opacity: 0.14,
        depthWrite: false,
      });
      const mesh = new THREE.InstancedMesh(geom, mat, ghostItems.length);
      for (let i = 0; i < ghostItems.length; i++) {
        const it = ghostItems[i];
        dummy.position.set(it.pos[0], it.pos[1], it.pos[2]);
        dummy.updateMatrix();
        mesh.setMatrixAt(i, dummy.matrix);
        color.set(palette[it.dom % palette.length]);
        mesh.setColorAt(i, color);
      }
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
      root.add(mesh);
    }

    // Crisp outer supercell wireframe bounding box
    const boxW = N1 * scale;
    const boxH = N3 * scale;
    const boxD = N2 * scale;
    const bbGeom = new THREE.EdgesGeometry(new THREE.BoxGeometry(boxW, boxH, boxD));
    const bbMat = new THREE.LineBasicMaterial({ color: 0x1b1a18, transparent: true, opacity: 0.35 });
    root.add(new THREE.LineSegments(bbGeom, bbMat));

    // Highlight frame for active z-slice plane when in 'slice' or 'cutaway' mode
    if (sliceMode === 'slice' || sliceMode === 'cutaway') {
      const planeY = (sliceZ - oz) * scale;
      const rectGeom = new THREE.EdgesGeometry(new THREE.BoxGeometry(boxW + 0.02, boxSize * 1.08, boxD + 0.02));
      const rectMat = new THREE.LineBasicMaterial({ color: 0x1b1a18, transparent: true, opacity: 0.85 });
      const sliceBox = new THREE.LineSegments(rectGeom, rectMat);
      sliceBox.position.set(0, planeY, 0);
      root.add(sliceBox);
    }
  }, [grid, dim, palette, activeDomain, sliceZ, sliceMode, gap]);

  return (
    <div
      ref={mountRef}
      style={{ width: '100%', height: '100%', minHeight: 160, cursor: 'grab' }}
      title="Drag to rotate 3D supercell domain cube · Scroll to zoom"
    />
  );
}
