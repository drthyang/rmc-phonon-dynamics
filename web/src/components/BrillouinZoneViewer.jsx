import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { Network } from 'lucide-react';

/**
 * Lattice-aware k-path picker.
 *
 * High-symmetry points come from the parent (derived from the detected crystal
 * system in math/reciprocal.js), in CONVENTIONAL-cell fractional reciprocal
 * coordinates. The user clicks spheres to build a path; the path of labels and
 * the points map are reported via onPathChange(labels, points).
 *
 * The geometry shown is a reference cube spanning the first BZ octants; it is a
 * schematic, not the exact BZ polyhedron (documented UI limitation).
 */
export default function BrillouinZoneViewer({ symSet, system, onPathChange }) {
  const mountRef = useRef(null);
  const [path, setPath] = useState([]);

  // Initialise the path to the system's default whenever the point set changes.
  useEffect(() => {
    if (!symSet) return;
    const def = symSet.defaultPath || [];
    setPath(def);
    if (onPathChange) onPathChange(def, symSet.points);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symSet]);

  useEffect(() => {
    if (!mountRef.current || !symSet) return;
    const width = mountRef.current.clientWidth;
    const height = mountRef.current.clientHeight;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    camera.position.set(1.4, 1.2, 1.8);
    camera.lookAt(0.25, 0.25, 0.25);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    mountRef.current.innerHTML = '';
    mountRef.current.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0.25, 0.25, 0.25);

    // Reference box [0,0.5]^3 (one octant of the BZ in fractional coords).
    const box = new THREE.BoxGeometry(0.5, 0.5, 0.5);
    box.translate(0.25, 0.25, 0.25);
    const edges = new THREE.EdgesGeometry(box);
    scene.add(new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x4f46e5 })));

    // High-symmetry point spheres.
    const sphereGeo = new THREE.SphereGeometry(0.025, 16, 16);
    const pointMat = new THREE.MeshBasicMaterial({ color: 0xf59e0b });
    const activeMat = new THREE.MeshBasicMaterial({ color: 0xef4444 });
    const pointsGroup = new THREE.Group();
    const pointMeshes = {};
    for (const [label, coords] of Object.entries(symSet.points)) {
      const mesh = new THREE.Mesh(sphereGeo, pointMat);
      mesh.position.set(coords[0], coords[1], coords[2]);
      mesh.userData = { label };
      pointsGroup.add(mesh);
      pointMeshes[label] = mesh;
    }
    scene.add(pointsGroup);

    const pathGroup = new THREE.Group();
    scene.add(pathGroup);
    const drawPath = (labels) => {
      pathGroup.clear();
      Object.values(pointMeshes).forEach(m => { m.material = pointMat; });
      labels.forEach(l => { if (pointMeshes[l]) pointMeshes[l].material = activeMat; });
      if (labels.length >= 2) {
        const pts = labels.map(l => new THREE.Vector3(...symSet.points[l]));
        const geo = new THREE.BufferGeometry().setFromPoints(pts);
        pathGroup.add(new THREE.Line(geo, new THREE.LineBasicMaterial({ color: 0xf59e0b })));
      }
    };
    drawPath(path);

    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    const onClick = (event) => {
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((event.clientX - rect.left) / width) * 2 - 1;
      mouse.y = -((event.clientY - rect.top) / height) * 2 + 1;
      raycaster.setFromCamera(mouse, camera);
      const hits = raycaster.intersectObjects(pointsGroup.children);
      if (hits.length > 0) {
        const label = hits[0].object.userData.label;
        setPath(prev => {
          const next = [...prev, label];
          if (onPathChange) onPathChange(next, symSet.points);
          return next;
        });
      }
    };
    renderer.domElement.addEventListener('click', onClick);

    let animId;
    const animate = () => {
      animId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      renderer.domElement.removeEventListener('click', onClick);
      cancelAnimationFrame(animId);
      renderer.dispose();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symSet, path]);

  return (
    <div className="bg-black/40 rounded-xl border border-white/5 overflow-hidden flex flex-col h-full relative">
      <div className="px-4 py-3 border-b border-white/10 bg-white/5 flex items-center justify-between">
        <h3 className="text-sm font-medium text-gray-300 flex items-center gap-2">
          <Network className="w-4 h-4 text-amber-500" />
          K-Path {system ? `(${system})` : ''}
        </h3>
        <div className="flex gap-1">
          <button
            onClick={() => { const d = symSet?.defaultPath || []; setPath(d); if (onPathChange) onPathChange(d, symSet.points); }}
            className="text-xs bg-white/10 text-gray-300 px-2 py-1 rounded hover:bg-white/20 transition-colors"
          >
            Default
          </button>
          <button
            onClick={() => { setPath([]); if (onPathChange) onPathChange([], symSet?.points || {}); }}
            className="text-xs bg-red-500/20 text-red-400 px-2 py-1 rounded hover:bg-red-500/30 transition-colors"
          >
            Clear
          </button>
        </div>
      </div>

      <div className="flex-1 relative">
        <div ref={mountRef} className="absolute inset-0 cursor-crosshair" />
        <div className="absolute top-2 left-2 pointer-events-none flex gap-1 flex-wrap max-w-[80%]">
          {path.map((p, i) => (
            <React.Fragment key={i}>
              <span className="bg-amber-500 text-black px-1.5 rounded font-bold text-xs">{p}</span>
              {i < path.length - 1 && <span className="text-gray-500 text-xs">→</span>}
            </React.Fragment>
          ))}
        </div>
        {path.length === 0 && (
          <div className="absolute bottom-2 left-0 right-0 text-center text-gray-500 text-xs pointer-events-none">
            Click spheres to build a k-path
          </div>
        )}
      </div>
    </div>
  );
}
