import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { Network } from 'lucide-react';

/**
 * Primitive Brillouin-zone k-path picker (seekpath-style).
 *
 * Renders the true primitive Wigner-Seitz polyhedron (faces + edges) with the
 * standard high-symmetry points at their cartesian positions and the suggested
 * path. Clicking a point extends the path from the current tip. Reports the path
 * as label segments {from,to} plus the label→conventional-fractional map (which
 * the calculation consumes).
 *
 * Props: bzModel = { points:{label:{cart,fracConv,display}}, path:[[a,b]], bz, code }
 */
export default function BrillouinZoneViewer({ bzModel, system, onPathChange }) {
  const mountRef = useRef(null);
  const sceneApi = useRef(null);   // { drawPath } set by the build effect
  const tipRef = useRef(null);
  const [segments, setSegments] = useState([]);

  const reportRef = useRef(onPathChange);
  reportRef.current = onPathChange;

  const emit = (segs) => {
    if (!bzModel || !reportRef.current) return;
    const conv = {};
    for (const [l, p] of Object.entries(bzModel.points)) conv[l] = p.fracConv;
    reportRef.current(segs, conv);
  };

  const resetToDefault = () => {
    const segs = (bzModel?.path || []).map(([from, to]) => ({ from, to }));
    tipRef.current = segs.length ? segs[segs.length - 1].to : null;
    setSegments(segs); emit(segs);
  };
  const clearPath = () => { tipRef.current = null; setSegments([]); emit([]); };

  // Build scene once per model.
  useEffect(() => {
    if (!mountRef.current || !bzModel) return;
    const { points, bz } = bzModel;
    const w = mountRef.current.clientWidth, h = mountRef.current.clientHeight;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, w / h, 0.001, 1000);
    const maxR = Math.max(0.1, ...bz.vertices.map(v => Math.hypot(...v)));
    camera.position.set(maxR * 2.2, maxR * 1.6, maxR * 2.6);
    camera.lookAt(0, 0, 0);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(w, h); renderer.setPixelRatio(window.devicePixelRatio);
    mountRef.current.innerHTML = '';
    mountRef.current.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    const faceMat = new THREE.MeshBasicMaterial({ color: 0x4f46e5, transparent: true, opacity: 0.08, side: THREE.DoubleSide });
    for (const face of bz.faces) {
      const pos = [];
      for (let i = 1; i < face.length - 1; i++) pos.push(...face[0], ...face[i], ...face[i + 1]);
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
      scene.add(new THREE.Mesh(g, faceMat));
    }
    const edgePos = [];
    for (const [a, b] of bz.edges) edgePos.push(...a, ...b);
    const eg = new THREE.BufferGeometry();
    eg.setAttribute('position', new THREE.Float32BufferAttribute(edgePos, 3));
    scene.add(new THREE.LineSegments(eg, new THREE.LineBasicMaterial({ color: 0x818cf8 })));

    const sphereGeo = new THREE.SphereGeometry(maxR * 0.03, 16, 16);
    const baseMat = new THREE.MeshBasicMaterial({ color: 0xf59e0b });
    const activeMat = new THREE.MeshBasicMaterial({ color: 0xef4444 });
    const pointsGroup = new THREE.Group();
    scene.add(pointsGroup);
    for (const [label, p] of Object.entries(points)) {
      const mesh = new THREE.Mesh(sphereGeo, baseMat);
      mesh.position.set(...p.cart);
      mesh.userData = { label };
      pointsGroup.add(mesh);
      const spr = makeLabel(p.display, maxR * 0.16);
      spr.position.set(p.cart[0] * 1.14, p.cart[1] * 1.14 + maxR * 0.04, p.cart[2] * 1.14);
      scene.add(spr);
    }

    const pathGroup = new THREE.Group();
    scene.add(pathGroup);
    const drawPath = (segs) => {
      pathGroup.clear();
      const onPath = new Set();
      if (tipRef.current) onPath.add(tipRef.current);
      for (const s of segs) {
        const a = points[s.from]?.cart, b = points[s.to]?.cart;
        if (!a || !b) continue;
        onPath.add(s.from); onPath.add(s.to);
        const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...a), new THREE.Vector3(...b)]);
        pathGroup.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0xf59e0b })));
      }
      pointsGroup.children.forEach(m => { m.material = onPath.has(m.userData.label) ? activeMat : baseMat; });
    };
    sceneApi.current = { drawPath };
    drawPath(segments);

    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    const onClick = (ev) => {
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((ev.clientX - rect.left) / w) * 2 - 1;
      mouse.y = -((ev.clientY - rect.top) / h) * 2 + 1;
      raycaster.setFromCamera(mouse, camera);
      const hit = raycaster.intersectObjects(pointsGroup.children)[0];
      if (!hit) return;
      const label = hit.object.userData.label;
      if (tipRef.current == null) { tipRef.current = label; drawPath(segments); return; }
      if (label === tipRef.current) return;
      const from = tipRef.current;
      tipRef.current = label;
      setSegments(prev => { const next = [...prev, { from, to: label }]; emit(next); return next; });
    };
    renderer.domElement.addEventListener('click', onClick);

    let id;
    const loop = () => { id = requestAnimationFrame(loop); controls.update(); renderer.render(scene, camera); };
    loop();
    const onResize = () => {
      if (!mountRef.current) return;
      const W = mountRef.current.clientWidth, H = mountRef.current.clientHeight;
      camera.aspect = W / H; camera.updateProjectionMatrix(); renderer.setSize(W, H);
    };
    window.addEventListener('resize', onResize);
    return () => { window.removeEventListener('resize', onResize); renderer.domElement.removeEventListener('click', onClick); cancelAnimationFrame(id); renderer.dispose(); sceneApi.current = null; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bzModel]);

  // Initialise path from the suggested path when the model loads.
  useEffect(() => { if (bzModel) resetToDefault(); /* eslint-disable-next-line */ }, [bzModel]);

  // Redraw path on segment change.
  useEffect(() => { sceneApi.current?.drawPath(segments); }, [segments]);

  if (!bzModel) return <div className="flex items-center justify-center h-full text-gray-500 text-sm">Load a dataset to build the Brillouin zone.</div>;

  return (
    <div className="bg-black/40 rounded-xl border border-white/5 overflow-hidden flex flex-col h-full relative">
      <div className="px-4 py-3 border-b border-white/10 bg-white/5 flex items-center justify-between">
        <h3 className="text-sm font-medium text-gray-300 flex items-center gap-2">
          <Network className="w-4 h-4 text-amber-500" />Primitive BZ — {bzModel.code}{system ? ` (${system})` : ''}
        </h3>
        <div className="flex gap-1">
          <button onClick={resetToDefault} className="text-xs bg-white/10 text-gray-300 px-2 py-1 rounded hover:bg-white/20">Default</button>
          <button onClick={clearPath} className="text-xs bg-red-500/20 text-red-400 px-2 py-1 rounded hover:bg-red-500/30">Clear</button>
        </div>
      </div>
      <div className="flex-1 relative">
        <div ref={mountRef} className="absolute inset-0 cursor-crosshair" />
        <div className="absolute top-2 left-2 pointer-events-none flex gap-1 flex-wrap max-w-[85%] items-center">
          {pathLabelSequence(segments).map((p, i) => p === '|'
            ? <span key={i} className="text-gray-500 text-xs px-0.5">|</span>
            : <span key={i} className="bg-amber-500 text-black px-1.5 rounded font-bold text-xs">{bzModel.points[p]?.display || p}</span>)}
        </div>
      </div>
    </div>
  );
}

function pathLabelSequence(segments) {
  const out = [];
  for (let i = 0; i < segments.length; i++) {
    const s = segments[i];
    if (i === 0) out.push(s.from);
    else if (segments[i - 1].to !== s.from) out.push('|', s.from);
    out.push(s.to);
  }
  return out;
}

function makeLabel(text, size) {
  const cv = document.createElement('canvas');
  cv.width = 128; cv.height = 128;
  const ctx = cv.getContext('2d');
  ctx.fillStyle = '#fbbf24'; ctx.font = 'bold 80px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(text, 64, 64);
  const tex = new THREE.CanvasTexture(cv);
  const spr = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true }));
  spr.scale.set(size, size, size);
  return spr;
}
