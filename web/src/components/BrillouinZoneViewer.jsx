import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

/**
 * Primitive/conventional Brillouin-zone k-path picker (Three.js).
 * Styled in MOTIF's clean white figure frame with quiet hairline controls.
 */
export default function BrillouinZoneViewer({ bzModel, system, onPathChange }) {
  const mountRef = useRef(null);
  const sceneApi = useRef(null);
  const tipRef = useRef(null);
  const [segments, setSegments] = useState([]);
  const segmentsRef = useRef(segments);
  segmentsRef.current = segments;

  const reportRef = useRef(onPathChange);
  reportRef.current = onPathChange;

  const emit = (segs) => {
    if (!bzModel || !reportRef.current) return;
    const conv = {};
    for (const [l, p] of Object.entries(bzModel.points)) conv[l] = p.frac;
    reportRef.current(segs, conv);
  };

  const resetToDefault = () => {
    const segs = (bzModel?.path || []).map(([from, to]) => ({ from, to }));
    tipRef.current = segs.length ? segs[segs.length - 1].to : null;
    setSegments(segs); emit(segs);
  };
  const clearPath = () => { tipRef.current = null; setSegments([]); emit([]); };
  const breakPath = () => { tipRef.current = null; sceneApi.current?.drawPath(segmentsRef.current); };

  useEffect(() => {
    if (!mountRef.current || !bzModel) return;
    const { points, bz } = bzModel;
    const w = Math.max(1, mountRef.current.clientWidth), h = Math.max(1, mountRef.current.clientHeight);
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

    const disposables = [];
    const faceMat = new THREE.MeshBasicMaterial({ color: 0x2563eb, transparent: true, opacity: 0.07, side: THREE.DoubleSide, depthWrite: false });
    disposables.push(faceMat);
    for (const face of bz.faces) {
      const pos = [];
      for (let i = 1; i < face.length - 1; i++) pos.push(...face[0], ...face[i], ...face[i + 1]);
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
      disposables.push(g);
      scene.add(new THREE.Mesh(g, faceMat));
    }
    const edgePos = [];
    for (const [a, b] of bz.edges) edgePos.push(...a, ...b);
    const eg = new THREE.BufferGeometry();
    eg.setAttribute('position', new THREE.Float32BufferAttribute(edgePos, 3));
    const edgeMat = new THREE.LineBasicMaterial({ color: 0x9a958d });
    disposables.push(eg, edgeMat);
    scene.add(new THREE.LineSegments(eg, edgeMat));

    const sphereGeo = new THREE.SphereGeometry(maxR * 0.032, 16, 16);
    const baseMat = new THREE.MeshBasicMaterial({ color: 0xd97706 });
    const GROUP_COLORS = [0xdc2626, 0x0d9488, 0x7c3aed, 0xdb2777, 0xd97706];
    const groupMats = GROUP_COLORS.map(c => new THREE.MeshBasicMaterial({ color: c }));
    const groupLineMats = GROUP_COLORS.map(c => new THREE.LineBasicMaterial({ color: c, linewidth: 2 }));
    disposables.push(sphereGeo, baseMat, ...groupMats, ...groupLineMats);
    const pointsGroup = new THREE.Group();
    scene.add(pointsGroup);
    for (const [label, p] of Object.entries(points)) {
      const mesh = new THREE.Mesh(sphereGeo, baseMat);
      mesh.position.set(...p.cart);
      mesh.userData = { label };
      pointsGroup.add(mesh);
      const spr = makeLabel(p.display, maxR * 0.16);
      disposables.push(spr.material.map, spr.material);
      spr.position.set(p.cart[0] * 1.14, p.cart[1] * 1.14 + maxR * 0.04, p.cart[2] * 1.14);
      scene.add(spr);
    }

    const pathGroup = new THREE.Group();
    scene.add(pathGroup);
    const arrowGeo = new THREE.ConeGeometry(maxR * 0.035, maxR * 0.11, 14);
    disposables.push(arrowGeo);
    const YUP = new THREE.Vector3(0, 1, 0);
    const clearPathGroup = () => {
      for (const child of pathGroup.children) {
        if (child.isLine && child.geometry) child.geometry.dispose();
      }
      pathGroup.clear();
    };
    const drawPath = (segs) => {
      clearPathGroup();
      const ptGroup = new Map();
      let group = 0;
      for (let si = 0; si < segs.length; si++) {
        const s = segs[si];
        if (si > 0 && segs[si - 1].to !== s.from) group++;
        const gi = group % GROUP_COLORS.length;
        const a = points[s.from]?.cart, b = points[s.to]?.cart;
        if (!a || !b) continue;
        ptGroup.set(s.from, gi); ptGroup.set(s.to, gi);
        const va = new THREE.Vector3(...a), vb = new THREE.Vector3(...b);
        const g = new THREE.BufferGeometry().setFromPoints([va, vb]);
        pathGroup.add(new THREE.Line(g, groupLineMats[gi]));
        const dir = new THREE.Vector3().subVectors(vb, va);
        const len = dir.length();
        if (len > 1e-6) {
          dir.divideScalar(len);
          const cone = new THREE.Mesh(arrowGeo, groupMats[gi]);
          cone.position.copy(va).addScaledVector(dir, len * 0.55);
          cone.quaternion.setFromUnitVectors(YUP, dir);
          pathGroup.add(cone);
        }
      }
      const tip = tipRef.current;
      if (tip && !ptGroup.has(tip)) ptGroup.set(tip, (segs.length ? group + 1 : 0) % GROUP_COLORS.length);
      pointsGroup.children.forEach(m => {
        const gi = ptGroup.get(m.userData.label);
        m.material = gi == null ? baseMat : groupMats[gi];
      });
    };
    sceneApi.current = { drawPath };
    drawPath(segmentsRef.current);

    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    let downPos = null;
    const onDown = (ev) => { downPos = { x: ev.clientX, y: ev.clientY }; };
    const onClick = (ev) => {
      if (downPos && Math.hypot(ev.clientX - downPos.x, ev.clientY - downPos.y) > 4) return;
      const rect = renderer.domElement.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      mouse.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(mouse, camera);
      const hit = raycaster.intersectObjects(pointsGroup.children)[0];
      if (!hit) return;
      const label = hit.object.userData.label;
      if (tipRef.current == null) { tipRef.current = label; drawPath(segmentsRef.current); return; }
      if (label === tipRef.current) return;
      const from = tipRef.current;
      tipRef.current = label;
      setSegments(prev => { const next = [...prev, { from, to: label }]; emit(next); return next; });
    };
    renderer.domElement.addEventListener('pointerdown', onDown);
    renderer.domElement.addEventListener('click', onClick);

    let id;
    const loop = () => {
      id = requestAnimationFrame(loop);
      controls.update();
      if (!renderer.getContext().isContextLost()) renderer.render(scene, camera);
    };
    loop();

    const onResize = () => {
      if (!mountRef.current) return;
      const W = mountRef.current.clientWidth, H = mountRef.current.clientHeight;
      if (W < 1 || H < 1) return;
      camera.aspect = W / H; camera.updateProjectionMatrix(); renderer.setSize(W, H);
    };
    window.addEventListener('resize', onResize);
    const ro = new ResizeObserver(onResize);
    ro.observe(mountRef.current);
    return () => {
      window.removeEventListener('resize', onResize);
      ro.disconnect();
      renderer.domElement.removeEventListener('pointerdown', onDown);
      renderer.domElement.removeEventListener('click', onClick);
      cancelAnimationFrame(id);
      clearPathGroup();
      for (const d of disposables) d?.dispose?.();
      controls.dispose();
      renderer.dispose();
      try { renderer.forceContextLoss(); } catch { /* ignore */ }
      sceneApi.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bzModel]);

  useEffect(() => { if (bzModel) resetToDefault(); /* eslint-disable-next-line */ }, [bzModel]);
  useEffect(() => { sceneApi.current?.drawPath(segments); }, [segments]);

  if (!bzModel) {
    return (
      <div className="m-frame" style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column' }}>
        <div style={{ padding: '10px 14px', borderBottom: '1px solid var(--line)', fontWeight: 600 }}>Brillouin zone</div>
        <div style={{ flex: 1, minHeight: 280, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--faint)', fontSize: 'var(--fz-control)' }}>
          Open an ensemble folder to construct the Brillouin zone.
        </div>
      </div>
    );
  }

  const seq = pathLabelSequence(segments);
  let pathStr = '— none —';
  if (seq.length) {
    pathStr = '';
    for (let i = 0; i < seq.length; i++) {
      const p = seq[i];
      if (p === '|') { pathStr += '  |  '; continue; }
      if (i > 0 && seq[i - 1] !== '|') pathStr += ' → ';
      pathStr += (bzModel.points[p]?.display || p);
    }
  }

  return (
    <div className="m-frame" style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 14px', borderBottom: '1px solid var(--line)' }}>
        <span style={{ fontWeight: 600 }}>Brillouin zone</span>
        <span style={{ fontSize: 'var(--fz-small)', fontFamily: 'var(--font-mono)', color: 'var(--faint)' }}>
          {bzModel.code}{system ? ` · ${system}` : ''}
        </span>
      </div>

      <div style={{ flex: 1, minHeight: 290, position: 'relative', background: 'var(--surface)' }}>
        <div ref={mountRef} style={{ position: 'absolute', inset: 0, cursor: 'crosshair' }} />
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '8px 14px', borderTop: '1px solid var(--line)', flexWrap: 'wrap' }}>
        <span style={{ fontSize: 'var(--fz-small)', color: 'var(--faint)' }}>path</span>
        <span
          style={{ fontFamily: 'var(--font-sym)', fontWeight: 600, fontSize: 13.5, color: 'var(--ink)', flex: 1, minWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
          title={pathStr}
        >
          {pathStr}
        </span>
        <div style={{ display: 'inline-flex', gap: 6 }}>
          <button type="button" onClick={resetToDefault} className="m-btn sm">Default</button>
          <button
            type="button"
            onClick={breakPath}
            title="Start a disconnected segment — the next point won't join the current tip"
            className="m-btn subtle sm"
          >Break</button>
          <button type="button" onClick={clearPath} className="m-btn subtle sm">Clear</button>
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
  ctx.fillStyle = '#1b1a18';
  ctx.font = "bold 78px 'Noto Sans', sans-serif";
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 64, 64);
  const tex = new THREE.CanvasTexture(cv);
  const spr = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true }));
  spr.scale.set(size, size, size);
  return spr;
}
