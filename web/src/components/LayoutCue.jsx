import { useEffect, useRef, useState } from 'react';
import { LAYOUTS } from '../layout';

function LayoutGlyph({ mode }) {
  const stroke = 'currentColor';
  if (mode === 'stack') {
    return (
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
        <rect x="1.5" y="1.5" width="11" height="3" rx="0.8" stroke={stroke} strokeWidth="1.2" />
        <rect x="1.5" y="5.5" width="11" height="3" rx="0.8" stroke={stroke} strokeWidth="1.2" />
        <rect x="1.5" y="9.5" width="11" height="3" rx="0.8" stroke={stroke} strokeWidth="1.2" />
      </svg>
    );
  }
  if (mode === 'deck') {
    return (
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
        <rect x="1.5" y="1.5" width="5" height="5" rx="0.8" stroke={stroke} strokeWidth="1.2" />
        <rect x="7.5" y="1.5" width="5" height="5" rx="0.8" stroke={stroke} strokeWidth="1.2" />
        <rect x="1.5" y="7.8" width="11" height="4.7" rx="0.8" stroke={stroke} strokeWidth="1.2" />
      </svg>
    );
  }
  if (mode === 'panoramic') {
    return (
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
        <rect x="1.2" y="2" width="3.3" height="10" rx="0.7" stroke={stroke} strokeWidth="1.2" />
        <rect x="5.35" y="2" width="3.3" height="10" rx="0.7" stroke={stroke} strokeWidth="1.2" />
        <rect x="9.5" y="2" width="3.3" height="10" rx="0.7" stroke={stroke} strokeWidth="1.2" />
      </svg>
    );
  }
  // 'split' (or fallback)
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
      <rect x="1.5" y="1.5" width="5" height="5" rx="0.8" stroke={stroke} strokeWidth="1.2" />
      <rect x="1.5" y="7.5" width="5" height="5" rx="0.8" stroke={stroke} strokeWidth="1.2" />
      <rect x="7.7" y="1.5" width="4.8" height="11" rx="0.8" stroke={stroke} strokeWidth="1.2" />
    </svg>
  );
}

/**
 * Topbar workspace layout switcher: Auto | Stack | Deck | Split | Panoramic.
 */
export default function LayoutCue({ layoutPref, effectiveLayout, onChange }) {
  const [pos, setPos] = useState(null);
  const box = useRef(null);

  useEffect(() => {
    if (!pos) return;
    const away = (e) => {
      if (box.current && !box.current.contains(e.target)) setPos(null);
    };
    const key = (e) => {
      if (e.key === 'Escape') setPos(null);
    };
    const onResize = () => setPos(null);
    document.addEventListener('pointerdown', away);
    document.addEventListener('keydown', key);
    window.addEventListener('resize', onResize);
    return () => {
      document.removeEventListener('pointerdown', away);
      document.removeEventListener('keydown', key);
      window.removeEventListener('resize', onResize);
    };
  }, [pos]);

  const activeMeta = LAYOUTS.find((l) => l.id === effectiveLayout) || LAYOUTS[3];
  const prefMeta = LAYOUTS.find((l) => l.id === layoutPref) || LAYOUTS[0];
  const buttonLabel = layoutPref === 'auto' ? `Auto · ${activeMeta.short}` : prefMeta.name;

  return (
    <div className="layout-cue" ref={box}>
      <button
        type="button"
        className={`m-btn sm subtle layout-cue-btn ${layoutPref !== 'auto' ? 'is-pinned' : ''}`}
        aria-label={`Workspace layout: ${buttonLabel}`}
        aria-expanded={pos !== null}
        title={`Workspace layout: ${buttonLabel}. Click to switch between Auto, Stack (iPhone), Deck (iPad), Split (MacBook/1K), and Panoramic (2K/4K).`}
        onClick={(e) => {
          if (pos) return setPos(null);
          const r = e.currentTarget.getBoundingClientRect();
          const menuW = Math.min(288, window.innerWidth - 16);
          setPos({
            top: r.bottom + 8,
            left: Math.max(8, Math.min(r.right - menuW, window.innerWidth - menuW - 8)),
          });
        }}
      >
        <LayoutGlyph mode={effectiveLayout} />
        <span className="layout-cue-label">{buttonLabel}</span>
      </button>

      {pos && (
        <div
          className="layout-menu"
          role="radiogroup"
          aria-label="Workspace layout"
          style={pos}
        >
          <div className="layout-menu-head">
            <span>Workspace Layout</span>
            <span className="layout-menu-active">active: {activeMeta.name}</span>
          </div>
          {LAYOUTS.map((l) => {
            const checked = layoutPref === l.id;
            const glyphMode = l.id === 'auto' ? effectiveLayout : l.id;
            return (
              <button
                key={l.id}
                type="button"
                role="radio"
                aria-checked={checked}
                className={`layout-opt ${checked ? 'on' : ''}`}
                onClick={() => {
                  onChange(l.id);
                  setPos(null);
                }}
              >
                <span className="layout-opt-icon">
                  <LayoutGlyph mode={glyphMode} />
                </span>
                <span className="layout-opt-text">
                  <span className="layout-opt-row">
                    <b>{l.name}</b>
                    <span className="layout-opt-tag">{l.target}</span>
                  </span>
                  <span className="layout-opt-desc">{l.desc}</span>
                </span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
