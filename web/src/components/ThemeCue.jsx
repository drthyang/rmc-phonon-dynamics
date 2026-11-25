import { useEffect, useRef, useState } from 'react';
import { applyTheme, readTheme, THEMES } from '../theme';

/**
 * MOTIF-style page colour dot: click to reveal the 6 pastel page accents.
 * Only the page chrome changes — scientific figures are never affected.
 */
export default function ThemeCue() {
  const [theme, setTheme] = useState(readTheme);
  const [open, setOpen] = useState(null);
  const box = useRef(null);
  const current = THEMES.find((t) => t.id === theme) ?? THEMES[0];

  useEffect(() => {
    applyTheme(theme, false);
  }, [theme]);

  useEffect(() => {
    if (!open) return;
    const away = (e) => {
      if (box.current && !box.current.contains(e.target)) setOpen(null);
    };
    const key = (e) => {
      if (e.key === 'Escape') setOpen(null);
    };
    document.addEventListener('pointerdown', away);
    document.addEventListener('keydown', key);
    return () => {
      document.removeEventListener('pointerdown', away);
      document.removeEventListener('keydown', key);
    };
  }, [open]);

  return (
    <div className="theme-cue" ref={box}>
      <button
        type="button"
        className="theme-dot"
        aria-label={`Page colour: ${current.name}`}
        aria-expanded={open !== null}
        title={`Page colour: ${current.name}. Only the page changes, never the figure.`}
        style={{ background: current.accent }}
        onClick={(e) => {
          if (open) return setOpen(null);
          const r = e.currentTarget.getBoundingClientRect();
          setOpen({
            top: r.bottom + 8,
            left: Math.max(8, Math.min(r.left - 8, window.innerWidth - 184)),
          });
        }}
      />
      {open && (
        <div className="theme-row" role="radiogroup" aria-label="Page colour" style={open}>
          {THEMES.map((t) => (
            <button
              key={t.id}
              type="button"
              role="radio"
              aria-checked={theme === t.id}
              aria-label={t.name}
              title={t.name}
              className={theme === t.id ? 'theme-dot on' : 'theme-dot'}
              style={{ background: t.accent }}
              onClick={() => {
                applyTheme(t.id);
                setTheme(t.id);
                setOpen(null);
              }}
            />
          ))}
        </div>
      )}
    </div>
  );
}
