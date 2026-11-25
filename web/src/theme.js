/**
 * MOTIF-style page accent colour: colours only the page's primary actions,
 * active hints, and notices — never the scientific figures.
 */
export const THEMES = [
  { id: 'lemon',    name: 'Lemon',    accent: '#fff27a', deep: '#f6e24a' },
  { id: 'marigold', name: 'Marigold', accent: '#ffd166', deep: '#f7bf3d' },
  { id: 'apricot',  name: 'Apricot',  accent: '#ffbe8a', deep: '#ffa86b' },
  { id: 'mint',     name: 'Mint',     accent: '#b5ecd4', deep: '#93e0bf' },
  { id: 'sky',      name: 'Sky',      accent: '#bcd8ff', deep: '#9cc4fb' },
  { id: 'lilac',    name: 'Lilac',    accent: '#d9ccff', deep: '#c6b4fb' },
];

export const DEFAULT_THEME = 'mint';
const STORE = 'rmcph.theme';

export function tint(hex, f) {
  return `#${[1, 3, 5]
    .map((i) => {
      const c = parseInt(hex.slice(i, i + 2), 16);
      return Math.round(c + (255 - c) * f).toString(16).padStart(2, '0');
    })
    .join('')}`;
}

export function themeVariables(id) {
  const theme = THEMES.find((t) => t.id === id);
  if (!theme || theme.id === DEFAULT_THEME) return {};
  return {
    '--accent': theme.accent,
    '--accent-deep': theme.deep,
    '--note-bg': tint(theme.accent, 0.72),
  };
}

export function readTheme() {
  try {
    const id = localStorage.getItem(STORE);
    return id && THEMES.some((t) => t.id === id) ? id : DEFAULT_THEME;
  } catch {
    return DEFAULT_THEME;
  }
}

export function applyTheme(id, keep = true) {
  const root = document.documentElement.style;
  for (const name of ['--accent', '--accent-deep', '--note-bg']) root.removeProperty(name);
  for (const [name, value] of Object.entries(themeVariables(id))) root.setProperty(name, value);
  if (!keep) return;
  try {
    if (id === DEFAULT_THEME) localStorage.removeItem(STORE);
    else localStorage.setItem(STORE, id);
  } catch { /* ignore private-mode storage errors */ }
}
