/**
 * Multi-layout workspace modes for RMC Phonon Dynamics.
 * Supports automatic device-class adaptation (iPhone -> iPad -> MacBook/1K -> 2K/4K)
 * as well as explicit user override persisted in localStorage.
 */
export const LAYOUTS = [
  {
    id: 'auto',
    name: 'Auto',
    short: 'Auto',
    target: 'Device-adaptive',
    desc: 'Adapts automatically to iPhone, iPad, MacBook, 1K, 2K & 4K screens',
  },
  {
    id: 'stack',
    name: 'Stack',
    short: 'Stack',
    target: 'iPhone · Narrow',
    desc: 'Single-column vertical scroll with touch-sized controls',
  },
  {
    id: 'deck',
    name: 'Deck',
    short: 'Deck',
    target: 'iPad · Portrait',
    desc: '2-up visual gallery on top with full-width analysis deck below',
  },
  {
    id: 'split',
    name: 'Split',
    short: 'Split',
    target: 'MacBook · 1K (1080p)',
    desc: 'Balanced 2-column studio with stacked visuals & right inspector',
  },
  {
    id: 'panoramic',
    name: 'Panoramic',
    short: 'Wide',
    target: '2K (1440p) · 4K (2160p)',
    desc: 'Full-height 3-column widescreen deck for high-resolution monitors',
  },
];

export const DEFAULT_LAYOUT = 'auto';
const STORE = 'rmcph.layout';

export function readLayout() {
  try {
    const id = localStorage.getItem(STORE);
    return id && LAYOUTS.some((l) => l.id === id) ? id : DEFAULT_LAYOUT;
  } catch {
    return DEFAULT_LAYOUT;
  }
}

export function saveLayout(id) {
  try {
    if (id === DEFAULT_LAYOUT) localStorage.removeItem(STORE);
    else localStorage.setItem(STORE, id);
  } catch {
    /* ignore private-mode storage errors */
  }
}

/**
 * Resolve the active layout archetype for a given viewport (CSS px):
 *   - iPhone portrait (<= 720px) & iPhone landscape (height <= 500px, width <= 960px) -> 'stack'
 *   - iPad portrait (<= 1024px portrait, or 721..959px)                                -> 'deck'
 *   - iPad landscape, MacBook 13"-16", & standard 1K (1080p, up to 2099px)             -> 'split'
 *   - Standard 2K (2560x1440) & 4K (3840x2160) widescreen monitors (>= 2100px)         -> 'panoramic'
 */
export function detectAutoLayout(width, height) {
  const w = width ?? (typeof window !== 'undefined' ? window.innerWidth : 1440);
  const h = height ?? (typeof window !== 'undefined' ? window.innerHeight : 900);
  if (w <= 720 || (h <= 500 && w <= 960)) return 'stack';
  if ((w <= 1024 && h >= w) || w < 960) return 'deck';
  if (w >= 2100) return 'panoramic';
  return 'split';
}
