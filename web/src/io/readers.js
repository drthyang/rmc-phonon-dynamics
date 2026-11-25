// web/src/io/readers.js

/**
 * Lists the applicable configuration files from a given FileSystemDirectoryHandle.
 * Detects either numbered .rmc6f files or Frac*.txt files.
 */
export async function listConfigs(dirHandle) {
  let rmc6fFiles = [];
  let fracFiles = [];
  
  for await (const entry of dirHandle.values()) {
    if (entry.kind === 'file') {
      if (entry.name.toLowerCase().endsWith('.rmc6f')) {
        if (!entry.name.toLowerCase().includes('average.rmc6f')) {
          const match = entry.name.match(/_(\d+)\.rmc6f$/i);
          if (match && parseInt(match[1]) >= 1) {
            rmc6fFiles.push({ name: entry.name, handle: entry, index: parseInt(match[1]) });
          }
        }
      } else if (entry.name.toLowerCase().startsWith('frac') && entry.name.toLowerCase().endsWith('.txt')) {
        fracFiles.push({ name: entry.name, handle: entry });
      }
    } else if (entry.kind === 'directory' && entry.name === 'configs') {
      for await (const subEntry of entry.values()) {
        if (subEntry.kind === 'file' && subEntry.name.toLowerCase().startsWith('frac') && subEntry.name.toLowerCase().endsWith('.txt')) {
          fracFiles.push({ name: subEntry.name, handle: subEntry });
        }
      }
    }
  }

  if (rmc6fFiles.length > 0) {
    rmc6fFiles.sort((a, b) => a.index - b.index);
    return { files: rmc6fFiles.map(f => f.handle), family: 'rmc6f' };
  }
  
  if (fracFiles.length > 0) {
    fracFiles.sort((a, b) => a.name.localeCompare(b.name));
    return { files: fracFiles.map(f => f.handle), family: 'frac' };
  }

  return { files: [], family: 'none' };
}

/**
 * Find ANY .rmc6f file in the directory to use as the structure/cell source.
 * Frac*.txt configs carry no lattice vectors or RN->element map, so the legacy
 * code always reads those from a companion .rmc6f (read_cell_vec + get_atom_idx).
 * Prefers the unnumbered base/structure file, then AVERAGE, then any numbered one.
 */
export async function findStructureFile(dirHandle) {
  let base = null, average = null, anyNumbered = null;
  for await (const entry of dirHandle.values()) {
    if (entry.kind !== 'file') continue;
    const lower = entry.name.toLowerCase();
    if (!lower.endsWith('.rmc6f')) continue;
    if (lower.includes('average')) { average = average || entry; continue; }
    if (/_\d+\.rmc6f$/i.test(entry.name)) { anyNumbered = anyNumbered || entry; continue; }
    base = base || entry; // unnumbered base/structure file
  }
  return base || anyNumbered || average || null;
}

/**
 * List every .rmc6f file in the directory as {name, handle}, sorted by name.
 * Used for the structure-file override and the file-reference picker.
 */
export async function listRmc6f(dirHandle) {
  const out = [];
  for await (const entry of dirHandle.values()) {
    if (entry.kind === 'file' && entry.name.toLowerCase().endsWith('.rmc6f')) out.push({ name: entry.name, handle: entry });
  }
  out.sort((a, b) => a.name.localeCompare(b.name));
  return out;
}

/**
 * Parse the base structure (cell vectors, atoms, basis) from raw .rmc6f text.
 * Works with both File objects (any browser) and FileSystemFileHandle.
 */
export function parseBaseStructureText(text) {
  const lines = text.split('\n');

  let dim = null;
  let v1, v2, v3;
  const rnSets = {};            // element -> Set of unique reference numbers
  const rnInfo = new Map();     // rn -> { el, frac:[within-cell xyz] } (first occurrence)

  let inAtomsBlock = false;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;

    const parts = line.split(/\s+/);

    if (parts[0] === 'Supercell') {
      dim = parts.slice(-3).map(Number);
    } else if (parts[0] === 'Lattice') {
      v1 = lines[++i].trim().split(/\s+/).map(Number);
      v2 = lines[++i].trim().split(/\s+/).map(Number);
      v3 = lines[++i].trim().split(/\s+/).map(Number);
    } else if (line.startsWith('Atoms:')) {
      inAtomsBlock = true;
      continue;
    }

    if (inAtomsBlock && parts.length >= 7) {
      const el = parts[1];
      const rn = parseInt(parts[parts.length - 4], 10);
      if (isNaN(rn)) continue;
      (rnSets[el] || (rnSets[el] = new Set())).add(rn);
      if (!rnInfo.has(rn)) {
        // within-cell fractional coordinate = mod(global_frac * dim, 1)
        const gf = [parseFloat(parts[parts.length - 7]), parseFloat(parts[parts.length - 6]), parseFloat(parts[parts.length - 5])];
        const frac = gf.map((g, k) => { let w = (g * dim[k]) % 1; if (w < 0) w += 1; return w; });
        rnInfo.set(rn, { el, frac });
      }
    }
  }

  if (!dim || !v1 || !v2 || !v3 || rnInfo.size === 0) {
    throw new Error('Could not parse Supercell, Lattice, and Atoms blocks from .rmc6f file.');
  }

  const atomDic = {};
  for (const el in rnSets) atomDic[el] = Array.from(rnSets[el]).sort((a, b) => a - b);

  // One representative basis site per reference number (for symmetry analysis).
  const basis = [...rnInfo.entries()].sort((a, b) => a[0] - b[0]).map(([rn, info]) => ({ rn, el: info.el, frac: info.frac }));

  return { v1, v2, v3, dim, atomDic, basis, v_super: [v1, v2, v3] };
}

/**
 * Read the base structure (cell vectors, atoms) from an rmc6f FileSystemFileHandle or File.
 */
export async function readBaseStructure(fileOrHandle) {
  const file = typeof fileOrHandle.getFile === 'function' ? await fileOrHandle.getFile() : fileOrHandle;
  const text = await file.text();
  return parseBaseStructureText(text);
}

/**
 * Parse all atom fractional coordinates + cell indices from a single .rmc6f text.
 * Browser-universal (no WebGPU / FileSystemFileHandle required).
 */
export function parseRmc6fFrame(text, dim) {
  const lines = text.split('\n');
  let inAtoms = false;
  const rnList = [];
  const xyzList = [];
  const cellList = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    if (!inAtoms) {
      if (line.startsWith('Atoms:')) inAtoms = true;
      continue;
    }
    const parts = line.split(/\s+/);
    if (parts.length < 7) continue;

    const rn = parseInt(parts[parts.length - 4], 10);
    const cellX = parseInt(parts[parts.length - 3], 10);
    const cellY = parseInt(parts[parts.length - 2], 10);
    const cellZ = parseInt(parts[parts.length - 1], 10);
    const fracX = parseFloat(parts[parts.length - 7]);
    const fracY = parseFloat(parts[parts.length - 6]);
    const fracZ = parseFloat(parts[parts.length - 5]);
    if (isNaN(rn) || isNaN(cellX) || isNaN(fracX)) continue;

    let x = (fracX * dim[0]) % 1.0; if (x < 0) x += 1.0;
    let y = (fracY * dim[1]) % 1.0; if (y < 0) y += 1.0;
    let z = (fracZ * dim[2]) % 1.0; if (z < 0) z += 1.0;

    rnList.push(rn);
    xyzList.push(x, y, z);
    cellList.push(cellX, cellY, cellZ);
  }

  return {
    xyz: Float32Array.from(xyzList),
    cellIdx: Float32Array.from(cellList),
    atomType: Int32Array.from(rnList),
  };
}

/**
 * Parse a Frac*.txt frame text into { xyz, cellIdx, atomType }.
 */
export function parseFracFrame(text, dim) {
  const lines = text.split('\n');
  const rnList = [];
  const xyzList = [];
  const cellList = [];

  for (let i = 5; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    const parts = line.split(/\s+/);
    if (parts.length < 7) continue;

    const rn = parseInt(parts[0], 10);
    const fracX = parseFloat(parts[1]);
    const fracY = parseFloat(parts[2]);
    const fracZ = parseFloat(parts[3]);
    const cellX = parseInt(parts[parts.length - 3], 10);
    const cellY = parseInt(parts[parts.length - 2], 10);
    const cellZ = parseInt(parts[parts.length - 1], 10);
    if (isNaN(rn) || isNaN(cellX) || isNaN(fracX)) continue;

    const x = fracX * dim[0] - cellX;
    const y = fracY * dim[1] - cellY;
    const z = fracZ * dim[2] - cellZ;

    rnList.push(rn);
    xyzList.push(x, y, z);
    cellList.push(cellX, cellY, cellZ);
  }

  return {
    xyz: Float32Array.from(xyzList),
    cellIdx: Float32Array.from(cellList),
    atomType: Int32Array.from(rnList),
  };
}

/**
 * Export any unit cell (3x3 lattice A + basis [{el, frac, rn?}]) as a CIF 1.1 string.
 */
export function generateUnitCellCIF(A, basis, opts = {}) {
  const { title = 'rmc_structure', spaceGroup = 'P 1', comment = '' } = opts;
  const norm = (v) => Math.hypot(v[0], v[1], v[2]);
  const dot = (u, v) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  const a = norm(A[0]), b = norm(A[1]), c = norm(A[2]);
  const alpha = Math.acos(Math.max(-1, Math.min(1, dot(A[1], A[2]) / (b * c)))) * 180 / Math.PI;
  const beta  = Math.acos(Math.max(-1, Math.min(1, dot(A[0], A[2]) / (a * c)))) * 180 / Math.PI;
  const gamma = Math.acos(Math.max(-1, Math.min(1, dot(A[0], A[1]) / (a * b)))) * 180 / Math.PI;

  const lines = [
    `data_${title.replace(/[^A-Za-z0-9_]/g, '_')}`,
    `_audit_creation_method  'rmc-phonon-dynamics'`,
  ];
  if (comment) {
    lines.push(`_publ_section_comment`, `;`, `  ${comment}`, `;`);
  }
  lines.push(
    ``,
    `_cell_length_a                     ${a.toFixed(6)}`,
    `_cell_length_b                     ${b.toFixed(6)}`,
    `_cell_length_c                     ${c.toFixed(6)}`,
    `_cell_angle_alpha                  ${alpha.toFixed(4)}`,
    `_cell_angle_beta                   ${beta.toFixed(4)}`,
    `_cell_angle_gamma                  ${gamma.toFixed(4)}`,
    ``,
    `_symmetry_space_group_name_H-M    'P 1'`,
    `_symmetry_Int_Tables_number         1`,
    `# Detected/target space group: ${spaceGroup}`,
    `loop_`,
    `_symmetry_equiv_pos_as_xyz`,
    `'x, y, z'`,
    ``,
    `loop_`,
    `_atom_site_label`,
    `_atom_site_type_symbol`,
    `_atom_site_occupancy`,
    `_atom_site_fract_x`,
    `_atom_site_fract_y`,
    `_atom_site_fract_z`,
  );

  basis.forEach((s, i) => {
    const label = `${s.el}${s.rn ?? (i + 1)}`;
    const wrap01 = (x) => ((x % 1) + 1) % 1;
    lines.push(
      `  ${label.padEnd(8)} ${s.el.padEnd(4)} 1.0000  ${wrap01(s.frac[0]).toFixed(6).padStart(10)}  ${wrap01(s.frac[1]).toFixed(6).padStart(10)}  ${wrap01(s.frac[2]).toFixed(6).padStart(10)}`
    );
  });

  return lines.join('\n') + '\n';
}
