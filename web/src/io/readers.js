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
 * Read the base structure (cell vectors, atoms) from an rmc6f file.
 */
export async function readBaseStructure(fileHandle) {
  const file = await fileHandle.getFile();
  const text = await file.text();
  const lines = text.split('\n');
  
  let dim = null;
  let v1, v2, v3;
  let atomDic = {};
  
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
    
    if (inAtomsBlock && parts.length >= 5) {
      const atom = parts[1];
      const rn = parseInt(parts[parts.length - 4]);
      if (!isNaN(rn)) {
        if (!atomDic[atom]) atomDic[atom] = [];
        atomDic[atom].push(rn);
      }
    }
  }
  
  for (let atom in atomDic) {
    atomDic[atom].sort((a, b) => a - b);
  }
  
  return {
    v1, v2, v3, dim, atomDic,
    v_super: [v1, v2, v3]
  };
}
