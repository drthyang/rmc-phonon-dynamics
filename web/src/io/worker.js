// web/src/io/worker.js

/**
 * Worker for parsing .rmc6f and Frac*.txt files efficiently without blocking the main UI thread.
 */

self.onmessage = async (e) => {
  const { fileHandle, family, atomDic, dim, atype } = e.data;
  
  try {
    const file = await fileHandle.getFile();
    const text = await file.text();
    
    let result;
    if (family === 'rmc6f') {
      result = parseRMC6F(text, atomDic, dim, atype);
    } else {
      result = parseFrac(text, atomDic, dim, atype);
    }
    
    // Transfer Float32Arrays to avoid structured cloning overhead
    self.postMessage({ success: true, ...result }, [result.xyz.buffer, result.cellIdx.buffer, result.atomType.buffer]);
  } catch (error) {
    self.postMessage({ success: false, error: error.message });
  }
};

function parseRMC6F(text, atomDic, dim, atype) {
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
    
    const rn = parseInt(parts[parts.length - 4]);
    const cellX = parseInt(parts[parts.length - 3]);
    const cellY = parseInt(parts[parts.length - 2]);
    const cellZ = parseInt(parts[parts.length - 1]);
    const fracX = parseFloat(parts[parts.length - 7]);
    const fracY = parseFloat(parts[parts.length - 6]);
    const fracZ = parseFloat(parts[parts.length - 5]);
    
    if (isNaN(rn) || isNaN(cellX) || isNaN(fracX)) continue;
    
    rnList.push(rn);
    xyzList.push([fracX, fracY, fracZ]);
    cellList.push([cellX, cellY, cellZ]);
  }
  
  return filterAndFormat(rnList, xyzList, cellList, atomDic, dim, atype, true);
}

function parseFrac(text, atomDic, dim, atype) {
  const lines = text.split('\n');
  
  const rnList = [];
  const xyzList = [];
  const cellList = [];
  
  // Skip header (5 lines)
  for (let i = 5; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    
    const parts = line.split(/\s+/);
    if (parts.length < 7) continue; // type, x, y, z, nx, ny, nz
    
    const rn = parseInt(parts[0]);
    const fracX = parseFloat(parts[1]);
    const fracY = parseFloat(parts[2]);
    const fracZ = parseFloat(parts[3]);
    const cellX = parseInt(parts[parts.length - 3]);
    const cellY = parseInt(parts[parts.length - 2]);
    const cellZ = parseInt(parts[parts.length - 1]);
    
    if (isNaN(rn) || isNaN(cellX) || isNaN(fracX)) continue;
    
    rnList.push(rn);
    xyzList.push([fracX, fracY, fracZ]);
    cellList.push([cellX, cellY, cellZ]);
  }
  
  return filterAndFormat(rnList, xyzList, cellList, atomDic, dim, atype, false);
}

function filterAndFormat(rnList, xyzList, cellList, atomDic, dim, atype, isRmc6f) {
  let allowed = null;
  if (atype !== 0) {
    allowed = new Set(atomDic[atype]);
  }
  
  const finalRn = [];
  const finalXyz = [];
  const finalCell = [];
  
  for (let i = 0; i < rnList.length; i++) {
    const rn = rnList[i];
    if (allowed && !allowed.has(rn)) continue;
    
    finalRn.push(rn);
    
    let x = xyzList[i][0] * dim[0];
    let y = xyzList[i][1] * dim[1];
    let z = xyzList[i][2] * dim[2];
    
    if (isRmc6f) {
      x = x % 1.0;
      y = y % 1.0;
      z = z % 1.0;
      if (x < 0) x += 1.0;
      if (y < 0) y += 1.0;
      if (z < 0) z += 1.0;
    } else {
      x = x % dim[0];
      y = y % dim[1];
      z = z % dim[2];
      if (x < 0) x += dim[0];
      if (y < 0) y += dim[1];
      if (z < 0) z += dim[2];
    }
    
    finalXyz.push(x, y, z);
    finalCell.push(cellList[i][0], cellList[i][1], cellList[i][2]);
  }
  
  return {
    atomType: new Int32Array(finalRn),
    xyz: new Float32Array(finalXyz),
    cellIdx: new Float32Array(finalCell)
  };
}
