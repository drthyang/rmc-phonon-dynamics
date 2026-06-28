// web/src/math/diagonalize.js
import { Matrix, EigenvalueDecomposition } from 'ml-matrix';

// ENERGY_CONV is imported by callers from ../constants.js and passed into
// eigenvaluesToMev(); it is intentionally NOT redefined here.

/**
 * Diagonalize a complex Hermitian matrix (A + iB) by constructing a 2N x 2N real symmetric matrix.
 * Returns eigenvalues and complex eigenvectors.
 * 
 * @param {Float64Array} Sk_real - A, flat array of size NxN
 * @param {Float64Array} Sk_imag - B, flat array of size NxN
 * @param {number} N - Dimension of the original matrix
 */
export function eigh(Sk_real, Sk_imag, N) {
  // Construct 2N x 2N matrix M = [A, -B; B, A]
  const M = new Matrix(2 * N, 2 * N);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const a = Sk_real[i * N + j];
      const b = Sk_imag[i * N + j];
      
      M.set(i, j, a);                 // Top-left: A
      M.set(i, j + N, -b);            // Top-right: -B
      M.set(i + N, j, b);             // Bottom-left: B
      M.set(i + N, j + N, a);         // Bottom-right: A
    }
  }
  
  // Symmetrize to fix tiny float32 accumulation asymmetries
  for (let i = 0; i < 2 * N; i++) {
    for (let j = i + 1; j < 2 * N; j++) {
      const val = (M.get(i, j) + M.get(j, i)) / 2;
      M.set(i, j, val);
      M.set(j, i, val);
    }
  }

  const evd = new EigenvalueDecomposition(M);
  const realEigenvalues = evd.realEigenvalues;
  const eigenvectorsMatrix = evd.eigenvectorMatrix;

  // The eigenvalues appear in identical pairs (lambda, lambda).
  // We need to extract N unique eigenvalues and their corresponding complex eigenvectors.
  
  // To robustly extract pairs, we sort them by eigenvalue.
  const paired = [];
  for (let i = 0; i < 2 * N; i++) {
    paired.push({ val: realEigenvalues[i], col: i });
  }
  paired.sort((a, b) => a.val - b.val);

  const finalEigenvalues = new Float64Array(N);
  
  // The complex eigenvectors will be represented as [real, imag] or separate arrays.
  // We'll return an array of { real: Float64Array, imag: Float64Array } of length N.
  const finalEigenvectors = [];

  for (let k = 0; k < N; k++) {
    // Take one from each pair (index 2*k)
    const idx = paired[2 * k].col;
    finalEigenvalues[k] = paired[2 * k].val;
    
    // For M v = lambda v, where v = [x; y], the complex eigenvector is x + iy.
    const vecReal = new Float64Array(N);
    const vecImag = new Float64Array(N);
    
    for (let i = 0; i < N; i++) {
      vecReal[i] = eigenvectorsMatrix.get(i, idx);
      vecImag[i] = eigenvectorsMatrix.get(i + N, idx);
    }
    
    // The Python np.linalg.eigh returns eigenvectors in columns, just like this.
    // However, it usually normalizes the complex vector to 1.
    let normSq = 0;
    for (let i = 0; i < N; i++) {
      normSq += vecReal[i] * vecReal[i] + vecImag[i] * vecImag[i];
    }
    const invNorm = 1.0 / Math.sqrt(Math.max(normSq, 1e-12));
    
    for (let i = 0; i < N; i++) {
      vecReal[i] *= invNorm;
      vecImag[i] *= invNorm;
    }
    
    finalEigenvectors.push({ real: vecReal, imag: vecImag });
  }

  return { eigenvalues: finalEigenvalues, eigenvectors: finalEigenvectors };
}

/**
 * Convert S(k) eigenvalues [amu A^2] to phonon energies [meV].
 * Mirrors python: E = ENERGY_CONV * sqrt(T / lambda)
 */
export function eigenvaluesToMev(eigenvalues, tempK, energyConv) {
  const result = new Float64Array(eigenvalues.length);
  const threshold = 1e-4;
  
  for (let i = 0; i < eigenvalues.length; i++) {
    const ev = eigenvalues[i];
    const absEv = Math.abs(ev);
    
    if (absEv < threshold) {
      result[i] = 0.0;
    } else {
      let energy = energyConv * Math.sqrt(tempK / absEv);
      if (isNaN(energy)) energy = 0.0;
      
      // Soft modes (ev < 0) returned as negative energies
      result[i] = ev >= 0 ? energy : -energy;
    }
  }
  return result;
}
