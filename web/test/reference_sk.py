#!/usr/bin/env python3
"""Independent (pure-Python, no numpy) reference for the S(k) computation.

Generates a small deterministic synthetic ensemble and computes the mass-weighted
displacement covariance S(k) exactly the way src_gpu/Calculators.Sk_avg does:

  - group atoms by RMC reference number (basis site), NOT by element,
  - displacement = (config - <config>) / dim @ v_super  (Cartesian Angstrom),
  - U_t(k) = (1/sqrt(N_t)) * sum_j sqrt(m_t) u_j exp(i * cell_j . kvec),
  - kvec = 2*pi * q_frac,
  - S(k) = < U(k)^dag U(k) > averaged over frames.

Writes web/test/reference.json (inputs + S(k) outputs). The Node validator
re-computes with the browser's engine-style real/imag split and must agree, and
also checks S(G) == S(Gamma) (the 2*pi periodicity validation).

This is intentionally a SECOND, independent implementation in a different
language so the JS result is validated against more than just itself.
"""
import json, math, os, random

random.seed(1234)
TWO_PI = 2.0 * math.pi

# ── Synthetic system ────────────────────────────────────────────────────────
# Non-contiguous reference numbers {5, 9} to exercise basis-site grouping.
DIM = [2, 1, 1]                      # 2 cells along x
V_SUPER = [[8.0, 0.0, 0.0],          # supercell lattice (rows), conventional a=4
           [0.0, 4.0, 0.0],
           [0.0, 0.0, 4.0]]
MASSES_BY_RN = {5: 28.0, 9: 16.0}    # amu per basis site

# Atoms: one per (rn, cell). 2 rns x 2 cells = 4 atoms.
BASE_FRAC = {5: [0.10, 0.20, 0.30], 9: [0.60, 0.55, 0.40]}  # within-cell frac
atoms = []  # {rn, cell, base_xyz(within-cell)}
for rn in (5, 9):
    for cx in range(DIM[0]):
        atoms.append({"rn": rn, "cell": [cx, 0, 0], "base": list(BASE_FRAC[rn])})

# Frames: small random displacements about the base (within-cell xyz).
N_FRAMES = 6
frames = []
for _ in range(N_FRAMES):
    xyz = []
    for a in atoms:
        xyz.append([a["base"][k] + random.uniform(-0.01, 0.01) for k in range(3)])
    frames.append(xyz)

K_POINTS = {"gamma": [0.0, 0.0, 0.0], "G": [1.0, 0.0, 0.0], "generic": [0.30, 0.0, 0.0]}


def matvec_rows(frac, M):
    return [frac[0]*M[0][i] + frac[1]*M[1][i] + frac[2]*M[2][i] for i in range(3)]


def compute_sk(qfrac):
    # unique reference numbers, sorted -> segment order (np.unique semantics)
    uniq = sorted({a["rn"] for a in atoms})
    seg = {rn: i for i, rn in enumerate(uniq)}
    counts = {rn: sum(1 for a in atoms if a["rn"] == rn) for rn in uniq}
    T = len(uniq)
    kvec = [TWO_PI * q for q in qfrac]

    # ensemble mean of within-cell xyz
    n = len(atoms)
    mean = [[0.0, 0.0, 0.0] for _ in range(n)]
    for fr in frames:
        for i in range(n):
            for c in range(3):
                mean[i][c] += fr[i][c]
    for i in range(n):
        for c in range(3):
            mean[i][c] /= len(frames)

    dim = DIM
    S = [[0j]*(3*T) for _ in range(3*T)]
    for fr in frames:
        # U_k accumulation per segment
        U = [[0j, 0j, 0j] for _ in range(T)]
        for i, a in enumerate(atoms):
            disp_frac = [(fr[i][c] - mean[i][c]) / dim[c] for c in range(3)]
            disp_cart = matvec_rows(disp_frac, V_SUPER)
            phase = sum(a["cell"][c] * kvec[c] for c in range(3))
            w = complex(math.cos(phase), math.sin(phase)) * math.sqrt(MASSES_BY_RN[a["rn"]])
            s = seg[a["rn"]]
            for c in range(3):
                U[s][c] += disp_cart[c] * w
        # normalize 1/sqrt(N_t)
        for rn in uniq:
            s = seg[rn]
            nf = 1.0 / math.sqrt(max(counts[rn], 1))
            for c in range(3):
                U[s][c] *= nf
        # flatten U (3T,) and accumulate S[i][j] += U_i * conj(U_j).
        # This matches the legacy src_gpu kernel convention
        #   Sk_real = A A^T + B B^T,  Sk_imag = B A^T - A B^T
        # (with U = A + iB). It is the complex conjugate / transpose of the plain
        # U^dag U; the matrix is Hermitian either way and the spectrum is identical.
        Uflat = [U[s][c] for s in range(T) for c in range(3)]
        for i in range(3*T):
            ui = Uflat[i]
            for j in range(3*T):
                S[i][j] += ui * Uflat[j].conjugate()
    inv = 1.0 / len(frames)
    return [[S[i][j]*inv for j in range(3*T)] for i in range(3*T)]


def serialize(S):
    return {"re": [[v.real for v in row] for row in S],
            "im": [[v.imag for v in row] for row in S]}


out = {
    "inputs": {"dim": DIM, "v_super": V_SUPER, "masses_by_rn": MASSES_BY_RN,
               "atoms": atoms, "frames": frames, "kpoints": K_POINTS},
    "outputs": {name: serialize(compute_sk(q)) for name, q in K_POINTS.items()},
}

dst = os.path.join(os.path.dirname(__file__), "reference.json")
with open(dst, "w") as f:
    json.dump(out, f)
print("wrote", dst, "with", len(out["outputs"]), "k-points; matrix dim =",
      len(out["outputs"]["gamma"]["re"]))
