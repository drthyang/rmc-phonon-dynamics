"""
Synthetic test for gen_phonopy_band_yaml and gen_vasp_phonon.
Uses small mock data (4 atoms: 2 Ta + 1 Ga + 1 Se) to stay fast.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src_gpu'))

import numpy as np
import Writers

# ── Synthetic system: 4 atoms, cubic cell ─────────────────────────────────────
# atom_dic: element -> sorted list of atom indices (1-based, matching RMC convention)
atom_dic = {'Ta': [1, 2], 'Ga': [3], 'Se': [4]}

# Primitive cell vectors (Å) and supercell dimension (1×1×1 for simplicity)
a = 5.178
v1 = np.array([a, 0.0, 0.0])
v2 = np.array([0.0, a, 0.0])
v3 = np.array([0.0, 0.0, a])
dim = np.array([1.0, 1.0, 1.0])

# hsym_test = (atom_type_list, xyz, cell_idx)
# xyz is fractional * dim (here dim=1 so xyz = fractional coords)
atom_type_list = [1, 2, 3, 4]   # atom indices in "file order"
xyz = np.array([
    [0.0,  0.0,  0.0],   # Ta1
    [0.5,  0.5,  0.5],   # Ta2
    [0.25, 0.25, 0.25],  # Ga3
    [0.75, 0.75, 0.75],  # Se4
])
cell_idx = np.zeros((4, 3), dtype=int)
hsym_test = (atom_type_list, xyz, cell_idx)

# ── Synthetic phonon data ──────────────────────────────────────────────────────
N_modes = 3 * len(atom_type_list)   # 12 modes for 4 atoms

sym_pnts = {
    'GM': np.array([0.0, 0.0, 0.0]),
    'X':  np.array([0.5, 0.0, 0.0]),
    'M':  np.array([0.5, 0.5, 0.0]),
}
k_path = ['GM', 'X', 'M']
kstep  = 4    # 4 q-points per segment → 8 total

n_qpts = (len(k_path) - 1) * kstep

rng = np.random.default_rng(42)

ph_band = []
eigenvectors_all = []

for qi in range(n_qpts):
    # Fake a Hermitian matrix and diagonalise it
    H = rng.standard_normal((N_modes, N_modes)) + 1j * rng.standard_normal((N_modes, N_modes))
    H = H + H.conj().T          # Hermitian
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    # Use abs so we don't get sqrt of negative
    freqs = np.sqrt(np.abs(eigenvalues)) * 0.5   # arbitrary fake THz-ish values
    ph_band.append(freqs)
    eigenvectors_all.append(eigenvectors)

# Also inject a NaN/inf to test graceful handling
ph_band[0][0] = float('nan')
ph_band[1][1] = float('inf')

out_dir = '/tmp/rmc_test_output/'

# ── Test gen_phonopy_band_yaml ─────────────────────────────────────────────────
print('=== Testing gen_phonopy_band_yaml ===')
Writers.gen_phonopy_band_yaml(
    atom_dic, hsym_test, v1, v2, v3, dim,
    ph_band, eigenvectors_all,
    k_path, sym_pnts, kstep,
    freq_factor=1.0,
    out_dir=out_dir,
)

yaml_path = out_dir + 'band.yaml'
with open(yaml_path) as f:
    content = f.read()

lines = content.splitlines()
print(f'  Lines written       : {len(lines)}')

# Spot-check key fields
checks = {
    'nqpoint': f'nqpoint: {n_qpts}',
    'npath':   f'npath: {len(k_path)-1}',
    'natom':   f'natom: {len(atom_type_list)}',
    'lattice present':       'lattice:',
    'reciprocal present':    'reciprocal_lattice:',
    'points present':        'points:',
    'phonon present':        'phonon:',
    'eigenvector present':   'eigenvector:',
    'Ta in points':          'symbol: Ta',
    'Ga in points':          'symbol: Ga',
    'Se in points':          'symbol: Se',
    'nan replaced with 0':   '    frequency:     0.0000000000',
    'distance monotone':     True,
}

# Check that distances are monotonically non-decreasing
dist_lines = [l for l in lines if l.strip().startswith('distance:')]
dists = [float(l.split(':')[1]) for l in dist_lines]
checks['distance monotone'] = all(dists[i] <= dists[i+1] for i in range(len(dists)-1))

# Count q-position entries
n_qpos = sum(1 for l in lines if 'q-position' in l)
checks['q-position count'] = n_qpos == n_qpts

# Count eigenvector blocks
n_ev = sum(1 for l in lines if 'eigenvector:' in l)
checks['eigenvector count'] = n_ev == n_qpts * N_modes

all_pass = True
for name, check in checks.items():
    if isinstance(check, bool):
        ok = check
    else:
        ok = check in content
    status = 'PASS' if ok else 'FAIL'
    if not ok:
        all_pass = False
    print(f'  [{status}] {name}')

# ── Print a small excerpt of the YAML ─────────────────────────────────────────
print('\n=== band.yaml excerpt (first 60 lines) ===')
print('\n'.join(lines[:60]))

# ── Test gen_vasp_phonon ───────────────────────────────────────────────────────
print('\n=== Testing gen_vasp_phonon ===')
Writers.gen_vasp_phonon(
    atom_dic, hsym_test, v1, v2, v3, dim,
    eigenvectors=eigenvectors_all[0],   # use first k-point eigenvectors
    mode_indices=[0, 1, 2],
    amplitude=0.03,
    out_dir=out_dir + 'VASP/',
)

vasp_files = os.listdir(out_dir + 'VASP/')
print(f'  Files in VASP/: {sorted(vasp_files)}')

with open(out_dir + 'VASP/POSCAR') as f:
    poscar = f.read()
print('\n=== POSCAR ===')
print(poscar)

print('\n=== All tests', 'PASSED' if all_pass else 'FAILED', '===')
