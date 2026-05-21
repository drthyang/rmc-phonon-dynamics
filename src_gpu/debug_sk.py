#!/usr/bin/env python3
"""Prints intermediate stats at every step from file load → Sk(k).

If Sk_avg outputs all-zero CSVs, run this and look for the first stage that's
unexpectedly zero. No caches read or written.

  cd src_gpu && python3 debug_sk.py
"""
import os
os.environ["JAX_ENABLE_X64"] = "True"

import sys
import glob
import numpy as np
import jax.numpy as jnp

import Readers
import Calculators
from Calculators import process_batch_kernel, get_mass_array

# ── Config (matches main.py for T=5) ──────────────────────────────────────────
T          = 5
STEM       = '../data/'
FPATH_EQ   = STEM + f'{T}K_ini/GTS_{T}K.rmc6f'
FPATH      = STEM + f'ensemble_20A_{T}K/configs/'
N_FRAMES   = 5                                  # tiny subset is enough
KPNT       = np.array([0.5, 0.0, 0.0])          # X-point — should be non-zero

# ── Load structure ────────────────────────────────────────────────────────────
atom_dic = Readers.get_atom_idx(FPATH_EQ, verbose=0)
v1, v2, v3, dim = Readers.read_cell_vec(FPATH_EQ, verbose=0)
v_super = np.array([v1, v2, v3])

fnames = sorted(glob.glob(FPATH + 'Frac*.txt'))[:N_FRAMES]
if not fnames:
    print(f'NO FILES at {FPATH}'); sys.exit(1)
print(f'Using {len(fnames)} frames from {FPATH}')
print(f'dim     = {dim}')
print(f'v_super = (max abs entry) {np.abs(v_super).max():.3f} Å')

# ── Average (equilibrium) configuration ───────────────────────────────────────
hsym = Readers.avg_frac_atom_ph(fnames, atom_dic, dim)
hsym_xyz = hsym[1]
print(f'\nhsym_xyz : shape {hsym_xyz.shape}  '
      f'min={hsym_xyz.min():.4f}  max={hsym_xyz.max():.4f}  mean={hsym_xyz.mean():.4f}')

# ── Static arrays (matches Sk_avg) ────────────────────────────────────────────
atype_static = Readers.read_frac_atom_ph(fnames[0], atom_dic, dim)[0]
masses_gpu = jnp.array(get_mass_array(atype_static, atom_dic))
unique_types, type_indices_cpu = np.unique(atype_static, return_inverse=True)
type_indices_gpu = jnp.array(type_indices_cpu, dtype=jnp.int32)
num_types = int(len(unique_types))
kvec_gpu = jnp.array(KPNT, dtype=jnp.float32)
print(f'\natype_static len = {len(atype_static)}  unique = {num_types}')
print(f'masses_gpu       : min={float(masses_gpu.min()):.3f}  '
      f'max={float(masses_gpu.max()):.3f}  '
      f'nonzero={int((masses_gpu > 0).sum())}/{masses_gpu.size}')

# ── Build batch ───────────────────────────────────────────────────────────────
disp_list, cell_list = [], []
for fname in fnames:
    _, config, cell_idx = Readers.read_frac_atom_ph(fname, atom_dic, dim)
    disp = (config - hsym_xyz) / dim @ v_super
    disp_list.append(disp)
    cell_list.append(cell_idx)

disp_all = np.stack(disp_list)
cell_all = np.stack(cell_list)
print(f'\ndisp batch : shape {disp_all.shape}')
print(f'  per-frame |disp| max:  {[float(np.abs(d).max()) for d in disp_all]}')
print(f'  per-frame |disp| mean: {[float(np.abs(d).mean()) for d in disp_all]}')
if np.allclose(disp_all, 0):
    print('  ⚠️  DISPLACEMENTS ARE ALL ZERO — frames equal the mean.')

print(f'cell batch : shape {cell_all.shape}  '
      f'min={cell_all.min()}  max={cell_all.max()}')

# ── Call the kernel ───────────────────────────────────────────────────────────
disp_batch_gpu = jnp.array(disp_all, dtype=jnp.float32)
cell_batch_gpu = jnp.array(cell_all, dtype=jnp.float32)

batch_real, batch_imag = process_batch_kernel(
    kvec_gpu, disp_batch_gpu, cell_batch_gpu,
    masses_gpu, type_indices_gpu, num_types)

br = np.array(batch_real)
bi = np.array(batch_imag)
print(f'\nkernel out : shape Sk_real {br.shape}, Sk_imag {bi.shape}')
print(f'  |Sk_real|: min={np.abs(br).min():.3e}  max={np.abs(br).max():.3e}  '
      f'mean={np.abs(br).mean():.3e}  nonzero={(br != 0).sum()}/{br.size}')
print(f'  |Sk_imag|: min={np.abs(bi).min():.3e}  max={np.abs(bi).max():.3e}  '
      f'mean={np.abs(bi).mean():.3e}  nonzero={(bi != 0).sum()}/{bi.size}')

if np.allclose(br, 0) and np.allclose(bi, 0):
    print('\n❌ KERNEL RETURNED ZERO — the bug is in process_batch_kernel.')
else:
    print('\n✓ Kernel produces non-zero Sk on this input. '
          'The zero-CSV must come from the Sk_avg cache/save path, not the kernel.')
