#!/usr/bin/env python3
"""Validate that reading a configuration from *.rmc6f matches reading the same
configuration from its Frac*.txt — the premise of the unified config reader.

read_frac_atom_ph and read_rmc6f_atom_ph must return IDENTICAL
(atom_type, within-cell xyz, cell_idx), because Sk_avg subtracts the reference
per atom; any convention drift (e.g. global vs within-cell coords) silently
corrupts displacements by integer cell offsets.

Pairing for the 5K ensemble: configs/Frac_coord_N.txt  <->  GTS_5K_N.rmc6f
(both are the same RMC snapshot; the rmc6f carries full precision, the Frac is
rounded to 5 decimals, so positions agree to ~1e-4).

Usage:  python validate_rmc6f_equiv.py [N]      (default N=1)
"""
import os
os.environ["JAX_ENABLE_X64"] = "True"
import sys
import numpy as np
import Readers

R = "/Users/tt9/Research/GitHub/rmc-phonon-dynamics"
D = R + "/data/ensemble_20A_5K"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1

sf = D + "/GTS_5K.rmc6f"
atom_dic = Readers.get_atom_idx(sf, verbose=0)
v1, v2, v3, dim = Readers.read_cell_vec(sf, verbose=0)
dimv = np.asarray(dim, float)

frac = Readers.read_frac_atom_ph(f"{D}/configs/Frac_coord_{N}.txt", atom_dic, dim)
rmc = Readers.read_rmc6f_atom_ph(f"{D}/GTS_5K_{N}.rmc6f", atom_dic, dim)

ok = True
# atom-type (RN) ordering
rn_ok = list(frac[0]) == list(rmc[0])
ok &= rn_ok
print(f"atom_type (RN) ordering match : {rn_ok}  ({len(frac[0])} atoms)")
# cell indices
cell_ok = np.array_equal(np.asarray(frac[2]), np.asarray(rmc[2]))
ok &= cell_ok
print(f"cell_idx match                : {cell_ok}")
# within-cell coords (minimum-image, to tolerate boundary wrap + Frac rounding)
d = np.asarray(rmc[1]) - np.asarray(frac[1])
d -= np.round(d / dimv) * dimv
maxd = float(np.abs(d).max())
xyz_ok = maxd < 1e-3
ok &= xyz_ok
print(f"within-cell xyz max|diff|     : {maxd:.2e}  (< 1e-3 ? {xyz_ok})")

print("\n" + ("PASS — readers are equivalent; unified reader is sound."
              if ok else "FAIL — readers disagree; do NOT use rmc6f and Frac interchangeably."))
sys.exit(0 if ok else 1)
