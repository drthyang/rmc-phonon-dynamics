import numpy as np
import glob
from tqdm.auto import trange
import pandas as pd
from joblib import Parallel, delayed

def read_cell_vec(fname, verbose=1):
    '''Read cell vectors and supercell dimension from a *.rmc6f'''
    if verbose != 0:
        print("📖 Reading cell information from {}".format(fname))
    with open(fname, 'r') as f:
        lines = f.readlines()
    
    v1, v2, v3, dim = None, None, None, None
    
    for ii in np.arange(len(lines)):
        if lines[ii].split()[0] == 'Lattice':
            v1 = np.array(np.float64(lines[ii+1].split()))
            v2 = np.array(np.float64(lines[ii+2].split()))
            v3 = np.array(np.float64(lines[ii+3].split()))
            if verbose != 0:
                print('Lattice vectors:\n', v1, '\n', v2, '\n', v3)
            break
        elif lines[ii].split()[0] == 'Supercell':
            dim = np.float64(lines[ii].split()[-3:])
            if verbose != 0:
                print('Supercell dimensions = {}'.format(dim))
                
    return v1, v2, v3, np.array(dim)

def get_atom_idx(fname, verbose=1):
    '''Get mapping of atom types to indices'''
    if verbose != 0:
        print("📖 Reading atom indices from {}".format(fname))
    with open(fname, 'r') as f:
        lines = f.readlines()
        
    atom_dic = {}
    idx_ini = 0
    for ii in np.arange(len(lines)):
        if lines[ii].split()[0] == 'Atoms:':
            idx_ini = ii
            break
            
    for ii in np.arange(idx_ini+1, len(lines), 1):
        ln = lines[ii].split()
        atom = ln[1]
        atom_idx = int(ln[-4])
        if atom not in atom_dic:
            atom_dic[atom] = [atom_idx]
        else:
            atom_dic[atom].append(atom_idx)
            
    for key in atom_dic:
        atom_dic[key] = list(set(atom_dic[key]))
    return atom_dic

def read_frac_atom_ph_legacy(fname, atom_dic, dim, atype=0, mode='Frac', v1_norm=None, v2_norm=None, v3_norm=None):
    '''Read fractional coordinates from Frac*.txt'''
    with open(fname, 'r') as f:
        lines = f.readlines()
        
    atmtype = []
    data = []
    cell_idx = []
    
    for ii in np.arange(5, len(lines), 1):
        ln = lines[ii].split()
        current_atom_type = int(ln[0])
        
        # Check if we should process this atom
        process = False
        if atype == 0:
            process = True
        elif current_atom_type in atom_dic[atype]:
            process = True
            
        if process:
            atmtype.append(current_atom_type)
            xyz = np.array(np.float64(ln[1:4])) * dim
            xyz = np.array([x-dim[0] if x > 1 else x for x in xyz])
            cell = np.array([int(ln[-3]), int(ln[-2]), int(ln[-1])])
            
            if mode == 'Frac':
                data.append(xyz)
                cell_idx.append(cell)
            else:
                # NOTE: cvt_pos was not defined in the original script provided.
                # You must define cvt_pos or ensure mode is always 'Frac'
                # xyz = np.array(cvt_pos(xyz, v1_norm, v2_norm, v3_norm))
                pass 
                
    return atmtype, np.array(data), np.array(cell_idx)

def avg_frac_atom_ph_legacy(fnames, atom_dic, dim, atype=0, mode='Frac'):
    '''Calculate average configuration from multiple files'''
    data_accum = None
    cell_tmp = None
    
    for fidx in trange(len(fnames), desc='📊 Calculating average configuration', disable=False):
        atmtype, data, cell_idx = read_frac_atom_ph(fnames[fidx], atom_dic, dim, atype, mode)
        
        if fidx == 0:
            data_accum = np.array(data)
        else:
            data_accum += np.array(data)
            
        if fidx > 1 and cell_tmp is not None:
             if cell_tmp.shape != np.array(cell_idx).shape or not (cell_tmp == np.array(cell_idx)).all():
                print('The cell indices do not match ... Please check ...')
        cell_tmp = np.array(cell_idx)
        
    data_avg = np.array(data_accum) / len(fnames)
    return atmtype, np.array(data_avg), np.array(cell_tmp)



def read_frac_atom_ph(fname, atom_dic, dim, atype=0, mode='Frac'):
    '''
    Fast vectorized reader using Pandas.
    '''
    # 1. Read the file efficiently
    # skiprows=5 skips the header lines
    # sep='\s+' handles variable whitespace
    try:
        df = pd.read_csv(fname, skiprows=5, header=None, sep='\s+', 
                         usecols=[0, 1, 2, 3, 4, 5, 6],
                         names=['type', 'x', 'y', 'z', 'c1', 'c2', 'c3'],
                         engine='c') # Engine 'c' is faster
    except pd.errors.EmptyDataError:
        # Handle empty files gracefully
        return np.array([]), np.array([]), np.array([])

    # 2. Vectorized Filtering
    if atype != 0:
        # Filter rows where 'type' is in the dictionary list
        valid_types = atom_dic[atype]
        df = df[df['type'].isin(valid_types)]
    
    # If filtered result is empty
    if df.empty:
        return np.array([]), np.array([]), np.array([])

    # 3. Coordinate Calculation (Vectorized)
    # Convert fractional to cartesian
    xyz = df[['x', 'y', 'z']].to_numpy(dtype=np.float64) * dim
    
    # Apply the specific boundary condition logic:
    # "if x > 1: x - dim[0]"
    # We use a boolean mask to do this for the whole array at once.
    mask = xyz > 1
    xyz[mask] -= dim[0] 

    # 4. Extract other arrays
    atmtype = df['type'].to_numpy(dtype=int)
    cell_idx = df[['c1', 'c2', 'c3']].to_numpy(dtype=int)

    # Note: mode='Frac' is the only one implemented in your original snippet
    # so we return directly.
    return atmtype, xyz, cell_idx

def avg_frac_atom_ph(fnames, atom_dic, dim, atype=0, mode='Frac', n_jobs=-1):
    '''
    Parallelized average configuration calculator.
    '''
    print(f"📊 Calculating average across {len(fnames)} files (Parallel)")

    # 1. Define the worker function for a single file
    def process_file(f):
        t, d, c = read_frac_atom_ph(f, atom_dic, dim, atype, mode)
        return d, c  # Return data and cell_idx
    
    # 2. Run in parallel
    # We only need the first file's atom types to return at the end
    first_atmtype, _, _ = read_frac_atom_ph(fnames[0], atom_dic, dim, atype, mode)
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_file)(f) for f in fnames
    )
    
    # results is a list of tuples: [(data1, cell1), (data2, cell2), ...]
    
    # 3. Validation and Aggregation
    # We unzip the results into two lists
    all_data = []
    all_cells = []
    
    for d, c in results:
        all_data.append(d)
        all_cells.append(c)

    # Convert to arrays for easier checking
    # Note: Using a loop for the consistency check is safer/easier than vectorizing 
    # the check across ragged arrays, and it's fast enough here.
    ref_cell = all_cells[0]
    for i, c in enumerate(all_cells[1:]):
        # Quick shape check first, then value check
        if c.shape != ref_cell.shape or not np.array_equal(c, ref_cell):
            print(f'⚠️ Warning: Cell indices do not match in file index {i+1} ... Please check ...')

    # 4. Summation and Averaging
    # Stack along a new axis and take the mean
    # Stack shape: (n_files, n_atoms, 3) -> mean axis 0 -> (n_atoms, 3)
    try:
        data_stack = np.array(all_data)
        data_avg = np.mean(data_stack, axis=0)
    except ValueError as e:
        print("❌ Error combining data: Atom counts might vary between frames.")
        raise e

    return first_atmtype, data_avg, ref_cell