import numpy as np
import glob
from tqdm.auto import trange

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

def read_frac_atom_ph(fname, atom_dic, dim, atype=0):
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
            xyz = np.mod(xyz, dim)  # wrap each component to [0, dim[i]) independently
            cell = np.array([int(ln[-3]), int(ln[-2]), int(ln[-1])])
            data.append(xyz)
            cell_idx.append(cell)

    return atmtype, np.array(data), np.array(cell_idx)

def avg_frac_atom_ph(fnames, atom_dic, dim, atype=0):
    '''Calculate average configuration from multiple files'''
    data_accum = None
    cell_tmp = None

    for fidx in trange(len(fnames), desc='📊 Calculating average configuration', disable=False):
        atmtype, data, cell_idx = read_frac_atom_ph(fnames[fidx], atom_dic, dim, atype)
        
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