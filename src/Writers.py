import numpy as np
# Assuming pymatgen is available as per original script comments
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter

def gen_ev_mcif(cifpath, atom_dic, vectors, eigen_num=np.arange(1), name=None):
    '''Generate MCIF files for eigenvectors'''
    structure = Structure.from_file(cifpath)
    
    # Reverse mapping
    rev_atm_dict = {}
    for element, numbers in atom_dic.items():
        for number in numbers:
            rev_atm_dict[number] = element

    labels_list = [f"{element}{number}" for element, numbers in atom_dic.items() for number in numbers]

    for ii in eigen_num:
        eigvecs = np.real(vectors[ii].reshape(len(vectors[ii])//3, 3))
        
        for jj in np.arange(len(labels_list)):
            atm_tmp = rev_atm_dict[jj+1]
            # Assumes 1-based indexing in atom_dic
            label_tmp = atm_tmp + str(jj+1) 
            
            # Find index in structure structure
            # Note: This requires the structure labels to match exactly
            try:
                idx_tmp = structure.labels.index(label_tmp)
                structure[idx_tmp].properties["magmom"] = eigvecs[jj]
            except ValueError:
                print(f"Warning: Label {label_tmp} not found in structure.")

        mcif_writer = CifWriter(structure, write_magmoms=True)
        
        out_dir = './Eigenvectors/'
        if name is None:
            fname = f'{out_dir}Eigenvector_#{ii}.mcif'
        else:
            fname = f'{out_dir}Eigenvector_#{ii}_{name}.mcif'
            
        mcif_writer.write_file(fname)