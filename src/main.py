#!/usr/bin/env python3
import numpy as np
import glob
from tqdm.auto import trange
from tqdm.auto import tqdm  # Make sure to import tqdm specifically

# Import modules
import Readers
import Calculators
import Writers
import Visualization

# --- Configuration Constants ---
amu = 1.66 * 10**-27 
kb = 8.6173303 * 10**-2 # meV/K
T = 5

stempath = '/Users/tt9/Research/LacunarSpinels/rmc/server_data/phonon/' 
fpath_eq = stempath + '5K_ini/GTS_5K.rmc6f'
fpath_eq_frac = stempath + '5K_ini/Frac_coord_GTS_5K.txt'
fpath = stempath + 'ensemble_20A_5K/configs/'

plot_PDOS = True
plot_PartialDOS = True

sym_pnts = {
    'A': np.array([0.5, 0.5, 0.5]),
    'GM': np.array([0.0, 0.0, 0.0]),
    'M': np.array([0.5, 0.5, 0.0]),
    'R': np.array([0.0, 0.5, 0.5]),
    'X': np.array([0.0, 0.5, 0.0]),
    'Z': np.array([0.0, 0.0, 0.5]),
    'hh-h': np.array([0.5, 0.5, -0.5]),
    'h00': np.array([0.5, 0.0, 0.0])
}

# --- Main Execution ---

if __name__ == "__main__":
    # 1. Read Initial Info
    atom_dic = Readers.get_atom_idx(fpath_eq)
    v1, v2, v3, dim = Readers.read_cell_vec(fpath_eq)

    # 2. Get Files and Average Configuration
    rmcfiles = glob.glob(fpath + 'Frac*.txt')
    print('🔎 Found ** {} ** configurations ... '.format(len(rmcfiles)))
    
    # Calculate average configuration (High Symmetry)
    hsym_test = Readers.avg_frac_atom_ph(rmcfiles, atom_dic, dim)

    # 3. Define k-path
    k_path = ['GM', 'h00']
    print('📊 Calculating phonon bands along : {} ...'.format(k_path))
    
    ph_band = []
    kstep = 16

    # 4. Loop over k-path
    # for ii in trange(len(k_path)-1, desc='Overall progress', disable=True):
    #     k_plot_vec = sym_pnts[k_path[ii+1]] - sym_pnts[k_path[ii]]
        
    #     for jj in trange(kstep, desc=f'k-path {k_path[ii]}–{k_path[ii+1]}', disable=True):
    #         current_k = sym_pnts[k_path[ii]] + jj * k_plot_vec / kstep
            
    #         # Calculate S(k)
    #         Sk = Calculators.Sk_avg(fpath, hsym_test, atom_dic, dim, current_k)
            
    #         # Diagonalize
    #         eigenvalues, eigenvectors = np.linalg.eigh(Sk)
            
    #         # Store bands (using meV conversion)
    #         # Note: Avoid division by zero if eigenvalues are very small/negative
    #         with np.errstate(divide='ignore', invalid='ignore'):
    #             ph_band.append(np.sqrt(kb * T / eigenvalues)) 

    #         # (Optional) Generate MCIF at Gamma point
    #         # if jj == 0:
    #         #     Writers.gen_ev_mcif('./test.cif', atom_dic, eigenvectors, name=k_path[ii])

    # 4-1. Calculate total expected iterations
    total_iterations = (len(k_path) - 1) * kstep

    # 4-2. Create the main progress bar
    with tqdm(total=total_iterations, desc='⏩️ Total Progress') as pbar:
        
        # Loop over path segments
        for ii in range(len(k_path)-1):  # Use range, not trange
            k_plot_vec = sym_pnts[k_path[ii+1]] - sym_pnts[k_path[ii]]
            
            # Loop over steps in segment
            for jj in range(kstep):      # Use range, not trange
                current_k = sym_pnts[k_path[ii]] + jj * k_plot_vec / kstep
                
                # Calculate S(k)
                Sk = Calculators.Sk_avg(fpath, hsym_test, atom_dic, dim, current_k)
                
                # Diagonalize
                eigenvalues, eigenvectors = np.linalg.eigh(Sk)
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    ph_band.append(np.sqrt(kb * T / eigenvalues)) 

                # 4-3. Manually update the bar by 1 step
                pbar.update(1)
                #(Optional) Generate MCIF at Gamma point
                if jj == 0:
                    Writers.gen_ev_mcif('../data/GTS_5K.cif', atom_dic, eigenvectors, name=k_path[ii])          
                       
    # 5. Plot Bands
    Visualization.plot_phonon_bands(ph_band, k_path, kstep)

    # 6. PDOS Calculation
    if plot_PDOS:
        qpnts = Calculators.gen_grid(5)
        wk = []
        for qpnt in qpnts:
            Sk = Calculators.Sk_avg(fpath, hsym_test, atom_dic, dim, qpnt)
            eigenvalues, _ = np.linalg.eigh(Sk)
            wk.append(1/np.sqrt(np.real(eigenvalues * eigenvalues.conj())))
        Visualization.plot_ph_dos(wk)

    # 7. Partial PDOS Calculation
    if plot_PartialDOS:
        elements = ['Ta', 'Ga', 'Se']
        for partial_type in elements:
            print(f'Calculating partial phonon DOS for {partial_type} atoms ...')
            hsym_partial = Readers.read_frac_atom_ph(fpath_eq_frac, atom_dic, dim, atype=partial_type)
            
            qpnts = Calculators.gen_grid(5)
            wk = []
            for qpnt in qpnts:
                Sk = Calculators.Partial_Sk_avg(fpath, hsym_partial, atom_dic, dim, qpnt, partial_type)
                eigenvalues, _ = np.linalg.eigh(Sk)
                wk.append(1/np.sqrt(np.real(eigenvalues * eigenvalues.conj())))
            
            print(f"Plotting Partial DOS for {partial_type}")
            Visualization.plot_ph_dos(wk)