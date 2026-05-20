import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import Readers

# Plot Styling
plt.rcParams['font.family'] = 'Dejavu Sans'
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['axes.facecolor'] = 'w'

def gen_3d_plot(fpath, atype='Ta'):
    try:
        from mayavi import mlab
    except ImportError:
        print("Mayavi not installed, skipping 3D plot")
        return

    atmtype, data, _ = Readers.read_frac_atom_ph(fpath, {}, np.array([1,1,1]), atype=atype) # simplified call
    xyz = np.transpose(data)
    kde = gaussian_kde(xyz)
    density = kde(xyz)

    figure = mlab.figure('DensityPlot')
    pts = mlab.points3d(xyz[0], xyz[1], xyz[2], density, scale_mode='none', scale_factor=0.01, color=(1,0,0))
    mlab.axes()
    mlab.show()

def plot_ph_dos(wks, Emin=0, Emax=8, binnum=50):
    fig = plt.figure(figsize=(3.375*4/3, 3.375))
    ax = fig.add_subplot(111)
    wks = np.array(wks)
    energies = wks.flatten()
    filtered_energies = energies[(energies >= Emin) & (energies <= Emax)]
    
    sns.histplot(data=filtered_energies, bins=binnum, alpha=0.6, kde=True, kde_kws={'bw_adjust':0.1}, ax=ax)
    
    ax.set_xlabel(r'Energy (arb. u.)', fontsize=10)
    ax.set_ylabel(r'Phonon DOS', fontsize=10)
    ax.set_xlim([Emin, Emax])
    plt.tight_layout()
    plt.show()

def plot_phonon_bands(ph_band, k_path, kstep, plot_PDOS=False):
    fig = plt.figure(figsize=(3.375, 3.375*16/9))
    ax = fig.add_subplot(111)
    
    tmp = np.transpose(ph_band)
    for ii in np.arange(len(tmp)):
        ax.plot(tmp[ii], color='r', lw=1.0)

    # x-axis ticks: segment boundaries at 0, kstep+1, 2*(kstep+1), …, last index
    # ph_band has (kstep+1) entries per segment (both endpoints included)
    n_segs = len(k_path) - 1
    tick_positions = [ii * (kstep + 1) for ii in range(n_segs)] + [n_segs * (kstep + 1) - 1]
    tick_labels = list(k_path)
    plt.xticks(tick_positions, tick_labels)

    # vertical lines
    for ii in tick_positions:
        ax.axvline(x=ii, color='gray', lw=0.5, linestyle='--', alpha=0.7)

    ax.tick_params(which='both', labelsize=9, direction='in', 
                   bottom=True, top=True, left=True, right=True)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['top'].set_linewidth(0.5)

    ax.set_ylabel(r'Energy (meV)', fontsize=10)
    ax.set_ylim([0, 20])
    plt.tight_layout()
    plt.show()