#!/usr/bin/env python3
import numpy as np
import sys, glob
import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.transforms import Affine2D
import seaborn as sns
from tqdm import tqdm
from tqdm.auto import trange
import csv
import os
from pathlib import Path

plt.rcParams['font.family'] = 'Dejavu Sans'
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['lines.linewidth']= 1
plt.rcParams['axes.facecolor'] = 'w'

# Constants
amu = 1.66 * 10**-27 # amu to kg
kb = 8.6173303 * 10**-5 # eV/K



#T = 250
#T = 5

plot_PDOS = True

##### Input section #####

# Initial configuration (high symmetry)
#fpath_eq = '/Users/tt9/Research/LacunarSpinels/structure/phonon/GTS_{}K/GTS_{}K.rmc6f'.format(T,T)
#fpath_eq_frac = '/Users/tt9/Research/LacunarSpinels/structure/phonon/GTS_{}K/Frac_coord_GTS_{}K.txt'.format(T,T)


# Folder contains all Frac*.txt
#fpath = '/Users/tt9/Research/LacunarSpinels/rmc/server_data/GTS/2_15A/{}K_try1/ensemble/'.format(T)
#fpath = '/Users/tt9/Research/LacunarSpinels/rmc/server_data/GTS/2_15A/250K_try1/ensemble_tmp/'

# NERSC GTS 250K 8*8*8 20A 
#fpath_eq = '/Users/tt9/Research/LacunarSpinels/rmc/NERSC/ensemble_20A_250K/initial/GTS_250K_0.rmc6f'
#fpath_eq_frac = '/Users/tt9/Research/LacunarSpinels/rmc/NERSC/ensemble_20A_250K/initial/Frac_coord_0.txt'
#fpath = '/Users/tt9/Research/LacunarSpinels/rmc/NERSC/ensemble_20A_250K/configs/'

# NERSC GTS 5K 8*8*8 20A 
# fpath_eq = '/Users/tt9/Research/LacunarSpinels/rmc/server_data/phonon/5K_ini/GTS_5K.rmc6f'
# fpath_eq_frac = '/Users/tt9/Research/LacunarSpinels/rmc/server_data/phonon/5K_ini/Frac_coord_GTS_5K.txt'
# fpath = '/Users/tt9/Research/LacunarSpinels/rmc/server_data/phonon/ensemble_20A_5K/configs/'

# NERSC GTS 250K 10*10*10 50A
fpath_eq = '/Users/tt9/Research/LacunarSpinels/rmc/NERSC/ensemble_250K/initial/GTS_250K_0.rmc6f'
fpath_eq_frac = '/Users/tt9/Research/LacunarSpinels/rmc/NERSC/ensemble_250K/initial/Frac_coord_0.txt'
fpath = '/Users/tt9/Research/LacunarSpinels/rmc/NERSC/ensemble_250K/configs/'


sym_pnts = {
'A': np.array([0.5000000000,0.5000000000,0.5000000000]),
'GM': np.array([0.0000000000,0.0000000000,0.0000000000]),
'M': np.array([0.5000000000,0.5000000000,0.0000000000]),
'R': np.array([0.0000000000,0.5000000000,0.5000000000]),
'X': np.array([0.0000000000,0.5000000000,0.0000000000]),
'Z': np.array([0.0000000000,0.0000000000,0.5000000000])
}


# fname is the file name of *.rmc6f 
# Extract the cell information
def read_cell_vec(fname,verbose=1) :
	#fname = glob.glob(fpath + '*.rmc6f')[0]
	if verbose!=0 :
		print(fname)
	f = open(fname,'r')
	lines = f.readlines()
	f.close()
	for ii in np.arange(len(lines)) :
		if (lines[ii].split()[0]=='Lattice') :
			v1 = np.array( np.float64(lines[ii+1].split()) )
			v2 = np.array( np.float64(lines[ii+2].split()) )
			v3 = np.array( np.float64(lines[ii+3].split()) )
			if verbose!=0 :
				print('Lattice vectors:')
				print(v1)
				print(v2)
				print(v3)
			break
		elif (lines[ii].split()[0]=='Supercell') :
			dim = np.float64(lines[ii].split()[-3:])
			if verbose!=0 :
				print('Supercell dimensions = {}'.format(dim))
	return v1,v2,v3,np.array(dim)

# fname is the file name of *.rmc6f 
# Get the index and make a dictionary
def get_atom_idx(fname,verbose=1) :
	#fname = glob.glob(fpath + '*.rmc6f')[0]
	if verbose!=0 :
		print(fname)
	f = open(fname,'r')
	lines = f.readlines()
	f.close()
	atom_dic = {}
	for ii in np.arange(len(lines)) :
		if (lines[ii].split()[0]=='Atoms:') :
			idx_ini = ii
			break
	for ii in np.arange(idx_ini+1,len(lines),1) :
		ln = lines[ii].split()
		#print(ln)
		atom = ln[1]
		atom_idx = int(ln[-4])
		if atom not in atom_dic :
			atom_dic[atom] = [atom_idx]
		else :
			atom_dic[atom].append(atom_idx)
	for key in atom_dic :
		atom_dic[key] = list(set(atom_dic[key]))
	return atom_dic

# fname is the file name of Frac*.txt
# Read the fractional atomic coordinates from Frac*.txt
def read_frac_atom_ph(fname,atom_dic,dim,atype=0,mode='Frac') :
	#stem_name = fname.split('/')[-1].split('.')[0]
	#fpath = fname.split(stem_name+'.rmc6f')[0]
	'''
	#############################################################
	# get atom idx
	atom_dic = get_atom_idx(fname)
	print(atom_dic)
	# get lattice vecs and supercell dims
	v1,v2,v3,dim = read_cell_vec(fname)
	UB = np.array([v1,v2,v3])
	v1_norm = v1/dim[0]
	v2_norm = v2/dim[1]
	v3_norm = v3/dim[2]
	v1_norm = v1_norm/np.sqrt(np.dot(v1_norm,v1_norm))
	v2_norm = v2_norm/np.sqrt(np.dot(v2_norm,v2_norm))
	v3_norm = v3_norm/np.sqrt(np.dot(v3_norm,v3_norm))
	############################################################
	'''
	#fname = glob.glob(fpath + 'Frac*.txt')[0]
	#fracname = fpath + 'Frac_coord_' + stem_name + '.txt'
	fracname = fname
	f = open(fracname,'r')
	lines = f.readlines()
	atmtype = []
	data = []
	cell_idx = []
	for ii in np.arange(5,len(lines),1) :
		ln = lines[ii].split()
		#print(ln[0])
		if atype==0 :
			atmtype.append(int(ln[0]))
			xyz = np.array(np.float64(ln[1:4]))*dim # Fractional coord. for unit cell
			#print(xyz)
			xyz = np.array( [x-dim[0] if x > 1 else x for x in xyz] )
			cell = np.array([int(ln[-3]),int(ln[-2]),int(ln[-1])])
			if mode=='Frac' :
				data.append(xyz)
				cell_idx.append(cell)
			else :
				xyz = np.array( cvt_pos(xyz,v1_norm,v2_norm,v3_norm) )
				data.append(xyz)
				cell_idx.append(cell)
		elif (int(ln[0]) in atom_dic[atype]) :
			atmtype.append(int(ln[0]))
			xyz = np.array(np.float64(ln[1:4]))*dim
			xyz = np.array( [x-dim[0] if x > 1 else x for x in xyz] )
			cell = np.array([int(ln[-3]),int(ln[-2]),int(ln[-1])])
			if mode=='Frac' :
				data.append(xyz)
				cell_idx.append(cell)
			else :
				xyz = np.array( cvt_pos(xyz,v1_norm,v2_norm,v3_norm) )
				data.append(xyz)
				cell_idx.append(cell)
	return atmtype, np.array(data), np.array(cell_idx)

def gen_3d_plot(fpath,atype='Ta') :
	# 3D plot
	atmtype, data = read_frac_atom(fpath,atype)
	xyz = np.transpose(data)
	kde = gaussian_kde(xyz)
	density = kde(xyz)

	from mayavi import mlab
	# Plot scatter with mayavi
	figure = mlab.figure('DensityPlot')
	pts = mlab.points3d(xyz[0], xyz[1], xyz[2], density, scale_mode='none', scale_factor=0.01, color=(1,0,0))
	mlab.axes()
	mlab.show()

def select_atom_type(tag,atype,config,cell_idx) :
	sel_idx = []
	for ii in np.arange(len(atype)) :
		if atype[ii]==tag :
			sel_idx.append(ii)
	return config[sel_idx], cell_idx[sel_idx]

# Generate a mass array for calculating U_k
def get_mass_array(atom_idx,atom_dic) :
	# Atomic mass in amu (g/mol)
	atomic_mass = {
		'H': 1.008, 'He': 4.0026, 'Li': 6.94, 'Be': 9.0122, 'B': 10.81, 'C': 12.011,
		'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180, 'Na': 22.990, 'Mg': 24.305,
		'Al': 26.982, 'Si': 28.085, 'P': 30.974, 'S': 32.06, 'Cl': 35.45, 'K': 39.098,
		'Ar': 39.948, 'Ca': 40.078, 'Sc': 44.956, 'Ti': 47.867, 'V': 50.942, 'Cr': 51.996,
		'Mn': 54.938, 'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.38,
		'Ga': 69.723, 'Ge': 72.63, 'As': 74.922, 'Se': 78.96, 'Br': 79.904, 'Kr': 83.798,
		'Rb': 85.468, 'Sr': 87.62, 'Y': 88.906, 'Zr': 91.224, 'Nb': 92.906, 'Mo': 95.96,
		'Tc': 98.0, 'Ru': 101.07, 'Rh': 102.91, 'Pd': 106.42, 'Ag': 107.87, 'Cd': 112.41,
		'In': 114.82, 'Sn': 118.71, 'Sb': 121.76, 'Te': 127.60, 'I': 126.90, 'Xe': 131.29,
		'Cs': 132.91, 'Ba': 137.33, 'La': 138.91, 'Ce': 140.12, 'Pr': 140.91, 'Nd': 144.24,
		'Pm': 145.0, 'Sm': 150.36, 'Eu': 151.96, 'Gd': 157.25, 'Tb': 158.93, 'Dy': 162.50,
		'Ho': 164.93, 'Er': 167.26, 'Tm': 168.93, 'Yb': 173.05, 'Lu': 174.97, 'Hf': 178.49,
		'Ta': 180.95, 'W': 183.84, 'Re': 186.21, 'Os': 190.23, 'Ir': 192.22, 'Pt': 195.08,
		'Au': 196.97, 'Hg': 200.59, 'Tl': 204.38, 'Pb': 207.2, 'Bi': 208.98, 'Po': 209.0,
		'At': 210.0, 'Rn': 222.0, 'Fr': 223.0, 'Ra': 226.0, 'Ac': 227.0, 'Th': 232.04,
		'Pa': 231.04, 'U': 238.03, 'Np': 237.0, 'Pu': 244.0, 'Am': 243.0, 'Cm': 247.0,
		'Bk': 247.0, 'Cf': 251.0, 'Es': 252.0, 'Fm': 257.0, 'Md': 258.0, 'No': 259.0,
		'Lr': 262.0
	}
	# Step 1: Create reverse mapping dictionary
	reverse_mapping = {v: k for k, values in atom_dic.items() for v in values}
	# Step 2: Replace numbers in the list with corresponding keys
	replaced_list = [reverse_mapping[num] for num in atom_idx]
	# Step 3: Replace atom type with mass
	mass_array = [atomic_mass[atom] for atom in replaced_list]
	return mass_array

# Generate a neutron cross-section array for each atoms
def get_nxs_array(atom_idx,atom_dic) :
 	# Total Neutron Scattering Cross-Sections (sigma_scatt) in Barns
	# Source: NIST Center for Neutron Research (Natural Abundance)
	neutron_scatt_sigma = {
		'H': 82.02,  'D': 7.64,   'He': 1.34,  'Li': 1.37,  'Be': 7.63,
		'B': 5.24,   'C': 5.551,  'N': 11.51,  'O': 4.232,  'F': 4.018,
		'Ne': 2.62,  'Na': 3.28,  'Mg': 3.71,  'Al': 1.503, 'Si': 2.167,
		'P': 3.31,   'S': 1.026,  'Cl': 16.8,  'K': 1.96,   'Ar': 0.68,
		'Ca': 2.83,  'Sc': 23.4,  'Ti': 4.35,  'V': 5.10,   'Cr': 3.49,
		'Mn': 2.15,  'Fe': 11.62, 'Co': 6.07,  'Ni': 18.5,  'Cu': 8.03,
		'Zn': 4.054, 'Ga': 6.83,  'Ge': 8.42,  'As': 5.48,  'Se': 8.30,
		'Br': 5.9,   'Kr': 7.66,  'Rb': 6.32,  'Sr': 6.25,  'Y': 7.76,
		'Zr': 6.46,  'Nb': 6.253, 'Mo': 5.71,  'Tc': 6.0,   'Ru': 5.1,
		'Rh': 4.81,  'Pd': 4.5,   'Ag': 4.99,  'Cd': 6.5,   'In': 2.61,
		'Sn': 4.89,  'Sb': 3.9,   'Te': 4.25,  'I': 3.55,   'Xe': 4.3,
		'Cs': 4.23,  'Ba': 3.38,  'La': 9.81,  'Ce': 2.94,  'Pr': 3.58,
		'Nd': 16.5,  'Pm': 16.0,  'Sm': 39.0,  'Eu': 9.2,   'Gd': 175.0, # High absorption!
		'Tb': 23.0,  'Dy': 34.4,  'Ho': 8.8,   'Er': 8.0,   'Tm': 8.5,
		'Yb': 23.4,  'Lu': 5.9,   'Hf': 5.88,  'Ta': 6.01,  'W': 4.87,
		'Re': 13.5,  'Os': 12.6,  'Ir': 14.0,  'Pt': 11.71, 'Au': 7.63,
		'Hg': 26.5,  'Tl': 9.89,  'Pb': 11.11, 'Bi': 9.16,  'Th': 12.63,
		'U': 14.16
	}
	# Step 1: Create reverse mapping dictionary
	reverse_mapping = {v: k for k, values in atom_dic.items() for v in values}
	# Step 2: Replace numbers in the list with corresponding keys
	replaced_list = [reverse_mapping[num] for num in atom_idx]
	# Step 3: Replace atom type with neutron corss-section
	nxs_array = [neutron_scatt_sigma[atom] for atom in replaced_list]
	return nxs_array

# Calculate the displacement 
#@njit(parallel=True)
def calc_collect_var(kvec,atype,configuration,cell_idx,hsymconfig,atom_dic) :
	kvec = np.array(kvec)
	# Step 1 : Calculate the displacements
	displacements = configuration - hsymconfig
	# Step 2 : Obtain types of atoms
	atom_idx = np.array(list(set(atype)))
	# Step 3 : Generate mass list
	mass_array = get_mass_array(atype,atom_dic)
	# Step 4 : Loop through atypes and calculate U_k_t[atype]
	# Might be able to speed up by getting rip of the double-loop
	U_k_t = []
	for ii in np.arange(len(atom_idx)) :
		tmp_config, tmp_cell_idx = select_atom_type(atom_idx[ii],atype,displacements,cell_idx)
		tmp_data = 0
		tmp_cnt = 0
		for jj in np.arange(len(tmp_cell_idx)) :
			tmp_data += np.sqrt(mass_array[ii]) * tmp_config[jj] * np.exp(1j * np.dot(kvec, tmp_cell_idx[jj]))
			tmp_cnt += 1
		U_k_t.append(tmp_data/np.sqrt(tmp_cnt))
	# Step 5 : Construct T(k)
	U_k_t = np.array(U_k_t)
	# Step 6 : Flatten the matrix to vector
	U_k_t = U_k_t.reshape(1,-1)
	return U_k_t

# Construct the matrix Sk
def calc_Sk(U_k_t) :
	# U_k_t is a one cloumn vector
	# Step 1 : make a column vector
	U_l = U_k_t.T
	# Step 2 : Take a complex conjugate
	U_r = U_k_t.conj()
	# Step 3 : Generate the matrix of S(k)
	result = U_l @ U_r
	return result

def Sk_avg(fpath,hsym_config,atom_dic,dim,kpnt,loadfile=True,save=True) :
	verbose=0
	fnames = glob.glob(fpath + 'Frac*.txt')
	# Load previously calculated S(k)
	saved_Sk = fpath + 'Sk_sum_kvec_{}_{}_{}.csv'.format(*kpnt)
	if (loadfile==True and os.path.exists(saved_Sk)) :
		#print('Found previously saved S(k) ... ')
		with open(saved_Sk, 'r') as file:
			reader = csv.reader(file)
			# Read the header
			header = next(reader)
			ini_idx = int(header[0]) + 1 # starting from the next configuration
			# Read the matrix
			Sk_sum = np.array([list(map(complex, row)) for row in reader])
		show_prog = False
	else :
		ini_idx = 0
		show_prog = False

	for ii in trange(ini_idx,len(fnames),desc='k={}'.format(kpnt),leave=None,position=0,disable=show_prog) :
		test = read_frac_atom_ph(fnames[ii],atom_dic,dim)
		U_k = calc_collect_var(kpnt,test[0],test[1],test[2],hsym_config[1],atom_dic)
		Sk = calc_Sk(U_k)
		if ii==0 :
			Sk_sum = Sk
		else :
			Sk_sum += Sk

	if (save==True and ini_idx<len(fnames)) :
		with open(fpath + 'Sk_sum_kvec_{}_{}_{}.csv'.format(*kpnt), 'w', newline='') as file:
			writer = csv.writer(file)
			# Write the number of configurations
			writer.writerow([len(fnames)])
			# Write the matrix rows
			writer.writerows(Sk_sum)
		#print('Saved the accumulated S(k) for k = {} '.format(kpnt))
	Sk_avg = Sk_sum/len(fnames)
	return Sk_avg

# generate grid in FBZ for Phonon DOS
def gen_grid(n_points=5) :
	# generate the grid in [-0.5,0.5]
	q_min = -0.5
	#q_min = 0
	q_max = 0.5
	# Generate linearly spaced grid points along each axis
	qx = np.linspace(q_min, q_max, n_points)
	qy = np.linspace(q_min, q_max, n_points)
	qz = np.linspace(q_min, q_max, n_points)
	# Create a 3D grid for q-points
	q_points = np.array(np.meshgrid(qx, qy, qz)).T.reshape(-1, 3)
	return q_points

# Input all eigenvalues (energies)
def PhDOS(wks,Emin=0,Emax=5,binnum=200) :
	fig = plt.figure(figsize=(3.375*4/3,3.375))
	ax = fig.add_subplot(111)
	wks = np.array(wks)
	energies = wks.flatten()
	filtered_energies = energies[(energies >= Emin) & (energies <= Emax)]
	ax = sns.histplot(data=filtered_energies, bins=binnum, alpha=0.6, kde=True, kde_kws={'bw_adjust':0.2})
	#ax = sns.kdeplot(data=filtered_energies, bw_adjust=0.1, cut=0)
	#ax.hist(filtered_energies,range=(Emin,Emax),bins=binnum)
	#ax.hist(energies,range=(Emin,Emax),bins=binnum)
	ax.set_xlabel(r'Energy (arb. u.)',fontsize=10)
	ax.set_ylabel(r'Phonon DOS',fontsize=10)
	# Add instrumental resolution function

# Eigenvectors


############################################################################################
############################################################################################
############################################################################################


#atom_dic = get_atom_idx(fpath)
#test = read_frac_atom_ph(fpath)
atom_dic = get_atom_idx(fpath_eq)
v1,v2,v3,dim = read_cell_vec(fpath_eq)
hsym_test = read_frac_atom_ph(fpath_eq_frac,atom_dic,dim)

# Γ—X—M—Γ—Z—R—A—Z
k_path = ['GM','X','M','GM','Z','R','A','Z']
#k_path = ['GM','X','M','GM']
#k_path = ['GM','X']
print('Calculating phonon bands along : {} ...'.format(k_path))
rmcfiles = glob.glob(fpath + 'Frac*.txt')
print('Found ** {} ** configurations ... '.format(len(rmcfiles)))
ph_band = []

kstep = 16

for ii in trange(len(k_path)-1,desc='Overall progress',disable=True) :
	k_plot_vec = sym_pnts[k_path[ii+1]] - sym_pnts[k_path[ii]]
	for jj in trange(kstep,desc='k-path {}–{}'.format(k_path[ii],k_path[ii+1]),disable=True) :
		Sk = Sk_avg(fpath,hsym_test,atom_dic,dim,sym_pnts[k_path[ii]]+jj*k_plot_vec/kstep)
		eigenvalues, eigenvectors = np.linalg.eig(Sk) # diagonalize the S(k)
		ph_band.append(1/np.sqrt(np.real(eigenvalues*eigenvalues.conj()))) # take dot product and only keep real part

fig = plt.figure(figsize=(3.375,3.375*16/9))
ax = fig.add_subplot(111)
tmp = np.transpose(ph_band)
for ii in np.arange(len(tmp)) :
	ax.plot(np.sqrt(tmp[ii]),color='r',lw=0.5)
	#plot(tmp[ii],ls=' ',marker='o')


# x-axis label
tick_positions = np.arange(0,len(k_path)*kstep,kstep)
tick_labels = [k_path[ii] for ii in range(len(tick_positions))]
plt.xticks(tick_positions, tick_labels)

# vertical lines
for ii in tick_positions :
	ax.axvline(x=ii,color='gray',lw=0.5,linestyle='--',alpha=0.7)

#ax.xaxis.set_major_locator(MultipleLocator(kstep))
#ax.xaxis.set_minor_locator(MultipleLocator(1))

ax.tick_params(axis='both',which='major',labelsize=8)
ax.spines['left'].set_linewidth(0.5)
ax.spines['right'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
ax.spines['top'].set_linewidth(0.5)

ax.tick_params(which='both', labelsize=9,
		labelbottom=True, labeltop=False, labelleft=False, labelright=False,
		bottom=True, top=True, left=True, right=True, direction='in')
#ax.set_xlabel(r'r ($\mathrm{\AA}$)',fontsize=10)
ax.set_ylabel(r'Energy (arb. u.)',fontsize=10)
#ax.set_xlabel(r'Q ($\mathrm{\AA^{-1}}$)',fontsize=10)
#ax.set_ylabel(r'S(Q)',fontsize=10)
ax.set_ylim([0,20])

if plot_PDOS==True :
	qpnts = gen_grid(5)
	wk = []
	for qpnt in qpnts :
		Sk = Sk_avg(fpath,hsym_test,atom_dic,dim,qpnt)
		eigenvalues, eigenvectors = np.linalg.eigh(Sk) # diagonalize the S(k)
		#wk.append(1/np.sqrt(real(eigenvalues*eigenvalues.conj()))) # take dot product and only keep real part
		wk.append(1/np.sqrt(eigenvalues))
	PhDOS(wk)



