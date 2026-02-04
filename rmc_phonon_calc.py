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
fpath_eq = '/Users/tt9/Research/LacunarSpinels/rmc/server_data/phonon/5K_ini/GTS_5K.rmc6f'
fpath_eq_frac = '/Users/tt9/Research/LacunarSpinels/rmc/server_data/phonon/5K_ini/Frac_coord_GTS_5K.txt'
fpath = '/Users/tt9/Research/LacunarSpinels/rmc/server_data/phonon/ensemble_20A_5K/configs/'

# NERSC GTS 250K 10*10*10 50A
#fpath_eq = '/Users/tt9/Research/LacunarSpinels/rmc/NERSC/ensemble_250K/initial/GTS_250K_0.rmc6f'
#fpath_eq_frac = '/Users/tt9/Research/LacunarSpinels/rmc/NERSC/ensemble_250K/initial/Frac_coord_0.txt'
#fpath = '/Users/tt9/Research/LacunarSpinels/rmc/NERSC/ensemble_250K/configs/'


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
	atomic_mass = {'Ga':69.723,
					'V':50.942,'Nb':92.906,'Ta':180.95,
					'Se':78.971}
	# Step 1: Create reverse mapping dictionary
	reverse_mapping = {v: k for k, values in atom_dic.items() for v in values}
	# Step 2: Replace numbers in the list with corresponding keys
	replaced_list = [reverse_mapping[num] for num in atom_idx]
	# Step 3: Replace atom type with mass
	mass_array = [atomic_mass[atom] for atom in replaced_list]
	return mass_array

# Generate a neutron cross-section array for each atoms
def get_nxs_array(atom_idx,atom_dic) :
	# unit = barn
	atomic_mass = {'Ga':6.83,
					'V':5.1,'Nb':6.255,'Ta':6.01,
					'Se':8.3}
	# Step 1: Create reverse mapping dictionary
	reverse_mapping = {v: k for k, values in atom_dic.items() for v in values}
	# Step 2: Replace numbers in the list with corresponding keys
	replaced_list = [reverse_mapping[num] for num in atom_idx]
	# Step 3: Replace atom type with mass
	mass_array = [atomic_mass[atom] for atom in replaced_list]
	return mass_array

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



