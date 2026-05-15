#!/usr/bin/env python3
import numpy as np
#import cupy as cp
#from numba import njit, prange
from scipy.stats import gaussian_kde
import sys, glob
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.transforms import Affine2D
#import seaborn as sns
from tqdm import tqdm
from tqdm.auto import trange
import csv
import os
from pathlib import Path
#from pymatgen.core import Structure
#from pymatgen.io.cif import CifWriter

plt.rcParams['font.family'] = 'Dejavu Sans'
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['lines.linewidth']= 1
plt.rcParams['axes.facecolor'] = 'w'

# Constants
amu = 1.66 * 10**-27 # amu to kg
kb = 8.6173303 * 10**-5 # eV/K

#T = 250
#T = 5

#stempath = '/home/tt9/rmc_thy/LS/phonon/' # big bird
stempath = '/Users/tt9/Research/LacunarSpinels/rmc/server_data/phonon/' # local mac

# 5 K 24 hrs
#fpath_eq = stempath + '5K_ini/GTS_5K.rmc6f'
#fpath_eq_frac = stempath + '5K_ini/Frac_coord_GTS_5K.txt'
#fpath = stempath + 'ensemble_20A_5K/configs/'

# 250 K 12 hrs
fpath_eq = stempath + 'ensemble_20A_250K/initial/GTS_250K_0.rmc6f'
fpath_eq_frac = stempath + 'ensemble_20A_250K/initial/Frac_coord_0.txt'
fpath = stempath + 'ensemble_20A_250K/configs/'



sym_pnts = {
'A': np.array([0.5000000000,0.5000000000,0.5000000000]),
'GM': np.array([0.0000000000,0.0000000000,0.0000000000]),
'M': np.array([0.5000000000,0.5000000000,0.0000000000]),
'R': np.array([0.0000000000,0.5000000000,0.5000000000]),
'X': np.array([0.0000000000,0.5000000000,0.0000000000]),
'Z': np.array([0.0000000000,0.0000000000,0.5000000000])
}


# fname is the file name of *.rmc6f 
def read_cell_vec(fname,verbose=1) :
	'''
	Read cell vectors and supercell dimension from a *.rmc6f
	
	Parameters :
	fname (string) : file name of *.rmc6f
	verbose (integer) : level of output messeges, 0 = no messege

	Return :
	v1, v2, v3, dim
	v1, v2, v3 (1 by 3 array) : cell vectors
	dim (1 by 3 array) : supercell dimensions 
	'''
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

# Read the fractional atomic coordinates from Frac*.txt
# fname is the file name of Frac*.txt
# use atype to select cetrain type of atom for partial PhDOS
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

# Calculate the average configuration
def avg_frac_atom_ph(fnames,atom_dic,dim,atype=0,mode='Frac') :
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
	#for fidx in np.arange(len(fnames)) :
	for fidx in trange(len(fnames),desc='Calculating average configuration',disable=False) :
		f = open(fnames[fidx],'r')
		lines = f.readlines()
		f.close()
		atmtype = []
		data = []
		cell_idx = []
		for ii in np.arange(5,len(lines),1) :
			ln = lines[ii].split()
			if atype==0 :
				atmtype.append(int(ln[0]))
				xyz = np.array(np.float64(ln[1:4]))*dim # Fractional coord. for unit cell
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
		if fidx==0 :
			data_accum = np.array(data)
		else :
			data_accum += np.array(data)
		if (fidx>1 and cell_tmp.all()!=np.array(cell_idx).all()) :
			print('The cell indecies do not match ... Please check ...')
		cell_tmp = np.array(cell_idx)
	data_avg = np.array(data_accum)/len(fnames)
	return atmtype, np.array(data_avg), np.array(cell_idx)

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


################ Parameters
atom_type = 0 # 0=all

num_configs = 100

xmin = 0
xmax = 1
ymin = 0
ymax = 1
zmin = 0.5
zmax = 0.75

num_bins = 90
bw_method = 0.02

################ Parameters

atom_dic = get_atom_idx(fpath_eq)
v1,v2,v3,dim = read_cell_vec(fpath_eq)
hsym_test = read_frac_atom_ph(fpath_eq_frac,atom_dic,dim,atype=atom_type)

rmcfiles = glob.glob(fpath + 'Frac*.txt')

X0_all = []
Y0_all = []
Z0_all = []
X_all = []
Y_all = []
Z_all = []
U_all = []
V_all = []
W_all = []

for ii in np.arange(num_configs) :
	test = read_frac_atom_ph(rmcfiles[ii],atom_dic,dim,atype=atom_type)
	diff = test[1] - hsym_test[1]
	hsym_coords = np.transpose(hsym_test[1])
	coords = np.transpose(test[1])
	diff = np.transpose(diff)

	X0_all.append( hsym_coords[0] )
	Y0_all.append( hsym_coords[1] )
	Z0_all.append( hsym_coords[2] )
	X_all.append( coords[0] )
	Y_all.append( coords[1] )
	Z_all.append( coords[2] )
	U_all.append( diff[0] )
	V_all.append( diff[1] )
	W_all.append( diff[2] )

X0_all = np.array(X0_all).flatten()
Y0_all = np.array(Y0_all).flatten()
Z0_all = np.array(Z0_all).flatten()
X_all = np.array(X_all).flatten()
Y_all = np.array(Y_all).flatten()
Z_all = np.array(Z_all).flatten()
U_all = np.array(U_all).flatten()
V_all = np.array(V_all).flatten()
W_all = np.array(W_all).flatten()
magnitude_all = np.sqrt(U_all**2 + V_all**2 + W_all**2)

idx_x = [i for i, x in enumerate(X0_all) if (x >= xmin and x <= xmax) ]
idx_y = [i for i, y in enumerate(Y0_all) if (y >= ymin and y <= ymax) ]
idx_z = [i for i, z in enumerate(Z0_all) if (z >= zmin and z <= zmax) ]


idx_plot = list(set(idx_x).intersection(idx_y, idx_z))

X0_all = X0_all[idx_plot]
Y0_all = Y0_all[idx_plot]
Z0_all = Z0_all[idx_plot]
X_all = X_all[idx_plot]
Y_all = Y_all[idx_plot]
Z_all = Z_all[idx_plot]
U_all = U_all[idx_plot]
V_all = V_all[idx_plot]
W_all = W_all[idx_plot]

# Calculate the magnitude of each vector
magnitude = np.sqrt(U_all**2 + V_all**2)

# Normalize the magnitudes to [0, 1] for color mapping
magnitude_norm = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())

# Define the line widths based on the magnitude (scaled up for visibility)
linewidths = 0.5 + 100 * magnitude_norm  # Adjust scaling factor as needed

# Create a colormap
cmap = plt.cm.coolwarm  # You can choose any colormap

plt.figure(figsize=(8, 6))

# Plot the vectors with color based on amplitude (magnitude)
plt.quiver(X0_all, Y0_all, U_all, V_all, magnitude_norm, cmap=cmap, angles='xy', scale_units='xy', scale=2, alpha=0.7, lw=2)
plt.xlim(0,1)
plt.ylim(0,1)

##########################
# Polar density map

fig, axes = plt.subplots(1, 3, subplot_kw={'projection': 'polar'}, figsize=(15, 5))

# Convert vectors to polar coordinates (angle in radians)
angles = np.arctan2(V_all, U_all)
counts, bin_edges = np.histogram(angles, bins=num_bins)
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

# Normalize counts for colormap
norm = Normalize(vmin=min(counts), vmax=max(counts))
cmap = plt.cm.coolwarm  # Choose a colormap

# Plot each bar with color based on its count
for count, angle in zip(counts, bin_centers):
    color = cmap(norm(count))  # Map count to color
    axes[0].bar(angle, count, width=(2 * np.pi) / num_bins, color=color, alpha=0.8)

# Add a color bar for reference
#sm = ScalarMappable(cmap=cmap, norm=norm)
#sm.set_array([])  # Required for ScalarMappable but unused here
#fig.colorbar(sm, label="Counts")
angles_extended = np.concatenate([angles, angles + 2 * np.pi, angles - 2 * np.pi])
# Calculate the KDE for the angle data
kde = gaussian_kde(angles_extended, bw_method=bw_method)  # Adjust bandwidth as needed
theta_vals = np.linspace(-np.pi, np.pi, 100)
kde_vals = kde(theta_vals)

kde_vals = kde_vals * max(counts) / max(kde_vals)

# Overlay the KDE on the polar plot
axes[0].plot(theta_vals, kde_vals * max(counts) / max(kde_vals), color='darkblue', linewidth=2)


# Labels and title
axes[0].set_title("XY-plane")
#plt.show()
# Set the maximum radial limit
max_limit = max(max(counts), max(kde_vals)) * 1.4  # Adjust multiplier as needed for padding
axes[0].set_ylim(0, max_limit)

#############################
#############################

# Polar density map
# Convert vectors to polar coordinates (angle in radians)
angles = np.arctan2(V_all, W_all)

counts, bin_edges = np.histogram(angles, bins=num_bins)
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

# Normalize counts for colormap
#norm = Normalize(vmin=min(counts), vmax=max(counts))
#cmap = plt.cm.coolwarm  # Choose a colormap

# Plot each bar with color based on its count
for count, angle in zip(counts, bin_centers):
    color = cmap(norm(count))  # Map count to color
    axes[1].bar(angle, count, width=(2 * np.pi) / num_bins, color=color, alpha=0.8)

# Add a color bar for reference
#sm = ScalarMappable(cmap=cmap, norm=norm)
#sm.set_array([])  # Required for ScalarMappable but unused here
#axes[1].colorbar(sm, label="Counts")

angles_extended = np.concatenate([angles, angles + 2 * np.pi, angles - 2 * np.pi])
# Calculate the KDE for the angle data
kde = gaussian_kde(angles_extended, bw_method=bw_method)  # Adjust bandwidth as needed
theta_vals = np.linspace(-np.pi, np.pi, 100)
kde_vals = kde(theta_vals)

kde_vals = kde_vals * max(counts) / max(kde_vals)

# Overlay the KDE on the polar plot
axes[1].plot(theta_vals, kde_vals * max(counts) / max(kde_vals), color='darkblue', linewidth=2)

# Labels and title
axes[1].set_title("XZ-plane")
#plt.show()
axes[1].set_ylim(0, max_limit)

#############################
#############################

# Polar density map
# Convert vectors to polar coordinates (angle in radians)
angles = np.arctan2(U_all, W_all)

counts, bin_edges = np.histogram(angles, bins=num_bins)
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

# Normalize counts for colormap
#norm = Normalize(vmin=min(counts), vmax=max(counts))
#cmap = plt.cm.coolwarm  # Choose a colormap

# Plot each bar with color based on its count
for count, angle in zip(counts, bin_centers):
    color = cmap(norm(count))  # Map count to color
    axes[2].bar(angle, count, width=(2 * np.pi) / num_bins, color=color, alpha=0.8)

# Add a color bar for reference
#sm = ScalarMappable(cmap=cmap, norm=norm)
#sm.set_array([])  # Required for ScalarMappable but unused here
#axes[2].colorbar(sm, label="Counts")

angles_extended = np.concatenate([angles, angles + 2 * np.pi, angles - 2 * np.pi])
# Calculate the KDE for the angle data
kde = gaussian_kde(angles_extended, bw_method=bw_method)  # Adjust bandwidth as needed
theta_vals = np.linspace(-np.pi, np.pi, 100)
kde_vals = kde(theta_vals)

kde_vals = kde_vals * max(counts) / max(kde_vals)

# Overlay the KDE on the polar plot
axes[2].plot(theta_vals, kde_vals * max(counts) / max(kde_vals), color='darkblue', linewidth=2)

# Labels and title
axes[2].set_title("YZ-plane")

axes[2].set_ylim(0, max_limit)

plt.show()
