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