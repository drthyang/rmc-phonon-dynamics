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

# Constants
amu = 1.66 * 10**-27 # amu to kg
kb = 8.6173303 * 10**-5 # eV/K

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