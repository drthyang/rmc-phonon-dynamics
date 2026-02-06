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