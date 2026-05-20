#!/bin/bash
#
#SBATCH --job-name=phbandpar
#SBATCH --output=phband_partial.out
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=256

ulimit -s unlimited
export OMP_NUM_THREADS=8

/home/tt9/rmc_thy/LS/phonon/rmc_phonon_calc_local.py

#RMCProfile_PATH=/home/tt9/RMCProfile_package
#export PGPLOT_DIR=$RMCProfile_PATH/exe/libs
#export LD_LIBRARY_PATH=$RMCProfile_PATH/exe/libs
#export LIBRARY_PATH=$RMCProfile_PATH/exe/libs
#export PATH=$PATH:$RMCProfile_PATH/exe
#$RMCProfile_PATH/exe/rmcprofile FeCoSn_100K
