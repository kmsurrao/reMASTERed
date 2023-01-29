import sys
import os
import subprocess
import numpy as np
import healpy as hp
import time
from input import Info
from bispectrum import *
from trispectrum import *
from test_remastered import *
from wigner3j import *
from plot_mask import *

start_time = time.time()

# main input file containing most specifications 
try:
    input_file = (sys.argv)[1]
except IndexError:
    input_file = 'moto.yaml'

# read in the input file and set up Info object
inp = Info(input_file, mask_provided=True)

# current environment, also environment in which to run subprocesses
my_env = os.environ.copy()

# base directory to save data and figures
base_dir = inp.output_dir

#get wigner 3j symbols
if inp.wigner_file != '':
    inp.wigner3j = pickle.load(open(inp.wigner_file, 'rb'))[:inp.ell_sum_max+1, :inp.ell_sum_max+1, :inp.ell_sum_max+1]
else:
    inp.wigner3j = compute_3j(inp.ell_sum_max)

#load map and mask
map_ = hp.ud_grade(hp.read_map(inp.map_file), inp.nside)
mask = hp.ud_grade(hp.read_map(inp.mask_file), inp.nside)

#get one point functions
alm_00 = hp.map2alm(map_)[0]
wlm_00 = hp.map2alm(mask)[0]

#get auto- and cross-spectra for map and mask
Cl_aa = hp.anafast(map_, lmax=inp.ell_sum_max)
Cl_ww = hp.anafast(mask, lmax=inp.ell_sum_max)
Cl_aw = hp.anafast(map_, mask, lmax=inp.ell_sum_max)
Cl_aa_mean_rem = hp.anafast(map_-np.mean(map_), lmax=inp.ell_sum_max)
Cl_ww_mean_rem = hp.anafast(mask-np.mean(mask), lmax=inp.ell_sum_max)
Cl_aw_mean_rem = hp.anafast(map_-np.mean(map_), mask-np.mean(mask), lmax=inp.ell_sum_max)


#get list of map, mask, masked map, and correlation coefficient
if inp.save_files or inp.plot:
    data = [map_, mask, map_*mask] #will contain map, mask, masked map, correlation coefficient
    if not os.path.isdir(base_dir):
        subprocess.call(f'mkdir {base_dir}', shell=True, env=my_env)
    corr = Cl_aw[:inp.ellmax+1]/np.sqrt(Cl_aa[:inp.ellmax+1]*Cl_ww[:inp.ellmax+1])
    data.append(corr)
    if inp.save_files:
        pickle.dump(data, open(f'{base_dir}/mask_data.p', 'wb'))
        print(f'saved {base_dir}/mask_data.p', flush=True)
    if inp.plot:
        plot_mask(inp, data, base_dir)

#Compute bispectrum for aaw and waw
print('Starting bispectrum calculation', flush=True)
bispectrum_aaw = Bispectrum(inp, map_-np.mean(map_), map_-np.mean(map_), mask-np.mean(mask), equal12=True)
bispectrum_waw = Bispectrum(inp, mask-np.mean(mask), map_-np.mean(map_), mask-np.mean(mask), equal13=True)

#Compute rho (unnormalized trispectrum)
print('Starting rho calculation', flush=True)
Rho = rho(inp, map_-np.mean(map_), mask-np.mean(mask), Cl_aw_mean_rem, Cl_aa_mean_rem, Cl_ww_mean_rem, remove_two_point=True)

#get MASTER LHS (directly computed power spectrum of masked map)
master_lhs = hp.anafast(map_*mask, lmax=inp.ellmax)

#Get all terms of reMASTERed equation
print('Starting reMASTERed comparison', flush=True)
compare_master(inp, master_lhs, wlm_00, alm_00, Cl_aa, Cl_ww, Cl_aw, bispectrum_aaw, bispectrum_waw, Rho, my_env, base_dir=base_dir)

print("--- %s seconds ---" % (time.time() - start_time), flush=True)