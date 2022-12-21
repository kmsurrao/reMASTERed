import sys
import os
import subprocess
import numpy as np
import healpy as hp
from input import Info
from bispectrum import *
from trispectrum import *
from test_remastered import *
import time
import string 
from plot_mask import *
start_time = time.time()

# main input file containing most specifications 
try:
    input_file = (sys.argv)[1]
except IndexError:
    input_file = 'moto.yaml'

# read in the input file and set up relevant info object
inp = Info(input_file, thresholding=False)

# current environment, also environment in which to run subprocesses
my_env = os.environ.copy()

lmax_data = 3*inp.nside-1

#load map and mask
map_ = hp.ud_grade(hp.read_map(inp.map_file), inp.nside)
mask = hp.ud_grade(hp.read_map(inp.mask_file), inp.nside)

#get alm and wlm for map and mask, respectively 
alm = hp.map2alm(map_)
wlm = hp.map2alm(mask)

#zero out modes above ellmax
l_arr,m_arr = hp.Alm.getlm(lmax_data)
alm = alm*(l_arr<=inp.ellmax)
wlm = wlm*(l_arr<=inp.ellmax)
map_ = hp.alm2map(alm, nside=inp.nside)
mask = hp.alm2map(wlm, nside=inp.nside)

#get auto- and cross-spectra for map and mask
Cl_aa = hp.alm2cl(alm, lmax_out=inp.ellmax)
Cl_ww = hp.alm2cl(wlm, lmax_out=inp.ellmax)
Cl_aw = hp.anafast(map_, mask, lmax=inp.ellmax)


#get list of map, mask, masked map, and correlation coefficient
if inp.save_files or inp.plot:
    data = [map_, mask, map_*mask] #contains map, mask, masked map, correlation coefficient
    base_dir = inp.output_dir
    if not os.path.isdir(base_dir):
        subprocess.call(f'mkdir {base_dir}', shell=True, env=my_env)
    corr = Cl_aw/np.sqrt(Cl_aa*Cl_ww)
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
Rho = rho(inp, map_-np.mean(map_), mask-np.mean(mask), Cl_aw, Cl_aa, Cl_ww)
# pickle.dump(Rho, open(f'rho/rho_isw_ellmax{inp.ellmax}.p', 'wb')) #remove
# Rho = pickle.load(open(f'rho/rho_isw_ellmax{inp.ellmax}.p', 'rb')) 

#get MASTER LHS (directly computed power spectrum of masked map)
master_lhs = hp.anafast(map_*mask, lmax=inp.ellmax)

#Get all terms of reMASTERed equation
print('Starting reMASTERed comparison', flush=True)
compare_master(inp, master_lhs, wlm[0], alm[0], Cl_aa, Cl_ww, Cl_aw, bispectrum_aaw, bispectrum_waw, Rho, my_env)

print("--- %s seconds ---" % (time.time() - start_time), flush=True)