import sys
sys.path.insert(0, "./../" )
import os
import subprocess
import numpy as np
import healpy as hp
import multiprocessing as mp
from input import Info
from generate_mask import *
from bispectrum import *
from interpolate_bispectrum import *
from test_remastered import *
import time
start_time = time.time()

# main input file containing most specifications 
try:
    input_file = (sys.argv)[1]
except IndexError:
    input_file = 'moto.yaml'

# read in the input file and set up relevant info object
inp = Info(input_file)

# current environment, also environment in which to run subprocesses
my_env = os.environ.copy()

def one_sim(inp, sim, offset):
    np.random.seed(sim)
    lmax_data = 3*inp.nside-1

    #get simulated map
    map_ = hp.read_map(inp.map_file) 
    map_cl = hp.anafast(map_, lmax=inp.ellmax)
    map_ = hp.synfast(map_cl, inp.nside)

    #create W=a+A mask for component map
    print('***********************************************************', flush=True)
    print('Starting mask generation', flush=True)
    mask = map_ + offset
    print('min of mask: ', np.amin(mask), flush=True)

    #get power spectra and bispectra
    print('***********************************************************', flush=True)
    print('Starting power spectra calculation', flush=True)
    alm = hp.map2alm(map_)
    wlm = hp.map2alm(mask)

    #zero out modes above ellmax
    l_arr,m_arr = hp.Alm.getlm(lmax_data)
    alm = alm*(l_arr<=inp.ellmax)
    wlm = wlm*(l_arr<=inp.ellmax)
    map_ = hp.alm2map(alm, nside=inp.nside)
    mask = hp.alm2map(wlm, nside=inp.nside)

    masked_map = map_*mask
    Cl_aa = hp.alm2cl(alm, lmax_out=inp.ellmax)
    Cl_ww = hp.alm2cl(wlm, lmax_out=inp.ellmax)
    Cl_aw = hp.anafast(map_, mask, lmax=inp.ellmax)

    lhs = hp.anafast(masked_map, lmax=inp.ellmax)

    return lhs, Cl_aa, Cl_ww, Cl_aw


#read map
map_ = hp.read_map(inp.map_file) 
map_ = hp.ud_grade(map_, inp.nside)

#find offset A for mask W=a+A
offset = 1.5*abs(np.amin(map_))
print('offset: ', offset, flush=True)

pool = mp.Pool(min(inp.nsims, 16))
results = pool.starmap(one_sim, [(inp, sim, offset) for sim in range(inp.nsims)])
pool.close()
lhs = np.mean(np.array([res[0] for res in results]), axis=0)
Cl_aa = np.mean(np.array([res[1] for res in results]), axis=0)
Cl_ww = np.mean(np.array([res[2] for res in results]), axis=0)
Cl_aw = np.mean(np.array([res[3] for res in results]), axis=0)


#make comparison plot of masked_map_cl and master_cl
#test modified MASTER
print('***********************************************************', flush=True)
print('Testing modified MASTER', flush=True)
compare_master(inp, lhs, 0, 0, Cl_aa, Cl_ww, Cl_aw, np.zeros((inp.ellmax+1, inp.ellmax+1, inp.ellmax+1)), np.zeros((inp.ellmax+1, inp.ellmax+1, inp.ellmax+1)), np.zeros((inp.ellmax+1, inp.ellmax+1, inp.ellmax+1, inp.ellmax+1, inp.ellmax+1)), my_env, base_dir=f'images/{inp.comp}_w_eq_a_plus_A_ellmax{inp.ellmax}')




print("--- %s seconds ---" % (time.time() - start_time), flush=True)