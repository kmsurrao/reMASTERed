import sys
import os
import subprocess
import numpy as np
import healpy as hp
import multiprocessing as mp
from input import Info
from generate_mask import *
from bispectrum_unbinned import *
from interpolate_bispectrum import *
from test_master import *
import time
print('imports complete in main.py', flush=True)
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

def one_sim(inp, sim):

    #get simulated map
    map_ = hp.read_map(inp.map_file) 
    map_cl = hp.anafast(map_, lmax=inp.ellmax)
    map_ = hp.synfast(map_cl, nside=inp.nside)

    #create threshold mask for component map
    print('***********************************************************', flush=True)
    print(f'Starting mask generation for sim {sim}', flush=True)
    mask = gen_mask(inp, map_)

    #get power spectra and bispectra
    print('***********************************************************', flush=True)
    print(f'Starting bispectrum calculation for sim {sim}', flush=True)
    min_l = 0
    Nl = int(inp.ellmax/inp.dl) # number of bins
    alm = hp.map2alm(map_, lmax=inp.ellmax)
    wlm = hp.map2alm(mask, lmax=inp.ellmax)
    Cl = hp.alm2cl(alm)
    Ml = hp.alm2cl(wlm)
    Wl = hp.anafast(map_, mask, lmax=inp.ellmax)
    mask_no_monopole = hp.remove_monopole(mask)
    wlm_no_monopole = hp.map2alm(mask_no_monopole, lmax=inp.ellmax)
    Ml_no_monopole = hp.alm2cl(wlm_no_monopole)
    # bispectrum_term3 = Bispectrum(alm, Cl, np.conj(alm), Cl, np.conj(wlm_no_monopole), Ml_no_monopole, inp.ellmax, inp.nside, 3, inp)
    # bispectrum_term4 = Bispectrum(alm, Cl, np.conj(alm), Cl, wlm_no_monopole, Ml_no_monopole, inp.ellmax, inp.nside, 4, inp)
    bispectrum_term3 = Bispectrum(alm, Cl, alm, Cl, wlm_no_monopole, Ml_no_monopole, inp.ellmax, inp.nside, 3, inp)
    bispectrum_term4 = Bispectrum(alm, Cl, alm, Cl, wlm_no_monopole, Ml_no_monopole, inp.ellmax, inp.nside, 4, inp)

    #get MASTER LHS
    master_lhs = hp.anafast(map_*mask, lmax=inp.ellmax)

    return [master_lhs, wlm[0], Cl, Ml, Wl, bispectrum_term3, bispectrum_term4]


#make plots of MASTER equation with new terms
pool = mp.Pool(min(inp.nsims, 16))
results = pool.starmap(one_sim, [(inp, sim) for sim in range(inp.nsims)])
pool.close()
print('len(results): ', len(results), flush=True)
master_lhs = np.mean(np.array([res[0] for res in results]), axis=0)
wlm_00 = np.mean(np.array([res[1] for res in results]), axis=0)
Cl = np.mean(np.array([res[2] for res in results]), axis=0)
Ml = np.mean(np.array([res[3] for res in results]), axis=0)
Wl = np.mean(np.array([res[4] for res in results]), axis=0)
bispectrum_term3 = np.mean(np.array([res[5] for res in results]), axis=0)
bispectrum_term4 = np.mean(np.array([res[6] for res in results]), axis=0)


print('***********************************************************', flush=True)
print('Starting MASTER comparison', flush=True)
compare_master(inp, master_lhs, wlm_00, Cl, Ml, Wl, bispectrum_term3, bispectrum_term4, my_env)

print("--- %s seconds ---" % (time.time() - start_time), flush=True)