import sys
import os
import subprocess
import numpy as np
import healpy as hp
import multiprocessing as mp
from input import Info
from generate_mask import *
from bispectrum import *
from trispectrum import *
from test_remastered import *
from helper import *
import time
import string 
from plot_mask import *
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

    np.random.seed(sim)

    lmax_data = 3*inp.nside-1

    #get simulated map
    map_ = hp.read_map(inp.map_file) 
    map_cl = hp.anafast(map_, lmax=inp.ellmax)
    map_ = hp.synfast(map_cl, nside=inp.nside)

    #create threshold mask for component map
    print('***********************************************************', flush=True)
    print(f'Starting mask generation for sim {sim}', flush=True)
    mask = gen_mask(inp, map_, sim)

    #get power spectra and bispectra
    alm = hp.map2alm(map_)
    wlm = hp.map2alm(mask)

    #zero out modes above ellmax
    l_arr,m_arr = hp.Alm.getlm(lmax_data)
    alm = alm*(l_arr<=inp.ellmax)
    wlm = wlm*(l_arr<=inp.ellmax)
    map_ = hp.alm2map(alm, nside=inp.nside)
    mask = hp.alm2map(wlm, nside=inp.nside)


    Cl_aa = hp.alm2cl(alm, lmax_out=inp.ellmax)
    Cl_ww = hp.alm2cl(wlm, lmax_out=inp.ellmax)
    Cl_aw = hp.anafast(map_, mask, lmax=inp.ellmax)


    #save list of map, mask, masked map, and correlation coefficient
    if sim==0:
        if inp.save_files or inp.plot:
            data = [map_, mask, map_*mask] #contains map, mask, masked map, correlation coefficient
            base_dir = f'images/{inp.comp}_cut{inp.cut}_ellmax{inp.ellmax}_nsims{inp.nsims}_nside{inp.nside}_nsideformasking{inp.nside_for_masking}'
            if not os.path.isdir(base_dir):
                subprocess.call(f'mkdir {base_dir}', shell=True, env=my_env)
            corr = Cl_aw/np.sqrt(Cl_aa*Cl_ww)
            data.append(corr)
            if inp.save_files:
                pickle.dump(data, open(f'{base_dir}/mask_data.p', 'wb'))
                print(f'saved {base_dir}/mask_data.p', flush=True)
            if inp.plot:
                plot_mask(inp, data, base_dir)

    print('***********************************************************', flush=True)
    print(f'Starting bispectrum calculation for sim {sim}', flush=True)
    bispectrum_aaw = Bispectrum(inp, map_-np.mean(map_), map_-np.mean(map_), mask-np.mean(mask), equal12=True)
    bispectrum_waw = Bispectrum(inp, mask-np.mean(mask), map_-np.mean(map_), mask-np.mean(mask), equal13=True)

    print('***********************************************************', flush=True)
    print(f'Starting rho calculation for sim {sim}', flush=True)
    Rho = rho(inp, map_-np.mean(map_), mask-np.mean(mask), Cl_aw, Cl_aa, Cl_ww)

    #get MASTER LHS
    master_lhs = hp.anafast(map_*mask, lmax=inp.ellmax)

    return master_lhs, wlm[0], alm[0], Cl_aa, Cl_ww, Cl_aw, bispectrum_aaw, bispectrum_waw, Rho 



#make plots of MASTER equation with new terms
pool = mp.Pool(min(inp.nsims, 16))
results = pool.starmap(one_sim, [(inp, sim) for sim in range(inp.nsims)])
pool.close()
master_lhs = np.mean(np.array([res[0] for res in results]), axis=0)
wlm_00 = np.mean(np.array([res[1] for res in results]), axis=0)
alm_00 = np.mean(np.array([res[2] for res in results]), axis=0)
Cl_aa = np.mean(np.array([res[3] for res in results]), axis=0)
Cl_ww = np.mean(np.array([res[4] for res in results]), axis=0)
Cl_aw = np.mean(np.array([res[5] for res in results]), axis=0)
bispectrum_aaw = np.mean(np.array([res[6] for res in results]), axis=0)
bispectrum_waw = np.mean(np.array([res[7] for res in results]), axis=0)
Rho = np.mean(np.array([res[8] for res in results]), axis=0)
pickle.dump(Rho, open(f'rho/rho_isw_ellmax{inp.ellmax}.p', 'wb')) #remove
# trispectrum = pickle.load(open(f'rho/rho_isw_ellmax{inp.ellmax}.p', 'rb')) 

print('***********************************************************', flush=True)
print('Starting MASTER comparison', flush=True)
compare_master(inp, master_lhs, wlm_00, alm_00, Cl_aa, Cl_ww, Cl_aw, bispectrum_aaw, bispectrum_waw, Rho, my_env)

print("--- %s seconds ---" % (time.time() - start_time), flush=True)