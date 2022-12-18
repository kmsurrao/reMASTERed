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
from interpolate_bispectrum import *
from test_master import *
import time
import pickle
print('imports complete in consistency_checks.py', flush=True)
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

base_dir = '/moto/hill/users/kms2320/repositories/halosky_maps/maps/'

def one_sim(inp, sim, offset, base_dir):
    np.random.seed(sim)
    lmax_data = 3*inp.nside-1

    #get simulated map
    map_ = hp.read_map(base_dir + f'tsz_0000{sim}.fits') 
    map_ = hp.ud_grade(map_, inp.nside)

    #create W=a+A mask for component map
    print('***********************************************************', flush=True)
    print('Starting mask generation', flush=True)
    mask = map_ + offset
    print('mask: ', mask, flush=True)
    print('min of mask: ', np.amin(mask), flush=True)

    #get power spectra and bispectra
    print('***********************************************************', flush=True)
    print('Starting power spectra and bispectrum calculation', flush=True)
    alm = hp.map2alm(map_)
    wlm = hp.map2alm(mask)

    #zero out modes above ellmax
    l_arr,m_arr = hp.Alm.getlm(lmax_data)
    alm = alm*(l_arr<=inp.ellmax)
    wlm = wlm*(l_arr<=inp.ellmax)
    map_ = hp.alm2map(alm, nside=inp.nside)
    mask = hp.alm2map(wlm, nside=inp.nside)

    masked_map = map_*mask
    masked_map_alm = hp.map2alm(masked_map, lmax=inp.ellmax)
    Cl_aa = hp.alm2cl(alm, lmax_out=inp.ellmax)
    Cl_ww = hp.alm2cl(wlm, lmax_out=inp.ellmax)
    Cl_aw = hp.anafast(map_, mask, lmax=inp.ellmax)

    lhs = hp.anafast(masked_map, lmax=inp.ellmax)
    bispectrum_aaw = Bispectrum(inp, map_-np.mean(map_), map_-np.mean(map_), mask-np.mean(mask), equal12=True)
    w00 = wlm[0]
    bispectrum_waw = Bispectrum(inp, mask-np.mean(mask), map_-np.mean(map_), mask-np.mean(mask), equal13=True)
    a00 = alm[0]

    print('***********************************************************', flush=True)
    print(f'Starting rho calculation for sim {sim}', flush=True)
    Rho = rho(inp, map_-np.mean(map_), mask-np.mean(mask), Cl_aw, Cl_aa, Cl_ww)


    return lhs, Cl_aa, Cl_ww, Cl_aw, bispectrum_aaw, w00, bispectrum_waw, a00, Rho


#read map
map_ = hp.read_map(base_dir + f'tsz_00000.fits') 
map_ = hp.ud_grade(map_, inp.nside)

#find offset A for mask W=a+A
offset = 1.e-6

pool = mp.Pool(min(inp.nsims, 16))
results = pool.starmap(one_sim, [(inp, sim, offset, base_dir) for sim in range(inp.nsims)])
pool.close()
master_lhs = np.mean(np.array([res[0] for res in results]), axis=0)
Cl_aa = np.mean(np.array([res[1] for res in results]), axis=0)
Cl_ww = np.mean(np.array([res[2] for res in results]), axis=0)
Cl_aw = np.mean(np.array([res[3] for res in results]), axis=0)
bispectrum_aaw = np.mean(np.array([res[4] for res in results]), axis=0)
w00 = np.mean(np.array([res[5] for res in results]), axis=0)
bispectrum_waw = np.mean(np.array([res[6] for res in results]), axis=0)
a00 = np.mean(np.array([res[7] for res in results]), axis=0)
Rho = np.mean(np.array([res[8] for res in results]), axis=0)
pickle.dump(Rho, open(f'rho_tsz_ellmax{inp.ellmax}.p', 'wb'))
# Rho = pickle.load(open(f'rho_tsz_ellmax{inp.ellmax}.p', 'rb'))


#test modified MASTER
print('***********************************************************', flush=True)
print('Testing modified MASTER', flush=True)
compare_master(inp, master_lhs, w00, a00, Cl_aa, Cl_ww, Cl_aw, bispectrum_aaw, bispectrum_waw, Rho, my_env, base_dir=f'images/tSZ_w_eq_a_plus_A_ellmax{inp.ellmax}')


print("--- %s seconds ---" % (time.time() - start_time), flush=True)