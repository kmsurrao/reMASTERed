import sys
sys.path.append("..")
import os
import subprocess
import numpy as np
import healpy as hp
import multiprocessing as mp
import time
import pickle
from input import Info
from generate_mask import *
from bispectrum import *
from trispectrum import *
from interpolate_bispectrum import *
from test_remastered import *
from wigner3j import *
print('imports complete in consistency_checks.py', flush=True)
start_time = time.time()

# main input file containing most specifications 
try:
    input_file = (sys.argv)[1]
except IndexError:
    input_file = 'threshold_moto.yaml'

# read in the input file and set up relevant info object
inp = Info(input_file, mask_provided=False)

# current environment, also environment in which to run subprocesses
my_env = os.environ.copy()

#get wigner 3j symbols
if inp.wigner_file:
    inp.wigner3j = pickle.load(open(inp.wigner_file, 'rb'))[:inp.ellmax+1, :inp.ellmax+1, :inp.ellmax+1]
else:
    inp.wigner3j = compute_3j(inp.ellmax)

base_dir = '/moto/hill/users/kms2320/repositories/halosky_maps/maps/'

def one_sim(inp, sim, offset, base_dir):
    '''
    PARAMETERS
    inp: Info object, contains input specifications
    sim: int, simulation number
    offset: float, offset = mask-map
    base_dir: str, directory to output plots and pickle files

    RETURNS
    lhs: 1D numpy array, directly computed power spectrum of masked map
    Cl_aa: 1D numpy array, auto-spectrum of the map
    Cl_ww: 1D numpy array, auto-spectrum of the mask
    Cl_aw: 1D numpy array, cross-spectrum of the map and mask
    bispectrum_aaw: 3D numpy array indexed as bispectrum_aaw[l1,l2,l3], bispectrum consisting of two factors of map and one factor of mask 
    w00: float, w_{00} for the mask
    bispectrum_waw: 3D numpy array indexed as bispectrum_waw[l1,l2,l3], bispectrum consisting of two factors of mask and one factor of map
    a00: float, a_{00} for the map   
    Rho: 5D numpy array indexed as Rho[l1,l2,l3,l4,L], estimator for unnormalized trispectrum
    '''
    np.random.seed(sim)

    #get simulated map
    map_ = hp.read_map(base_dir + f'tsz_0000{sim}.fits') 
    map_ = hp.ud_grade(map_, inp.nside)

    #create W=a+A mask for component map
    print('Starting mask generation', flush=True)
    mask = map_ + offset

    #get alm and wlm for map and mask, respectively 
    alm = hp.map2alm(map_)
    wlm = hp.map2alm(mask)

    #zero out modes above ellmax
    lmax_data = 3*inp.nside-1
    l_arr,m_arr = hp.Alm.getlm(lmax_data)
    alm = alm*(l_arr<=inp.ellmax)
    wlm = wlm*(l_arr<=inp.ellmax)
    map_ = hp.alm2map(alm, nside=inp.nside)
    mask = hp.alm2map(wlm, nside=inp.nside)
    masked_map = map_*mask
    
    #get auto- and cross-spectra for map, mask, and masked map
    Cl_aa = hp.alm2cl(alm, lmax_out=inp.ellmax)
    Cl_ww = hp.alm2cl(wlm, lmax_out=inp.ellmax)
    Cl_aw = hp.anafast(map_, mask, lmax=inp.ellmax)
    lhs = hp.anafast(masked_map, lmax=inp.ellmax)

    #calculate bispectra and one-point functions
    print(f'Starting bispectrum calculation for sim {sim}', flush=True)
    bispectrum_aaw = Bispectrum(inp, map_-np.mean(map_), map_-np.mean(map_), mask-np.mean(mask), equal12=True)
    w00 = wlm[0]
    bispectrum_waw = Bispectrum(inp, mask-np.mean(mask), map_-np.mean(map_), mask-np.mean(mask), equal13=True)
    a00 = alm[0]

    #calculate rho
    print(f'Starting rho calculation for sim {sim}', flush=True)
    Rho = rho(inp, map_-np.mean(map_), mask-np.mean(mask), Cl_aw, Cl_aa, Cl_ww)


    return lhs, Cl_aa, Cl_ww, Cl_aw, bispectrum_aaw, w00, bispectrum_waw, a00, Rho


#read map
map_ = hp.read_map(base_dir + f'tsz_00000.fits') 
map_ = hp.ud_grade(map_, inp.nside)

#find offset A for mask W=a+A
offset = 1.e-6

#do ensemble averaging
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
pickle.dump(Rho, open(f'rho_tsz_ellmax{inp.ellmax}.p', 'wb')) #remove
# Rho = pickle.load(open(f'rho_tsz_ellmax{inp.ellmax}.p', 'rb')) #remove


#test reMASTERed
print('Testing reMASTERed', flush=True)
compare_master(inp, master_lhs, w00, a00, Cl_aa, Cl_ww, Cl_aw, bispectrum_aaw, bispectrum_waw, Rho, my_env, base_dir=f'images/tSZ_w_eq_a_plus_A_ellmax{inp.ellmax}')


print("--- %s seconds ---" % (time.time() - start_time), flush=True)