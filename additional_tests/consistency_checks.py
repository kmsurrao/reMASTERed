#This script assumes zero-mean fields.

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
from test_remastered import *
import time
import pickle
from plot_consistency import *
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

    lmax_data = 3*inp.nside-1

    np.random.seed(sim)

    #get simulated map
    map_ = hp.read_map(inp.map_file) 
    map_cl = hp.anafast(map_, lmax=lmax_data)
    map_ = hp.synfast(map_cl, nside=inp.nside)

    #create threshold mask for component map
    print('***********************************************************', flush=True)
    print(f'Starting mask generation sim {sim}', flush=True)
    mask = gen_mask(inp, map_, sim, testing_aniso=True)

    #get power spectra and bispectra
    print('***********************************************************', flush=True)
    print(f'Starting bispectrum calculation sim {sim}', flush=True)
    alm = hp.map2alm(map_)
    wlm = hp.map2alm(mask)
    
    #added below
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

    #load 3j symbols and set up arrays
    l2 = np.arange(inp.ellmax+1)
    l3 = np.arange(inp.ellmax+1)
    wigner = pickle.load(open(inp.wigner_file, 'rb'))[:inp.ellmax+1, :inp.ellmax+1, :inp.ellmax+1]
    
    #compare <a tilde(a)> to representation in terms of bispectrum
    bispectrum_aaw = Bispectrum(inp, map_-np.mean(map_), map_-np.mean(map_), mask-np.mean(mask), equal12=True)
    print(f'finished bispectrum calculation aaw sim {sim}', flush=True)
    lhs_atildea = hp.anafast(map_*mask, map_, lmax=inp.ellmax)
    aaw_term_atildea = float(1/(4*np.pi))*np.einsum('a,b,lab,lab,lab->l',2*l2+1,2*l3+1,wigner,wigner,bispectrum_aaw,optimize=True)
    w_aa_term_atildea = np.real(wlm[0])/np.sqrt(4*np.pi)*Cl_aa

    #compare <w tilde(a)> to representation in terms of Claw and w00
    bispectrum_waw = Bispectrum(inp,mask-np.mean(mask),map_-np.mean(map_),mask-np.mean(mask),equal13=True)
    print(f'finished bispectrum calculation waw sim {sim}', flush=True)
    wigner = pickle.load(open(inp.wigner_file, 'rb'))[:inp.ellmax+1, :inp.ellmax+1, :inp.ellmax+1]
    lhs_wtildea = hp.anafast(map_*mask, mask, lmax=inp.ellmax)
    w_aw_term_wtildea = float(1/np.sqrt(4*np.pi))*np.real(wlm[0])*Cl_aw #added no monopole here to test
    waw_term_wtildea = 1/(4*np.pi)*np.einsum('a,b,lab,lab,lab->l',2*l2+1,2*l3+1,wigner,wigner,bispectrum_waw)
    
    return lhs_atildea, w_aa_term_atildea, aaw_term_atildea, lhs_wtildea, w_aw_term_wtildea, waw_term_wtildea

#do ensemble averaging
pool = mp.Pool(min(inp.nsims, 16))
results = pool.starmap(one_sim, [(inp, sim) for sim in range(inp.nsims)])
pool.close()
lhs_atildea = np.mean(np.array([res[0] for res in results]), axis=0)
w_aa_term_atildea = np.mean(np.array([res[1] for res in results]), axis=0)
aaw_term_atildea = np.mean(np.array([res[2] for res in results]), axis=0)
lhs_wtildea = np.mean(np.array([res[3] for res in results]), axis=0)
w_aw_term_wtildea = np.mean(np.array([res[4] for res in results]), axis=0)
waw_term_wtildea = np.mean(np.array([res[5] for res in results]), axis=0)

#save files and plot
if inp.save_files or inp.plot:
    to_save = [lhs_atildea, w_aa_term_atildea, aaw_term_atildea, lhs_wtildea, w_aw_term_wtildea, waw_term_wtildea]
    base_dir = f'images/consistency_{inp.comp}_cut{inp.cut}_ellmax{inp.ellmax}_nsims{inp.nsims}_nside{inp.nside}_nsideformasking{inp.nside_for_masking}'
    if not os.path.isdir(base_dir):
        subprocess.call(f'mkdir {base_dir}', shell=True, env=my_env)
    if inp.save_files:
        pickle.dump(to_save, open(f'{base_dir}/consistency.p', 'wb'))
        print(f'saved {base_dir}/consistency.p', flush=True)
    if inp.plot:
        plot_consistency(inp, to_save, base_dir, start=2, logx=True, logy=False)




print("--- %s seconds ---" % (time.time() - start_time), flush=True)