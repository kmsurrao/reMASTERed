import sys
sys.path.insert(0, "./../" )
sys.path.insert(0, "./" )
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
from test_remastered import *
from plot_consistency import *
from wigner3j import *


def one_sim(inp, sim):
    '''
    PARAMETERS
    inp: Info object, contains input specifications
    sim: int, simulation number

    RETURNS
    lhs_atildea: 1D numpy array, directly computed <a tilde(a)>
    w_aa_term_atildea: 1D numpy array, <w><aa> term for <a tilde(a)>
    a_aw_term_atildea: 1D numpy array, <a><aw> term for <a tilde(a)>
    aaw_term_atildea: 1D numpy array, <aaw> term for <a tilde(a)>
    lhs_wtildea: 1D numpy array, directly computed <w tilde(a)>
    w_aw_term_wtildea: 1D numpy array, <w><aw> term for <w tilde(a)>
    a_ww_term_wtildea: 1D numpy array, <a><ww> term for <w tilde(a)>
    waw_term_wtildea: 1D numpy array, <waw> term for <w tilde(a)>
    '''

    np.random.seed(sim)

    #get simulated map
    lmax_data = 3*inp.nside-1
    map_ = hp.read_map(inp.map_file) 
    map_cl = hp.anafast(map_, lmax=lmax_data)
    map_ = hp.synfast(map_cl, nside=inp.nside)

    #create threshold mask for component map
    print(f'Starting mask generation sim {sim}', flush=True)
    mask = gen_mask(inp, map_)

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

    #load 3j symbols and set up arrays
    l2 = np.arange(inp.ellmax+1)
    l3 = np.arange(inp.ellmax+1)
    wigner = pickle.load(open(inp.wigner_file, 'rb'))[:inp.ellmax+1, :inp.ellmax+1, :inp.ellmax+1]
    
    #compare <a tilde(a)> to representation in terms of bispectrum
    print(f'Starting bispectrum aaw calculation sim {sim}', flush=True)
    bispectrum_aaw = Bispectrum(inp, map_-np.mean(map_), map_-np.mean(map_), mask-np.mean(mask), equal12=True)
    lhs_atildea = hp.anafast(map_*mask, map_, lmax=inp.ellmax)
    aaw_term_atildea = float(1/(4*np.pi))*np.einsum('a,b,lab,lab,lab->l',2*l2+1,2*l3+1,wigner,wigner,bispectrum_aaw,optimize=True)
    w_aa_term_atildea = np.real(wlm[0])/np.sqrt(4*np.pi)*Cl_aa
    a_aw_term_atildea = np.real(alm[0])/np.sqrt(4*np.pi)*Cl_aw

    #compare <w tilde(a)> to representation in terms of Claw and w00
    print(f'Starting bispectrum waw calculation sim {sim}', flush=True)
    bispectrum_waw = Bispectrum(inp,mask-np.mean(mask),map_-np.mean(map_),mask-np.mean(mask),equal13=True)
    lhs_wtildea = hp.anafast(map_*mask, mask, lmax=inp.ellmax)
    w_aw_term_wtildea = float(1/np.sqrt(4*np.pi))*np.real(wlm[0])*Cl_aw
    a_ww_term_wtildea = float(1/np.sqrt(4*np.pi))*np.real(alm[0])*Cl_ww
    waw_term_wtildea = 1/(4*np.pi)*np.einsum('a,b,lab,lab,lab->l',2*l2+1,2*l3+1,wigner,wigner,bispectrum_waw)
    
    return lhs_atildea, w_aa_term_atildea, a_aw_term_atildea, aaw_term_atildea, lhs_wtildea, w_aw_term_wtildea, a_ww_term_wtildea, waw_term_wtildea


if __name__ == '__main__':

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

    #do ensemble averaging
    pool = mp.Pool(min(inp.nsims, 16))
    results = pool.starmap(one_sim, [(inp, sim) for sim in range(inp.nsims)])
    pool.close()
    lhs_atildea = np.mean(np.array([res[0] for res in results]), axis=0)
    w_aa_term_atildea = np.mean(np.array([res[1] for res in results]), axis=0)
    a_aw_term_atildea = np.mean(np.array([res[2] for res in results]), axis=0)
    aaw_term_atildea = np.mean(np.array([res[3] for res in results]), axis=0)
    lhs_wtildea = np.mean(np.array([res[4] for res in results]), axis=0)
    w_aw_term_wtildea = np.mean(np.array([res[5] for res in results]), axis=0)
    a_ww_term_wtildea = np.mean(np.array([res[6] for res in results]), axis=0)
    waw_term_wtildea = np.mean(np.array([res[7] for res in results]), axis=0)

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