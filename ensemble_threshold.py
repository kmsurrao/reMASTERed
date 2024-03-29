import os
import subprocess
import numpy as np
import healpy as hp
import multiprocessing as mp
import time
import pickle
import argparse
from input import Info
from generate_mask import *
from bispectrum import *
from trispectrum import *
from test_remastered import * 
from plot_mask import *
from wigner3j import *


def get_one_map_and_mask(inp, sim):
    '''
    PARAMETERS
    inp: Info object, contains input specifications
    sim: int, simulation number

    RETURNS
    map_: 1D numpy array, contains simulated map
    mask: 1D numpy array, contains threshold mask
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
    if inp.remove_high_ell_power:
        l_arr,m_arr = hp.Alm.getlm(lmax_data)
        alm = alm*(l_arr<=inp.ellmax)
        wlm = wlm*(l_arr<=inp.ellmax)
        map_ = hp.alm2map(alm, nside=inp.nside)
        mask = hp.alm2map(wlm, nside=inp.nside)

    return map_, mask


def one_sim(inp, sim, map_, mask, map_avg, mask_avg):
    '''
    PARAMETERS
    inp: Info object, contains input specifications
    sim: int, simulation number
    map_: 1D numpy array, contains simulated map
    mask: 1D numpy array, contains threshold mask
    map_avg: float, average pixel value over all map realizations
    mask_avg: float, average pixel value over all mask realizations

    RETURNS
    master_lhs: 1D numpy array, directly computed power spectrum of masked map
    wlm_00: float, w_{00} for the mask
    alm_00: float, a_{00} for the map
    Cl_aa: 1D numpy array, auto-spectrum of the map
    Cl_ww: 1D numpy array, auto-spectrum of the mask
    Cl_aw: 1D numpy array, cross-spectrum of the map and mask 
    bispectrum_aaw: 3D numpy array indexed as bispectrum_aaw[l1,l2,l3], bispectrum consisting of two factors of map and one factor of mask (with means removed)
    bispectrum_waw: 3D numpy array indexed as bispectrum_waw[l1,l2,l3], bispectrum consisting of two factors of mask and one factor of map (with means removed)
    Rho: 5D numpy array indexed as Rho[l2,l4,l3,l5,l1], estimator for unnormalized trispectrum (with means removed from map and mask)
    '''
    
    #get one point functions
    alm_00 = hp.map2alm(map_)[0]
    wlm_00 = hp.map2alm(mask)[0]

    #get auto- and cross-spectra for map and mask
    Cl_aa = hp.anafast(map_, lmax=inp.ell_sum_max)
    Cl_ww = hp.anafast(mask, lmax=inp.ell_sum_max)
    Cl_aw = hp.anafast(map_, mask, lmax=inp.ell_sum_max)
    Cl_aa_mean_rem = hp.anafast(map_-map_avg, lmax=inp.ell_sum_max)
    Cl_ww_mean_rem = hp.anafast(mask-mask_avg, lmax=inp.ell_sum_max)
    Cl_aw_mean_rem = hp.anafast(map_-map_avg, mask-mask_avg, lmax=inp.ell_sum_max)

    #get list of map, mask, masked map, and correlation coefficient
    if sim==0:
        if inp.save_files or inp.plot:
            data = [map_, mask, map_*mask] #contains map, mask, masked map, correlation coefficient
            if inp.output_dir:
                base_dir = inp.output_dir
            else:
                base_dir = f'{inp.comp}_cut{inp.cut}_ellmax{inp.ellmax}_nsims{inp.nsims}_nside{inp.nside}_nsideformasking{inp.nside_for_masking}'
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
    print(f'Starting bispectrum calculation for sim {sim}', flush=True)
    bispectrum_aaw = Bispectrum(inp, map_-map_avg, map_-map_avg, mask-mask_avg, equal12=True)
    bispectrum_waw = Bispectrum(inp, mask-mask_avg, map_-map_avg, mask-mask_avg, equal13=True)

    #Compute rho (unnormalized trispectrum)
    print(f'Starting rho calculation for sim {sim}', flush=True)
    Rho = rho(inp, map_-map_avg, mask-mask_avg, Cl_aw_mean_rem, Cl_aa_mean_rem, Cl_ww_mean_rem, remove_two_point=True)

    #get MASTER LHS (directly computed power spectrum of masked map)
    master_lhs = hp.anafast(map_*mask, lmax=inp.ellmax)

    return master_lhs, wlm_00, alm_00, Cl_aa, Cl_ww, Cl_aw, bispectrum_aaw, bispectrum_waw, Rho 


def main_ensemble_threshold():

    start_time = time.time()

    # main input file containing most specifications 
    parser = argparse.ArgumentParser(description="Contributions of n-point functions to pseudo-Cl.")
    parser.add_argument("--config", default="threshold_moto.yaml")
    args = parser.parse_args()
    input_file = args.config

    # read in the input file and set up Info object
    inp = Info(input_file, mask_provided=False)

    # current environment, also environment in which to run subprocesses
    my_env = os.environ.copy()

    #get wigner 3j symbols
    if inp.wigner_file != '':
        inp.wigner3j = pickle.load(open(inp.wigner_file, 'rb'))[:inp.ell_sum_max+1, :inp.ell_sum_max+1, :inp.ell_sum_max+1]
    else:
        inp.wigner3j = compute_3j(inp.ell_sum_max)

    #get all maps and masks
    pool = mp.Pool(min(inp.nsims, 16))
    results = pool.starmap(get_one_map_and_mask, [(inp, sim) for sim in range(inp.nsims)])
    pool.close()
    results = np.array(results)
    maps = results[:,0,:]
    masks = results[:,1,:]
    map_avg = np.mean(maps)
    mask_avg = np.mean(masks)
    
    #Run inp.nsims simulations
    pool = mp.Pool(min(inp.nsims, 16))
    results = pool.starmap(one_sim, [(inp, sim, maps[sim], masks[sim], map_avg, mask_avg) for sim in range(inp.nsims)])
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

    #Get all terms of reMASTERed equation
    print('Starting reMASTERed comparison', flush=True)
    if inp.output_dir:
        base_dir = inp.output_dir
    else:
        base_dir = f'{inp.comp}_cut{inp.cut}_ellmax{inp.ellmax}_nsims{inp.nsims}_nside{inp.nside}_nsideformasking{inp.nside_for_masking}'
    compare_master(inp, master_lhs, wlm_00, alm_00, Cl_aa, Cl_ww, Cl_aw, bispectrum_aaw, bispectrum_waw, Rho, my_env, base_dir=base_dir)

    print("--- %s seconds ---" % (time.time() - start_time), flush=True)


if __name__ == '__main__': 
    main_ensemble_threshold()