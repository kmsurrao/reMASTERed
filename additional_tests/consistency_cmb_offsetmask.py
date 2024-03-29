# import sys
# sys.path.insert(0, "./../" )
# sys.path.insert(0, "./" )
# import os
# import subprocess
# import numpy as np
# import healpy as hp
# import multiprocessing as mp
# import time
# from input import Info
# from generate_mask import *
# from bispectrum import *
# from test_remastered import *
# from wigner3j import *


# def one_sim(inp, sim, offset):
#     '''
#     PARAMETERS
#     inp: Info object, contains input specifications
#     sim: int, simulation number
#     offset: float, offset = mask-map

#     RETURNS
#     lhs: 1D numpy array, directly computed power spectrum of masked map
#     Cl_aa: 1D numpy array, auto-spectrum of the map
#     Cl_ww: 1D numpy array, auto-spectrum of the mask
#     Cl_aw: 1D numpy array, cross-spectrum of the map and mask 
#     '''

#     np.random.seed(sim)

#     #get simulated map
#     lmax_data = 3*inp.nside-1
#     map_ = hp.read_map(inp.map_file) 
#     map_cl = hp.anafast(map_, lmax=lmax_data)
#     map_ = hp.synfast(map_cl, inp.nside)

#     #create W=a+A mask for component map
#     print('Starting mask generation', flush=True)
#     mask = map_ + offset

#     #get alm and wlm for map and mask, respectively 
#     alm = hp.map2alm(map_)
#     wlm = hp.map2alm(mask)

#     #zero out modes above ellmax
#     if inp.remove_high_ell_power:
#         l_arr,m_arr = hp.Alm.getlm(lmax_data)
#         alm = alm*(l_arr<=inp.ellmax)
#         wlm = wlm*(l_arr<=inp.ellmax)
#         map_ = hp.alm2map(alm, nside=inp.nside)
#         mask = hp.alm2map(wlm, nside=inp.nside)

#     #get auto- and cross-spectra for map and mask
#     Cl_aa = hp.anafast(map_, lmax=inp.ell_sum_max)
#     Cl_ww = hp.anafast(mask, lmax=inp.ell_sum_max)
#     Cl_aw = hp.anafast(map_, mask, lmax=inp.ell_sum_max)

#     lhs = hp.anafast(map_*mask, lmax=inp.ellmax)

#     return lhs, Cl_aa, Cl_ww, Cl_aw


# if __name__=='__main__':
#     start_time = time.time()

#     # main input file containing most specifications 
#     try:
#         input_file = (sys.argv)[1]
#     except IndexError:
#         input_file = 'threshold_moto.yaml'

#     # read in the input file and set up relevant info object
#     inp = Info(input_file, mask_provided=False)

#     # current environment, also environment in which to run subprocesses
#     my_env = os.environ.copy()

#     #get wigner 3j symbols
#     if inp.wigner_file != '':
#         inp.wigner3j = pickle.load(open(inp.wigner_file, 'rb'))[:inp.ell_sum_max+1, :inp.ell_sum_max+1, :inp.ell_sum_max+1]
#     else:
#         inp.wigner3j = compute_3j(inp.ell_sum_max)

#     #read map
#     map_ = hp.read_map(inp.map_file) 
#     map_ = hp.ud_grade(map_, inp.nside)

#     #find offset A for mask W=a+A
#     offset = 1.5*abs(np.amin(map_))
#     print('offset: ', offset, flush=True)

#     #do ensemble averaging
#     pool = mp.Pool(min(inp.nsims, 16))
#     results = pool.starmap(one_sim, [(inp, sim, offset) for sim in range(inp.nsims)])
#     pool.close()
#     lhs = np.mean(np.array([res[0] for res in results]), axis=0)
#     Cl_aa = np.mean(np.array([res[1] for res in results]), axis=0)
#     Cl_ww = np.mean(np.array([res[2] for res in results]), axis=0)
#     Cl_aw = np.mean(np.array([res[3] for res in results]), axis=0)

#     #test reMASTERed
#     print('Testing reMASTERed', flush=True)
#     compare_master(inp, lhs, 0, 0, Cl_aa, Cl_ww, Cl_aw, np.zeros((inp.ellmax+1, inp.ell_sum_max+1, inp.ell_sum_max+1)), np.zeros((inp.ellmax+1, inp.ell_sum_max+1, inp.ell_sum_max+1)), np.zeros((inp.ell_sum_max+1, inp.ell_sum_max+1, inp.ell_sum_max+1, inp.ell_sum_max+1, inp.ellmax+1)), my_env, base_dir=f'images/{inp.comp}_w_eq_a_plus_A_ellmax{inp.ellmax}_{inp.nsims}sims')


#     print("--- %s seconds ---" % (time.time() - start_time), flush=True)



import sys
sys.path.insert(0, "./../" )
sys.path.insert(0, "./" )
import os
import subprocess
import numpy as np
import healpy as hp
import multiprocessing as mp
import time
from input import Info
from generate_mask import *
from bispectrum import *
from trispectrum import *
from test_remastered import *
from wigner3j import *


def get_one_map_and_mask(inp, sim, offset):
    '''
    PARAMETERS
    inp: Info object, contains input specifications
    sim: int, simulation number
    offset: float, offset = mask-map

    RETURNS
    map_: 1D numpy array, contains simulated map
    mask: 1D numpy array, contains offset mask
    '''

    np.random.seed(sim)

    #get simulated map
    lmax_data = 3*inp.nside-1
    map_ = hp.read_map(inp.map_file) 
    map_cl = hp.anafast(map_, lmax=lmax_data)
    map_ = hp.synfast(map_cl, inp.nside)

    #create W=a+A mask for component map
    print('Starting mask generation', flush=True)
    mask = map_ + offset

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
    mask: 1D numpy array, contains offset mask
    map_avg: float, average pixel value over all map realizations
    mask_avg: float, average pixel value over all mask realizations

    RETURNS
    lhs: 1D numpy array, directly computed power spectrum of masked map
    Cl_aa: 1D numpy array, auto-spectrum of the map
    Cl_ww: 1D numpy array, auto-spectrum of the mask
    Cl_aw: 1D numpy array, cross-spectrum of the map and mask
    bispectrum_aaw: 3D numpy array indexed as bispectrum_aaw[l1,l2,l3], bispectrum consisting of two factors of map and one factor of mask 
    w00: float, w_{00} for the mask
    bispectrum_waw: 3D numpy array indexed as bispectrum_waw[l1,l2,l3], bispectrum consisting of two factors of mask and one factor of map
    a00: float, a_{00} for the map   
    Rho: 5D numpy array indexed as Rho[l2,l4,l3,l5,l1], estimator for unnormalized trispectrum
    '''

    #get auto- and cross-spectra for map, mask, and masked map
    Cl_aa = hp.anafast(map_, lmax=inp.ell_sum_max)
    Cl_ww = hp.anafast(mask, lmax=inp.ell_sum_max)
    Cl_aw = hp.anafast(map_, mask, lmax=inp.ell_sum_max)
    Cl_aa_mean_rem = hp.anafast(map_-map_avg, lmax=inp.ell_sum_max)
    Cl_ww_mean_rem = hp.anafast(mask-mask_avg, lmax=inp.ell_sum_max)
    Cl_aw_mean_rem = hp.anafast(map_-map_avg, mask-mask_avg, lmax=inp.ell_sum_max)
    lhs = hp.anafast(map_*mask, lmax=inp.ellmax)

    #calculate bispectra and one-point functions
    print(f'Starting bispectrum calculation for sim {sim}', flush=True)
    bispectrum_aaw = Bispectrum(inp, map_-map_avg, map_-map_avg, mask-mask_avg, equal12=True)
    w00 = hp.map2alm(mask)[0]
    bispectrum_waw = Bispectrum(inp, mask-mask_avg, map_-map_avg, mask-mask_avg, equal13=True)
    a00 = hp.map2alm(map_)[0]

    #calculate rho
    print(f'Starting rho calculation for sim {sim}', flush=True)
    Rho = rho(inp, map_-map_avg, mask-mask_avg, Cl_aw_mean_rem, Cl_aa_mean_rem, Cl_ww_mean_rem)


    return lhs, Cl_aa, Cl_ww, Cl_aw, bispectrum_aaw, w00, bispectrum_waw, a00, Rho


if __name__=='__main__':
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
    if inp.wigner_file != '':
        inp.wigner3j = pickle.load(open(inp.wigner_file, 'rb'))[:inp.ell_sum_max+1, :inp.ell_sum_max+1, :inp.ell_sum_max+1]
    else:
        inp.wigner3j = compute_3j(inp.ell_sum_max)

    #read map
    map_ = hp.read_map(inp.map_file) 
    map_ = hp.ud_grade(map_, inp.nside)

    #find offset A for mask W=a+A
    offset = 1.5*abs(np.amin(map_))
    print('offset: ', offset, flush=True)

    #get all maps and masks
    pool = mp.Pool(min(inp.nsims, 16))
    results = pool.starmap(get_one_map_and_mask, [(inp, sim, offset) for sim in range(inp.nsims)])
    pool.close()
    results = np.array(results)
    maps = results[:,0,:]
    masks = results[:,1,:]
    map_avg = np.mean(maps)
    mask_avg = np.mean(masks)

    #do ensemble averaging
    pool = mp.Pool(min(inp.nsims, 16))
    results = pool.starmap(one_sim, [(inp, sim, maps[sim], masks[sim], map_avg, mask_avg) for sim in range(inp.nsims)])
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
    pickle.dump(Rho, open(f'rho/rho_cmb_offsetmask_ellmax{inp.ellmax}_1sim.p', 'wb')) #remove
    # Rho = pickle.load(open(f'rho/rho_cmb_offsetmask_ellmax{inp.ellmax}_1sim.p', 'rb')) #remove


    #test reMASTERed
    print('Testing reMASTERed', flush=True)
    compare_master(inp, master_lhs, w00, a00, Cl_aa, Cl_ww, Cl_aw, bispectrum_aaw, bispectrum_waw, Rho, my_env, base_dir=f'images/CMB_w_eq_a_plus_A_ellmax{inp.ellmax}_{inp.nsims}sims')


    print("--- %s seconds ---" % (time.time() - start_time), flush=True)