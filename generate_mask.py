import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt
import pickle
import os.path
import sys
from input import Info
import multiprocessing as mp

def mask_above_cut(inp, cut_high, cut_low, map_, nside, nside_for_masking, sim, testing_aniso=False):
    #downgrade resolution of map for create mask initially
    map_ = hp.ud_grade(map_, nside_for_masking)
    if inp.cut_high and inp.cut_low:
        #set mask to 0 below cut_low or above cut_high, and 1 elsewhere
        m = np.where(np.logical_or(map_>cut_high, map_<cut_low), 0, 1)
    elif inp.cut_high:
        m = np.where(map_>cut_high, 0, 1)
    elif inp.cut_low:
        m = np.where(map_<cut_low, 0, 1)
    #return mask to nside of original map
    m = hp.ud_grade(m, nside)
    print('zeroed out pixels above cut', flush=True)
    aposcale = inp.aposcale # Apodization scale in degrees, original
    m = nmt.mask_apodization(m, aposcale, apotype="C1")
    #added section below to artificially generate anisotropic mask for each realization, remove later
    if testing_aniso:
        npix = 12*inp.nside**2
        if sim%2==0:
            m[npix//2:]=1.
        else:
            m[:npix//2]=1.
    #added section below to artificially create same ensemble average as half mask thersholding, remove
    else:
        m = (1.+m)/2
    return m

def gen_mask(inp, map_, sim, testing_aniso=False):
   
    ells = np.arange(inp.ellmax+1)

    #create mask
    mean = np.mean(map_)
    std_dev = np.std(map_)
    cut_high = mean + inp.cut*std_dev
    cut_low = mean - inp.cut*std_dev
    mask = mask_above_cut(inp, cut_high, cut_low, map_, inp.nside, inp.nside_for_masking, sim, testing_aniso=testing_aniso)
    return mask

if __name__=="__main__":
    # main input file containing most specifications 
    try:
        input_file = (sys.argv)[1]
    except IndexError:
        input_file = 'moto.yaml'

    # read in the input file and set up relevant info object
    inp = Info(input_file)

    #get mask and plot
    def one_sim(inp, sim):
        np.random.seed(sim)
        map_ = hp.read_map(inp.map_file)
        map_cl = hp.anafast(map_, lmax=inp.ellmax)
        map_ = hp.synfast(map_cl, nside=inp.nside)
        mask = gen_mask(inp, map_, sim, testing_aniso=True)
        return mask, map_
    
    # mask = one_sim(inp, 0)
    # plt.clf()
    # hp.mollview(mask)
    # plt.savefig('cmb_anisotropic_mask.png')


    pool = mp.Pool(min(inp.nsims, 16))
    results = pool.starmap(one_sim, [(inp, sim) for sim in range(inp.nsims)])
    pool.close()

    mask = np.mean(np.array([res[0] for res in results]), axis=0)
    plt.clf()
    hp.mollview(mask)
    plt.savefig(f'ensemble_averaged_mask.png')
    print(f'saved ensemble_averaged_mask.png', flush=True)

    wa = np.mean(np.array([res[0]*res[1] for res in results]), axis=0)
    plt.clf()
    hp.mollview(wa)
    plt.savefig(f'ensemble_averaged_wa.png')
    print(f'saved ensemble_averaged_wa.png', flush=True)