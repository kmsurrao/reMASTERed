import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import h5py
import pymaster as nmt
import pickle

def mask_above_cut(inp, cut_high, cut_low, map_, nside, nside_for_masking, save_file=False):
    #downgrade resolution of map for create mask initially
    map_ = hp.ud_grade(map_, nside_for_masking)
    #set mask to 0 below cut_low or above cut_high, and 1 elsewhere
    m = np.where(np.logical_or(map_>cut_high, map_<cut_low), 0, 1)
    #return mask to nside of original map
    m = hp.ud_grade(m, nside)
    print('zeroed out pixels above cut', flush=True)
    plt.clf()
    hp.mollview(m)
    plt.savefig('images/mask_not_apodized.png')
    print('saved images/mask_not_apodized.png', flush=True)
    aposcale = inp.aposcale # Apodization scale in degrees, original
    m = nmt.mask_apodization(m, aposcale, apotype="C1")
    plt.clf()
    hp.mollview(m)
    plt.savefig('images/apodized_mask.png')
    print('saved images/apodized_mask.png', flush=True)
    if save_file:
        hp.write_map(f'mask_{inp.comp}_{inp.cut}.fits', m, overwrite=True)
        print(f'wrote mask_{inp.comp}_{inp.cut}.fits', flush=True)
    return m

def gen_mask(inp):
    Nl = int((inp.ellmax+1)/inp.dl)
    nside_for_masking = inp.nside//2
    ells = np.arange(inp.ellmax+1)

    #load maps
    map_ = hp.read_map(inp.map_file) 
    map_ = hp.ud_grade(map_, inp.nside)
    plt.clf()
    hp.mollview(map_)
    plt.savefig('images/map.png')
    print('saved images/map.png', flush=True)

    #create mask
    mean = np.mean(map_)
    std_dev = np.std(map_)
    cut_high = mean + 0.7*std_dev
    cut_low = mean - 0.7*std_dev
    mask = mask_above_cut(inp, cut_high, cut_low, map_, inp.nside, nside_for_masking, save_file=True)
    return mask
