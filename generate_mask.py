import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt
import pickle
import os.path

def mask_above_cut(inp, cut_high, cut_low, map_, nside, nside_for_masking, save_file=False):
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
    plt.clf()
    hp.mollview(m)
    plt.savefig(f'images/mask_not_apodized_comp{inp.comp}_nside{nside_for_masking}_cut{inp.cut}.png')
    print(f'saved images/mask_not_apodized_comp{inp.comp}_nside{nside_for_masking}_cut{inp.cut}.png', flush=True)
    aposcale = inp.aposcale # Apodization scale in degrees, original
    m = nmt.mask_apodization(m, aposcale, apotype="C1")
    plt.clf()
    hp.mollview(m)
    plt.savefig(f'images/apodized_mask_comp{inp.comp}_nside{nside_for_masking}_cut{inp.cut}.png')
    print(f'saved images/apodized_mask_comp{inp.comp}_nside{nside_for_masking}_cut{inp.cut}.png', flush=True)
    if save_file:
        hp.write_map(f'masks/mask_{inp.comp}_cut{inp.cut}_nside{nside_for_masking}.fits', m, overwrite=True)
        print(f'wrote masks/mask_{inp.comp}_cut{inp.cut}_nside{nside_for_masking}.fits', flush=True)
    return m

def gen_mask(inp):
   
    Nl = int((inp.ellmax+1)/inp.dl)
    nside_for_masking = inp.nside//2
    ells = np.arange(inp.ellmax+1)

    #check if mask already exists
    if os.path.exists(f'mask_{inp.comp}_cut{inp.cut}_nside{nside_for_masking}.fits'):
        print('mask already exists', flush=True)
        mask = hp.read_map(f'mask_{inp.comp}_cut{inp.cut}_nside{nside_for_masking}.fits')
        return mask

    #load maps
    map_ = hp.read_map(inp.map_file) 
    map_ = hp.ud_grade(map_, inp.nside)
    plt.clf()
    hp.mollview(map_)
    plt.savefig(f'images/map_{inp.comp}_nside{inp.nside}.png')
    print(f'saved images/map_{inp.comp}_nside{inp.nside}.png', flush=True)

    #create mask
    mean = np.mean(map_)
    std_dev = np.std(map_)
    cut_high = mean + inp.cut*std_dev
    cut_low = mean - inp.cut*std_dev
    mask = mask_above_cut(inp, cut_high, cut_low, map_, inp.nside, nside_for_masking, save_file=True)
    return mask
