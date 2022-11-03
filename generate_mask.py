import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt
import pickle
import os.path
import sys
from input import Info

def mask_above_cut(inp, cut_high, cut_low, map_, nside, nside_for_masking):
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
    return m

def gen_mask(inp, map_):
   
    Nl = int((inp.ellmax+1)/inp.dl)
    ells = np.arange(inp.ellmax+1)

    #create mask
    mean = np.mean(map_)
    std_dev = np.std(map_)
    cut_high = mean + inp.cut*std_dev
    cut_low = mean - inp.cut*std_dev
    mask = mask_above_cut(inp, cut_high, cut_low, map_, inp.nside, inp.nside_for_masking)
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
    map_ = hp.read_map(inp.map_file)
    map_ = hp.ud_grade(map_, inp.nside)
    mask = gen_mask(inp, map_)
    plt.clf()
    hp.mollview(mask)
    plt.savefig(f'mask_{inp.comp}_cut{inp.cut}_high{inp.cut_high}_low{inp.cut_low}_ellmax{inp.ellmax}_nsims{inp.nsims}_nside{inp.nside}_nsideformasking{inp.nside_for_masking}.png')
    print(f'saved mask_{inp.comp}_cut{inp.cut}_high{inp.cut_high}_low{inp.cut_low}_ellmax{inp.ellmax}_nsims{inp.nsims}_nside{inp.nside}_nsideformasking{inp.nside_for_masking}.png', flush=True)