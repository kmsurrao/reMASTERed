import numpy as np
import healpy as hp
import pymaster as nmt

def mask_above_cut(inp, cut_high, cut_low, map_, nside, nside_for_masking):
    '''
    PARAMETERS
    inp: Info object, contains input specifications
    cut_high: float, value of map pixels above which to mask if inp.cut_high is True
    cut_low: float, value of map pixels below which to mask if inp.cut_low is True
    map_ : 1D numpy array, map for which to create mask
    nside: int, healpy NSIDE parameter of the input map
    nside_for_masking: int, healpy NSIDE parameter to use for mask creation

    RETURNS
    1D numpy array containing the apodized mask 
    '''

    #downgrade resolution of map to create mask initially
    map_tmp = hp.ud_grade(map_, nside_for_masking)
    if inp.cut_high and inp.cut_low:
        #set mask to 0 below cut_low or above cut_high, and 1 elsewhere
        m = np.where(np.logical_or(map_tmp>cut_high, map_tmp<cut_low), 0, 1)
    elif inp.cut_high:
        m = np.where(map_tmp>cut_high, 0, 1)
    elif inp.cut_low:
        m = np.where(map_tmp<cut_low, 0, 1)
    #return mask to nside of original map
    m = hp.ud_grade(m, nside)
    aposcale = inp.aposcale # Apodization scale in degrees
    m = nmt.mask_apodization(m, aposcale, apotype="C1")
    m[m>1.]=1.
    m[m<0.]=0.
    return m

def gen_mask(inp, map_):
    '''
    PARAMETERS
    inp: Info object containing input specifications
    map_: 1D numpy array, map for which to create mask

    RETURNS
    1D numpy array containing the apodized mask 
    '''

    mean = np.mean(map_)
    std_dev = np.std(map_)
    cut_high = mean + inp.cut*std_dev
    cut_low = mean - inp.cut*std_dev
    mask = mask_above_cut(inp, cut_high, cut_low, map_, inp.nside, inp.nside_for_masking)
    return mask
