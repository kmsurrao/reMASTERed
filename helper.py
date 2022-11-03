import numpy as np
import healpy as hp
import py3nj

def get_w_array(inp, hp_wlm):
    '''
    PARAMETERS
    inp: Info() object containing parameters of the run
    hp_wlm: wlm in default healpy ordering
    
    RETURNS
    wlm: 2D array with dimensions (ellmax+1, 2*ellmax+1) indexed as wlm[l][ellmax+m] to get w_{lm}
    '''
    ellmax = inp.ellmax
    wlm = np.zeros((ellmax+1, 2*ellmax+1))
    ells, ms = hp.Alm.getlm(ellmax)
    for i in range(hp_wlm):
        l,m = ells[i], ms[i]
        wlm[l][ellmax+m] = hp_wlm[i]
        wlm[l][ellmax-m] = (-1)**m*np.conj(hp_wlm[i])
    return wlm

def get_aw_array(inp, hp_alm, hp_wlm):
    '''
    PARAMETERS
    inp: Info() object containing parameters of the run
    hp_alm: alm in default healpy ordering
    hp_wlm: wlm in default healpy ordering
    
    RETURNS
    aw_array: 4D array with dimensions (ellmax+1, 2*ellmax+1, ellmax+1, 2*ellmax+1) indexed as aw[l1][ellmax+m1][l2][ellmax+m2] to get a_{l1m1}w_{l2m2}
    '''
    ellmax = inp.ellmax
    aw_array = np.zeros((ellmax+1, 2*ellmax+1, ellmax+1, 2*ellmax+1))
    ells, ms = hp.Alm.getlm(ellmax)
    for i in range(hp_alm):
        for j in range(hp_wlm):
            l1,m1 = ells[i], ms[i]
            l2,m2 = ells[j], ms[j]
            aw_array[l1][ellmax+m1][l2][ellmax+m2] = hp_alm[i]*hp_wlm[j]
            aw_array[l1][ellmax-m1][l2][ellmax+m2] = (-1)**m1*np.conj(hp_alm[i])*hp_wlm[j]
            aw_array[l1][ellmax+m1][l2][ellmax-m2] = (-1)**m2*hp_alm[i]*np.conj(hp_wlm[j])
            aw_array[l1][ellmax-m1][l2][ellmax-m2] = (-1)**(m1+m2)*np.conj(hp_alm[i])*np.conj(hp_wlm[j])
    return aw_array

def get_neg_ones_array(inp):
    '''
    PARAMETERS
    inp: Info() object containing parameters of the run
    
    RETURNS
    neg_ones_array: 1D array of shape 2*ellmax+1, indexed as neg_ones_array[ellmax+m]
    '''
    ellmax = inp.ellmax
    neg_ones_array = np.ones(2*ellmax+1)
    if ellmax%2==0:
        neg_ones_array[1::2] = -1
    else:
        neg_ones_array[0::2] = -1
    return neg_ones_array

def get_3j_nonzero_m(inp):
    '''
    PARAMETERS
    inp: Info() object containing parameters of the run
    
    RETURNS
    wigner_3j_arr: 6D array of shape (ellmax+1, 2*ellmax+1, ellmax+1, 2*ellmax+1, ellmax+1, 2*ellmax+1) indexed as wigner_3j_array[l1][ellmax+m1][l2][ellmax+m2][l3][ellamx+m3]
    '''
    l1_to_compute = 0


