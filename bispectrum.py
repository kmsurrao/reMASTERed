#Code adapted from PolyBin O. H. E. Philcox (2022), in prep.

import healpy as hp
import numpy as np

def Bl_norm(inp):
    '''
    Computes bispectrum normalization

    PARAMETER
    inp: Info() object, contains information about input parameters

    RETURNS
    norm: 3D numpy array, indexed as norm[l1,l2,l3]
    '''
    
    # Compute normalization matrix
    lmax = inp.ellmax
    lmax_sum = inp.ell_sum_max
    norm = np.zeros((lmax+1,lmax_sum+1,lmax_sum+1))

    # Iterate over bins
    for l1 in range(lmax+1):
        for l2 in range(lmax_sum+1):
            for l3 in range(abs(l1-l2),min(l1+l2+1,lmax_sum+1)):
                if (-1)**(l1+l2+l3)==-1: continue # 3j = 0 here
                norm[l1,l2,l3] += inp.wigner3j[l1,l2,l3]**2*(2.*l1+1.)*(2.*l2+1.)*(2.*l3+1.)/(4.*np.pi)

    return norm



def Bl_numerator(inp, data1, data2, data3, equal12=False,equal23=False,equal13=False):
    """
    Compute the numerator of the idealized bispectrum estimator. 
    NB: this doesn't subtract off the disconnected terms, so requires mean-zero maps!

    PARAMETERS
    inp: Info() object, contains information about input parameters
    data{i}: 1D numpy array, ith map input to bispectrum
    equal{i}{j}: Bool, whether data{i}==data{j}

    RETURNS
    b_num_ideal: 3D numpy array, indexed as b_num_ideal[l1,l2,l3]
    """
    
    lmax = inp.ellmax
    lmax_sum = inp.ell_sum_max
    lmax_data = 3*inp.nside-1
    Nside = inp.nside
    l_arr,m_arr = hp.Alm.getlm(lmax_data)
    

    # Define pixel area
    Npix = 12*Nside**2
    A_pix = 4.*np.pi/Npix
    
    ## Transform to harmonic space + compute I maps
    
    # Map 1
    data1_lm = hp.map2alm(data1)
    I1_map = [hp.alm2map((l_arr==l)*data1_lm,Nside) for l in range(lmax+1)]
    
    # Map 2
    if equal12:
        data2_lm = hp.map2alm(data2)
        I2_map = I1_map + [hp.alm2map((l_arr==l)*data2_lm,Nside) for l in range(lmax+1, lmax_sum+1)]
    else:
        data2_lm = hp.map2alm(data2)
        I2_map = [hp.alm2map((l_arr==l)*data2_lm,Nside) for l in range(lmax_sum+1)]

    # Map 3
    if equal13:
        data3_lm = hp.map2alm(data3)
        I3_map = I1_map + [hp.alm2map((l_arr==l)*data3_lm,Nside) for l in range(lmax+1, lmax_sum+1)]
    elif equal23:
        I3_map = I2_map
    else:
        data3_lm = hp.map2alm(data3)
        I3_map = [hp.alm2map((l_arr==l)*data3_lm,Nside) for l in range(lmax_sum+1)]
    
    # Combine to find numerator
    b_num_ideal = np.zeros((lmax+1,lmax_sum+1,lmax_sum+1))

    # Iterate over bins
    for l1 in range(lmax+1):
        for l2 in range(lmax_sum+1):
            for l3 in range(abs(l1-l2),min(l1+l2+1,lmax_sum+1)):

                # skip odd modes which vanish on average
                if (-1)**(l1+l2+l3)==-1: continue
                
                # compute numerators
                b_num_ideal[l1,l2,l3] = A_pix*np.sum(I1_map[l1]*I2_map[l2]*I3_map[l3])

    return b_num_ideal



def Bispectrum(inp, data1, data2, data3, equal12=False,equal23=False,equal13=False):
    '''
    Computes bispectrum

    PARAMETERS
    inp: Info() object, contains information about input parameters
    data{i}: 1D numpy array, ith map input to bispectrum
    equal{i}{j}: Bool, whether data{i}==data{j}

    RETURNS
    bl_out: 3D numpy array, indexed as bl_out[l1,l2,l3]
    '''
    bl_norm = Bl_norm(inp)
    bl_out = Bl_numerator(inp, data1, data2, data3, equal12=equal12, equal23=equal23, equal13=equal13)
    # Normalize bispectrum
    bl_out[bl_norm!=0] /= bl_norm[bl_norm!=0]
    return bl_out