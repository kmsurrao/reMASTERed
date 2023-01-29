#Code adapted from PolyBin O. H. E. Philcox (2022), in prep.

import healpy as hp
import numpy as np


def Tl_numerator(inp, lmax, lmax_sum, data1, data2, data3, data4,
                 Cl12_th, Cl13_th, Cl14_th, Cl23_th, Cl24_th, Cl34_th,
                 lmin=0, verb=False,
                 equal12=False,equal13=False,equal14=False,equal23=False,equal24=False,equal34=False,
                 remove_two_point=True):
    """
    Compute the numerator of the idealized trispectrum estimator. 
    Note that we weight according to *spin-zero* Gaunt symbols, which is different to Philcox (in prep.).
    This necessarily subtracts off the disconnected pieces, given input theory Cl_th spectra (plus noise, if appropriate).
    Note that we index the output as t[l1,l2,l3,l4,L] for diagonal momentum L.
    We also drop the L=0 pieces, since these would require non-mean-zero correlators.
    If lmin > 0, we don't calculate any pieces with l<lmin; the output array still contains these pieces, just filled with zeros.

    PARAMETERS
    inp: Info() object, contains information about input parameters
    lmax: int, maximum ell for which to calculate output
    lmax_sum: int, maximum ell for summing over
    data{i}: 1D numpy array, ith map input to trispectrum
    Cl{i}{j}_th: 1D numpy array, power spectrum of data{i} and data{j}
    lmin: int, minimum ell for which to calculate output
    verb: Bool, whether to print
    equal{i}{j}: Bool, whether data{i}==data{j}
    remove_two_point: Bool, whether to subtract two-point disconnected pieces

    RETURNS
    t4_num_ideal+t2_num_ideal+t0_num_ideal: 5D numpy array, indexed with [l1,l2,l3,l4,L]
    """

    # Define 3j calculation
    lmax_data = 3*inp.nside-1
    l_arr,m_arr = hp.Alm.getlm(lmax_data)
    Nside = inp.nside
    
    
    ## Transform to harmonic space + compute I maps
    if verb: print("Computing I^a maps")
    
    # Map 1
    data1_lm = hp.map2alm(data1)
    I1_map = [hp.alm2map((l_arr==l)*data1_lm,Nside) for l in range(lmin,lmax_sum+1)]
    
    # Map 2
    if equal12:
        I2_map = I1_map
    else:
        data2_lm = hp.map2alm(data2)
        I2_map = [hp.alm2map((l_arr==l)*data2_lm,Nside) for l in range(lmin,lmax_sum+1)]

    # Map 3
    if equal13:
        I3_map = I1_map
    elif equal23:
        I3_map = I2_map
    else:
        data3_lm = hp.map2alm(data3)
        I3_map = [hp.alm2map((l_arr==l)*data3_lm,Nside) for l in range(lmin,lmax_sum+1)]
    
    # Map 4
    if equal14:
        I4_map = I1_map
    elif equal24:
        I4_map = I2_map
    elif equal34:
        I4_map = I3_map
    else:
        data4_lm = hp.map2alm(data4)
        I4_map = [hp.alm2map((l_arr==l)*data4_lm,Nside) for l in range(lmin,lmax_sum+1)]
    
    ## Define maps of A^{ab}_lm = int[dn Y_lm(n) I^a(n)I^b(n)] for two I maps
    if verb: print("Computing A^{ab} maps")
    A12_lm = [[hp.map2alm(I1_map[l1-lmin]*I2_map[l2-lmin]) for l2 in range(lmin,lmax_sum+1)] for l1 in range(lmin,lmax_sum+1)]
    A34_lm = [[hp.map2alm(I3_map[l3-lmin]*I4_map[l4-lmin]) for l4 in range(lmin,lmax_sum+1)] for l3 in range(lmin,lmax_sum+1)]
    
    # Create output arrays (for 4-field, 2-field and 0-field terms)
    t4_num_ideal = np.zeros((lmax_sum+1,lmax_sum+1,lmax_sum+1,lmax_sum+1,lmax+1), dtype=np.float32)
    t2_num_ideal = np.zeros_like(t4_num_ideal, dtype=np.float32)
    t0_num_ideal = np.zeros_like(t4_num_ideal, dtype=np.float32)
    
    ## Compute four-field term
    if verb: print("Computing four-field term")
    
    # Iterate over bins
    for l1 in range(lmin,lmax_sum+1):
        for l2 in range(lmin,lmax_sum+1):
            for l3 in range(lmin,lmax_sum+1):
                for l4 in range(lmin,lmax_sum+1):
                    if (-1)**(l1+l2+l3+l4)==-1: continue
                    
                    summand = A12_lm[l1-lmin][l2-lmin]*A34_lm[l3-lmin][l4-lmin].conj()
                    
                    for L in range(max(abs(l1-l2),lmin),min(l1+l2+1,lmax+1)):
                        if L<abs(l3-l4) or L>l3+l4: continue
                        if (-1)**(l1+l2+L)==-1: continue # drop parity-odd modes
                        if (-1)**(l3+l4+L)==-1: continue 
   
                        # Compute four-field term
                        t4_num_ideal[l1,l2,l3,l4,L] = np.sum(summand*(l_arr==L)*(1.+(m_arr>0))).real
    
    if not remove_two_point:
        return t4_num_ideal

    ## Compute two-field term
    if verb: print("Computing two-field and zero-field terms")
    
    # Compute empirical power spectra
    Cl12 = hp.anafast(data1, data2, lmax=lmax_sum)
    Cl13 = hp.anafast(data1, data3, lmax=lmax_sum)
    Cl14 = hp.anafast(data1, data4, lmax=lmax_sum)
    Cl23 = hp.anafast(data2, data3, lmax=lmax_sum)
    Cl24 = hp.anafast(data2, data4, lmax=lmax_sum)
    Cl34 = hp.anafast(data3, data4, lmax=lmax_sum)
    
    # Iterate over bins
    for l1 in range(lmin,lmax_sum+1):
        for l2 in range(lmin,lmax_sum+1):
            for l3 in range(lmin,lmax_sum+1):
                for l4 in range(lmin,lmax_sum+1):
                        
                    # second permutation
                    if (l1==l3 and l2==l4):
                        for L in range(max(abs(l1-l2),lmin),min(l1+l2+1,lmax+1)):
                            if L<abs(l3-l4) or L>l3+l4: continue
                            if (-1)**(l1+l2+L)==-1: continue # drop parity-odd modes
                            if (-1)**(l3+l4+L)==-1: continue 

                            # Compute two-field term
                            prefactor = (2*l1+1)*(2*l2+1)*(2*L+1)/(4.*np.pi)*inp.wigner3j[l1,l2,L]**2
                            t2_num_ideal[l1,l2,l3,l4,L] += -prefactor*(Cl13_th[l1]*Cl24[l2]+Cl13[l1]*Cl24_th[l2])
                            t0_num_ideal[l1,l2,l3,l4,L] += prefactor*Cl13_th[l1]*Cl24_th[l2]
                            
                    # third permutation
                    if (l1==l4 and l2==l3):
                        for L in range(max(abs(l1-l2),lmin),min(l1+l2+1,lmax+1)):
                            if L<abs(l3-l4) or L>l3+l4: continue
                            if (-1)**(l1+l2+L)==-1: continue # drop parity-odd modes
                            if (-1)**(l3+l4+L)==-1: continue 

                            # Compute two-field term
                            prefactor = (2*l1+1)*(2*l2+1)*(2*L+1)/(4.*np.pi)*inp.wigner3j[l1,l2,L]**2
                            t2_num_ideal[l1,l2,l3,l4,L] += -prefactor*(Cl14_th[l1]*Cl23[l2]+Cl14[l1]*Cl23_th[l2])
                            t0_num_ideal[l1,l2,l3,l4,L] += prefactor*Cl14_th[l1]*Cl23_th[l2]
                         
    return t4_num_ideal+t2_num_ideal+t0_num_ideal


def rho(inp, a_map, w_map, Cl_aw, Cl_aa, Cl_ww, remove_two_point=True):
    '''
    Compute trispectrum without normalization

    PARAMETERS
    inp: Info() object, contains information about input parameters
    a_map: 1D numpy array, map of signal with average subtracted
    w_map: 1D numpy array, map of mask with average subtracted
    Cl_aw: 1D numpy array, cross-spectrum of the map and mask with averages subtracted
    Cl_aa: 1D numpy array, auto-spectrum of the map with average subtracted
    Cl_ww: 1D numpy array, auto-spectrum of the mask with average subtracted
    remove_two_point: Bool, whether to subtract two-point disconnected pieces

    RETURNS
    tl_out: 5D numpy array, indexed as tl_out[l2,l4,l3,l5,l1]
    '''
    tl_out = Tl_numerator(inp, inp.ellmax, inp.ell_sum_max, 
                          a_map,w_map,a_map,w_map,
                          Cl_aw,Cl_aa,Cl_aw,Cl_aw,Cl_ww,Cl_aw,
                          verb=False, equal13=True, equal24=True, 
                          remove_two_point=remove_two_point)
    return tl_out