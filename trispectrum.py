#Code adapted from PolyBin O. H. E. Philcox (2022), in prep.

import pywigxjpf as wig
import healpy as hp
import numpy as np

def compute_Cl(inp, lmax, data1, data2, equal12=False):
    """Compute the power spectrum up to some l-max"""
    lmax_data = 3*inp.nside-1
    l_arr,m_arr = hp.Alm.getlm(lmax_data)
    data1_lm = hp.map2alm(data1)
    if equal12:
        data2_lm = data1_lm
    else:
        data2_lm = hp.map2alm(data2)
    Cl_summand = (1.+(m_arr>0))*data1_lm*data2_lm.conj()/(2.*l_arr+1.)
    Cl = [np.sum(Cl_summand*(l_arr==l)).real for l in range(lmax+1)]
    return Cl

def Tl_numerator(inp,lmax, data1, data2, data3, data4,
                 Cl12_th, Cl13_th, Cl14_th, Cl23_th, Cl24_th, Cl34_th,
                 lmin=0,verb=False,
                 equal12=False,equal13=False,equal14=False,equal23=False,equal24=False,equal34=False):
    """
    Compute the numerator of the idealized trispectrum estimator. 
    Note that we weight according to *spin-zero* Gaunt symbols, which is different to Philcox (in prep.).
    This necessarily subtracts off the disconnected pieces, given input theory Cl_the spectra (plus noise, if appropriate).
    Note that we index the output as t[l1,l2,l3,l4,L] for diagonal momentum L.
    We also drop the L=0 pieces, since these would require non-mean-zero correlators.
    If lmin > 0, we don't calculate any pieces with l<lmin; the output array still contains these pieces, just filled with zeros.
    """
    
    if np.abs(data1.mean())>0.1*data1.std() or np.abs(data2.mean())>0.1*data2.std() or np.abs(data3.mean())>0.1*data3.std() or np.abs(data4.mean())>0.1*data4.std():
        raise Exception("Need mean-zero inputs!")

    # Define 3j calculation
    lmax_data = 3*inp.nside-1
    l_arr,m_arr = hp.Alm.getlm(lmax_data)
    wig.wig_table_init(lmax_data*2,9)
    wig.wig_temp_init(lmax_data*2)
    tj0 = lambda l1,l2,l3: wig.wig3jj(2*l1,2*l2,2*l3,0,0,0)
    Nside = inp.nside
    
    
    ## Transform to harmonic space + compute I maps
    if verb: print("Computing I^a maps")
    
    # Map 1
    data1_lm = hp.map2alm(data1)
    I1_map = [hp.alm2map((l_arr==l)*data1_lm,Nside) for l in range(lmin,lmax+1)]
    
    # Map 2
    if equal12:
        I2_map = I1_map
    else:
        data2_lm = hp.map2alm(data2)
        I2_map = [hp.alm2map((l_arr==l)*data2_lm,
        
        de) for l in range(lmin,lmax+1)]

    # Map 3
    if equal13:
        I3_map = I1_map
    elif equal23:
        I3_map = I2_map
    else:
        data3_lm = hp.map2alm(data3)
        I3_map = [hp.alm2map((l_arr==l)*data3_lm,Nside) for l in range(lmin,lmax+1)]
    
    # Map 4
    if equal14:
        I4_map = I1_map
    elif equal24:
        I4_map = I2_map
    elif equal34:
        I4_map = I3_map
    else:
        data4_lm = hp.map2alm(data4)
        I4_map = [hp.alm2map((l_arr==l)*data4_lm,Nside) for l in range(lmin,lmax+1)]
    
    ## Define maps of A^{ab}_lm = int[dn Y_lm(n) I^a(n)I^b(n)] for two I maps
    if verb: print("Computing A^{ab} maps")
    A12_lm = [[hp.map2alm(I1_map[l1-lmin]*I2_map[l2-lmin]) for l2 in range(lmin,lmax+1)] for l1 in range(lmin,lmax+1)]
    A34_lm = [[hp.map2alm(I3_map[l3-lmin]*I4_map[l4-lmin]) for l4 in range(lmin,lmax+1)] for l3 in range(lmin,lmax+1)]
    
    # Create output arrays (for 4-field, 2-field and 0-field terms)
    t4_num_ideal = np.zeros((lmax+1,lmax+1,lmax+1,lmax+1,lmax+1), dtype=np.float32)
    t2_num_ideal = np.zeros_like(t4_num_ideal, dtype=np.float32)
    t0_num_ideal = np.zeros_like(t4_num_ideal, dtype=np.float32)
    
    ## Compute four-field term
    if verb: print("Computing four-field term")
    
    # Iterate over bins
    for l1 in range(lmin,lmax+1):
        for l2 in range(lmin,lmax+1):
            for l3 in range(lmin,lmax+1):
                for l4 in range(lmin,lmax+1):
                    if (-1)**(l1+l2+l3+l4)==-1: continue
                    
                    summand = A12_lm[l1-lmin][l2-lmin]*A34_lm[l3-lmin][l4-lmin].conj()
                    
                    for L in range(max(abs(l1-l2),lmin),min(l1+l2+1,lmax+1)):
                        if L<abs(l3-l4) or L>l3+l4: continue
                        if (-1)**(l1+l2+L)==-1: continue # drop parity-odd modes
                        if (-1)**(l3+l4+L)==-1: continue 
   
                        # Compute four-field term
                        t4_num_ideal[l1,l2,l3,l4,L] = np.sum(summand*(l_arr==L)*(1.+(m_arr>0))).real
        
    ## Compute two-field term
    if verb: print("Computing two-field and zero-field terms")
    
    # Compute empirical power spectra
    Cl12 = compute_Cl(inp, lmax, data1, data2, equal12=equal12)
    Cl13 = compute_Cl(inp, lmax, data1, data3, equal12=equal13)
    Cl14 = compute_Cl(inp, lmax, data1, data4, equal12=equal14)
    Cl23 = compute_Cl(inp, lmax, data2, data3, equal12=equal23)
    Cl24 = compute_Cl(inp, lmax, data2, data4, equal12=equal24)
    Cl34 = compute_Cl(inp, lmax, data3, data4, equal12=equal34)
    
    # Iterate over bins
    for l1 in range(lmin,lmax+1):
        for l2 in range(lmin,lmax+1):
            for l3 in range(lmin,lmax+1):
                for l4 in range(lmin,lmax+1):
                    
                    # first permutation
                    if (l1==l2) and (l3==l4):
                        t2_num_ideal[l1,l2,l3,l4,0] += -(2*l1+1)*(2*l3+1)/(4.*np.pi)*(Cl12[l1]*Cl34_th[l3]+Cl12_th[l1]*Cl34[l3])
                        t0_num_ideal[l1,l2,l3,l4,0] += (2*l1+1)*(2*l3+1)/(4.*np.pi)*(Cl12_th[l1]*Cl34_th[l3])
                        
                    # second permutation
                    if (l1==l3 and l2==l4):
                        for L in range(max(abs(l1-l2),lmin),min(l1+l2+1,lmax+1)):
                            if L<abs(l3-l4) or L>l3+l4: continue
                            if (-1)**(l1+l2+L)==-1: continue # drop parity-odd modes
                            if (-1)**(l3+l4+L)==-1: continue 

                            # Compute two-field term
                            prefactor = (2*l1+1)*(2*l2+1)*(2*L+1)/(4.*np.pi)*tj0(l1,l2,L)**2
                            t2_num_ideal[l1,l2,l3,l4,L] += -prefactor*(Cl13_th[l1]*Cl24[l2]+Cl13[l1]*Cl24_th[l2])
                            t0_num_ideal[l1,l2,l3,l4,L] += prefactor*Cl13_th[l1]*Cl24_th[l2]
                            
                    # third permutation
                    if (l1==l4 and l2==l3):
                        for L in range(max(abs(l1-l2),lmin),min(l1+l2+1,lmax+1)):
                            if L<abs(l3-l4) or L>l3+l4: continue
                            if (-1)**(l1+l2+L)==-1: continue # drop parity-odd modes
                            if (-1)**(l3+l4+L)==-1: continue 

                            # Compute two-field term
                            prefactor = (2*l1+1)*(2*l2+1)*(2*L+1)/(4.*np.pi)*tj0(l1,l2,L)**2
                            t2_num_ideal[l1,l2,l3,l4,L] += -prefactor*(Cl14_th[l1]*Cl23[l2]+Cl14[l1]*Cl23_th[l2])
                            t0_num_ideal[l1,l2,l3,l4,L] += prefactor*Cl14_th[l1]*Cl23_th[l2]
                         
    return t4_num_ideal+t2_num_ideal+t0_num_ideal


def rho(inp, a_map, w_map, Cl_aw, Cl_aa, Cl_ww):
    # Compute trispectrum without normalization
    tl_out = Tl_numerator(inp, inp.ellmax,
                          a_map,a_map,w_map,w_map,
                          Cl_aa,Cl_aw,Cl_aw,Cl_aw,Cl_aw,Cl_ww,
                          verb=True,
                          equal12=True,equal34=True)
    return tl_out