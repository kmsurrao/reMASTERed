import numpy as np
import healpy as hp
import pickle
from scipy.interpolate import InterpolatedUnivariateSpline
import time
import os.path
start_time = time.time()


def Bispectrum(alm1, Cl1, alm2, Cl2, alm3, Cl3, lmax, Nside, term, inp=None):

    #check if bispectrum already exists
    if inp and os.path.exists(f'bispectra/bispectrum_{inp.comp}_{inp.cut}_ellmax{lmax}_term{term}.p'):
        print('bispectrum already exists', flush=True)
        bispectrum = pickle.load(open(f'bispectra/bispectrum_{inp.comp}_{inp.cut}_ellmax{lmax}_term{term}.p', 'rb'))
        return bispectrum

    A_pix = 4.*np.pi/(12*Nside**2)

    # Define ell arrays
    l = np.arange(lmax+1)
    l_arr,m_arr = hp.Alm.getlm(lmax)
    # Interpolate to all ell, m grid
    Cl1_interp = InterpolatedUnivariateSpline(l,Cl1)
    Cl1_lm = Cl1_interp(l_arr)
    Cl2_interp = InterpolatedUnivariateSpline(l,Cl2)
    Cl2_lm = Cl2_interp(l_arr)
    Cl3_interp = InterpolatedUnivariateSpline(l,Cl3)
    Cl3_lm = Cl3_interp(l_arr)


    # Basic HEALPix utilities
    def to_lm(input_map):
        """Convert from map-space to harmonic-space"""
        return hp.map2alm(input_map,pol=False)

    def to_map(input_lm):
        """Convert from harmonic-space to map-space"""
        return hp.alm2map(input_lm,Nside,pol=False)

    def safe_divide(x,y):
        """Function to divide maps without zero errors."""
        out = np.zeros_like(x)
        out[y!=0] = x[y!=0]/y[y!=0]
        return out

    # Define ell bins
    ell_bins = [(l_arr==l) for l in range(lmax+1)]
    print('ell_bins: ', ell_bins, flush=True)

    # Compute I maps
    I_map1 = [to_map(ell_bins[bin1]*safe_divide(alm1,Cl1_lm)) for bin1 in range(len(ell_bins))]
    I_map2 = [to_map(ell_bins[bin1]*safe_divide(alm2,Cl2_lm)) for bin1 in range(len(ell_bins))]
    I_map3 = [to_map(ell_bins[bin1]*safe_divide(alm3, Cl3_lm)) for bin1 in range(len(ell_bins))]
    print('computed I maps', flush=True)


    # Load pre-computed 3j symbols
    assert lmax<=1000, "Higher-l 3j symbols not yet precomputed!"
    # tj_arr = pickle.load(open('/global/homes/k/kmsurrao/NILC-Parameter-Pipeline/wigner3j_ellmax1000.p', 'rb'))[:lmax+1,:lmax+1,:lmax+1] #for cori
    tj_arr = pickle.load(open('/moto/hill/users/kms2320/wigner3j_ellmax1000.p', 'rb'))[:lmax+1,:lmax+1,:lmax+1] #for moto
    print('loaded tj_arr', flush=True)

    def check_triangle(lmax):
        """Array is one if modes satisfy the even-parity triangle conditions, or zero else.
        """
        output = np.ones((lmax+1, lmax+1, lmax+1))
        for l1 in range(lmax+1):
            for l2 in range(lmax+1):
                for l3 in range(lmax+1):
                    if l3<abs(l1-l2) or l3>l1+l2:
                        output[l1,l2,l3] = 0
                    elif l2<abs(l1-l3) or l2>l1+l3:
                        output[l1,l2,l3] = 0
                    elif l1<abs(l2-l3) or l1>l2+l3:
                        output[l1,l2,l3] = 0
        return output
    
    def get_sym_factors(lmax):
        '''
        computes symmetry factors 
        '''
        output = np.ones((lmax+1, lmax+1, lmax+1))
        for l1 in range(lmax+1):
            for l2 in range(lmax+1):
                for l3 in range(lmax+1):
                    if l1==l2 or l2==l3 or l1==l3:
                        output[l1,l2,l3] = 2
                    if l1==l2==l3:
                        output[l2,l2,l3] = 6
        return output


    # Combine to find numerator
    # notation: a=bin1, b=bin2, c=bin3, n indexes pixel
    check_triangle_array = check_triangle(lmax)
    print('got check_triangle_array', flush=True)
    sym_factors_array = get_sym_factors(lmax)
    print('got sym_factors_array', flush=True)
    b_num_ideal = A_pix*np.einsum('ijk,ijk,in,jn,kn->ijk', check_triangle_array, 1/sym_factors_array, I_map1, I_map2, I_map3, optimize=True)
    print('got b_num_ideal', flush=True)


    # C and M vectors 
    Cl1_vec = np.array([(li>=2)*(Cl1_interp(li)) for li in l])
    Cl2_vec = np.array([(li>=2)*(Cl2_interp(li)) for li in l])
    Cl3_vec = np.array([(li>=2)*(Cl3_interp(li)) for li in l])
    Cl1_vec[Cl1_vec==0]=np.inf
    Cl2_vec[Cl2_vec==0]=np.inf
    Cl3_vec[Cl3_vec==0]=np.inf
    print('got Cl_vec and Ml_vec', flush=True)


    # compute denominator     
    #notation: i=l1, j=l2, k=l3
    ells = np.arange(lmax+1)
    b_denom = 1/(4*np.pi)*np.einsum('ijk,ijk,i,j,k,i,j,k,ijk->ijk', tj_arr, tj_arr, (2*ells+1), (2*ells+1), (2*ells+1), 1/Cl1_vec, 1/Cl2_vec, 1/Cl3_vec, 1/sym_factors_array, optimize=True)
    print('got b_denom', flush=True)

    b_ideal = b_num_ideal/b_denom
    b_ideal[b_ideal==np.inf]=0.
    b_ideal[b_ideal==-np.inf]=0.
    b_ideal = np.nan_to_num(b_ideal)
    # print('b_ideal: ', b_ideal, flush=True)
    print('b_ideal.shape: ', b_ideal.shape, flush=True)
    if inp:
        pickle.dump(b_ideal, open(f'bispectra/bispectrum_{inp.comp}_{inp.cut}_ellmax{lmax}_term{term}.p', 'wb'))
        print(f'saved bispectra/bispectrum_{inp.comp}_{inp.cut}_ellmax{lmax}_term{term}.p', flush=True)
    return b_ideal

if __name__=="__main__":
    ellmax = 10
    Nside = 4
    # isw_map = hp.read_map('/global/cscratch1/sd/kmsurrao/Correlated-Mask-Power-Spectrum/maps/isw.fits') #for cori
    # mask = hp.read_map('/global/homes/k/kmsurrao/Correlated-Mask-Power-Spectrum/mask_isw_threshold.fits') #for cori
    isw_map = hp.read_map('isw.fits') #for moto
    mask = hp.read_map('mask_isw_threshold.fits') #for moto
    alm = hp.map2alm(isw_map, lmax=ellmax)
    wlm = hp.map2alm(mask, lmax=ellmax)
    Cl = hp.alm2cl(alm)
    Ml = hp.alm2cl(wlm)
    print('calling Bispectrum() term 4', flush=True)
    b_ideal = Bispectrum(alm, Cl, np.conj(alm), Cl, wlm, Ml, ellmax, Nside, 4)
    print("--- %s seconds ---" % (time.time() - start_time), flush=True)


    # import matplotlib.pyplot as plt

    # plt.clf()

    # ellmax = 50
    # Nside = 32
    # # isw_map = hp.read_map('/global/cscratch1/sd/kmsurrao/Correlated-Mask-Power-Spectrum/maps/isw.fits') #for cori
    # # mask = hp.read_map('/global/homes/k/kmsurrao/Correlated-Mask-Power-Spectrum/mask_isw_threshold.fits') #for cori
    # isw_map = hp.read_map('isw.fits') #for moto
    # mask = hp.read_map('mask_isw_threshold.fits') #for moto
    # alm = hp.map2alm(isw_map, lmax=ellmax)
    # wlm = hp.map2alm(mask, lmax=ellmax)
    # Cl = hp.alm2cl(alm)
    # Ml = hp.alm2cl(wlm)
    # print('calling Bispectrum() term 4', flush=True)
    # b_ideal = Bispectrum(alm, Cl, np.conj(alm), Cl, wlm, Ml, ellmax, Nside, 4)
    # print("--- %s seconds ---" % (time.time() - start_time), flush=True)


