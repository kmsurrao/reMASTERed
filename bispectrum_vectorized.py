import numpy as np
import healpy as hp
import pickle
from scipy.interpolate import InterpolatedUnivariateSpline


def Bispectrum(alm, Cl, wlm, Ml, lmax, Nside, Nl, dl, min_l):

    print("binned lmax: %d"%(min_l+dl*Nl), flush=True)
    A_pix = 4.*np.pi/(12*Nside**2)

    # Define ell arrays
    l = np.arange(lmax+1)
    l_arr,m_arr = hp.Alm.getlm(lmax)
    # Interpolate to all ell, m grid
    Cl_interp = InterpolatedUnivariateSpline(l,Cl)
    Cl_lm = Cl_interp(l_arr)
    Ml_interp = InterpolatedUnivariateSpline(l,Ml)
    Ml_lm = Ml_interp(l_arr)
    # Zero out ell = 0 and ell = 1
    Cl_lm[l_arr<min_l] = 0.
    Ml_lm[l_arr<min_l] = 0.

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
    ell_bins = [(l_arr>=min_l+dl*bin1)&(l_arr<min_l+dl*(bin1+1)) for bin1 in range(Nl)]

    # Compute I maps
    I_map1 = [to_map(ell_bins[bin1]*safe_divide(alm,Cl_lm)) for bin1 in range(Nl)]
    I_map2 = I_map1
    I_map3 = [to_map(ell_bins[bin1]*safe_divide(wlm,Ml_lm)) for bin1 in range(Nl)]
    print('computed I maps', flush=True)


    def check_bins(bins):
        """Array is one if modes in the bin satisfy the even-parity triangle conditions, or zero else.
        
        This is used either for all triangles in the bin, or just the center of the bin.
        """
        output = np.ones((len(bins), len(bins), len(bins)))
        l1 = min_l+(bins+0.5)*dl
        l2 = min_l+(bins+0.5)*dl
        l3 = min_l+(bins+0.5)*dl
        output[l3<abs(l1-l2)] = 0
        output[l3>l1+l2] = 0
        return output
    
    def get_sym_factors(Nl):
        '''
        computes symmetry factors 
        '''
        output = np.ones((Nl, Nl, Nl))
        bins1 = np.arange(Nl)
        bins2 = np.arange(Nl)
        bins3 = np.arange(Nl)
        output[np.logical_or(np.logical_or(bins1==bins2, bins2==bins3), bins1==bins3)] = 2
        output[np.logical_and(bins1==bins2, bins2==bins3)] = 6
        return output


    # Combine to find numerator
    # notation: a=bin1, b=bin2, c=bin3, n indexes pixel
    check_bins_array = check_bins(np.arange(Nl))
    print('got check_bins_array', flush=True)
    sym_factors_array = get_sym_factors(Nl)
    print('got sym_factors_array', flush=True)
    b_num_ideal = A_pix*np.einsum('abc,abc,an,bn,cn->abc', check_bins_array, 1/sym_factors_array, I_map1, I_map2, I_map3, optimize=True)
    print('got b_num_ideal', flush=True)

    # number of bins
    N_b = len(b_num_ideal)
    print("%d bispectrum bins"%N_b, flush=True)

    # Load pre-computed 3j symbols
    assert lmax<=1000, "Higher-l 3j symbols not yet precomputed!"
    # tj_arr = pickle.load(open('/global/homes/k/kmsurrao/NILC-Parameter-Pipeline/wigner3j_ellmax1000.p', 'rb'))[:lmax+1,:lmax+1,:lmax+1] #for cori
    tj_arr = pickle.load(open('/moto/hill/users/kms2320/wigner3j_ellmax1000.p', 'rb'))[:lmax+1,:lmax+1,:lmax+1] #for moto
    print('loaded tj_arr', flush=True)

    # C and M vectors 
    Cl_vec = np.array([(li>=2)*(Cl_interp(li)) for li in l])
    Ml_vec = np.array([(li>=2)*(Ml_interp(li)) for li in l])
    Cl_vec[Cl_vec==0]=np.inf
    Ml_vec[Ml_vec==0]=np.inf
    print('got Cl_vec and Ml_vec', flush=True)
    print('Cl_vec: ', Cl_vec, flush=True)
    print('Ml_vec: ', Ml_vec, flush=True)




    # compute denominator     
    #notation: a=bin1, b=bin2, c=bin3, i=l1, j=l2, k=l3
    ells = np.arange(lmax+1)
    ells_in_bin = np.zeros((Nl, lmax+1))
    for bin in range(Nl):
        ells_in_bin[bin][min_l+bin*dl : min_l+(bin+1)*dl] = 1
    print('got ells_in_bin', flush=True)
    print('ells_in_bin.shape: ', ells_in_bin.shape, flush=True)
    print('tj_arr.shape: ', tj_arr.shape, flush=True)
    print('ells.shape: ', ells.shape, flush=True)
    print('Cl_vec.shape: ', Cl_vec.shape, flush=True)
    print('Ml_vec.shape: ', Ml_vec.shape, flush=True)
    print('sym_factors_array.shape: ', sym_factors_array.shape, flush=True)
    b_denom = 1/(4*np.pi)*np.einsum('ai,bj,ck,ijk,ijk,i,j,k,i,j,k,abc->abc', ells_in_bin, ells_in_bin, ells_in_bin, tj_arr, tj_arr, (2*ells+1), (2*ells+1), (2*ells+1), 1/Cl_vec, 1/Cl_vec, 1/Ml_vec, 1/sym_factors_array, optimize=True)
    print('got b_denom', flush=True)

    b_ideal = b_num_ideal/b_denom
    b_ideal[b_ideal==np.inf]=0.
    b_ideal[b_ideal==-np.inf]=0.
    print('b_ideal: ', b_ideal, flush=True)
    print('b_ideal.shape: ', b_ideal.shape, flush=True)
    return b_ideal

if __name__=="__main__":
    ellmax = 500
    Nside = 256
    # ellmax = 100
    # Nside = 64
    # Binning parameters
    dl = 10 # bin width
    Nl = int(ellmax/dl) # number of bins
    min_l = 0 # minimum l
    # isw_map = hp.read_map('/global/cscratch1/sd/kmsurrao/Correlated-Mask-Power-Spectrum/maps/isw.fits') #for cori
    # mask = hp.read_map('/global/homes/k/kmsurrao/Correlated-Mask-Power-Spectrum/mask_isw_threshold.fits') #for cori
    isw_map = hp.read_map('isw.fits') #for moto
    mask = hp.read_map('mask_isw_threshold.fits') #for moto
    alm = hp.map2alm(isw_map, lmax=ellmax)
    wlm = hp.map2alm(mask, lmax=ellmax)
    Cl = hp.alm2cl(alm)
    Ml = hp.alm2cl(wlm)
    print('calling Bispectrum()', flush=True)
    b_ideal = Bispectrum(alm, Cl, wlm, Ml, ellmax, Nside, Nl, dl, min_l)
    pickle.dump(b_ideal, open(f'bispectrum_isw_maskisw0p7_ellmax{ellmax}_{Nl}bins.p', 'wb'))
    print(f'saved bispectrum_isw_maskisw0p7_ellmax{ellmax}_{Nl}bins.p', flush=True)
