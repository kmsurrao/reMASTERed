import numpy as np
import healpy as hp
import pickle
from scipy.interpolate import InterpolatedUnivariateSpline


#need to input alm, Cl, wlm, Ml, lmax
def Bispectrum(alm, Cl, wlm, Ml, lmax, Nside):

    # Binning parameters
    dl = 10 # bin width
    Nl = 100 # number of bins
    min_l = 0 # minimum l
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

    def check_bin(bin1,bin2,bin3):
        """Return one if modes in the bin satisfy the even-parity triangle conditions, or zero else.
        
        This is used either for all triangles in the bin, or just the center of the bin.
        """

        l1 = min_l+(bin1+0.5)*dl
        l2 = min_l+(bin2+0.5)*dl
        l3 = min_l+(bin3+0.5)*dl
        if l3<abs(l1-l2) or l3>l1+l2:
            return 0
        else:
            return 1

    # Combine to find numerator
    # b_num_ideal = []
    b_num_ideal = np.zeros((Nl,Nl,Nl))
    sym_factor = []
    for bin1 in range(Nl):
        for bin2 in range(bin1,Nl):
            for bin3 in range(Nl):
                # skip bins outside the triangle conditions
                if not check_bin(bin1,bin2,bin3): continue
                    
                # compute symmetry factor
                if bin1==bin2 and bin2==bin3:
                    sym = 6
                elif bin1==bin2 or bin2==bin3:
                    sym = 2
                else:
                    sym = 1
                sym_factor.append(sym)
                    
                # compute numerators
                # b_num_ideal.append(A_pix*np.sum(I_map1[bin1]*I_map2[bin2]*I_map3[bin3])/sym)
                b_num_ideal[bin1,bin2,bin3] = (A_pix*np.sum(I_map1[bin1]*I_map2[bin2]*I_map3[bin3])/sym) 
                b_num_ideal[bin2,bin1,bin3] = b_num_ideal[bin1,bin2,bin3]        
                
    # b_num_ideal = np.asarray(b_num_ideal)
                
    # number of bins
    N_b = len(b_num_ideal)
    print("%d bispectrum bins"%N_b, flush=True)

    # Load pre-computed 3j symbols
    assert lmax<=1000, "Higher-l 3j symbols not yet precomputed!"
    tj_arr = pickle.load(open('/global/homes/k/kmsurrao/NILC-Parameter-Pipeline/wigner3j_ellmax1000.p', 'rb'))

    # C and M vectors 
    Cl_vec = [(li>=2)*(Cl_interp(li)) for li in l]
    Ml_vec = [(li>=2)*(Ml_interp(li)) for li in l]


    # compute denominator
    # b_denom = []
    b_denom = np.ones((Nl,Nl,Nl))
    for bin1 in range(Nl):
        for bin2 in range(bin1,Nl):
            for bin3 in range(Nl):
                # skip bins outside the triangle conditions
                if not check_bin(bin1,bin2,bin3): continue
                    
                if (bin1%5==0): print("Computing bin %d of %d"%(bin1,Nl), flush=True)
                value = 0.
                
                # Now iterate over l bins
                for l1 in range(min_l+bin1*dl,min_l+(bin1+1)*dl):
                    for l2 in range(min_l+bin2*dl,min_l+(bin2+1)*dl):
                        for l3 in range(min_l+bin3*dl,min_l+(bin3+1)*dl):
                            if (-1)**(l1+l2+l3)==-1: continue # 3j = 0 here
                            if l3<abs(l1-l2) or l3>l1+l2: continue
                            # if l1>ellmax or l2>ellmax or l3>ellmax: continue
                            tj = tj_arr[l1,l2,l3]
                            # print('len(Ml_vec): ', len(Ml_vec), flush=True)
                            # print(l3)
                            value += tj**2*(2.*l1+1.)*(2.*l2+1.)*(2.*l3+1.)/(4.*np.pi)/Cl_vec[l1]/Cl_vec[l2]/Ml_vec[l3]/sym_factor[len(b_denom)]
                # b_denom.append(value)
                b_denom[bin1,bin2,bin3] = value
                b_denom[bin2,bin1,bin3] = value
    # b_denom = np.asarray(b_denom)

    b_ideal = b_num_ideal/b_denom
    print('b_ideal: ', b_ideal, flush=True)
    print('b_ideal.shape: ', b_ideal.shape, flush=True)
    return b_ideal

if __name__=="__main__":
    ellmax = 1000
    Nside = 512
    isw_map = hp.read_map('/global/cscratch1/sd/kmsurrao/Correlated-Mask-Power-Spectrum/maps/isw.fits')
    mask = hp.read_map('/global/homes/k/kmsurrao/Correlated-Mask-Power-Spectrum/mask_isw_threshold.fits')
    alm = hp.map2alm(isw_map, lmax=ellmax)
    wlm = hp.map2alm(mask, lmax=ellmax)
    Cl = hp.alm2cl(alm)
    Ml = hp.alm2cl(wlm)
    print('calling Bispectrum()', flush=True)
    b_ideal = Bispectrum(alm, Cl, wlm, Ml, ellmax, Nside)
    pickle.dump(b_ideal, open(f'bispectrum_isw_maskisw0p7_ellmax{ellmax}.p', 'wb'))
