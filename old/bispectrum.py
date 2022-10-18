import numpy as np
import healpy as hp
import pickle
from scipy.interpolate import InterpolatedUnivariateSpline
import time
start_time = time.time()


def Bispectrum(alm1, Cl1, alm2, Cl2, alm3, Cl3, lmax, Nside, Nl, dl, min_l):

    print("binned lmax: %d"%(min_l+dl*Nl), flush=True)
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
    # Zero out ell = 0 and ell = 1 if min_l==2
    Cl1_lm[l_arr<min_l] = 0.
    Cl2_lm[l_arr<min_l] = 0.
    Cl3_lm[l_arr<min_l] = 0.

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
    I_map1 = [to_map(ell_bins[bin1]*safe_divide(alm1,Cl1_lm)) for bin1 in range(Nl)]
    I_map2 = [to_map(ell_bins[bin1]*safe_divide(alm2,Cl2_lm)) for bin1 in range(Nl)]
    I_map3 = [to_map(ell_bins[bin1]*safe_divide(alm3, Cl3_lm)) for bin1 in range(Nl)]
    print('computed I maps', flush=True)

    def check_bin(bin1,bin2,bin3):
        """Return one if modes in the bin satisfy the even-parity triangle conditions, or zero else.
        
        This is used either for all triangles in the bin, or just the center of the bin.
        """

        l1 = min_l+(bin1+0.5)*dl
        l2 = min_l+(bin2+0.5)*dl
        l3 = min_l+(bin3+0.5)*dl
        if l3<abs(l1-l2) or l3>l1+l2:
            return 0
        elif l2<abs(l1-l3) or l2>l1+l3:
            return 0
        elif l1<abs(l2-l3) or l1>l2+l3:
            return 0
        else:
            return 1

    # Combine to find numerator
    # b_num_ideal = []
    b_num_ideal = np.zeros((Nl,Nl,Nl))
    sym_factor = []
    for bin1 in range(Nl):
        for bin2 in range(Nl):
            for bin3 in range(Nl):
                # skip bins outside the triangle conditions
                if not check_bin(bin1,bin2,bin3): continue
                    
                # compute symmetry factor
                if bin1==bin2 and bin2==bin3:
                    sym = 6
                elif bin1==bin2 or bin2==bin3 or bin1==bin3:
                    sym = 2
                else:
                    sym = 1
                sym_factor.append(sym)
                    
                # compute numerators
                # b_num_ideal.append(A_pix*np.sum(I_map1[bin1]*I_map2[bin2]*I_map3[bin3])/sym)
                b_num_ideal[bin1,bin2,bin3] = (A_pix*np.sum(I_map1[bin1]*I_map2[bin2]*I_map3[bin3])/sym)        
                
    # b_num_ideal = np.asarray(b_num_ideal)
                
    # number of bins
    N_b = len(b_num_ideal)
    print("%d bispectrum bins"%N_b, flush=True)

    # Load pre-computed 3j symbols
    assert lmax<=1000, "Higher-l 3j symbols not yet precomputed!"
    # tj_arr = pickle.load(open('/global/homes/k/kmsurrao/NILC-Parameter-Pipeline/wigner3j_ellmax1000.p', 'rb')) #for cori
    tj_arr = pickle.load(open('/moto/hill/users/kms2320/wigner3j_ellmax1000.p', 'rb')) #for moto

    # C and M vectors
    Cl1_vec = np.array([(li>=2)*(Cl1_interp(li)) for li in l])
    Cl2_vec = np.array([(li>=2)*(Cl2_interp(li)) for li in l])
    Cl3_vec = np.array([(li>=2)*(Cl3_interp(li)) for li in l])
    Cl1_vec[Cl1_vec==0]=np.inf
    Cl2_vec[Cl2_vec==0]=np.inf
    Cl3_vec[Cl3_vec==0]=np.inf
    print('got Cl_vec and Ml_vec', flush=True)


    # compute denominator
    # b_denom = []
    b_denom = np.ones((Nl,Nl,Nl))
    count = -1
    for bin1 in range(Nl):
        for bin2 in range(Nl):
            for bin3 in range(Nl):
                # skip bins outside the triangle conditions
                if not check_bin(bin1,bin2,bin3): continue
                count += 1
                    
                if (bin1%5==0): print("Computing bin %d of %d"%(bin1,Nl), flush=True)
                value = 0.
                
                # Now iterate over l bins
                for l1 in range(min_l+bin1*dl,min_l+(bin1+1)*dl):
                    for l2 in range(min_l+bin2*dl,min_l+(bin2+1)*dl):
                        for l3 in range(min_l+bin3*dl,min_l+(bin3+1)*dl):
                            if (-1)**(l1+l2+l3)==-1: continue # 3j = 0 here
                            if l3<abs(l1-l2) or l3>l1+l2: continue
                            if l2<abs(l1-l3) or l2>l1+l3: continue
                            if l1<abs(l2-l3) or l1>l2+l3: continue
                            tj = tj_arr[l1,l2,l3]
                            value += tj**2*(2.*l1+1.)*(2.*l2+1.)*(2.*l3+1.)/(4.*np.pi)/Cl1_vec[l1]/Cl2_vec[l2]/Cl3_vec[l3]/sym_factor[count]
                b_denom[bin1,bin2,bin3] = value

    b_ideal = b_num_ideal/b_denom
    print('b_ideal: ', b_ideal, flush=True)
    print('b_ideal.shape: ', b_ideal.shape, flush=True)
    return b_ideal

if __name__=="__main__":
    ellmax = 50
    Nside = 32
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
    print('calling Bispectrum() term 4', flush=True)
    b_ideal = Bispectrum(alm, Cl, np.conj(alm), Cl, wlm, Ml, ellmax, Nside, Nl, dl, min_l)
    print("--- %s seconds ---" % (time.time() - start_time), flush=True)
    # pickle.dump(b_ideal, open(f'bispectrum_isw_maskisw0p7_ellmax{ellmax}_{Nl}bins_notvectorized.p', 'wb'))
    # print(f'saved bispectrum_isw_maskisw0p7_ellmax{ellmax}_{Nl}bins_notvectorized.p', flush=True)

