import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import h5py
import pymaster as nmt
import pickle
from scipy.interpolate import griddata
print('imports complete', flush=True)

scratch_path = '/global/cscratch1/sd/kmsurrao/Correlated-Mask-Power-Spectrum/'
nside = 1024 #original nside of isw.fits is 4096
ellmax = 499
dl = 10
Nl = int((ellmax+1)/dl)
nside_for_masking = 1024
ells = np.arange(ellmax+1)

#load binned bispectrum
bispectrum = pickle.load(open(f'linear_interpolated_bispectrum_ellmax{ellmax+1}_{Nl}origbins.p', 'rb'))[:ellmax+1, :ellmax+1, :ellmax+1]

#load maps
isw_map = hp.read_map(f'{scratch_path}/maps/isw.fits') #for cori
# isw_map = hp.read_map('isw.fits') #for moto
isw_map = hp.ud_grade(isw_map, nside)
# isw_map = hp.remove_monopole(isw_map) #todo, remove?
isw_cl = hp.anafast(isw_map, lmax=ellmax)
print('loaded maps', flush=True)
plt.clf()
hp.mollview(isw_map)
plt.savefig('images/isw_map.png')
print('saved images/isw_map.png', flush=True)


#load wigner3j symbols and bispectrum
wigner_file = '/global/homes/k/kmsurrao/NILC-Parameter-Pipeline/wigner3j_ellmax1000.p' #for cori
# wigner_file = '/moto/hill/users/kms2320/wigner3j_ellmax1000.p' #for moto
wigner_zero_m = pickle.load(open(wigner_file, 'rb'))[:ellmax+1, :ellmax+1, :ellmax+1]




def mask_above_cut(cut_high, cut_low, map_, nside, nside_for_masking, save_file=False):
    #downgrade resolution of map for create mask initially
    map_ = hp.ud_grade(map_, nside_for_masking)
    #set mask to 0 below cut_low or above cut_high, and 1 elsewhere
    m = np.where(np.logical_or(map_>cut_high, map_<cut_low), 0, 1)
    #return mask to nside of original map
    m = hp.ud_grade(m, nside)
    print('zeroed out pixels above cut', flush=True)
    plt.clf()
    hp.mollview(m)
    plt.savefig('images/mask_not_apodized.png')
    print('saved images/mask_not_apodized.png', flush=True)
    #to do: apodize mask so that it goes smoothly from 0 to 1
    aposcale = 2.5 # Apodization scale in degrees, original
    # aposcale = 1.
    # aposcale = 5.*(1./60.) # Apodization scale in degrees, based on hole radius
    m = nmt.mask_apodization(m, aposcale, apotype="C1")
    print('apodized mask', flush=True)
    plt.clf()
    hp.mollview(m)
    plt.savefig('images/apodized_mask.png')
    print('saved images/apodized_mask.png', flush=True)
    if save_file:
        hp.write_map('mask_isw_threshold.fits', m, overwrite=True)
        print('wrote mask_isw_threshold.fits', flush=True)
    return m

mean = np.mean(isw_map)
std_dev = np.std(isw_map)
cut_high = mean + 0.7*std_dev
cut_low = mean - 0.7*std_dev
print('cut_high: ', cut_high, flush=True)
print('cut_low: ', cut_low, flush=True)
print('max: ', np.max(isw_map), flush=True)
print('min: ', np.min(isw_map), flush=True)
print('means: ', mean, flush=True)
print('std dev: ', std_dev, flush=True)
mask = mask_above_cut(cut_high, cut_low, isw_map, nside, nside_for_masking, save_file=True)
# mask = isw_map #remove, use for testing perfectly correlated mask
isw_masked = mask*isw_map
isw_masked_cl = hp.anafast(isw_masked, lmax=ellmax)
print('done cut', flush=True)


plt.clf()
plt.plot(ells[0:], (ells*(ells+1)*isw_cl/(2*np.pi))[0:], label='Unmasked ISW')
plt.plot(ells[0:], (ells*(ells+1)*isw_masked_cl/(2*np.pi))[0:], label='Masked ISW cut ')
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.grid()
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\frac{\ell(\ell+1)C_{\ell}^{\rm{ISW}, \rm{ISW}}}{2\pi}$')
plt.savefig('images/masked_spectrum_diff_cuts.png')
print('saved fig images/masked_spectrum_diff_cuts.png', flush=True)


#############################################################################
#############################################################################
#############################################################################

#start comparison with analytic equation
comp_map = isw_map
comp_cl = isw_cl
masked_map = isw_masked
masked_map_cl = isw_masked_cl


#get <wa> W spectra
comp_map_alm = hp.map2alm(comp_map)
mask_alm = hp.map2alm(mask)
W  = hp.alm2cl(comp_map_alm, mask_alm, lmax=ellmax)
M = hp.alm2cl(mask_alm, lmax=ellmax)
print('W: ', W[50:70], flush=True)
print('M: ', M[50:70], flush=True)
print('C: ', comp_cl[50:70], flush=True)
ells = np.arange(ellmax+1)
plt.clf()
plt.plot(ells[0:], abs(W[0:]), label='abs(W) (component, mask cross spectrum)')
plt.plot(ells[0:], M[0:], label='M (mask auto spectrum)')
plt.plot(ells[0:], comp_cl[0:], label='C (component auto spectrum)')
plt.plot(ells[0:], M[0:]*comp_cl[0:], label='M*C')
plt.plot(ells[0:], W[0:]*W[0:], label='W*W')
plt.legend()
plt.yscale('log')
plt.grid()
plt.xlabel(r'$\ell$')
plt.ylabel('Power Spectrum')
plt.savefig('images/power_spectra.png')



#calculate spectra of masked map from MASTER approach
l2 = np.arange(ellmax+1)
l3 = np.arange(ellmax+1)
# m_array = np.zeros((ellmax+1,2*ellmax+1)) #acount for (-1)^{m_2+m_3} factor in term3
# zero_idx = ellmax
# for l in range(ellmax+1):
#     for i in range(zero_idx-l, zero_idx+l+1):
#         if abs(i-zero_idx)%2==1:
#             m_array[l][i] = -1
#         else:
#             m_array[l][i] = 1
term1 = float(1/(4*np.pi))*np.einsum('a,b,lab,lab,a,b->l',2*l2+1,2*l3+1,wigner_zero_m,wigner_zero_m,comp_cl,M,optimize=True)
term2 = float(1/(4*np.pi))*np.einsum('a,b,lab,lab,a,b->l',2*l2+1,2*l3+1,wigner_zero_m,wigner_zero_m,W,W,optimize=True)
print('wigner_zero_m.shape: ', wigner_zero_m.shape, flush=True)
print('bispectrum.shape: ', bispectrum.shape, flush=True)
# term3 = float(2/(4*np.pi)**1.5)*mask_alm[0]*np.einsum('b,lab,lab,lab->l',2*l3+1,wigner_zero_m,wigner_zero_m,bispectrum,optimize=True)
term3 = float(2/(4*np.pi)**1)*mask_alm[0]*np.einsum('a,b,lab,lab,lab->l',2*l2+1,2*l3+1,wigner_zero_m,wigner_zero_m,bispectrum,optimize=True)
master_cl = term1 + term2 + term3


#make comparison plot of masked_map_cl and master_cl
ells = np.arange(ellmax+1)
plt.clf()
plt.plot(ells[2:], (ells*(ells+1)*term1/(2*np.pi))[2:], label='term1 contribution (original MASTER equation)')
plt.plot(ells[2:], (ells*(ells+1)*term2/(2*np.pi))[2:], label='term2 contribution', linestyle='dotted')
plt.plot(ells[2:], (ells*(ells+1)*term3/(2*np.pi))[2:], label='term3 contribution', linestyle='dotted')
plt.plot(ells[2:], (ells*(ells+1)*masked_map_cl/(2*np.pi))[2:], label='LHS of MASTER equation')
plt.plot(ells[2:], (ells*(ells+1)*master_cl/(2*np.pi))[2:], label='RHS of MASTER equation with new terms', linestyle='dotted')
plt.legend()
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\frac{\ell(\ell+1)C_{\ell}^{TT}}{2\pi}$ [$\mathrm{K}^2$]')
# plt.yscale('log')
plt.xscale('log')
plt.grid()
plt.savefig(f'images/isw_corr_mask.png')
print(f'saved fig images/isw_corr_mask.png')
plt.close('all')
print((masked_map_cl/master_cl)[50:70])
print('masked_map_cl: ', masked_map_cl[50:70])
print('master_cl: ', master_cl[50:70])
print('comp_cl: ', comp_cl[50:70])
print('term1: ', term1[50:70])
print('term2: ', term2[50:70])
print('term3: ', term3[50:70])


#plot ratios
plt.plot(ells[0:], (term1/masked_map_cl)[0:], label='orig RHS/ LHS')
plt.plot(ells[0:], (master_cl/masked_map_cl)[0:], label='new RHS/ LHS', linestyle='dotted')
plt.legend()
# plt.xscale('log')
plt.xlabel(r'$\ell$')
plt.ylabel('Ratio')
plt.grid()
plt.savefig('images/ratios.png')
print('saved fig images/ratios.png', flush=True)


#plot correlation of component map with mask
corr = W/np.sqrt(comp_cl*M)
plt.clf()
plt.plot(ells[0:], corr[0:])
plt.xlabel(r'$\ell$')
plt.ylabel(r'$r_{\ell}$')
plt.grid()
plt.xscale('log')
plt.savefig('images/component_mask_corr_coeff.png')
print('saved fig images/component_mask_corr_coeff.png')
print('corr component map and mask: ', corr, flush=True)
