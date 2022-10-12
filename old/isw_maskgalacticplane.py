import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import h5py
import pymaster as nmt
import pickle
print('imports complete', flush=True)

scratch_path = '/global/cscratch1/sd/kmsurrao/Correlated-Mask-Power-Spectrum/'
nside = 4096 #original nside of isw.fits is 4096
ellmax = 1000
nside_for_masking = 32

#load maps
isw_map = hp.read_map(f'{scratch_path}/maps/isw.fits')
isw_map = hp.ud_grade(isw_map, nside)
# isw_map = hp.remove_monopole(isw_map) #todo, remove?
isw_cl = hp.anafast(isw_map, lmax=ellmax)
print('loaded maps', flush=True)
plt.clf()
hp.mollview(isw_map)
plt.savefig('isw_map.png')
print('saved isw_map.png', flush=True)


mask = hp.read_map(f'{scratch_path}/wmap/wmap_temperature_analysis_mask_r9_7yr_v4.fits')
mask = hp.ud_grade(mask, nside)
mask[mask!=0]=1.
plt.clf()
hp.mollview(mask)
plt.savefig('mask_not_apodized.png')
print('saved mask_not_apodized.png', flush=True)
aposcale = 2.5 # Apodization scale in degrees, original
aposcale = 1.
mask = nmt.mask_apodization(mask, aposcale, apotype="C1")
print('apodized mask', flush=True)
plt.clf()
hp.mollview(mask)
plt.savefig('apodized_mask.png')
isw_masked = mask*isw_map
isw_masked_cl = hp.anafast(isw_masked, lmax=ellmax)
print('done cut', flush=True)


ells = np.arange(ellmax+1)
plt.clf()
plt.plot(ells[0:], (ells*(ells+1)*isw_cl/(2*np.pi))[0:], label='Unmasked ISW')
plt.plot(ells[0:], (ells*(ells+1)*isw_masked_cl/(2*np.pi))[0:], label='Masked ISW cut ')
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.grid()
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\frac{\ell(\ell+1)C_{\ell}^{\rm{ISW}, \rm{ISW}}}{2\pi}$')
plt.savefig('masked_spectrum_diff_cuts.png')
print('saved fig masked_spectrum_diff_cuts.png', flush=True)


#############################################################################
#############################################################################
#############################################################################

#start comparison with analytic equation
comp_map = isw_map
comp_cl = isw_cl
masked_map = isw_masked
masked_map_cl = isw_masked_cl
# mask = -isw_map #remove
# masked_map = mask*isw_map #remove
# masked_map_cl = hp.anafast(masked_map, lmax=ellmax) #remove


#load wigner3j symbols
wigner_file = '/global/homes/k/kmsurrao/NILC-Parameter-Pipeline/wigner3j_ellmax1000.p'
wigner_nonzero_m_file = '/global/homes/k/kmsurrao/NILC-Parameter-Pipeline/wigner3j_nonzero_m_ellmax1000.p'
wigner_zero_m = pickle.load(open(wigner_file, 'rb'))[:ellmax+1, :ellmax+1, :ellmax+1]
wigner_nonzero_m = pickle.load(open(wigner_nonzero_m_file, 'rb'))[:ellmax+1, :ellmax+1, :2*ellmax+1]

#get <wa> W spectra
W  = hp.anafast(comp_map, mask, lmax=ellmax)
M = hp.anafast(mask, lmax=ellmax)
print('W: ', W, flush=True)
print('M: ', M, flush=True)
print('C: ', comp_cl, flush=True)
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
plt.savefig('power_spectra.png')



#calculate spectra of masked map from MASTER approach
l2 = np.arange(ellmax+1)
l3 = np.arange(ellmax+1)
m_array = np.ones(2*ellmax+1) #acount for (-1)^{m_2+m_3} factor in term3
zero_idx = ellmax
for i in range(2*ellmax+1):
    if abs(i-zero_idx)%2==1:
        m_array[i] = -1
m_array = np.array(m_array)
term1 = float(1/(4*np.pi))*np.einsum('a,b,lab,lab,a,b->l',2*l2+1,2*l3+1,wigner_zero_m,wigner_zero_m,comp_cl,M,optimize=True)
term2 = float(1/(4*np.pi))*np.einsum('a,b,lab,lab,a,b->l',2*l2+1,2*l3+1,wigner_zero_m,wigner_zero_m,W,W,optimize=True)
term3 = float(1/(4*np.pi))*np.einsum('a,b,laa,lbb,lac,lbd,a,b,c,d->l',2*l2+1,2*l3+1,wigner_zero_m,wigner_zero_m,wigner_nonzero_m,wigner_nonzero_m,W,W,m_array,m_array,optimize=True)
master_cl = term1 + term2 + term3

# #calculate spectra of masked map from MASTER approach not summing over monopole and dipole
# l2 = np.arange(2,ellmax+1)
# l3 = np.arange(2,ellmax+1)
# # m_array = np.ones(2*ellmax+1) #acount for (-1)^{m_2+m_3} factor in term3
# # zero_idx = ellmax
# # for i in range(2*ellmax+1):
# #     if abs(i-zero_idx)%2==1:
# #         m_array[i] = -1
# # m_array = np.array(m_array)
# wigner_zero_m = wigner_zero_m[:,2:,2:]
# comp_cl = 
# term1 = float(1/(4*np.pi))*np.einsum('a,b,lab,lab,a,b->l',2*l2+1,2*l3+1,wigner_zero_m,wigner_zero_m,comp_cl,M,optimize=True)
# term2 = float(1/(4*np.pi))*np.einsum('a,b,lab,lab,a,b->l',2*l2+1,2*l3+1,wigner_zero_m,wigner_zero_m,W,W,optimize=True)
# # term3 = float(1/(4*np.pi))*np.einsum('a,b,laa,lbb,lac,lbd,a,b,c,d->l',2*l2+1,2*l3+1,wigner_zero_m,wigner_zero_m,wigner_nonzero_m,wigner_nonzero_m,W,W,m_array,m_array,optimize=True)
# master_cl = term1 + term2 

#make comparison plot of masked_map_cl and master_cl
ells = np.arange(ellmax+1)
plt.clf()
plt.plot(ells[2:], (ells*(ells+1)*term1/(2*np.pi))[2:], label='term1 contribution (original MASTER equation)')
plt.plot(ells[2:], (ells*(ells+1)*term2/(2*np.pi))[2:], label='term2 contribution', linestyle='dotted')
plt.plot(ells[2:], (ells*(ells+1)*masked_map_cl/(2*np.pi))[2:], label='LHS of MASTER equation')
plt.plot(ells[2:], (ells*(ells+1)*master_cl/(2*np.pi))[2:], label='RHS of MASTER equation with new terms', linestyle='dotted')
plt.legend()
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\frac{\ell(\ell+1)C_{\ell}^{TT}}{2\pi}$ [$\mathrm{K}^2$]')
# plt.yscale('log')
plt.xscale('log')
plt.grid()
plt.savefig(f'isw_corr_mask.png')
print(f'saved fig isw_corr_mask.png')
plt.close('all')
print(masked_map_cl/master_cl)
print('masked_map_cl: ', masked_map_cl)
print('master_cl: ', master_cl)
print('comp_cl: ', comp_cl)
print('term1: ', term1)
print('term2: ', term2)
print('term3: ', term3)


#plot ratios
plt.plot(ells[0:], (term1/masked_map_cl)[0:], label='orig RHS/ LHS')
plt.plot(ells[0:], (master_cl/masked_map_cl)[0:], label='new RHS/ LHS', linestyle='dotted')
plt.legend()
# plt.xscale('log')
plt.xlabel(r'$\ell$')
plt.ylabel('Ratio')
plt.grid()
plt.savefig('ratios.png')
print('saved fig ratios.png', flush=True)


#plot correlation of component map with mask
corr = W/np.sqrt(comp_cl*M)
plt.clf()
plt.plot(ells[0:], corr[0:])
plt.xlabel(r'$\ell$')
plt.ylabel(r'$r_{\ell}$')
plt.grid()
plt.xscale('log')
plt.savefig('component_mask_corr_coeff.png')
print('saved fig component_mask_corr_coeff.png')
print('corr component map and mask: ', corr, flush=True)
