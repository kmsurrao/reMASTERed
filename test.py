import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import h5py
import pymaster as nmt
import pickle
print('imports complete', flush=True)

scratch_path = '/global/cscratch1/sd/kmsurrao/Correlated-Mask-Power-Spectrum/'
nside = 1024
ellmax = 1000

#load maps
isw_map = hp.read_map(f'{scratch_path}/maps/isw.fits')
tsz_map = hp.read_map(f'{scratch_path}/maps/tsz_8192.fits')
isw_map = hp.ud_grade(isw_map, nside)
tsz_map = hp.ud_grade(tsz_map, nside)
isw_cl = hp.anafast(isw_map, lmax=ellmax)
print('loaded maps', flush=True)

#read catalog file
f = h5py.File(f'{scratch_path}/maps/catalog_153.0.h5', 'r')
print(f.keys(), flush=True)
flux = f['flux'][()]
phi = f['phi'][()]
polarized_flux = f['polarized flux'][()]
theta = f['theta'][()]
print('read catalog', flush=True)

def mask_above_fluxcut(fluxCut, nside, flux, theta, phi):
    #initialize mask of all ones
    m = np.ones(hp.nside2npix(nside))
    #find where flux > fluxCut
    tmp = flux > fluxCut
    #find theta and phi values of point sources where flux > fluxCut
    theta_new = theta[tmp]
    phi_new = phi[tmp]
    for i in range(len(theta_new)):
        vec = hp.ang2vec(theta_new[i], phi_new[i])
        radius = 4.*(1./60.)*(np.pi/180.) #5.*(1./60.)*(np.pi/180.)
        ipix = hp.query_disc(nside, vec, radius)
        m[ipix] = 0
    print('zeroed out point sources', flush=True)
    plt.clf()
    hp.mollview(m)
    plt.savefig('mask_not_apodized.png')
    print('saved mask_not_apodized.png', flush=True)
    #to do: apodize mask so that it goes smoothly from 0 to 1
    # aposcale = 2.5 # Apodization scale in degrees, original
    aposcale = 6.*(1./60.) # Apodization scale in degrees, based on hole radius
    m = nmt.mask_apodization(m, aposcale, apotype="C1")
    print('apodized mask', flush=True)
    plt.clf()
    hp.mollview(m)
    plt.savefig('apodized_mask.png')
    print('saved apodized_mask.png', flush=True)
    return m

#get masks at different flux cuts
# fluxCut = 7*10**(-3) #7*10**(-3)
# mask_7mJy = mask_above_fluxcut(fluxCut, nside, flux, theta, phi)
# isw_masked_7mJy = mask_7mJy*isw_map
# isw_masked_7mJy_cl = hp.anafast(isw_masked_7mJy, lmax=ellmax)
# print('done 7mJy flux cut', flush=True)

mask_1mJy = mask_above_fluxcut(0.01*10**(-3), nside, flux, theta, phi)
isw_masked_1mJy = mask_1mJy*isw_map
isw_masked_1mJy_cl = hp.anafast(isw_masked_1mJy, lmax=ellmax)
print('done 1mJy flux cut', flush=True)

# mask_20mJy = mask_above_fluxcut(20*10**(-3), nside, flux, theta, phi)
# isw_masked_20mJy = mask_20mJy*isw_map
# isw_masked_20mJy_cl = hp.anafast(isw_masked_20mJy, lmax=ellmax)
# print('done 20mJy flux cut', flush=True)

# mask_200mJy = mask_above_fluxcut(200*10**(-3), nside, flux, theta, phi)
# isw_masked_200mJy = mask_200mJy*isw_map
# isw_masked_200mJy_cl = hp.anafast(isw_masked_200mJy, lmax=ellmax)
# print('done 200mJy flux cut', flush=True)

# mask_2000mJy = mask_above_fluxcut(2000*10**(-3), nside, flux, theta, phi)
# isw_masked_2000mJy = mask_2000mJy*isw_map
# isw_masked_2000mJy_cl = hp.anafast(isw_masked_2000mJy, lmax=ellmax)
# print('done 2000mJy flux cut', flush=True)

ells = np.arange(ellmax+1)
plt.clf()
plt.plot(ells[0:], (ells*(ells+1)*isw_cl/(2*np.pi))[0:], label='Unmasked ISW')
plt.plot(ells[0:], (ells*(ells+1)*isw_masked_1mJy_cl/(2*np.pi))[0:], label='Masked ISW fluxCut 0.01mJy')
# plt.plot(ells[0:], (ells*(ells+1)*isw_masked_20mJy_cl/(2*np.pi))[0:], label='Masked ISW fluxCut 20mJy')
# plt.plot(ells[0:], (ells*(ells+1)*isw_masked_200mJy_cl/(2*np.pi))[0:], label='Masked ISW fluxCut 200mJy')
# plt.plot(ells[0:], (ells*(ells+1)*isw_masked_2000mJy_cl/(2*np.pi))[0:], label='Masked ISW fluxCut 2000mJy')
# plt.plot(ells[0:], (ells*(ells+1)*isw_masked_7mJy_cl/(2*np.pi))[0:], label='Masked ISW fluxCut 7mJy')
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.grid()
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\frac{\ell(\ell+1)C_{\ell}^{\rm{ISW}, \rm{ISW}}}{2\pi}$')
plt.savefig('masked_spectrum_diff_fluxcuts.png')
print('saved fig masked_spectrum_diff_fluxcuts.png', flush=True)


#############################################################################
#############################################################################
#############################################################################

#start comparison with analytic equation
comp_map = isw_map
comp_cl = isw_cl
mask = mask_1mJy
masked_map = mask_1mJy
masked_map_cl = isw_masked_1mJy_cl

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
plt.plot(ells[2:], abs(W[2:]), label='abs(W) (component, mask cross spectrum)')
plt.plot(ells[2:], M[2:], label='M (mask auto spectrum)')
plt.plot(ells[2:], comp_cl[2:], label='C (component auto spectrum)')
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

#make comparison plot of masked_map_cl and master_cl
ells = np.arange(ellmax+1)
plt.clf()
plt.plot(ells[2:], (ells*(ells+1)*term1/(2*np.pi))[2:], label='term1 contribution (original MASTER equation)')
plt.plot(ells[2:], (ells*(ells+1)*term2/(2*np.pi))[2:], label='term2 contribution', linestyle='dotted')
plt.plot(ells[2:], (ells*(ells+1)*term3/(2*np.pi))[2:], label='term3 contribution')
plt.plot(ells[2:], (ells*(ells+1)*masked_map_cl/(2*np.pi))[2:], label='LHS of MASTER equation')
plt.plot(ells[2:], (ells*(ells+1)*master_cl/(2*np.pi))[2:], label='RHS of MASTER equation with new terms', linestyle='dotted')
plt.legend()
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\frac{\ell(\ell+1)C_{\ell}^{TT}}{2\pi}$ [$\mathrm{K}^2$]')
# plt.yscale('log')
plt.xscale('log')
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
plt.plot(ells[2:], (term1/masked_map_cl)[2:], label='orig RHS/ LHS')
plt.plot(ells[2:], (master_cl/masked_map_cl)[2:], label='new RHS/ LHS')
plt.legend()
plt.xscale('log')
plt.xlabel(r'$\ell$')
plt.ylabel('Ratio')
plt.savefig('ratios.png')
print('saved fig ratios.png', flush=True)


#plot correlation of component map with mask
corr = W/np.sqrt(comp_cl*M)
plt.clf()
plt.plot(ells[2:], corr[2:])
plt.xlabel(r'$\ell$')
plt.ylabel(r'$r_{\ell}$')
plt.grid()
plt.savefig('component_mask_corr_coeff.png')
print('saved fig component_mask_corr_coeff.png')
print('corr component map and mask: ', corr, flush=True)

#plot correlation of tSZ and mask
tsz_corr = hp.anafast(tsz_map, mask, lmax=ellmax)/np.sqrt(M*hp.anafast(tsz_map, lmax=ellmax))
plt.clf()
plt.plot(ells[2:], tsz_corr[2:])
plt.xlabel(r'$\ell$')
plt.ylabel(r'$r_{\ell}$')
plt.grid()
plt.savefig('tsz_mask_corr_coeff.png')
print('saved fig tsz_mask_corr_coeff.png')
print('corr tsz map and mask: ', tsz_corr, flush=True)
