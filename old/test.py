import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import h5py
import pymaster as nmt
print('imports complete', flush=True)

scratch_path = ''
nside = 1024
ellmax = 1500

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
        radius = 5.*(1./60.)*(np.pi/180.)
        ipix = hp.query_disc(nside, vec, radius)
        m[ipix] = 0
    print('zeroed out point sources', flush=True)
    #to do: apodize mask so that it goes smoothly from 0 to 1
    # aposcale = 2.5 # Apodization scale in degrees, original
    aposcale = 50.*(1./60.) # Apodization scale in degrees, based on hole radius
    m = nmt.mask_apodization(m, aposcale, apotype="Smooth")
    print('apodized mask', flush=True)
    return m

#get masks at different flux cuts
fluxCut = 7*10**(-3)
mask_7mJy = mask_above_fluxcut(fluxCut, nside, flux, theta, phi)
isw_masked_7mJy = mask_7mJy*isw_map
isw_masked_7mJy_cl = hp.anafast(isw_masked_7mJy, lmax=ellmax)
print('done 7mJy flux cut', flush=True)

# mask_2mJy = mask_above_fluxcut(2*10**(-3), nside, flux, theta, phi)
# isw_masked_2mJy = mask_2mJy*isw_map
# isw_masked_2mJy_cl = hp.anafast(isw_masked_2mJy, lmax=ellmax)
# print('done 2mJy flux cut', flush=True)

# mask_20mJy = mask_above_fluxcut(20*10**(-3), nside, flux, theta, phi)
# isw_masked_20mJy = mask_20mJy*isw_map
# isw_masked_20mJy_cl = hp.anafast(isw_masked_20mJy, lmax=ellmax)
# print('done 20mJy flux cut', flush=True)

# mask_200mJy = mask_above_fluxcut(200*10**(-3), nside, flux, theta, phi)
# isw_masked_200mJy = mask_200mJy*isw_map
# isw_masked_200mJy_cl = hp.anafast(isw_masked_200mJy, lmax=ellmax)
# print('done 200mJy flux cut', flush=True)

mask_2000mJy = mask_above_fluxcut(2000*10**(-3), nside, flux, theta, phi)
isw_masked_2000mJy = mask_2000mJy*isw_map
isw_masked_2000mJy_cl = hp.anafast(isw_masked_2000mJy, lmax=ellmax)
print('done 2000mJy flux cut', flush=True)

ells = np.arange(ellmax+1)
# plt.plot(ells[0:], (ells*(ells+1)*isw_masked_2mJy_cl/(2*np.pi))[0:], label='Masked ISW fluxCut 2mJy')
plt.plot(ells[0:], (ells*(ells+1)*isw_masked_7mJy_cl/(2*np.pi))[0:], label='Masked ISW fluxCut 7mJy')
# plt.plot(ells[0:], (ells*(ells+1)*isw_masked_20mJy_cl/(2*np.pi))[0:], label='Masked ISW fluxCut 20mJy')
# plt.plot(ells[0:], (ells*(ells+1)*isw_masked_200mJy_cl/(2*np.pi))[0:], label='Masked ISW fluxCut 200mJy')
plt.plot(ells[0:], (ells*(ells+1)*isw_masked_2000mJy_cl/(2*np.pi))[0:], label='Masked ISW fluxCut 2000mJy')
plt.plot(ells[0:], (ells*(ells+1)*isw_cl/(2*np.pi))[0:], label='Unmasked ISW')
plt.legend()
plt.yscale('log')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\frac{\ell(\ell+1)C_{\ell}^{\rm{ISW}, \rm{ISW}}}{2\pi}$')
plt.savefig('masked_spectrum_diff_fluxcuts.png')
print('saved fig masked_spectrum_diff_fluxcuts.png', flush=True)