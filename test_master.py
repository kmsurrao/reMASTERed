import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pickle


def compare_master(inp, comp_map, mask, bispectrum_term3, bispectrum_term4):

    ellmax = inp.ellmax-1

    #load wigner3j symbols
    wigner = pickle.load(open(inp.wigner_file, 'rb'))[:ellmax+1, :ellmax+1, :ellmax+1]


    comp_cl = hp.anafast(comp_map, lmax=ellmax)
    masked_map = mask*comp_map
    masked_map_cl = hp.anafast(masked_map, lmax=ellmax)


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
    term1 = float(1/(4*np.pi))*np.einsum('a,b,lab,lab,a,b->l',2*l2+1,2*l3+1,wigner,wigner,comp_cl,M,optimize=True)
    term2 = float(1/(4*np.pi))*np.einsum('a,b,lab,lab,a,b->l',2*l2+1,2*l3+1,wigner,wigner,W,W,optimize=True)
    print('wigner.shape: ', wigner.shape, flush=True)
    print('bispectrum_term3.shape: ', bispectrum_term3.shape, flush=True)
    term3 = float(1/(4*np.pi)**1.5)*mask_alm[0]*np.einsum('b,lab,lab,lab->l',2*l3+1,wigner,wigner,bispectrum_term3,optimize=True)
    term4 = float(1/(4*np.pi)**1.5)*mask_alm[0]*np.einsum('b,lab,lab,lab->l',2*l3+1,wigner,wigner,bispectrum_term4,optimize=True)
    print('mask_alm[0]: ', mask_alm[0], flush=True)
    # term3 = float(1/(4*np.pi)**1.5)*mask_alm[0]*np.einsum('a,b,lab,lab,lab->l',2*l2+1,2*l3+1,wigner,wigner,bispectrum_term3,optimize=True)
    # term4 = float(1/(4*np.pi)**1.5)*mask_alm[0]*np.einsum('a,b,lab,lab,lab->l',2*l2+1,2*l3+1,wigner,wigner,bispectrum_term4,optimize=True)
    master_cl = (term1 + term2 + term3 + term4)


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
