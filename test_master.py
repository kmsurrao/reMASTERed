import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pickle
import subprocess
import os


def compare_master(inp, master_lhs, wlm_00, Cl, Ml, Wl, bispectrum, env):

    base_dir = f'images/{inp.comp}_cut{inp.cut}_high{inp.cut_high}_low{inp.cut_low}_ellmax{inp.ellmax}_nsims{inp.nsims}_nside{inp.nside}_nsideformasking{inp.nside_for_masking}'
    if not os.path.isdir(base_dir):
        subprocess.call(f'mkdir {base_dir}', shell=True, env=env)

    ellmax = inp.ellmax

    #load wigner3j symbols
    wigner = pickle.load(open(inp.wigner_file, 'rb'))[:ellmax+1, :ellmax+1, :ellmax+1]

    #get <wa> W spectra
    print('W: ', Wl[30:40], flush=True)
    print('M: ', Ml[30:40], flush=True)
    print('C: ', Cl[30:40], flush=True)
    ells = np.arange(ellmax+1)
    plt.clf()
    plt.plot(ells[0:], abs(Wl[0:]), label='abs(W) (component, mask cross spectrum)')
    plt.plot(ells[0:], Ml[0:], label='M (mask auto spectrum)')
    plt.plot(ells[0:], Cl[0:], label='C (component auto spectrum)')
    plt.plot(ells[0:], Ml[0:]*Cl[0:], label='M*C')
    plt.plot(ells[0:], Wl[0:]*Wl[0:], label='W*W')
    plt.legend()
    plt.yscale('log')
    plt.grid()
    plt.xlabel(r'$\ell$')
    plt.ylabel('Power Spectrum')
    plt.savefig(f'{base_dir}/power_spectra.png')



    #calculate spectra of masked map from MASTER approach
    l2 = np.arange(ellmax+1)
    l3 = np.arange(ellmax+1)
    term1 = float(1/(4*np.pi))*np.einsum('a,b,lab,lab,a,b->l',2*l2+1,2*l3+1,wigner,wigner,Cl,Ml,optimize=True)
    term2 = float(1/(4*np.pi))*np.einsum('a,b,lab,lab,a,b->l',2*l2+1,2*l3+1,wigner,wigner,Wl,Wl,optimize=True)
    term3 = 2.*float(1/(4*np.pi)**1.5)*wlm_00*np.einsum('a,b,lab,lab,lab->l',2*l2+1,2*l3+1,wigner,wigner,bispectrum,optimize=True)
    master_cl = (term1 + term2 + term3)


    #make comparison plot of masked_map_cl and master_cl
    ells = np.arange(ellmax+1)
    plt.clf()
    plt.plot(ells[10:], (ells*(ells+1)*term1/(2*np.pi))[10:], label='term1 (original MASTER)', color='c')
    plt.plot(ells[10:], (ells*(ells+1)*term2/(2*np.pi))[10:], label='term2', linestyle='dotted')
    plt.plot(ells[10:], (ells*(ells+1)*term3/(2*np.pi))[10:], label='term3', color='r')
    plt.plot(ells[10:], (ells*(ells+1)*master_lhs/(2*np.pi))[10:], label='MASTER LHS', color='g')
    plt.plot(ells[10:], (ells*(ells+1)*master_cl/(2*np.pi))[10:], label='Modified MASTER RHS', linestyle='dotted', color='m')
    plt.legend()
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$\frac{\ell(\ell+1)C_{\ell}^{TT}}{2\pi}$ [$\mathrm{K}^2$]')
    # plt.yscale('log')
    # plt.xscale('log')
    plt.grid()
    plt.savefig(f'{base_dir}/master.png')
    print(f'saved {base_dir}/master.png')
    plt.close('all')
    print((master_lhs/master_cl)[30:40])
    print('master lhs: ', master_lhs[30:40])
    print('master_cl: ', master_cl[30:40])
    print('Cl: ', Cl[30:40])
    print('term1: ', term1[30:40])
    print('term2: ', term2[30:40])
    print('term3: ', term3[30:40])


    #plot ratios
    plt.plot(ells[10:], (term1/master_lhs)[10:], label='orig RHS/ LHS')
    plt.plot(ells[10:], (master_cl/master_lhs)[10:], label='new RHS/ LHS', linestyle='dotted')
    plt.legend()
    # plt.xscale('log')
    plt.xlabel(r'$\ell$')
    plt.ylabel('Ratio')
    plt.grid()
    plt.savefig(f'{base_dir}/ratios.png')
    print(f'saved {base_dir}/ratios.png', flush=True)


    #plot correlation of component map with mask
    corr = Wl/np.sqrt(Cl*Ml)
    corr = np.convolve(corr, np.ones(10)/10, mode='same')
    plt.clf()
    plt.plot(ells[0:], corr[0:])
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$r_{\ell}$')
    plt.grid()
    # plt.xscale('log')
    plt.savefig(f'{base_dir}/component_mask_corr_coeff.png')
    print(f'saved {base_dir}/component_mask_corr_coeff.png')
    print('corr component map and mask: ', corr, flush=True)
