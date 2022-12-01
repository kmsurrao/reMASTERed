import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pickle
import subprocess
import os


def compare_master(inp, master_lhs, wlm_00, alm_00, Cl, Ml, Wl, bispectrum_aaw, bispectrum_waw, trispectrum, env, base_dir=None, plot_log=False):
    start = 0
    if base_dir is None:
        base_dir = f'images/{inp.comp}_cut{inp.cut}_ellmax{inp.ellmax}_nsims{inp.nsims}_nside{inp.nside}_nsideformasking{inp.nside_for_masking}'
    if not os.path.isdir(base_dir):
        subprocess.call(f'mkdir {base_dir}', shell=True, env=env)

    ellmax = inp.ellmax

    #load wigner3j and wigner6j symbols
    wigner3j = pickle.load(open(inp.wigner_file, 'rb'))[:ellmax+1, :ellmax+1, :ellmax+1]
    # wigner6j = pickle.load(open('wigner6j_ellmax20.p', 'rb'))[:ellmax+1, :ellmax+1, :ellmax+1, :ellmax+1, :ellmax+1, :ellmax+1]


    #calculate spectra of masked map from MASTER approach
    l1 = np.arange(ellmax+1)
    l2 = np.arange(ellmax+1)
    l3 = np.arange(ellmax+1)
    l4 = np.arange(ellmax+1)
    l5 = np.arange(ellmax+1)
    L = np.arange(ellmax+1)
    neg_ones_array = np.ones(ellmax+1)
    neg_ones_array[1::2] = -1
    aa_ww_term = float(1/(4*np.pi))*np.einsum('a,b,lab,lab,a,b->l',2*l2+1,2*l3+1,wigner3j,wigner3j,Cl,Ml,optimize=True)
    aw_aw_term = float(1/(4*np.pi))*np.einsum('a,b,lab,lab,a,b->l',2*l2+1,2*l3+1,wigner3j,wigner3j,Wl,Wl,optimize=True)
    w_aaw_term = 2.*float(1/(4*np.pi)**1.5)*wlm_00*np.einsum('a,b,lab,lab,lab->l',2*l2+1,2*l3+1,wigner3j,wigner3j,bispectrum_aaw,optimize=True)
    a_waw_term = 2.*float(1/(4*np.pi)**1.5)*alm_00*np.einsum('a,b,lab,lab,lab->l',2*l2+1,2*l3+1,wigner3j,wigner3j,bispectrum_waw,optimize=True)
    # aaww_term_parta = 16.*float(1/(4*np.pi)**2)*np.einsum('l,L,a,b,c,d,L,calbdL,alc,cdL,dlb,baL,abcdL->l', neg_ones_array,neg_ones_array,2*l2+1,2*l3+1,2*l4+1,2*l5+1,2*L+1,wigner6j,wigner3j,wigner3j,wigner3j,wigner3j,trispectrum,optimize=True)
    # aaww_term_partc = 8.*float(1/(4*np.pi)**2)*np.einsum('a,b,c,d,lac,lac,lbd,lbd,abcdl->l', 2*l2+1,2*l3+1,2*l4+1,2*l5+1,wigner3j,wigner3j,wigner3j,wigner3j,trispectrum, optimize=True)
    # print('aaww_term_parta: ', aaww_term_parta, flush=True)
    # print('aaww_term_partc: ', aaww_term_partc, flush=True)
    # aaww_term = aaww_term_parta + aaww_term_partc
    aaww_term = np.einsum('l,acbdl->l', 1/(2*l1+1), trispectrum) #use when using rho for trispectrum
    master_cl = (aa_ww_term + aw_aw_term + w_aaw_term + a_waw_term + aaww_term)
    without_trispectrum = aa_ww_term + aw_aw_term + w_aaw_term + a_waw_term


    #make comparison plot of masked_map_cl and master_cl
    start = 0
    ells = np.arange(ellmax+1)
    plt.clf()
    plt.plot(ells[start:], (ells*(ells+1)*aa_ww_term/(2*np.pi))[start:], label='<aa><ww> term', color='c')
    plt.plot(ells[start:], (ells*(ells+1)*aw_aw_term/(2*np.pi))[start:], label='<aw><aw> term', linestyle='dotted')
    plt.plot(ells[start:], (ells*(ells+1)*w_aaw_term/(2*np.pi))[start:], label='<w><aaw> term', color='r')
    plt.plot(ells[start:], (ells*(ells+1)*a_waw_term/(2*np.pi))[start:], label='<a><waw> term', color='mediumpurple')
    plt.plot(ells[start:], (ells*(ells+1)*aaww_term/(2*np.pi))[start:], label='<aaww> term', color='y', linestyle='dotted')
    plt.plot(ells[start:], (ells*(ells+1)*master_lhs/(2*np.pi))[start:], label='MASTER LHS', color='g')
    plt.plot(ells[start:], (ells*(ells+1)*master_cl/(2*np.pi))[start:], label='Modified MASTER RHS', linestyle='dotted', color='m')
    plt.plot(ells[start:], (ells*(ells+1)*without_trispectrum/(2*np.pi))[start:], label='Modified RHS w/o trispectrum', linestyle='dotted', color='k')
    plt.legend()
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$\frac{\ell(\ell+1)C_{\ell}^{TT}}{2\pi}$ [$\mathrm{K}^2$]')
    if plot_log:
        plt.yscale('log')
    # plt.xscale('log')
    plt.grid()
    plt.savefig(f'{base_dir}/master.png')
    print(f'saved {base_dir}/master.png')
    plt.close('all')
    print((master_lhs/master_cl)[start:start+10])
    print('master lhs: ', master_lhs[start:start+10])
    print('master_cl: ', master_cl[start:start+10])
    print('Cl: ', Cl[start:start+10])
    print('aa_ww_term: ', aa_ww_term[start:start+10])
    print('aw_aw_term: ', aw_aw_term[start:start+10])
    print('w_aaw_term: ', w_aaw_term[start:start+10])
    print('aaww_term: ', aaww_term[start:start+10])


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
