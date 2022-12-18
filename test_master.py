import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pickle
import subprocess
import os


def compare_master(inp, master_lhs, wlm_00, alm_00, Cl_aa, Cl_ww, Cl_aw, bispectrum_aaw, bispectrum_waw, Rho, env, base_dir=None):
    start = 0
    if base_dir is None:
        base_dir = f'images/{inp.comp}_cut{inp.cut}_ellmax{inp.ellmax}_nsims{inp.nsims}_nside{inp.nside}_nsideformasking{inp.nside_for_masking}'
    if not os.path.isdir(base_dir):
        subprocess.call(f'mkdir {base_dir}', shell=True, env=env)

    ellmax = inp.ellmax

    #load wigner3j symbols
    wigner3j = pickle.load(open(inp.wigner_file, 'rb'))[:ellmax+1, :ellmax+1, :ellmax+1]


    #calculate spectra of masked map from MASTER approach
    l1 = np.arange(ellmax+1)
    l2 = np.arange(ellmax+1)
    l3 = np.arange(ellmax+1)
    l4 = np.arange(ellmax+1)
    l5 = np.arange(ellmax+1)
    L = np.arange(ellmax+1)
    neg_ones_array = np.ones(ellmax+1)
    neg_ones_array[1::2] = -1
    aa_ww_term = float(1/(4*np.pi))*np.einsum('a,b,lab,lab,a,b->l',2*l2+1,2*l3+1,wigner3j,wigner3j,Cl_aa,Cl_ww,optimize=True)
    aw_aw_term = float(1/(4*np.pi))*np.einsum('a,b,lab,lab,a,b->l',2*l2+1,2*l3+1,wigner3j,wigner3j,Cl_aw,Cl_aw,optimize=True)
    w_aaw_term = 2.*float(1/(4*np.pi)**1.5)*wlm_00*np.einsum('a,b,lab,lab,lab->l',2*l2+1,2*l3+1,wigner3j,wigner3j,bispectrum_aaw,optimize=True)
    a_waw_term = 2.*float(1/(4*np.pi)**1.5)*alm_00*np.einsum('a,b,lab,lab,lab->l',2*l2+1,2*l3+1,wigner3j,wigner3j,bispectrum_waw,optimize=True)
    aaww_term = np.einsum('l,acbdl->l', 1/(2*l1+1), Rho) #use when using rho for trispectrum
    remastered = (aa_ww_term + aw_aw_term + w_aaw_term + a_waw_term + aaww_term)
    remastered_curves = [aa_ww_term, aw_aw_term, w_aaw_term, a_waw_term, aaww_term, master_lhs, remastered]
    pickle.dump(remastered_curves, open(f'{base_dir}/remastered_curves.p', 'wb'))
    print(f'saved {base_dir}/remastered_curves.p', flush=True)
   
    print('Directly Calculated: ', master_lhs[start:start+10])
    print('reMASTERed: ', remastered[start:start+10])
    print('aa_ww_term: ', aa_ww_term[start:start+10])
    print('aw_aw_term: ', aw_aw_term[start:start+10])
    print('w_aaw_term: ', w_aaw_term[start:start+10])
    print('aaww_term: ', aaww_term[start:start+10])

    return remastered_curves




   