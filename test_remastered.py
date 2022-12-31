import numpy as np
import pickle
import subprocess
import os
from plot_remastered import *

def compare_master(inp, master_lhs, wlm_00, alm_00, Cl_aa, Cl_ww, Cl_aw, bispectrum_aaw, bispectrum_waw, Rho, env, base_dir=None):
    '''
    PARAMETERS
    inp: Info() object, contains information about input parameters
    master_lhs: 1D numpy array, directly computed power spectrum of masked map
    wlm_00: float, w_{00} for the mask
    alm_00: float, a_{00} for the map
    Cl_aa: 1D numpy array, auto-spectrum of the map
    Cl_ww: 1D numpy array, auto-spectrum of the mask
    Cl_aw: 1D numpy array, cross-spectrum of the map and mask
    bispectrum_aaw: 3D numpy array indexed as bispectrum_aaw[l1,l2,l3], bispectrum consisting of two factors of map and one factor of mask 
    bispectrum_waw: 3D numpy array indexed as bispectrum_waw[l1,l2,l3], bispectrum consisting of two factors of mask and one factor of map 
    Rho: 5D numpy array indexed as Rho[l1,l2,l3,l4,L], estimator for unnormalized trispectrum
    env: dict, contains features of the current environment
    base_dir: str, directory to save plots and pickle files

    RETURNS
    aa_ww_term: 1D numpy array of length ellmax+1, <aa><ww> term
    aw_aw_term: 1D numpy array of length ellmax+1, <aw><aw> term
    w_aaw_term: 1D numpy array of length ellmax+1, <w><aaw> term
    a_waw_term: 1D numpy array of length ellmax+1, <a><waw> term
    aaww_term: 1D numpy array of length ellmax+1, <aaww> term
    directly_computed: 1D numpy array of length ellmax+1, directly computed power spectrum of masked map
    remastered: 1D numpy array of length ellmax+1, reMASTERed result for power spectrum of masked map
    '''

    if base_dir is None:
        base_dir = inp.output_dir
    if not os.path.isdir(base_dir):
        subprocess.call(f'mkdir {base_dir}', shell=True, env=env)

    #calculate spectra of masked map from MASTER approach
    l1 = np.arange(inp.ellmax+1)
    l2 = np.arange(inp.ell_sum_max+1)
    l3 = np.arange(inp.ell_sum_max+1)
    aa_ww_term = float(1/(4*np.pi))*np.einsum('a,b,lab,lab,a,b->l',2*l2+1,2*l3+1,inp.wigner3j[:inp.ellmax+1,:,:],inp.wigner3j[:inp.ellmax+1,:,:],Cl_aa,Cl_ww,optimize=True)
    aw_aw_term = float(1/(4*np.pi))*np.einsum('a,b,lab,lab,a,b->l',2*l2+1,2*l3+1,inp.wigner3j[:inp.ellmax+1,:,:],inp.wigner3j[:inp.ellmax+1,:,:],Cl_aw,Cl_aw,optimize=True)
    w_aaw_term = float(1/(4*np.pi))*1/np.sqrt(np.pi)*wlm_00*np.einsum('a,b,lab,lab,lab->l',2*l2+1,2*l3+1,inp.wigner3j[:inp.ellmax+1,:,:],inp.wigner3j[:inp.ellmax+1,:,:],bispectrum_aaw,optimize=True)
    a_waw_term = float(1/(4*np.pi))*1/np.sqrt(np.pi)*alm_00*np.einsum('a,b,lab,lab,lab->l',2*l2+1,2*l3+1,inp.wigner3j[:inp.ellmax+1,:,:],inp.wigner3j[:inp.ellmax+1,:,:],bispectrum_waw,optimize=True)
    aaww_term = np.einsum('l,acbdl->l', 1/(2*l1+1), Rho)
    remastered = (aa_ww_term + aw_aw_term + w_aaw_term + a_waw_term + aaww_term)
    
    #save and plot data
    remastered_curves = [aa_ww_term, aw_aw_term, w_aaw_term, a_waw_term, aaww_term, master_lhs, remastered]
    if inp.save_files:
        pickle.dump(remastered_curves, open(f'{base_dir}/remastered_curves.p', 'wb'))
        print(f'saved {base_dir}/remastered_curves.p', flush=True)
    if inp.plot:
        plot_remastered(inp, remastered_curves, base_dir, start=2, logx=False, logy=False)

    return remastered_curves




   