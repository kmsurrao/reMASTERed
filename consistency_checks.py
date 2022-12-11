import sys
import os
import subprocess
import numpy as np
import healpy as hp
import multiprocessing as mp
from input import Info
from generate_mask import *
from bispectrum import *
from test_master import *
import time
print('imports complete in consistency_checks.py', flush=True)
start_time = time.time()

# main input file containing most specifications 
try:
    input_file = (sys.argv)[1]
except IndexError:
    input_file = 'moto.yaml'

# read in the input file and set up relevant info object
inp = Info(input_file)

# current environment, also environment in which to run subprocesses
my_env = os.environ.copy()


def one_sim(inp, sim, atildea=True, wtildea=True):

    lmax_data = 3*inp.nside-1

    np.random.seed(sim)

    #get simulated map
    map_ = hp.read_map(inp.map_file) 
    map_cl = hp.anafast(map_, lmax=lmax_data)
    map_ = hp.synfast(map_cl, nside=inp.nside)

    #create threshold mask for component map
    print('***********************************************************', flush=True)
    print(f'Starting mask generation sim {sim}', flush=True)
    mask = gen_mask(inp, map_, sim, testing_aniso=True)

    #get power spectra and bispectra
    print('***********************************************************', flush=True)
    print(f'Starting bispectrum calculation sim {sim}', flush=True)
    alm = hp.map2alm(map_)
    wlm = hp.map2alm(mask)
    
    #added below
    l_arr,m_arr = hp.Alm.getlm(lmax_data)
    alm = alm*(l_arr<=inp.ellmax)
    wlm = wlm*(l_arr<=inp.ellmax)
    map_ = hp.alm2map(alm, nside=inp.nside)
    mask = hp.alm2map(wlm, nside=inp.nside)

    masked_map = map_*mask
    masked_map_alm = hp.map2alm(masked_map, lmax=inp.ellmax)
    Cl = hp.alm2cl(alm, lmax_out=inp.ellmax)
    Ml = hp.alm2cl(wlm, lmax_out=inp.ellmax)
    Wl = hp.anafast(map_, mask, lmax=inp.ellmax)

    #load 3j symbols and set up arrays
    l2 = np.arange(inp.ellmax+1)
    l3 = np.arange(inp.ellmax+1)
    wigner = pickle.load(open(inp.wigner_file, 'rb'))[:inp.ellmax+1, :inp.ellmax+1, :inp.ellmax+1]
    
    if atildea:
        #compare <a tilde(a)> to representation in terms of bispectrum
        bispectrum_aaw = Bispectrum(inp, map_-np.mean(map_), map_-np.mean(map_), mask-np.mean(mask), equal12=True)
        print(f'finished bispectrum calculation aaw sim {sim}', flush=True)
        lhs_atildea = hp.anafast(map_*mask, map_, lmax=inp.ellmax)
        bispectrum_term_atildea = float(1/(4*np.pi))*np.einsum('a,b,lab,lab,lab->l',2*l2+1,2*l3+1,wigner,wigner,bispectrum_aaw,optimize=True)
        cl_term_atildea = np.real(wlm[0])/np.sqrt(4*np.pi)*Cl

    if wtildea:
        #compare <w tilde(a)> to representation in terms of Claw and w00
        bispectrum_waw = Bispectrum(inp,mask-np.mean(mask),map_-np.mean(map_),mask-np.mean(mask),equal13=True)
        print(f'finished bispectrum calculation waw sim {sim}', flush=True)
        wigner = pickle.load(open(inp.wigner_file, 'rb'))[:inp.ellmax+1, :inp.ellmax+1, :inp.ellmax+1]
        lhs_wtildea = hp.anafast(map_*mask, mask, lmax=inp.ellmax)
        wl_term_wtildea = float(1/np.sqrt(4*np.pi))*np.real(wlm[0])*Wl #added no monopole here to test
        bispectrum_term_wtildea = 1/(4*np.pi)*np.einsum('a,b,lab,lab,lab->l',2*l2+1,2*l3+1,wigner,wigner,bispectrum_waw)
        ml_term_wtildea = float(1/np.sqrt(4*np.pi))*np.real(alm[0])*Ml
    
    if atildea and wtildea:
        return lhs_atildea,cl_term_atildea, bispectrum_term_atildea, lhs_wtildea, wl_term_wtildea, bispectrum_term_wtildea, ml_term_wtildea
    elif atildea:
        return lhs_atildea, cl_term_atildea, bispectrum_term_atildea, rhs_atildea
    elif wtildea:
        return lhs_wtildea, wl_term_wtildea, bispectrum_term_wtildea, ml_term_wtildea

#do ensemble averaging
atildea = True
wtildea = True
pool = mp.Pool(min(inp.nsims, 16))
results = pool.starmap(one_sim, [(inp, sim, atildea, wtildea) for sim in range(inp.nsims)])
pool.close()
if atildea:
    lhs_atildea = np.mean(np.array([res[0] for res in results]), axis=0)
    cl_term_atildea = np.mean(np.array([res[1] for res in results]), axis=0)
    bispectrum_term_atildea = np.mean(np.array([res[2] for res in results]), axis=0)
    if wtildea:
        lhs_wtildea = np.mean(np.array([res[3] for res in results]), axis=0)
        wl_term_wtildea = np.mean(np.array([res[4] for res in results]), axis=0)
        bispectrum_term_wtildea = np.mean(np.array([res[5] for res in results]), axis=0)
        ml_term_wtildea = np.mean(np.array([res[6] for res in results]), axis=0)
elif wtildea:
    lhs_wtildea = np.mean(np.array([res[0] for res in results]), axis=0)
    wl_term_wtildea = np.mean(np.array([res[1] for res in results]), axis=0)
    bispectrum_term_wtildea = np.mean(np.array([res[2] for res in results]), axis=0)
    ml_term_wtildea = np.mean(np.array([res[3] for res in results]), axis=0)
start = 2
ells = np.arange(inp.ellmax+1)

if atildea and wtildea:
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(1, 2, layout="constrained")
    plt.axes(ax1)
    rhs_atildea = cl_term_atildea + bispectrum_term_atildea
    plt.plot(ells[start:], cl_term_atildea[start:], label='<w><aa> term', color='c')
    plt.plot(ells[start:], bispectrum_term_atildea[start:], label='<aaw> term', color='r')
    plt.plot(ells[start:], lhs_atildea[start:], label='Directly Computed', color='g')
    plt.plot(ells[start:], rhs_atildea[start:], label='reMASTERed', linestyle='dotted', color='m')
    plt.legend()
    plt.xscale('log')
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$C_{\ell}^{\tilde{a}a}$')
    print('lhs_atildea[start:]: ', lhs_atildea[start:], flush=True)
    print('rhs_atildea[start:]: ', rhs_atildea[start:], flush=True)
    print()
    plt.axes(ax2)
    rhs_wtildea = wl_term_wtildea + bispectrum_term_wtildea
    plt.plot(ells[start:], wl_term_wtildea[start:], label='<w><aw> term')
    plt.plot(ells[start:], bispectrum_term_wtildea[start:], label='<waw> term', color='r')
    plt.plot(ells[start:], lhs_wtildea[start:], label='Directly Computed', color='g')
    plt.plot(ells[start:], rhs_wtildea[start:], label='ReMASTERed', color='m', linestyle='dotted')
    plt.legend()
    plt.xscale('log')
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$C_{\ell}^{\tilde{a}w}$')
    print('lhs_wtildea[start:]: ', lhs_wtildea[start:], flush=True)
    print('rhs_wtildea[start:]: ', rhs_wtildea[start:], flush=True)
    print('wl_term_wtildea[start:]: ', wl_term_wtildea[start:], flush=True)
    print('bispectrum_term_wtildea[start:]: ', bispectrum_term_wtildea[start:], flush=True)
    print('ml_term_wtildea[start:]: ', ml_term_wtildea[start:], flush=True)
    print()
    plt.savefig(f'images/consistency/consistency_{inp.comp}_{inp.ellmax}.png')

elif atildea:
    #plot <a tilde(a)>
    plt.clf()
    rhs_atildea = cl_term_atildea + bispectrum_term_atildea
    plt.plot(ells[start:], cl_term_atildea[start:], label='<w><aa> term', color='c')
    plt.plot(ells[start:], bispectrum_term_atildea[start:], label='<aaw> term', color='r')
    plt.plot(ells[start:], lhs_atildea[start:], label='Directly Computed', color='g')
    plt.plot(ells[start:], rhs_atildea[start:], label='reMASTERed', linestyle='dotted', color='m')
    plt.legend()
    plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$C_{\ell}^{\tilde{a}a}$')
    plt.savefig(f'images/consistency/consistency_atildea_{inp.comp}.png')
    print(f'saved images/consistency/consistency_atildea_{inp.comp}.png', flush=True)
    print('lhs_atildea[start:]: ', lhs_atildea[start:], flush=True)
    print('rhs_atildea[start:]: ', rhs_atildea[start:], flush=True)
    print()

elif wtildea:
    #plot <w tilde(a)>
    rhs_wtildea = wl_term_wtildea + bispectrum_term_wtildea
    plt.clf()
    plt.plot(ells[start:], wl_term_wtildea[start:], label='<w><aw> term')
    plt.plot(ells[start:], bispectrum_term_wtildea[start:], label='<waw> term', color='r')
    plt.plot(ells[start:], lhs_wtildea[start:], label='Directly Computed', color='g')
    plt.plot(ells[start:], rhs_wtildea[start:], label='ReMASTERed', color='m', linestyle='dotted')
    plt.legend()
    plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$C_{\ell}^{\tilde{a}w}$')
    plt.savefig(f'images/consistency/consistency_wtildea_{inp.comp}.png')
    print(f'saved images/consistency/consistency_wtildea_{inp.comp}.png', flush=True)
    print('lhs_wtildea[start:]: ', lhs_wtildea[start:], flush=True)
    print('rhs_wtildea[start:]: ', rhs_wtildea[start:], flush=True)
    print('wl_term_wtildea[start:]: ', wl_term_wtildea[start:], flush=True)
    print('bispectrum_term_wtildea[start:]: ', bispectrum_term_wtildea[start:], flush=True)
    print('ml_term_wtildea[start:]: ', ml_term_wtildea[start:], flush=True)
    print()



print("--- %s seconds ---" % (time.time() - start_time), flush=True)