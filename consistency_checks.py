import sys
import os
import subprocess
import numpy as np
import healpy as hp
import multiprocessing as mp
from input import Info
from generate_mask import *
from bispectrum_unbinned import *
from interpolate_bispectrum import *
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


def one_sim(inp, sim):

    np.random.seed(sim)

    #get simulated map
    map_ = hp.read_map(inp.map_file) 
    map_cl = hp.anafast(map_, lmax=inp.ellmax)
    map_ = hp.synfast(map_cl, nside=inp.nside)

    #create threshold mask for component map
    print('***********************************************************', flush=True)
    print(f'Starting mask generation sim {sim}', flush=True)
    mask = gen_mask(inp, map_)

    #get power spectra and bispectra
    print('***********************************************************', flush=True)
    print(f'Starting bispectrum calculation sim {sim}', flush=True)
    alm = hp.map2alm(map_, lmax=inp.ellmax)
    wlm = hp.map2alm(mask, lmax=inp.ellmax)
    masked_map = map_*mask
    masked_map_alm = hp.map2alm(masked_map, lmax=inp.ellmax)
    Cl = hp.alm2cl(alm)
    Ml = hp.alm2cl(wlm)
    Wl = hp.anafast(map_, mask, lmax=inp.ellmax)
    mask_no_monopole = hp.remove_monopole(mask)
    print('mean of mask: ', np.mean(mask), flush=True)
    print('mean of mask no monopole: ', np.mean(mask_no_monopole), flush=True)
    wlm_no_monopole = hp.map2alm(mask_no_monopole, lmax=inp.ellmax)
    Ml_no_monopole = hp.alm2cl(wlm_no_monopole)
    Wl_no_monopole = hp.alm2cl(alm, wlm_no_monopole, lmax=inp.ellmax)

    #load 3j symbols and set up arrays
    l2 = np.arange(inp.ellmax+1)
    l3 = np.arange(inp.ellmax+1)
    wigner = pickle.load(open(inp.wigner_file, 'rb'))[:inp.ellmax+1, :inp.ellmax+1, :inp.ellmax+1]
    
    #compare <a tilde(a)> to representation in terms of bispectrum
    bispectrum_aaw = Bispectrum(alm, Cl, alm, Cl, wlm_no_monopole, Ml_no_monopole, inp.ellmax, inp.nside, inp)
    print(f'finished bispectrum calculation aaw sim {sim}', flush=True)
    lhs_atildea = hp.alm2cl(masked_map_alm, alm)
    rhs_atildea = float(1/(4*np.pi))*np.einsum('a,b,lab,lab,lab->l',2*l2+1,2*l3+1,wigner,wigner,bispectrum_aaw,optimize=True)
    rhs_atildea += np.real(wlm[0])/np.sqrt(4*np.pi)*Cl

    #compare <w tilde(a)> to representation in terms of Claw and w00
    bispectrum_waw = Bispectrum(wlm_no_monopole, Ml_no_monopole, alm, Cl, wlm_no_monopole, Ml_no_monopole, inp.ellmax, inp.nside, inp)
    print(f'finished bispectrum calculation waw sim {sim}', flush=True)
    wigner = pickle.load(open(inp.wigner_file, 'rb'))[:inp.ellmax+1, :inp.ellmax+1, :inp.ellmax+1]
    lhs_wtildea = hp.anafast(map_*mask, mask, lmax=inp.ellmax)
    rhs_wtildea = float(1/np.sqrt(4*np.pi))*np.real(wlm[0])*Wl #added no monopole here to test
    rhs_wtildea += 1/(4*np.pi)*np.einsum('a,b,lab,lab,lab->l',2*l2+1,2*l3+1,wigner,wigner,bispectrum_waw)

    return lhs_atildea, rhs_atildea, lhs_wtildea, rhs_wtildea

#do ensemble averaging
pool = mp.Pool(min(inp.nsims, 16))
results = pool.starmap(one_sim, [(inp, sim) for sim in range(inp.nsims)])
pool.close()
lhs_atildea = np.mean(np.array([res[0] for res in results]), axis=0)
rhs_atildea = np.mean(np.array([res[1] for res in results]), axis=0)
lhs_wtildea = np.mean(np.array([res[2] for res in results]), axis=0)
rhs_wtildea = np.mean(np.array([res[3] for res in results]), axis=0)
start = 10
ells = np.arange(inp.ellmax+1)

#plot <a tilde(a)>
plt.clf()
plt.plot(ells[start:], lhs_atildea[start:], label='directly computed')
plt.plot(ells[start:], rhs_atildea[start:], label='in terms of n-point expansions', linestyle='dotted')
plt.legend()
plt.yscale('log')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_{\ell}^{\tilde{a}a}$')
plt.savefig('consistency_atildea.png')
print('saved consistency_atildea.png', flush=True)
print('lhs_atildea[start:]: ', lhs_atildea[start:], flush=True)
print('rhs_atildea[start:]: ', rhs_atildea[start:], flush=True)
print()


#plot <w tilde(a)>
plt.clf()
plt.plot(ells[start:], lhs_wtildea[start:], label='directly computed')
plt.plot(ells[start:], rhs_wtildea[start:], label='in terms of n-point expansions', linestyle='dotted')
plt.legend()
plt.yscale('log')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_{\ell}^{\tilde{a}w}$')
plt.savefig('consistency_wtildea.png')
print('saved consistency_wtildea.png', flush=True)
print('lhs_wtildea[start:]: ', lhs_wtildea[start:], flush=True)
print('rhs_wtildea[start:]: ', rhs_wtildea[start:], flush=True)
print()



print("--- %s seconds ---" % (time.time() - start_time), flush=True)