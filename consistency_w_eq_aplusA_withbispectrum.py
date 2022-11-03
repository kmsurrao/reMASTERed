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

base_dir = '/moto/hill/users/kms2320/repositories/halosky_maps/maps/'

def one_sim(inp, sim, offset, base_dir):
    np.random.seed(sim)

    #get simulated map
    map_ = hp.read_map(base_dir + f'tsz_0000{sim}.fits') 
    map_ = hp.ud_grade(map_, inp.nside)

    #create W=a+A mask for component map
    print('***********************************************************', flush=True)
    print('Starting mask generation', flush=True)
    mask = map_ + offset
    print('mask: ', mask, flush=True)
    print('min of mask: ', np.amin(mask), flush=True)

    #get power spectra and bispectra
    print('***********************************************************', flush=True)
    print('Starting power spectra and bispectrum calculation', flush=True)
    alm = hp.map2alm(map_, lmax=inp.ellmax)
    wlm = hp.map2alm(mask, lmax=inp.ellmax)
    masked_map = map_*mask
    masked_map_alm = hp.map2alm(masked_map, lmax=inp.ellmax)
    Cl = hp.alm2cl(alm)
    Ml = hp.alm2cl(wlm)
    Wl = hp.anafast(map_, mask, lmax=inp.ellmax)
    mask_no_monopole = hp.remove_monopole(mask)
    wlm_no_monopole = hp.map2alm(mask_no_monopole, lmax=inp.ellmax)
    Ml_no_monopole = hp.alm2cl(wlm_no_monopole)
    lhs = hp.anafast(map_*mask, lmax=inp.ellmax)
    bispectrum = Bispectrum(alm, Cl, alm, Cl, wlm_no_monopole, Ml_no_monopole, inp.ellmax, inp.nside, inp)
    w00 = wlm[0]

    return lhs, Cl, Ml, Wl, bispectrum, w00


#read map
map_ = hp.read_map(base_dir + f'tsz_00000.fits') 
map_ = hp.ud_grade(map_, inp.nside)

#find offset A for mask W=a+A
offset = 1.e-6
print('offset: ', offset, flush=True)

pool = mp.Pool(min(inp.nsims, 16))
results = pool.starmap(one_sim, [(inp, sim, offset, base_dir) for sim in range(inp.nsims)])
pool.close()
lhs = np.mean(np.array([res[0] for res in results]), axis=0)
Cl = np.mean(np.array([res[1] for res in results]), axis=0)
Ml = np.mean(np.array([res[2] for res in results]), axis=0)
Wl = np.mean(np.array([res[3] for res in results]), axis=0)
bispectrum = np.mean(np.array([res[4] for res in results]), axis=0)
w00 = np.mean(np.array([res[5] for res in results]), axis=0)


#test modified MASTER
print('***********************************************************', flush=True)
print('Testing modified MASTER', flush=True)
start = 10
l1 = np.arange(inp.ellmax+1)
l2 = np.arange(inp.ellmax+1)
l3 = np.arange(inp.ellmax+1)
ells = np.arange(inp.ellmax+1)
wigner = pickle.load(open(inp.wigner_file, 'rb'))[:inp.ellmax+1, :inp.ellmax+1, :inp.ellmax+1]
term1 = float(1/(4*np.pi))*np.einsum('a,b,lab,lab,a,b->l',2*l2+1,2*l3+1,wigner,wigner,Cl,Ml,optimize=True)
term2 = float(1/(4*np.pi))*np.einsum('a,b,lab,lab,a,b->l',2*l2+1,2*l3+1,wigner,wigner,Wl,Wl,optimize=True)
term3 = 2.*float(1/(4*np.pi)**1.5)*w00*np.einsum('a,b,lab,lab,lab->l',2*l2+1,2*l3+1,wigner,wigner,bispectrum,optimize=True)
rhs = term1+term2+term3


#make comparison plot of masked_map_cl and master_cl
print('***********************************************************', flush=True)
print('Plotting', flush=True)
ells = np.arange(inp.ellmax+1)
plt.clf()
plt.plot(ells[start:], (ells*(ells+1)*term1/(2*np.pi))[start:], label='term1 (original MASTER)', color='c')
plt.plot(ells[start:], (ells*(ells+1)*term2/(2*np.pi))[start:], label='term2', linestyle='dotted')
plt.plot(ells[start:], (ells*(ells+1)*term3/(2*np.pi))[start:], label='term3', color='r')
plt.plot(ells[start:], (ells*(ells+1)*lhs/(2*np.pi))[start:], label='MASTER LHS', color='g')
plt.plot(ells[start:], (ells*(ells+1)*rhs/(2*np.pi))[start:], label='Modified MASTER RHS', linestyle='dotted', color='m')
plt.legend()
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\frac{\ell(\ell+1)C_{\ell}^{TT}}{2\pi}$ [$\mathrm{K}^2$]')
plt.yscale('log')
# plt.xscale('log')
plt.grid()
plt.savefig(f'consistency_w_eq_aplusA_withbispectrum.png')
print(f'saved consistency_w_eq_aplusA_withbispectrum.png')
plt.close('all')
print((lhs/rhs)[30:40])
print('master lhs: ', lhs[30:40])
print('master rhs: ', rhs[30:40])
print('Cl: ', Cl[30:40])
print('term1: ', term1[30:40])
print('term2: ', term2[30:40])
print('term3: ', term3[30:40])



print("--- %s seconds ---" % (time.time() - start_time), flush=True)