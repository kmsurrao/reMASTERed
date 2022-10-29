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


#get simulated map
map_ = hp.read_map(inp.map_file) 
map_ = hp.ud_grade(map_, inp.nside)

#create threshold mask for component map
print('***********************************************************', flush=True)
print('Starting mask generation', flush=True)
mask = gen_mask(inp, map_)

#get power spectra and bispectra
print('***********************************************************', flush=True)
print('Starting bispectrum calculation', flush=True)
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
# bispectrum_aaw = Bispectrum(alm, Cl, alm, Cl, wlm_no_monopole, Ml_no_monopole, inp.ellmax, inp.nside, inp)
bispectrum_waw = Bispectrum(wlm_no_monopole, Ml_no_monopole, alm, Cl, wlm_no_monopole, Ml_no_monopole, inp.ellmax, inp.nside, inp)
# bispectrum = Bispectrum(alm, Cl, alm, Cl, wlm, Ml, inp.ellmax, inp.nside, inp)
print('finished bispectrum calculation', flush=True)

# #compare <a tilde(a)> to representation in terms of bispectrum
# start = 0
# l2 = np.arange(inp.ellmax+1)
# l3 = np.arange(inp.ellmax+1)
# ells = np.arange(inp.ellmax+1)
# wigner = pickle.load(open(inp.wigner_file, 'rb'))[:inp.ellmax+1, :inp.ellmax+1, :inp.ellmax+1]
# lhs = hp.alm2cl(masked_map_alm, alm)
# rhs = float(1/(4*np.pi))*np.einsum('a,b,lab,lab,lab->l',2*l2+1,2*l3+1,wigner,wigner,bispectrum_aaw,optimize=True)
# rhs += np.real(wlm[0])/np.sqrt(4*np.pi)*Cl
# plt.clf()
# plt.plot(ells[start:], lhs[start:], label='directly computed')
# plt.plot(ells[start:], rhs[start:], label='using bispectrum', linestyle='dotted')
# plt.legend()
# plt.yscale('log')
# plt.xlabel(r'$\ell$')
# plt.ylabel(r'$C_{\ell}^{\tilde{a}a}$')
# plt.savefig('consistency_atildea.png')
# print('saved consistency_atildea.png', flush=True)
# print('lhs[start:]: ', lhs[start:], flush=True)
# print('rhs[start:]: ', rhs[start:], flush=True)
# print()

#compare <w tilde(a)> to representation in terms of Claw and w00
start = 10
l1 = np.arange(inp.ellmax+1)
l2 = np.arange(inp.ellmax+1)
l3 = np.arange(inp.ellmax+1)
ells = np.arange(inp.ellmax+1)
wigner = pickle.load(open(inp.wigner_file, 'rb'))[:inp.ellmax+1, :inp.ellmax+1, :inp.ellmax+1]
lhs = hp.anafast(map_*mask, mask, lmax=inp.ellmax)
rhs = float(1/np.sqrt(4*np.pi))*np.real(wlm[0])*Wl
rhs += 1/(4*np.pi)*np.einsum('a,b,lab,lab,lab->l',2*l2+1,2*l3+1,wigner,wigner,bispectrum_waw)
plt.clf()
plt.plot(ells[start:], lhs[start:], label='directly computed')
plt.plot(ells[start:], rhs[start:], label='using Claw and w00', linestyle='dotted')
plt.legend()
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_{\ell}^{\tilde{a}w}$')
plt.savefig('consistency_wtildea.png')
print('saved consistency_wtildea.png', flush=True)
print('lhs[start:]: ', lhs[start:], flush=True)
print('rhs[start:]: ', rhs[start:], flush=True)
print()




print("--- %s seconds ---" % (time.time() - start_time), flush=True)