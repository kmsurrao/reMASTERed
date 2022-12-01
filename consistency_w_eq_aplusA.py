import sys
import os
import subprocess
import numpy as np
import healpy as hp
import multiprocessing as mp
from input import Info
from generate_mask import *
from bispectrum import *
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

def one_sim(inp, sim, offset):
    np.random.seed(sim)
    lmax_data = 3*inp.nside-1

    #get simulated map
    map_ = hp.read_map(inp.map_file) 
    map_cl = hp.anafast(map_, lmax=inp.ellmax)
    map_ = hp.synfast(map_cl, inp.nside)
    print('map: ', map_, flush=True)

    #create W=a+A mask for component map
    print('***********************************************************', flush=True)
    print('Starting mask generation', flush=True)
    mask = map_ + offset
    print('min of mask: ', np.amin(mask), flush=True)

    #get power spectra and bispectra
    print('***********************************************************', flush=True)
    print('Starting power spectra and bispectrum calculation', flush=True)
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

    lhs = hp.anafast(masked_map, lmax=inp.ellmax)
    lhs_wtildea = hp.anafast(masked_map, mask, lmax=inp.ellmax)

    return lhs, Cl, Ml, Wl, wlm[0], lhs_wtildea


#read map
map_ = hp.read_map(inp.map_file) 
map_ = hp.ud_grade(map_, inp.nside)

#find offset A for mask W=a+A
offset = 2*abs(np.amin(map_))
print('offset: ', offset, flush=True)

pool = mp.Pool(min(inp.nsims, 16))
results = pool.starmap(one_sim, [(inp, sim, offset) for sim in range(inp.nsims)])
pool.close()
lhs = np.mean(np.array([res[0] for res in results]), axis=0)
Cl = np.mean(np.array([res[1] for res in results]), axis=0)
Ml = np.mean(np.array([res[2] for res in results]), axis=0)
Wl = np.mean(np.array([res[3] for res in results]), axis=0)
w00 = np.mean(np.array([res[4] for res in results]), axis=0)
lhs_wtildea = np.mean(np.array([res[5] for res in results]), axis=0)
# bispectrum = np.mean(np.array([res[4] for res in results]), axis=0)
# w00 = np.mean(np.array([res[5] for res in results]), axis=0)


#test modified MASTER
print('***********************************************************', flush=True)
print('Testing modified MASTER', flush=True)
# print('finished bispectrum calculation', flush=True)
start = 10
l1 = np.arange(inp.ellmax+1)
l2 = np.arange(inp.ellmax+1)
l3 = np.arange(inp.ellmax+1)
ells = np.arange(inp.ellmax+1)
wigner = pickle.load(open(inp.wigner_file, 'rb'))[:inp.ellmax+1, :inp.ellmax+1, :inp.ellmax+1]
term1 = float(1/(4*np.pi))*np.einsum('a,b,lab,lab,a,b->l',2*l2+1,2*l3+1,wigner,wigner,Cl,Ml,optimize=True)
term2 = float(1/(4*np.pi))*np.einsum('a,b,lab,lab,a,b->l',2*l2+1,2*l3+1,wigner,wigner,Wl,Wl,optimize=True)
term3 = 2.*float(1/(4*np.pi)**1.5)*wlm_00*np.einsum('a,b,lab,lab,lab->l',2*l2+1,2*l3+1,wigner,wigner,bispectrum,optimize=True)
rhs = term1+term2+term3


#make comparison plot of masked_map_cl and master_cl
print('***********************************************************', flush=True)
print('Plotting', flush=True)
ells = np.arange(inp.ellmax+1)
plt.clf()
plt.plot(ells[start:], (ells*(ells+1)*term1/(2*np.pi))[start:], label='term1 (original MASTER)', color='c')
plt.plot(ells[start:], (ells*(ells+1)*term2/(2*np.pi))[start:], label='term2', linestyle='dotted')
# plt.plot(ells[start:], (ells*(ells+1)*term3/(2*np.pi))[start:], label='term3', color='r')
plt.plot(ells[start:], (ells*(ells+1)*lhs/(2*np.pi))[start:], label='MASTER LHS', color='g')
plt.plot(ells[start:], (ells*(ells+1)*rhs/(2*np.pi))[start:], label='Modified MASTER RHS', linestyle='dotted', color='m')
plt.legend()
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\frac{\ell(\ell+1)C_{\ell}^{TT}}{2\pi}$ [$\mathrm{K}^2$]')
# plt.yscale('log')
# plt.xscale('log')
plt.grid()
plt.savefig(f'consistency_w_eq_aplusA.png')
print(f'saved consistency_w_eq_aplusA.png')
plt.close('all')
print((lhs/rhs)[30:40])
print('master lhs: ', lhs[30:40])
print('master rhs: ', rhs[30:40])
print('Cl: ', Cl[30:40])
print('term1: ', term1[30:40])
print('term2: ', term2[30:40])
# print('term3: ', term3[30:40])


#check consistency <tilde(a) w> for this result
rhs_wtildea = float(1/np.sqrt(4*np.pi))*np.real(w00)*Wl 
plt.clf()
plt.plot(ells[start:], lhs_wtildea[start:], label='directly computed', color='g')
plt.plot(ells[start:], rhs_wtildea[start:], label='in terms of n-point expansions', linestyle='dotted', color='m')
plt.legend()
plt.yscale('log')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_{\ell}^{\tilde{a}w}$')
plt.grid()
plt.savefig(f'consistency_wtileda_for_wapA.png')
print(f'saved consistency_wtildea_for_wapA.png')
plt.close('all')




print("--- %s seconds ---" % (time.time() - start_time), flush=True)