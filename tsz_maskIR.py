import sys
import os
import subprocess
import numpy as np
import healpy as hp
import multiprocessing as mp
from input import Info
from generate_mask import *
from bispectrum import *
from trispectrum import *
from test_master import *
from helper import *
import h5py
import pymaster as nmt
import pickle
import time
print('imports complete', flush=True)
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

def get_flux_theta_phi():
    #read catalog file
    f1 = h5py.File('/moto/hill/users/kms2320/repositories/halosky_maps/cen_chunk1_flux_353.h5', 'r')
    f2 = h5py.File('/moto/hill/users/kms2320/repositories/halosky_maps/cen_chunk2_flux_353.h5', 'r')
    flux = np.concatenate((f1['flux'][()],  f2['flux'][()]))
    f3 = h5py.File('/moto/hill/users/kms2320/repositories/halosky_maps/cen_chunk1.h5', 'r')
    f4 = h5py.File('/moto/hill/users/kms2320/repositories/halosky_maps/cen_chunk2.h5', 'r')
    theta = np.concatenate((f3['theta'][()],  f4['theta'][()]))
    phi = np.concatenate((f3['phi'][()],  f4['phi'][()]))
    print('read catalog', flush=True)
    print("np.amax(flux): ", np.amax(flux), flush=True)
    print('np.mean(flux): ', np.mean(flux), flush=True)
    print('np.std(flux): ', np.std(flux), flush=True)
    return flux, theta, phi

def mask_above_fluxcut(inp, fluxCut, flux, theta, phi):
    #initialize mask of all ones
    m = np.ones(hp.nside2npix(inp.nside))
    #find where flux > fluxCut
    tmp = flux > fluxCut
    #find theta and phi values of point sources where flux > fluxCut
    theta_new = theta[tmp]
    phi_new = phi[tmp]
    print('len(theta_new): ', len(theta_new), flush=True)
    for i in range(len(theta_new)):
        vec = hp.ang2vec(theta_new[i], phi_new[i])
        radius = 20.*(1./60.)*(np.pi/180.)
        ipix = hp.query_disc(inp.nside, vec, radius)
        print("ipix: ", ipix, flush=True)
        m[ipix] = 0.
    print('zeroed out point sources', flush=True)
    aposcale = inp.aposcale # Apodization scale in degrees, based on hole radius
    m = nmt.mask_apodization(m, aposcale, apotype="C1")
    print('apodized mask', flush=True)
    m[m>1.]=1.
    m[m<0.]=0.
    return m



def one_sim(inp, sim, fluxCut, flux, theta, phi):

    scratch_path = '/moto/hill/users/kms2320/repositories/halosky_maps'

    lmax_data = 3*inp.nside-1

    #get simulated map
    # map_ = hp.read_map(f'{scratch_path}/maps/tsz_{sim:05d}.fits')
    map_ = hp.read_map(f'{scratch_path}/tsz.fits')  
    map_ = hp.ud_grade(map_, inp.nside)

    #create threshold mask for component map
    print('***********************************************************', flush=True)
    print(f'Starting mask generation for sim {sim}', flush=True)
    mask = mask_above_fluxcut(inp, fluxCut, flux, theta, phi)

    #get power spectra and bispectra
    alm = hp.map2alm(map_)
    wlm = hp.map2alm(mask)

    #added below
    l_arr,m_arr = hp.Alm.getlm(lmax_data)
    alm = alm*(l_arr<=inp.ellmax)
    wlm = wlm*(l_arr<=inp.ellmax)
    map_ = hp.alm2map(alm, nside=inp.nside)
    mask = hp.alm2map(wlm, nside=inp.nside)


    Cl_aa = hp.alm2cl(alm, lmax_out=inp.ellmax)
    Cl_ww = hp.alm2cl(wlm, lmax_out=inp.ellmax)
    Cl_aw = hp.anafast(map_, mask, lmax=inp.ellmax)


    #make plot of map, mask, masked map, and correlation coefficient
    data = [map_, mask, map_*mask] #contains map, mask, masked map, correlation coefficient
    base_dir = f'images/tSZ_mask_IR_ellmax{inp.ellmax}_nside{inp.nside}'
    if not os.path.isdir(base_dir):
        subprocess.call(f'mkdir {base_dir}', shell=True, env=my_env)
    corr = Cl_aw/np.sqrt(Cl_aa*Cl_ww)
    data.append(corr)
    pickle.dump(data, open(f'{base_dir}/mask_data.p', 'wb'))
    print(f'saved {base_dir}/mask_data.p', flush=True)

    print('***********************************************************', flush=True)
    print(f'Starting bispectrum calculation for sim {sim}', flush=True)
    bispectrum_aaw = Bispectrum(inp, map_-np.mean(map_), map_-np.mean(map_), mask-np.mean(mask), equal12=True)
    bispectrum_waw = Bispectrum(inp, mask-np.mean(mask), map_-np.mean(map_), mask-np.mean(mask), equal13=True)

    print(f'Starting trispectrum calculation for sim {sim}', flush=True)
    Rho = rho(inp, map_-np.mean(map_), mask-np.mean(mask), Cl_aw, Cl_aa, Cl_ww)

    #get MASTER LHS
    master_lhs = hp.anafast(map_*mask, lmax=inp.ellmax)

    return master_lhs, wlm[0], alm[0], Cl_aa, Cl_ww, Cl,_aw bispectrum_aaw, bispectrum_waw, Rho 



#make plots of MASTER equation with new terms
# fluxCut = 7*10**(-3) #7 mJy
fluxCut = 3.*10**(-6) 
flux, theta, phi = get_flux_theta_phi()
master_lhs, wlm_00, alm_00, Cl_aa, Cl_ww, Cl_aw, bispectrum_aaw, bispectrum_waw, Rho = one_sim(inp, 0, fluxCut, flux, theta, phi)
pickle.dump(Rho, open(f'rho_tszmaskIR_ellmax{inp.ellmax}.p', 'wb')) #remove
# Rho = pickle.load(open(f'rho_tszmaskIR_ellmax{inp.ellmax}.p', 'rb')) 

print('***********************************************************', flush=True)
print('Starting MASTER comparison', flush=True)
base_dir = f'images/tSZ_mask_IR_ellmax{inp.ellmax}_nside{inp.nside}'
compare_master(inp, master_lhs, wlm_00, alm_00, Cl_aa, Cl_ww, Cl_aw, bispectrum_aaw, bispectrum_waw, Rho, my_env, base_dir=base_dir)

print("--- %s seconds ---" % (time.time() - start_time), flush=True)





