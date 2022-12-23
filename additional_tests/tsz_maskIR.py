import sys
sys.path.append("..")
import os
import subprocess
import numpy as np
import healpy as hp
import multiprocessing as mp
import h5py
import pymaster as nmt
import pickle
import time
from input import Info
from generate_mask import *
from bispectrum import *
from trispectrum import *
from test_remastered import *
from wigner3j import *
from plot_mask import *
print('imports complete', flush=True)
start_time = time.time()

# main input file containing most specifications 
try:
    input_file = (sys.argv)[1]
except IndexError:
    input_file = 'threshold_moto.yaml'

# read in the input file and set up relevant info object
inp = Info(input_file, mask_provided=False)

# current environment, also environment in which to run subprocesses
my_env = os.environ.copy()

#get wigner 3j symbols
if inp.wigner_file:
    inp.wigner3j = pickle.load(open(inp.wigner_file, 'rb'))[:inp.ellmax+1, :inp.ellmax+1, :inp.ellmax+1]
else:
    inp.wigner3j = compute_3j(inp.ellmax)

def get_flux_theta_phi():
    '''
    Reads IR source catalogs and gets information about the sources

    RETURNS
    flux: 1D numpy array, contains flux of sources in MJy
    theta: 1D numpy array, contains theta values of sources
    phi: 1D numpy array, containns phi values of sources
    '''

    f1Flux = h5py.File('/moto/hill/users/kms2320/repositories/halosky_maps/cen_chunk1_flux_153.h5', 'r')
    f2Flux = h5py.File('/moto/hill/users/kms2320/repositories/halosky_maps/cen_chunk2_flux_153.h5', 'r')
    f3Flux = h5py.File('/moto/hill/users/kms2320/repositories/halosky_maps/sat_chunk1_flux_153.h5', 'r')
    f4Flux = h5py.File('/moto/hill/users/kms2320/repositories/halosky_maps/sat_chunk2_flux_153.h5', 'r')
    flux = np.concatenate((f1Flux['flux'][()],  f2Flux['flux'][()], f3Flux['flux'][()], f4Flux['flux'][()]))
    f1 = h5py.File('/moto/hill/users/kms2320/repositories/halosky_maps/cen_chunk1.h5', 'r')
    f2 = h5py.File('/moto/hill/users/kms2320/repositories/halosky_maps/cen_chunk2.h5', 'r')
    f3 = h5py.File('/moto/hill/users/kms2320/repositories/halosky_maps/sat_chunk1.h5', 'r')
    f4 = h5py.File('/moto/hill/users/kms2320/repositories/halosky_maps/sat_chunk2.h5', 'r')
    theta = np.concatenate((f1['theta'][()], f2['theta'][()], f3['theta'][()],  f4['theta'][()]))
    phi = np.concatenate((f1['phi'][()], f2['phi'][()], f3['phi'][()],  f4['phi'][()]))
    print('read catalogs', flush=True)
    print("np.amax(flux): ", np.amax(flux), flush=True)
    print('np.mean(flux): ', np.mean(flux), flush=True)
    print('np.std(flux): ', np.std(flux), flush=True)
    return flux, theta, phi

def mask_above_fluxcut(inp, fluxCut, flux, theta, phi):
    '''
    PARAMETERS
    inp: Info() object, contains information about input parameters
    fluxCut: float, minimum flux above which to mask sources, in MJy
    flux: 1D numpy array, contains flux of sources in MJy
    theta: 1D numpy array, contains theta values of sources
    phi: 1D numpy array, containns phi values of sources

    RETURNS
    m: 1D numpy array, apodized mask that masks IR sources above fluxCut with 20 arcmin holes
    '''
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



def one_sim(inp, fluxCut, flux, theta, phi):
    '''
    PARAMETERS
    inp: Info() object, contains information about input parameters
    fluxCut: float, minimum flux above which to mask sources, in MJy
    flux: 1D numpy array, contains flux of sources in MJy
    theta: 1D numpy array, contains theta values of sources
    phi: 1D numpy array, containns phi values of sources

    RETURNS
    master_lhs: 1D numpy array, directly computed power spectrum of masked map
    wlm[0]: float, w_{00} for the mask
    alm[0]: float, a_{00} for the map
    Cl_aa: 1D numpy array, auto-spectrum of the map
    Cl_ww: 1D numpy array, auto-spectrum of the mask
    Cl_aw: 1D numpy array, cross-spectrum of the map and mask 
    bispectrum_aaw: 3D numpy array indexed as bispectrum_aaw[l1,l2,l3], bispectrum consisting of two factors of map and one factor of mask 
    bispectrum_waw: 3D numpy array indexed as bispectrum_waw[l1,l2,l3], bispectrum consisting of two factors of mask and one factor of map  
    Rho: 5D numpy array indexed as Rho[l1,l2,l3,l4,L], estimator for unnormalized trispectrum
    '''

    scratch_path = '/moto/hill/users/kms2320/repositories/halosky_maps'

    #load compton-y map
    map_ = hp.read_map(f'{scratch_path}/tsz.fits')  
    map_ = hp.ud_grade(map_, inp.nside)

    #create IR source mask for component map
    print('Starting mask generation', flush=True)
    mask = mask_above_fluxcut(inp, fluxCut, flux, theta, phi)

    #get alm and wlm for map and mask, respectively 
    alm = hp.map2alm(map_)
    wlm = hp.map2alm(mask)

    #zero out modes above ellmax
    lmax_data = 3*inp.nside-1
    l_arr,m_arr = hp.Alm.getlm(lmax_data)
    alm = alm*(l_arr<=inp.ellmax)
    wlm = wlm*(l_arr<=inp.ellmax)
    map_ = hp.alm2map(alm, nside=inp.nside)
    mask = hp.alm2map(wlm, nside=inp.nside)

    #get auto- and cross-spectra for map and mask
    Cl_aa = hp.alm2cl(alm, lmax_out=inp.ellmax)
    Cl_ww = hp.alm2cl(wlm, lmax_out=inp.ellmax)
    Cl_aw = hp.anafast(map_, mask, lmax=inp.ellmax)


    #get list of map, mask, masked map, and correlation coefficient
    if inp.save_files or inp.plot:
        data = [map_, mask, map_*mask] #will contain map, mask, masked map, correlation coefficient
        if inp.output_dir:
            base_dir = inp.output_dir
        else:
            base_dir = f'images/tSZ_mask_IR_ellmax{inp.ellmax}_nside{inp.nside}'
        if not os.path.isdir(base_dir):
            subprocess.call(f'mkdir {base_dir}', shell=True, env=my_env)
        corr = Cl_aw/np.sqrt(Cl_aa*Cl_ww)
        data.append(corr)
        if inp.save_files:
            pickle.dump(data, open(f'{base_dir}/mask_data.p', 'wb'))
            print(f'saved {base_dir}/mask_data.p', flush=True)
        if inp.plot:
            plot_mask(inp, mask_data, base_dir)

    print('Starting bispectrum calculation', flush=True)
    bispectrum_aaw = Bispectrum(inp, map_-np.mean(map_), map_-np.mean(map_), mask-np.mean(mask), equal12=True)
    bispectrum_waw = Bispectrum(inp, mask-np.mean(mask), map_-np.mean(map_), mask-np.mean(mask), equal13=True)

    print('Starting trispectrum calculation', flush=True)
    Rho = rho(inp, map_-np.mean(map_), mask-np.mean(mask), Cl_aw, Cl_aa, Cl_ww)

    #get MASTER LHS
    master_lhs = hp.anafast(map_*mask, lmax=inp.ellmax)

    return master_lhs, wlm[0], alm[0], Cl_aa, Cl_ww, Cl_aw, bispectrum_aaw, bispectrum_waw, Rho 



#make plots of reMASTERed equation with new terms
fluxCut = 1.*10**(-7) #0.1 mJy
flux, theta, phi = get_flux_theta_phi()
master_lhs, wlm_00, alm_00, Cl_aa, Cl_ww, Cl_aw, bispectrum_aaw, bispectrum_waw, Rho = one_sim(inp, fluxCut, flux, theta, phi)
pickle.dump(Rho, open(f'rho/rho_tszmaskIR_ellmax{inp.ellmax}.p', 'wb')) #remove
# Rho = pickle.load(open(f'rho/rho_tszmaskIR_ellmax{inp.ellmax}.p', 'rb')) #remove

print('Starting reMASTERed comparison', flush=True)
if inp.output_dir:
    base_dir = inp.output_dir
else:
    base_dir = f'images/tSZ_mask_IR_ellmax{inp.ellmax}_nside{inp.nside}'
compare_master(inp, master_lhs, wlm_00, alm_00, Cl_aa, Cl_ww, Cl_aw, bispectrum_aaw, bispectrum_waw, Rho, my_env, base_dir=base_dir)

print("--- %s seconds ---" % (time.time() - start_time), flush=True)





