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
import time
import string 
print('imports complete in main.py', flush=True)
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

    font = {'size'   : 20, 'family':'STIXGeneral'}
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.sans-serif': ['Computer Modern']})
    plt.rc_context({'axes.autolimit_mode': 'round_numbers'})

    np.random.seed(sim)

    lmax_data = 3*inp.nside-1

    #get simulated map
    map_ = hp.read_map(inp.map_file) 
    map_cl = hp.anafast(map_, lmax=inp.ellmax)
    map_ = hp.synfast(map_cl, nside=inp.nside)

    #create threshold mask for component map
    print('***********************************************************', flush=True)
    print(f'Starting mask generation for sim {sim}', flush=True)
    mask = gen_mask(inp, map_, sim)

    #get power spectra and bispectra
    alm = hp.map2alm(map_)
    wlm = hp.map2alm(mask)

    #added below
    l_arr,m_arr = hp.Alm.getlm(lmax_data)
    alm = alm*(l_arr<=inp.ellmax)
    wlm = wlm*(l_arr<=inp.ellmax)
    map_ = hp.alm2map(alm, nside=inp.nside)
    mask = hp.alm2map(wlm, nside=inp.nside)


    Cl = hp.alm2cl(alm, lmax_out=inp.ellmax)
    Ml = hp.alm2cl(wlm, lmax_out=inp.ellmax)
    Wl = hp.anafast(map_, mask, lmax=inp.ellmax)


    #make plot of map, mask, masked map, and correlation coefficient
    if sim==0:
        base_dir = f'images/{inp.comp}_cut{inp.cut}_ellmax{inp.ellmax}_nsims{inp.nsims}_nside{inp.nside}_nsideformasking{inp.nside_for_masking}'
        if not os.path.isdir(base_dir):
            subprocess.call(f'mkdir {base_dir}', shell=True, env=my_env)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(9,6))
        plt.axes(ax1)
        hp.mollview(map_, fig=1, hold=True, title='', format='%.03g')
        ax1.text(-0.1, 1.05, string.ascii_uppercase[0], transform=ax1.transAxes, 
            size=18, weight='bold')
        plt.axes(ax2)
        hp.mollview(mask, fig=2, hold=True, title='', format='%.03g', min=0.0, max=1.0)
        ax2.text(-0.1, 1.05, string.ascii_uppercase[1], transform=ax2.transAxes, 
            size=18, weight='bold')
        plt.axes(ax3)
        hp.mollview(mask*map_, fig=3, hold=True, title='', format='%.03g', min=np.amin(map_), max=np.amax(map_))
        ax3.text(-0.1, 1.05, string.ascii_uppercase[2], transform=ax3.transAxes, 
            size=18, weight='bold')
        plt.axes(ax4)
        ax4.text(-0.1, 1.05, string.ascii_uppercase[3], transform=ax4.transAxes, 
            size=18, weight='bold')
        corr = Wl/np.sqrt(Cl*Ml)
        # corr = np.convolve(corr, np.ones(10)/10, mode='same')
        ells = np.arange(inp.ellmax+1)
        plt.plot(ells[2:], corr[2:])
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$r_{\ell}$')
        # plt.title('Map and Mask Correlation Coefficient', font=font)
        plt.grid()
        plt.savefig(f'{base_dir}/maps.png')
        print(f'saved {base_dir}/maps.png', flush=True)
        print('corr component map and mask: ', corr, flush=True)

    print('***********************************************************', flush=True)
    print(f'Starting bispectrum calculation for sim {sim}', flush=True)
    bispectrum_aaw = Bispectrum(inp, map_-np.mean(map_), map_-np.mean(map_), mask-np.mean(mask), equal12=True)
    bispectrum_waw = Bispectrum(inp, mask-np.mean(mask), map_-np.mean(map_), mask-np.mean(mask), equal13=True)

    # print(f'Starting trispectrum calculation for sim {sim}', flush=True)
    # #trispectrum = Trispectrum(inp, map_-np.mean(map_), mask-np.mean(mask), Wl, Cl, Ml)
    # trispectrum = rho(inp, map_-np.mean(map_), mask-np.mean(mask), Wl, Cl, Ml)

    #get MASTER LHS
    master_lhs = hp.anafast(map_*mask, lmax=inp.ellmax)

    return master_lhs, wlm[0], alm[0], Cl, Ml, Wl, bispectrum_aaw, bispectrum_waw#, trispectrum 



#make plots of MASTER equation with new terms
pool = mp.Pool(min(inp.nsims, 16))
results = pool.starmap(one_sim, [(inp, sim) for sim in range(inp.nsims)])
pool.close()
print('len(results): ', len(results), flush=True)
master_lhs = np.mean(np.array([res[0] for res in results]), axis=0)
wlm_00 = np.mean(np.array([res[1] for res in results]), axis=0)
alm_00 = np.mean(np.array([res[2] for res in results]), axis=0)
Cl = np.mean(np.array([res[3] for res in results]), axis=0)
Ml = np.mean(np.array([res[4] for res in results]), axis=0)
Wl = np.mean(np.array([res[5] for res in results]), axis=0)
bispectrum_aaw = np.mean(np.array([res[6] for res in results]), axis=0)
bispectrum_waw = np.mean(np.array([res[7] for res in results]), axis=0)
# trispectrum = np.mean(np.array([res[8] for res in results]), axis=0)
# pickle.dump(trispectrum, open(f'trispectrum_isw_rho_ellmax{inp.ellmax}.p', 'wb')) #remove
trispectrum = pickle.load(open(f'trispectrum_isw_rho_ellmax{inp.ellmax}.p', 'rb')) 

print('***********************************************************', flush=True)
print('Starting MASTER comparison', flush=True)
plot_logx = False
if inp.comp=='ISW':
    plot_logx=True
compare_master(inp, master_lhs, wlm_00, alm_00, Cl, Ml, Wl, bispectrum_aaw, bispectrum_waw, trispectrum, my_env, plot_logx=plot_logx)

print("--- %s seconds ---" % (time.time() - start_time), flush=True)