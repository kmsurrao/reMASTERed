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
print('imports complete in main.py', flush=True)

# main input file containing most specifications 
try:
    input_file = (sys.argv)[1]
except IndexError:
    input_file = 'moto.yaml'

# read in the input file and set up relevant info object
inp = Info(input_file)

#create threshold mask for component map
print('***********************************************************', flush=True)
print('Starting mask generation', flush=True)
mask = gen_mask(inp)

#get <aaw> bispectra
print('***********************************************************', flush=True)
print('Starting bispectrum calculation', flush=True)
min_l = 0
Nl = int(inp.ellmax/inp.dl) # number of bins
map_ = hp.read_map(inp.map_file)
map_ = hp.ud_grade(map_, inp.nside)
alm = hp.map2alm(map_, lmax=inp.ellmax)
wlm = hp.map2alm(mask, lmax=inp.ellmax)
Cl = hp.alm2cl(alm)
Ml = hp.alm2cl(wlm)
bispectrum_term3 = Bispectrum(alm, Cl, np.conj(alm), Cl, np.conj(wlm), Ml, inp.ellmax, inp.nside, 3, inp)
bispectrum_term4 = Bispectrum(alm, Cl, np.conj(alm), Cl, wlm, Ml, inp.ellmax, inp.nside, 4, inp)

#make plots of MASTER equation with new terms
print('***********************************************************', flush=True)
print('Starting MASTER comparison', flush=True)
compare_master(inp, map_, mask, bispectrum_term3, bispectrum_term4)