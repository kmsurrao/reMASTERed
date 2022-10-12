import sys
import os
import subprocess
import numpy as np
import healpy as hp
import multiprocessing as mp
from input import Info
from generate_mask import *
from bispectrum_vectorized import *
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
mask = gen_mask(inp)

#get <aaw> bispectra
min_l = 0
Nl = int(inp.ellmax/inp.dl) # number of bins
map_ = hp.read_map(inp.map_file)
map_ = hp.ud_grade(map_, inp.nside)
alm = hp.map2alm(map_, lmax=inp.ellmax)
wlm = hp.map2alm(mask, lmax=inp.ellmax)
Cl = hp.alm2cl(alm)
Ml = hp.alm2cl(wlm)
bispectrum_term3 = Bispectrum(inp, alm, Cl, np.conj(alm), Cl, np.conj(wlm), Ml, inp.ellmax, inp.nside, Nl, inp.dl, min_l, 3)
bispectrum_term4 = Bispectrum(inp, alm, Cl, np.conj(alm), Cl, wlm, Ml, inp.ellmax, inp.nside, Nl, inp.dl, min_l, 4)

#interpolate bispectra
bispectrum_term3 = Interpolate(inp, bispectrum_term3, 3)
bispectrum_term4 = Interpolate(inp, bispectrum_term4, 4)

#make plots of MASTER equation with new terms
compare_master(inp, map_, mask, bispectrum_term3, bispectrum_term4)