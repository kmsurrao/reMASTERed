import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import griddata
import scipy.ndimage as ndimage
print('imports complete', flush=True)

#set parameters that were used in bispectrum calculation
ellmax = 500
dl = 10 #bin width
Nl = int(ellmax/dl) #number of bins

#load binned bispectrum
bispectrum = pickle.load(open(f'bispectrum_isw_maskisw0p7_ellmax{ellmax}_{Nl}bins.p', 'rb'))

#using ndimage.zoom
bispectrum = ndimage.zoom(bispectrum, dl, order=1)
print('bispectrum: ', bispectrum, flush=True)
print('bispectrum.shape: ', bispectrum.shape, flush=True)

#save interpolated bispectrum
pickle.dump(bispectrum, open(f'linear_interpolated_bispectrum_ellmax{ellmax}_{Nl}origbins.p', 'wb'))
print(f'saved linear_interpolated_bispectrum_ellmax{ellmax}_{Nl}origbins.p', flush=True)