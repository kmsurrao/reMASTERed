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
min_l = 0
Nl = int(ellmax/dl) #number of bins

#load binned bispectrum
bispectrum = pickle.load(open(f'bispectrum_isw_maskisw0p7_ellmax{ellmax}_{Nl}bins.p', 'rb'))

#using ndimage.zoom
bispectrum = ndimage.zoom(bispectrum, dl, order=1)
print('bispectrum: ', bispectrum, flush=True)
print('bispectrum.shape: ', bispectrum.shape, flush=True)

#zero out all ells that don't satisfy triangle inequality 
ells = np.arange(ellmax)
tri = np.ones((ellmax, ellmax, ellmax)) 
for l1 in ells:
    for l2 in ells:
        for l3 in ells:
            if l3<abs(l1-l2) or l3>l1+l2:
                tri[l1,l2,l3] = 0
            elif l2<abs(l1-l3) or l2>l1+l3:
                tri[l1,l2,l3] = 0
            elif l1<abs(l2-l3) or l1>l2+l3:
                tri[l1,l2,l3] = 0
bispectrum = np.einsum('ijk,ijk->ijk', bispectrum, tri)

#save interpolated bispectrum
pickle.dump(bispectrum, open(f'linear_interpolated_bispectrum_ellmax{ellmax}_{Nl}origbins.p', 'wb'))
print(f'saved linear_interpolated_bispectrum_ellmax{ellmax}_{Nl}origbins.p', flush=True)