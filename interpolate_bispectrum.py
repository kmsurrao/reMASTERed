import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import griddata
print('imports complete', flush=True)

#set parameters that were used in bispectrum calculation
ellmax = 1000
dl = 10 #bin width
Nl = 100 #number of bins

#load binned bispectrum
bispectrum = pickle.load(open('bispectrum_isw_maskisw0p7_ellmax1000_100bins.p', 'rb'))

#set up mesh grid and interpolate
points_b1 = np.repeat(dl*np.arange(Nl)+dl/2, Nl**2)
points_b2 = np.tile( np.repeat(dl*np.arange(Nl)+dl/2, Nl), Nl)
points_b3 = np.tile(dl*np.arange(Nl)+dl/2, Nl**2)
points = np.array([points_b1, points_b2, points_b3]).transpose()
values = np.reshape(bispectrum, -1)
grid_l1, grid_l2, grid_l3 = np.mgrid[0:ellmax+1, 0:ellmax+1, 0:ellmax+1]
print('got mgrid', flush=True)
grid_z1 = griddata(points, values, (grid_l1, grid_l2, grid_l3), method='linear')
print('grid_z1: ', grid_z1, flush=True)
print('grid_z1.shape: ', grid_z1.shape, flush=True)
bispectrum = grid_z1
print('loaded bispectrum', flush=True)

#save interpolated bispectrum
pickle.dump(bispectrum, open('linear_interpolated_bispectrum_ellmax1000_100origbins.p', 'wb'))
print('saved linear_interpolated_bispectrum_ellmax1000_100origbins.p', flush=True)