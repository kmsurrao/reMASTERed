import matplotlib.pyplot as plt
import numpy as np
import pickle

ell = 299
dl = 10
bispectrum_binned = pickle.load(open('bispectrum_isw_maskisw0p7_ellmax500_50bins.p', 'rb'))[ell//dl,:,:]
bispectrum_interp = pickle.load(open('linear_interpolated_bispectrum_ellmax500_50origbins.p', 'rb'))[ell,:,:]

fig = plt.figure()
ax1 = fig.add_subplot(121)  # left side
ax2 = fig.add_subplot(122)  # right side
im = ax1.imshow(bispectrum_binned, vmin=0, vmax=255, norm='linear')
im = ax2.imshow(bispectrum_interp, vmin=0, vmax=255, norm='linear')
plt.savefig(f'bispectrum_plot_ell{ell}.png')
print(f'saved bispectrum_plot_ell{ell}.png')