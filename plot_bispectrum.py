import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import pickle

twoD = False
oneD = True

#code for 2D image at some ell slice
if twoD:

    ell = 300
    dl = 10
    bispectrum_binned = pickle.load(open('bispectrum_isw_maskisw0p7_ellmax500_50bins.p', 'rb'))[ell//dl,:,:]
    bispectrum_interp = pickle.load(open('linear_interpolated_bispectrum_ellmax500_50origbins.p', 'rb'))[ell,:,:]

    fig = plt.figure()
    ax1 = fig.add_subplot(121)  # left side
    ax2 = fig.add_subplot(122)  # right side
    logthresh = 10**(-100)
    im = ax1.imshow(bispectrum_binned, norm=matplotlib.colors.SymLogNorm(10**-logthresh))
    im = ax2.imshow(bispectrum_interp, norm=matplotlib.colors.SymLogNorm(10**-logthresh))
    plt.savefig(f'bispectrum_plots/ell{ell}.png')
    print(f'saved bispectrum_plots/ell{ell}.png')
    print(bispectrum_interp)


#code for 1D plot at slice ell1, ell2
if oneD:

    ell1 = 300
    ell2 = 400
    dl = 10
    bispectrum_binned = pickle.load(open('bispectrum_isw_maskisw0p7_ellmax500_50bins.p', 'rb'))[ell1//dl,ell2//dl,:]
    bispectrum_interp = pickle.load(open('linear_interpolated_bispectrum_ellmax500_50origbins.p', 'rb'))[ell1,ell2,:]

    plt.plot((dl+dl+1)/2*np.arange(len(bispectrum_binned)), bispectrum_binned, 'o', label='binned')
    plt.plot(np.arange(len(bispectrum_interp)), bispectrum_interp, label='interpolated')
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$b_{\ell_1,\ell_2,\ell_3}$')
    plt.title(f'Bispectrum at ell1={ell1}, ell2={ell2}')
    plt.legend()
    plt.savefig(f'bispectrum_plots/ell1_{ell1}_ell2_{ell2}.png')
    print(f'saved bispectrum_plots/ell1_{ell1}_ell2_{ell2}.png')