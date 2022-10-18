import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import griddata
import scipy.ndimage as ndimage
import skimage
import os.path



def Interpolate(inp, bispectrum, term):
    Nl = int(inp.ellmax/inp.dl) #number of bins

    #check if interpolated bispectrum already exists
    if os.path.exists(f'bispectra/linear_interpolated_bispectrum_comp{inp.comp}_ellmax{inp.ellmax}_{Nl}origbins_term{term}.p'):
        print('interpolated bispectrum already exists', flush=True)
        bispectrum = pickle.load(open(f'bispectra/linear_interpolated_bispectrum_comp{inp.comp}_ellmax{inp.ellmax}_{Nl}origbins_term{term}.p', 'rb'))
        return bispectrum
    

    # #using ndimage.zoom
    # bispectrum = ndimage.zoom(bispectrum, inp.dl, order=0)

    #using skimage
    bispectrum = skimage.transform.rescale(bispectrum, inp.dl, order=0)
    print('bispectrum: ', bispectrum, flush=True)
    print('bispectrum.shape: ', bispectrum.shape, flush=True)


    #save interpolated bispectrum
    pickle.dump(bispectrum, open(f'bispectra/linear_interpolated_bispectrum_comp{inp.comp}_ellmax{inp.ellmax}_{Nl}origbins_term{term}.p', 'wb'))
    print(f'saved bispectra/linear_interpolated_bispectrum_comp{inp.comp}_ellmax{inp.ellmax}_{Nl}origbins_term{term}.p', flush=True)

    return bispectrum