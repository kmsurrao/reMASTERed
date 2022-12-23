import numpy as np
import pywigxjpf as wig
import multiprocessing as mp

def compute_3j_symbol_one_ell(l1, ellmax):
    '''
    PARAMETERS
    l1: int, value of first ell for Wigner 3j symbol
    ellmax: int, maximum ell for which to calculate Wigner 3j symbols

    RETURNS
    wig3j_one_ell: 2D numpy array, indexed as wig3j_one_ell[l2,l3], contains all 3j symbols for l1
    '''
    tj0 = lambda l1,l2,l3: wig.wig3jj(2*l1,2*l2,2*l3,0,0,0)
    wig3j_one_ell = np.zeros((ellmax+1, ellmax+1), dtype=np.float32)
    for l2 in range(ellmax+1):
        for l3 in range(ellmax+1):
            wig3j_one_ell[l2,l3] = tj0(l1,l2,l3)
    return wig3j_one_ell

def compute_3j(ellmax):
    '''
    Computes Wigner 3j symbols with multiprocessing

    PARAMETERS
    ellmax: int, maximum ell for which to calculate 3j symbols

    RETURNS
    wig3j: 3D numpy array, contains wigner 3j symbols, indexed as wig3j[l1,l2,l3]
    '''
    wig.wig_table_init(ellmax*2,9)
    wig.wig_temp_init(ellmax*2)
    pool = mp.Pool(16)
    wig3j = pool.starmap(compute_3j_symbol_one_ell, [(l1, ellmax) for l1 in range(ellmax+1)])
    pool.close()
    return np.array(wig3j)

def compute_3j_no_multiprocessing(ellmax):
    '''
    Computes Wigner 3j symbols without the multiprocessing module

    PARAMETERS
    ellmax: int, maximum ell for which to calculate 3j symbols

    RETURNS
    wig3j: 3D numpy array, contains wigner 3j symbols, indexed as wig3j[l1,l2,l3]
    '''
    wig.wig_table_init(ellmax*2,9)
    wig.wig_temp_init(ellmax*2)
    tj0 = lambda l1,l2,l3: wig.wig3jj(2*l1,2*l2,2*l3,0,0,0)
    wig3j = np.zeros((ellmax+1, ellmax+1, ellmax+1), dtype=np.float32)
    for l1 in range(ellmax+1):
        for l2 in range(ellmax+1):
            for l3 in range(ellmax+1):
                wig3j[l1,l2,l3] = tj0(l1,l2,l3)
    return wig3j



