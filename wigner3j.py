import numpy as np
import pywigxjpf as wig
import multiprocessing as mp

def compute_3j_symbol_one_ell(l1, ell_sum_max):
    '''
    PARAMETERS
    l1: int, value of first ell for Wigner 3j symbol
    ell_sum_max: int, maximum l2,l3 for which to calculate Wigner 3j symbols

    RETURNS
    wig3j_one_ell: 2D numpy array, indexed as wig3j_one_ell[l2,l3], contains all 3j symbols for l1
    '''
    wig.wig_table_init(ell_sum_max*2,9)
    wig.wig_temp_init(ell_sum_max*2)
    tj0 = lambda l1,l2,l3: wig.wig3jj(2*l1,2*l2,2*l3,0,0,0)
    wig3j_one_ell = np.zeros((ell_sum_max+1, ell_sum_max+1), dtype=np.float32)
    for l2 in range(ell_sum_max+1):
        for l3 in range(ell_sum_max+1):
            wig3j_one_ell[l2,l3] = tj0(l1,l2,l3)
    return wig3j_one_ell

def compute_3j(ell_sum_max):
    '''
    Computes Wigner 3j symbols with multiprocessing

    PARAMETERS
    ell_sum_max: int, maximum ell for which to calculate Wigner 3j symbols

    RETURNS
    wig3j: 3D numpy array, contains wigner 3j symbols, indexed as wig3j[l1,l2,l3]
    '''
    pool = mp.Pool(16)
    wig3j = pool.starmap(compute_3j_symbol_one_ell, [(l1, ell_sum_max) for l1 in range(ell_sum_max+1)])
    pool.close()
    return np.array(wig3j)

def compute_3j_no_multiprocessing(ell_sum_max):
    '''
    Computes Wigner 3j symbols without the multiprocessing module

    PARAMETERS 
    ell_sum_max: int, maximum l2,l3 for which to calculate Wigner 3j symbols

    RETURNS
    wig3j: 3D numpy array, contains wigner 3j symbols, indexed as wig3j[l1,l2,l3]
    '''
    wig.wig_table_init(ell_sum_max*2,9)
    wig.wig_temp_init(ell_sum_max*2)
    tj0 = lambda l1,l2,l3: wig.wig3jj(2*l1,2*l2,2*l3,0,0,0)
    wig3j = np.zeros((ell_sum_max+1, ell_sum_max+1, ell_sum_max+1), dtype=np.float32)
    for l1 in range(ell_sum_max+1):
        for l2 in range(ell_sum_max+1):
            for l3 in range(ell_sum_max+1):
                wig3j[l1,l2,l3] = tj0(l1,l2,l3)
    return wig3j



