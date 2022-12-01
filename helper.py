import numpy as np
import healpy as hp
import sys
from input import Info
import py3nj
import time
import pickle
import multiprocessing as mp

def get_wigner6j(lmax):
    '''
    l4 l2 l1
    l3 l5 L
    '''
    num_ls = lmax+1
    l4s = np.repeat(np.arange(num_ls), num_ls**5)
    l2s = np.tile(np.repeat(np.arange(num_ls), num_ls**4), num_ls)
    l1s = np.tile(np.repeat(np.arange(num_ls), num_ls**3), num_ls**2)
    l3s = np.tile(np.repeat(np.arange(num_ls), num_ls**2), num_ls**3)
    l5s = np.tile(np.repeat(np.arange(num_ls), num_ls), num_ls**4)
    Ls = np.tile(np.arange(num_ls), num_ls**5)
    wigner = py3nj.wigner6j(2*l4s, 2*l2s, 2*l1s, 2*l3s, 2*l5s, 2*Ls, ignore_invalid=True)
    wigner_6j_arr = wigner.reshape((num_ls, num_ls, num_ls, num_ls, num_ls, num_ls))
    return wigner_6j_arr




if __name__=="__main__":
    # # main input file containing most specifications 
    # try:
    #     input_file = (sys.argv)[1]
    # except IndexError:
    #     input_file = 'moto.yaml'
    start_time = time.time()
    lmax = 25
    wigner6j = get_wigner6j(lmax)
    pickle.dump(wigner6j, open('wigner6j.p', 'wb'))
    print("--- %s seconds ---" % (time.time() - start_time), flush=True)


