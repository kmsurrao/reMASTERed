import numpy as np
import healpy as hp
import sys
from input import Info
import pywigxjpf as wig
import time
import pickle
import multiprocessing as mp

def compute_3j(ellmax):
    wig3j = np.zeros((ellmax+1, ellmax+1, ellmax+1), dtype=np.float32)
    wig.wig_table_init(ellmax*2,9)
    wig.wig_temp_init(ellmax*2)
    tj0 = lambda l1,l2,l3: wig.wig3jj(2*l1,2*l2,2*l3,0,0,0)
    wig3j = np.zeros((ellmax+1, ellmax+1, ellmax+1))
    for l1 in range(ellmax+1):
        for l2 in range(ellmax+1):
            for l3 in range(ellmax+1):
                wig3j[l1,l2,l3] = tj0(l1,l2,l3)
    return wig3j


if __name__=="__main__":
    ellmax = 300
    t1 = time.time()
    wig3j = compute_3j(ellmax)
    t2 = time.time()
    print("---%s seconds ---" % (t2 - t1), flush=True)




