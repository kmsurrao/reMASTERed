import healpy as hp
import numpy as np

lmax = 1000
tsz_map = 1000*hp.read_map('/global/cscratch1/sd/kmsurrao/halosky/scripts/maps/tsz_00002.fits')
cmb_map = hp.read_map('/global/cscratch1/sd/kmsurrao/NILC-Parameter-Pipeline-Outputs/maps/0_cmb_map.fits')
wt_map = hp.read_map('/global/cscratch1/sd/kmsurrao/NILC-Parameter-Pipeline-Outputs/wt_maps/CMB/4_weightmap_freq0_scale5_component_CMB.fits')
# wt_map = hp.remove_monopole(wt_map) #use to test if arbitrary map with no monopole has al0=0 for l!=0

tsz_alm = hp.map2alm(tsz_map, lmax=lmax)
cmb_alm = hp.map2alm(cmb_map, lmax=lmax)
wt_alm = hp.map2alm(wt_map, lmax=lmax)

def expectation_alm(alm, l, lmax):
    tot = 0.+0.j
    tot += alm[hp.Alm.getidx(lmax,l,0)]
    for i in range(1,l+1):
        if i%2==0:
            tot += np.real(alm[hp.Alm.getidx(lmax,l,i)])
        else:
            tot += 1.j*np.imag(alm[hp.Alm.getidx(lmax,l,i)])
    return tot/(2*l+1)

for i, alm in enumerate([tsz_alm, cmb_alm, wt_alm]):
    print()
    print('****************************************')
    if i==0:
        print('tSZ')
    elif i==1:
        print('CMB')
    elif i==2:
        print('weight map')
    for tmp in range(3):
        print()
        for l in [0, 10, 11, 50, 100, 500, 1000]:
            if tmp==0:
                al0 = alm[hp.Alm.getidx(lmax,l,0)]
                print(f'a_{l},0={al0}')
            # elif tmp==1:
            #     #check if alm is just mean of map multiplied by some factor
            #     if i==0:
            #         map_mean = np.mean(tsz_map)
            #     elif i==1:
            #         map_mean = np.mean(cmb_map)
            #     else:
            #         map_mean = np.mean(wt_map)
            #     alm_const_factor = alm[hp.Alm.getidx(lmax,l,0)]/map_mean
            #     print(f'alm is mean multiplied by {alm_const_factor} for l={l}')
            elif tmp==2:
                print(f'<a{l},m> = ', expectation_alm(alm, l, lmax))
        if tmp==2:
            print(alm)
