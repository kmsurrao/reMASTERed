import matplotlib.pyplot as plt
import string
import healpy as hp
import numpy as np

def plot_mask(inp, mask_data, base_dir):

    '''
    PARAMETERS
    inp: Info() object, contains information about input parameters
    mask_data: list, contains [map, mask, masked map, and correlation coefficient]
    base_dir: str, directory to save plots
    '''

    font = {'size'   : 20, 'family':'STIXGeneral'}
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.sans-serif': ['Computer Modern']})
    plt.rc_context({'axes.autolimit_mode': 'round_numbers'})

    map_, mask, masked_map, corr = mask_data
    ells = np.arange(inp.ellmax+1)
    fig, axs = plt.subplots(2,2, figsize=(9,6))
    axs = axs.flatten()
    for n, ax in enumerate(axs):
        plt.axes(ax)
        if n==0:
            hp.mollview(map_, fig=1, hold=True, title='', format='%.03g')
        elif n==1:
            hp.mollview(mask, fig=2, hold=True, title='', format='%.03g', min=0.0, max=1.0)
        elif n==2:
            hp.mollview(mask*map_, fig=3, hold=True, title='', format='%.03g', min=np.amin(map_), max=np.amax(map_))
        else:
            plt.plot(ells[2:], corr[2:])
            plt.xlabel(r'$\ell$')
            plt.ylabel(r'$r_{\ell}$')
            plt.grid()
        plt.text(-0.1, 1.05, string.ascii_uppercase[n], transform=ax.transAxes, 
            size=18, weight='bold')
        if n==0 or n==2:
            plt.text(0.47, -0.02, '[K]', transform=ax.transAxes,size=12)
            
    plt.savefig(f'{base_dir}/maps_{inp.comp}.pdf')
    print(f'saved {base_dir}/maps_{inp.comp}.pdf', flush=True)
    return