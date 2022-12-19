import matplotlib.pyplot as plt
import string
import healpy as hp
import numpy as np

def plot_remastered(inp, remastered_data, base_dir, start=2, logx=False, logy=False):
    
    '''
    PARAMETERS
    inp: Info() object, contains information about input parameters
    remastered_data: list, contains [aa_ww_term, aw_aw_term, w_aaw_term, a_waw_term, aaww_term, directly_computed, remastered]
    base_dir: str, directory to save plots
    start: int, ell at which to start plotting
    logx: Bool, if True plot on log scale for x
    logy: Bool, if True plot on log scale for y
    '''

    font = {'size'   : 20, 'family':'STIXGeneral'}
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.sans-serif': ['Computer Modern']})
    plt.rc_context({'axes.autolimit_mode': 'round_numbers'})

    aa_ww_term, aw_aw_term, w_aaw_term, a_waw_term, aaww_term, directly_computed, remastered = remastered_data
    ells = np.arange(inp.ellmax+1)
    unit_conv = 10.**12

    fig, ax = plt.subplots(1,1, figsize=(9,5))
    plt.axes(ax)
    plt.plot(ells[start:], unit_conv*(ells*(ells+1)*aa_ww_term/(2*np.pi))[start:], label=r'$\langle aa \rangle \langle ww \rangle$ term', color='c')
    plt.plot(ells[start:], unit_conv*(ells*(ells+1)*aw_aw_term/(2*np.pi))[start:], label=r'$\langle aw \rangle \langle aw \rangle$ term')
    plt.plot(ells[start:], unit_conv*(ells*(ells+1)*w_aaw_term/(2*np.pi))[start:], label=r'$\langle w \rangle \langle aaw \rangle$ term', color='r')
    plt.plot(ells[start:], unit_conv*(ells*(ells+1)*a_waw_term/(2*np.pi))[start:], label=r'$\langle a \rangle \langle waw \rangle$ term', color='mediumpurple')
    plt.plot(ells[start:], unit_conv*(ells*(ells+1)*directly_computed/(2*np.pi))[start:], label='Directly Computed', color='g')
    plt.plot(ells[start:], unit_conv*(ells*(ells+1)*remastered/(2*np.pi))[start:], label='ReMASTERed', linestyle='dotted', color='m')
    plt.legend(fontsize=12,  bbox_to_anchor=(1.02, 0.7))
    plt.xlabel(r'$\ell$', font=font)
    plt.ylabel(r'$\langle \tilde{D}_{\ell} \rangle $ [$\mu \mathrm{K}^2$]', font=font)
    if logx:
        plt.xscale('log')
    if logy:
        plt.yscale('log')
    plt.grid()
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.tight_layout()
    
    plt.savefig(f'{base_dir}/master_{inp.comp}.pdf')
    print(f'saved {base_dir}/master_{inp.comp}.pdf', flush=True)
    return