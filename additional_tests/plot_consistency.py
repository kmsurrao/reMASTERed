import matplotlib.pyplot as plt
import string
import healpy as hp
import numpy as np

def plot_consistency(inp, data, base_dir, start=2, logx=True, logy=False):
    
    '''
    PARAMETERS
    inp: Info() object, contains information about input parameters
    data: list, contains [lhs_atildea, w_aa_term_atildea, aaw_term_atildea, lhs_wtildea, w_aw_term_wtildea, waw_term_wtildea]
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

    lhs_atildea, w_aa_term_atildea, aaw_term_atildea, lhs_wtildea, w_aw_term_wtildea, waw_term_wtildea = data
    ells = np.arange(inp.ellmax+1)

    fig, (ax1, ax2) = plt.subplots(1, 2, layout="constrained", figsize=(9,4))
    plt.axes(ax1)
    rhs_atildea = w_aa_term_atildea + aaw_term_atildea
    plt.plot(ells[start:], 10**12*w_aa_term_atildea[start:], label=r'$\langle w \rangle \langle aa \rangle$ term', color='c')
    plt.plot(ells[start:], 10**12*aaw_term_atildea[start:], label=r'$\langle aaw \rangle$ term', color='r')
    plt.plot(ells[start:], 10**12*lhs_atildea[start:], label='Directly Computed', color='g')
    plt.plot(ells[start:], 10**12*rhs_atildea[start:], label='reMASTERed', linestyle='dotted', color='m')
    plt.legend(fontsize=12)
    if logx:
        plt.xscale('log')
    if logy:
        plt.yscale('log')
    plt.xlabel(r'$\ell$', font=font)
    plt.ylabel(r'$ \langle C_{\ell}^{\tilde{a}a} \rangle$ [$\mu K^2$]', font=font)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)

    plt.axes(ax2)
    rhs_wtildea = w_aw_term_wtildea + waw_term_wtildea
    plt.plot(ells[start:], 10**6*w_aw_term_wtildea[start:], label=r'$\langle w \rangle \langle aw \rangle$ term')
    plt.plot(ells[start:], 10**6*waw_term_wtildea[start:], label=r'$\langle waw \rangle$ term', color='mediumpurple')
    plt.plot(ells[start:], 10**6*lhs_wtildea[start:], label='Directly Computed', color='g')
    plt.plot(ells[start:], 10**6*rhs_wtildea[start:], label='ReMASTERed', color='m', linestyle='dotted')
    plt.legend(fontsize=12)
    if logx:
        plt.xscale('log')
    if logy:
        plt.yscale('log')
    plt.xlabel(r'$\ell$', font=font)
    plt.ylabel(r'$\langle C_{\ell}^{\tilde{a}w} \rangle$ [$\mu K$]', font=font)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)

    plt.savefig(f'{base_dir}/consistency_{comp}_{inp.ellmax}.pdf')
    print(f'saved {base_dir}/consistency_{comp}_{inp.ellmax}.pdf', flush=True)
    return