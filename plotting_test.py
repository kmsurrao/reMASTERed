import numpy as np
import matplotlib.pyplot as plt
print('imports complete', flush=True)


font = {'size'   : 20, 'family':'STIXGeneral'}
print('set font var', flush=True)
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.sans-serif': ['Computer Modern']})
print('updated params', flush=True)
plt.rc_context({'axes.autolimit_mode': 'round_numbers'})
print('rc context', flush=True)

x = np.arange(10)
y = 2*np.arange(10)
plt.plot(x,y)
plt.xlabel(r'$x$', fontsize=20)
plt.ylabel(r'$y$', fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.tight_layout()


plt.savefig('basic_plot.png')
print('saved fig', flush=True)