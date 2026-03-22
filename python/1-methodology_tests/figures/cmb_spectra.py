#%%
import sys
import numpy as np

sys.path.append('../')
from data import data, masks, path_map
import healpy as hp
import matplotlib.pyplot as plt

plt.rcParams.update({
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'axes.titlecolor': 'white',
    'legend.facecolor': 'black',
    'legend.edgecolor': 'white',
    'legend.fontsize': 'medium',
    'text.color': 'white',
    'figure.facecolor': 'black',
    'figure.edgecolor': 'white',
    'axes.facecolor': 'black',
    'axes.edgecolor': 'white',
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,  
})

save_path_ppt = '/home/pablo/Desktop/master/tfm/figures_ppt/spectra/'


cmb_path = '/home/pablo/Desktop/master/tfm/spectra/COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt'

cmb_spectra = np.loadtxt(cmb_path, skiprows=1).T

ell = cmb_spectra[0]

fig = plt.figure(figsize=(8,5))

plt.plot(ell, cmb_spectra[3], color='steelblue')
plt.title(r'CMB EE $\Lambda CDM$', fontsize=15)
plt.yscale('log')
plt.xlim(0,300)
plt.xlabel(r'$\ell$', fontsize=15)
plt.ylabel(r'$C_{\ell} \ [\rm \mu K^2]$', fontsize=15)
plt.tight_layout()
plt.show()

fig.savefig(save_path_ppt + 'CMB_EE.png', dpi=300, transparent=True)

fig = plt.figure(figsize=(8,5))

plt.plot(ell, cmb_spectra[4], color='steelblue')
plt.title(r'CMB BB lensing', fontsize=15)
plt.yscale('log')
plt.xlim(0,300)
plt.xlabel(r'$\ell$', fontsize=15)
plt.ylabel(r'$C_{\ell} \ [\rm \mu K^2]$', fontsize=15)
plt.tight_layout()
plt.show()
fig.savefig(save_path_ppt + 'CMB_BB.png', dpi=300, transparent=True)
