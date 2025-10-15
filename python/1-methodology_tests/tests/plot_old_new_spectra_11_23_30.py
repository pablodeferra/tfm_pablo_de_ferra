#%%
import numpy as np
import os
import sys

sys.path.append('../') 
from data import data, path_map, masks, path_masks
import functions 
import matplotlib.pyplot as plt

plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

galcuts = ['10']
freq_auto = ['11', '23', '30']
freq_cross = ['11-23', '11-30', '23-30']

cl_auto = np.zeros([3, 3, 7, 102])
cl_cross = np.zeros([3, 3, 7, 102])
error_auto = np.zeros([3, 3, 7, 102])
error_cross = np.zeros([3, 3, 7, 102])

for ii in range(len(galcuts)):
    for jj in range(len(freq_auto)):
        cl_auto[ii,jj] = np.loadtxt('/home/pablo/Desktop/Paper/1_wide_survey/data/spectra/cl_' + freq_auto[jj] + 'ghz_galcut' + galcuts[ii] + '.txt', skiprows=1)
        cl_cross[ii,jj] = np.loadtxt('/home/pablo/Desktop/Paper/1_wide_survey/data/spectra/cross_' + freq_cross[jj] + 'ghz_galcut' + galcuts[ii] + '.txt', skiprows=1)
        error_auto[ii,jj] = np.loadtxt('/home/pablo/Desktop/Paper/1_wide_survey/data/errorbars/errorbar_' + freq_auto[jj] + 'ghz_galcut' + galcuts[ii] + '.txt', skiprows=1)
        error_cross[ii,jj] = np.loadtxt('/home/pablo/Desktop/Paper/1_wide_survey/data/errorbars/errorbar_cross_' + freq_cross[jj] + 'ghz_galcut' + galcuts[ii] + '.txt', skiprows=1)


# Bands to use
quijote_bands = ['11']
wmap_bands = ['23', '33', '41', '61', '94']
planck_bands = ['30', '44', '70', '30', '143', '217', '217']

band_list = quijote_bands + wmap_bands + planck_bands

mask_select = masks['quijote_galcut']['galcut10']
mask_name = mask_select['name']
out_path = '/home/pablo/Desktop/master/tfm/spectra/'

# Load original corrected spectra (without suffix)
path_corrected_spectra_old = os.path.join(out_path, f'corrected_power_spectra_{mask_name}.fits')
# Load new corrected spectra from notebook (with _11_23_30 suffix)
path_corrected_spectra_new = os.path.join(out_path, f'corrected_power_spectra_{mask_name}_11_23_30.fits')

save_path = '/home/pablo/Desktop/master/tfm/figures/spectra_test/'

# Read both sets of corrected spectra
band_list_old = quijote_bands + wmap_bands + planck_bands  # Original band list
band_list_new = ['11', '23', '30']  # New band list from notebook

cl_old = functions.read_corrected_cls(path_corrected_spectra_old, band_list_old)
cl_new = functions.read_corrected_cls(path_corrected_spectra_new, band_list_new)

ell = cl_auto[0,0,0]

ylims = [[1e-13, 1e-3]]
ylims_cross = [[1e-11, 1e-5]]
titles = ['$|b|>10^{\circ}$']
fig_names = ['galcut10']


def plot_auto_sim(ell, cl_auto_ii, error_auto_ii, cl_sim_old, cl_sim_new, bands, title, ylims):
    """Plot autos (EE and BB) comparing WS measurements with old and new simulations.
    Left: EE, Right: BB. Uses error bars instead of shaded regions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    colors = ['steelblue', 'k', 'goldenrod']

    # EE
    ax = axes[0]
    for j, b in enumerate(bands):
        y = cl_auto_ii[j, 2]
        yerr = error_auto_ii[j, 2] * 0
        ax.errorbar(ell, y, yerr=np.abs(yerr), fmt='o', ms=4, color=colors[j], alpha=0.8, label=f'WS {b} GHz')

    # Simulations (squares without fill)
    for j, b in enumerate(bands):
        key = f"{b}_{b}"
        if key in cl_sim_old:
            sim = cl_sim_old[key]['EE']['SPECTRUM']
            simerr = cl_sim_old[key]['EE']['ERROR'] * 0
            ax.errorbar(ell, sim, yerr=np.abs(simerr), fmt='s', ms=4, fillstyle='none', 
                       color=colors[j], alpha=0.8, label=f'Sim {b} GHz')

    # Data (triangles without fill)  
    for j, b in enumerate(bands):
        key = f"{b}_{b}"
        if key in cl_sim_new:
            sim = cl_sim_new[key]['EE']['SPECTRUM']
            simerr = cl_sim_new[key]['EE']['ERROR'] * 0
            ax.errorbar(ell, sim, yerr=np.abs(simerr), fmt='^', ms=4, fillstyle='none',
                       color=colors[j], alpha=0.8, label=f'Dat {b} GHz')

    ax.set_yscale('log')
    ax.set_xlim(20, 210)
    ax.set_ylim(1e-10, 1e-4)
    ax.set_ylim(ylims[0])
    ax.set_title('EE, ' + title, fontsize=16)
    ax.set_xlabel(r'$\ell$', fontsize=14)
    ax.set_ylabel(r'$C_{\ell}\ [mK^{2}]$', fontsize=14)

    # BB
    ax = axes[1]
    for j, b in enumerate(bands):
        y = cl_auto_ii[j, 3]
        yerr = error_auto_ii[j, 3] * 0
        ax.errorbar(ell, y, yerr=np.abs(yerr), fmt='o', ms=4, color=colors[j], alpha=0.8, label=f'WS {b} GHz')

    # Simulations (squares without fill)
    for j, b in enumerate(bands):
        key = f"{b}_{b}"
        if key in cl_sim_old:
            sim = cl_sim_old[key]['BB']['SPECTRUM']
            simerr = cl_sim_old[key]['BB']['ERROR'] * 0
            ax.errorbar(ell, sim, yerr=np.abs(simerr), fmt='s', ms=4, fillstyle='none',
                       color=colors[j], alpha=0.8, label=f'Sim {b} GHz')

    # Data (triangles without fill)
    for j, b in enumerate(bands):
        key = f"{b}_{b}"
        if key in cl_sim_new:
            sim = cl_sim_new[key]['BB']['SPECTRUM']
            simerr = cl_sim_new[key]['BB']['ERROR'] * 0
            ax.errorbar(ell, sim, yerr=np.abs(simerr), fmt='^', ms=4, fillstyle='none',
                       color=colors[j], alpha=0.8, label=f'Dat {b} GHz')

    ax.set_yscale('log')
    ax.set_xlim(20, 210)
    ax.set_ylim(1e-10, 1e-4)
    ax.set_ylim(ylims[0])
    ax.set_title('BB, ' + title, fontsize=16)
    ax.set_xlabel(r'$\ell$', fontsize=14)

    # Legend to the right
    axes[1].legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False, fontsize=14)
    plt.tight_layout()
    plt.show()
    # fig.savefig('/home/pablo/Desktop/master/tfm/figures/1-methodology_tests/spectra/autos_11_23_30_WS_Sim.pdf')



def plot_cross_sim(ell, cl_cross_ii, error_cross_ii, cl_sim_old, cl_sim_new, cross_keys, title, ylims):
    """Plot crossspectra (EE and BB) comparing WS measurements with old and new simulations.
    Uses error bars instead of shaded regions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    colors = ['steelblue', 'k', 'goldenrod']

    # EE
    ax = axes[0]
    for j, key in enumerate(cross_keys):
        y = cl_cross_ii[j, 2]
        yerr = error_cross_ii[j, 2] * 0
        ax.errorbar(ell, y, yerr=np.abs(yerr), fmt='o', ms=4, color=colors[j], alpha=0.8, label=f'WS {key.replace("_","-")} GHz')

    # Simulations (squares without fill)
    for j, key in enumerate(cross_keys):
        if key in cl_sim_old:
            sim = cl_sim_old[key]['EE']['SPECTRUM']
            simerr = cl_sim_old[key]['EE']['ERROR'] * 0
            ax.errorbar(ell, sim, yerr=np.abs(simerr), fmt='s', ms=4, fillstyle='none',
                       color=colors[j], alpha=0.8, label=f'Sim {key.replace("_","-")} GHz')

    # Data (triangles without fill)
    for j, key in enumerate(cross_keys):
        if key in cl_sim_new:
            sim = cl_sim_new[key]['EE']['SPECTRUM']
            simerr = cl_sim_new[key]['EE']['ERROR'] * 0
            ax.errorbar(ell, sim, yerr=np.abs(simerr), fmt='^', ms=4, fillstyle='none',
                       color=colors[j], alpha=0.8, label=f'Dat {key.replace("_","-")} GHz')

    ax.set_yscale('log')
    ax.set_xlim(20, 210)
    ax.set_ylim(1e-10, 1e-4)
    ax.set_ylim(ylims[0])
    ax.set_title('EE, ' + title, fontsize=16)
    ax.set_xlabel(r'$\ell$', fontsize=14)
    ax.set_ylabel(r'$C_{\ell}\ [mK^{2}]$', fontsize=14)

    # BB
    ax = axes[1]
    for j, key in enumerate(cross_keys):
        y = cl_cross_ii[j, 3]
        yerr = error_cross_ii[j, 3] * 0
        ax.errorbar(ell, y, yerr=np.abs(yerr), fmt='o', ms=4, color=colors[j], alpha=0.8, label=f'WS {key.replace("_","-")} GHz')

    # Simulations (squares without fill)
    for j, key in enumerate(cross_keys):
        if key in cl_sim_old:
            sim = cl_sim_old[key]['BB']['SPECTRUM']
            simerr = cl_sim_old[key]['BB']['ERROR'] * 0
            ax.errorbar(ell, sim, yerr=np.abs(simerr), fmt='s', ms=4, fillstyle='none',
                       color=colors[j], alpha=0.8, label=f'Sim {key.replace("_","-")} GHz')

    # Data (triangles without fill)
    for j, key in enumerate(cross_keys):
        if key in cl_sim_new:
            sim = cl_sim_new[key]['BB']['SPECTRUM']
            simerr = cl_sim_new[key]['BB']['ERROR'] * 0
            ax.errorbar(ell, sim, yerr=np.abs(simerr), fmt='^', ms=4, fillstyle='none',
                       color=colors[j], alpha=0.8, label=f'Dat {key.replace("_","-")} GHz')

    ax.set_yscale('log')
    ax.set_xlim(20, 210)
    ax.set_ylim(1e-10, 1e-4)
    ax.set_ylim(ylims[0])
    ax.set_title('BB, ' + title, fontsize=16)
    ax.set_xlabel(r'$\ell$', fontsize=14)

    axes[1].legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False, fontsize=14)
    plt.tight_layout()
    plt.show()
    # fig.savefig('/home/pablo/Desktop/master/tfm/figures/1-methodology_tests/spectra/cross_11_23_30_WS_Sim.pdf')



# Plot autos (EE and BB) with simulations
for ii in range(len(galcuts)):
    bands = freq_auto
    plot_auto_sim(ell, cl_auto[ii], error_auto[ii], cl_old, cl_new, bands, titles[ii], ylims)

# Plot crossspectra with simulations
cross_keys = ['11_23', '11_30', '23_30']
for ii in range(len(galcuts)):
    plot_cross_sim(ell, cl_cross[ii], error_cross[ii], cl_old, cl_new, cross_keys, titles[ii], ylims)

#%%

# =============================================================================
# ERROR COMPARISON PLOTS
# =============================================================================

def plot_auto_errors(ell, error_auto_ii, cl_sim_old, cl_sim_new, bands, title, ylims):
    """Plot auto errors (EE and BB) comparing WS measurements with old and new simulations.
    Left: EE, Right: BB. Shows only the error bars/uncertainties.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    colors = ['steelblue', 'k', 'goldenrod']

    # EE Errors
    ax = axes[0]
    for j, b in enumerate(bands):
        y = error_auto_ii[j, 2]
        ax.plot(ell, np.abs(y), 'o', ms=4, color=colors[j], alpha=0.8, label=f'WS {b} GHz')

    # Simulations errors (squares without fill)
    # for j, b in enumerate(bands):
    #     key = f"{b}_{b}"
    #     if key in cl_sim_old:
    #         simerr = cl_sim_old[key]['EE']['ERROR']
    #         ax.plot(ell, np.abs(simerr), 's', ms=4, fillstyle='none', 
    #                color=colors[j], alpha=0.8, label=f'Sim {b} GHz')

    # Data errors (triangles without fill)  
    for j, b in enumerate(bands):
        key = f"{b}_{b}"
        if key in cl_sim_new:
            simerr = cl_sim_new[key]['EE']['ERROR']
            ax.plot(ell, np.abs(simerr), '^', ms=4, fillstyle='none',
                   color=colors[j], alpha=0.8, label=f'Dat {b} GHz')

    ax.set_yscale('log')
    ax.set_xlim(20, 210)
    ax.set_ylim(ylims[0])
    ax.set_title('EE Errors, ' + title, fontsize=16)
    ax.set_xlabel(r'$\ell$', fontsize=14)
    ax.set_ylabel(r'$\sigma(C_{\ell})\ [mK^{2}]$', fontsize=14)

    # BB Errors
    ax = axes[1]
    for j, b in enumerate(bands):
        y = error_auto_ii[j, 3]
        ax.plot(ell, np.abs(y), 'o', ms=4, color=colors[j], alpha=0.8, label=f'WS {b} GHz')

    # Simulations errors (squares without fill)
    # for j, b in enumerate(bands):
    #     key = f"{b}_{b}"
    #     if key in cl_sim_old:
    #         simerr = cl_sim_old[key]['BB']['ERROR']
    #         ax.plot(ell, np.abs(simerr), 's', ms=4, fillstyle='none',
    #                color=colors[j], alpha=0.8, label=f'Sim {b} GHz')

    # Data errors (triangles without fill)
    for j, b in enumerate(bands):
        key = f"{b}_{b}"
        if key in cl_sim_new:
            simerr = cl_sim_new[key]['BB']['ERROR']
            ax.plot(ell, np.abs(simerr), '^', ms=4, fillstyle='none',
                   color=colors[j], alpha=0.8, label=f'Dat {b} GHz')

    ax.set_yscale('log')
    ax.set_xlim(20, 210)
    ax.set_ylim(ylims[0])
    ax.set_title('BB Errors, ' + title, fontsize=16)
    ax.set_xlabel(r'$\ell$', fontsize=14)

    # Legend to the right
    axes[1].legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False, fontsize=14)
    plt.tight_layout()
    plt.show()
    # fig.savefig('/home/pablo/Desktop/master/tfm/figures/1-methodology_tests/spectra/autos_errors_11_23_30_WS_Sim.pdf')


def plot_cross_errors(ell, error_cross_ii, cl_sim_old, cl_sim_new, cross_keys, title, ylims):
    """Plot cross errors (EE and BB) comparing WS measurements with old and new simulations.
    Shows only the error bars/uncertainties.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    colors = ['steelblue', 'k', 'goldenrod']

    # EE Errors
    ax = axes[0]
    for j, key in enumerate(cross_keys):
        y = error_cross_ii[j, 2]
        ax.plot(ell, np.abs(y), 'o', ms=4, color=colors[j], alpha=0.8, label=f'WS {key.replace("_","-")} GHz')

    # Simulations errors (squares without fill)
    # for j, key in enumerate(cross_keys):
    #     if key in cl_sim_old:
    #         simerr = cl_sim_old[key]['EE']['ERROR']
    #         ax.plot(ell, np.abs(simerr), 's', ms=4, fillstyle='none',
    #                color=colors[j], alpha=0.8, label=f'Sim {key.replace("_","-")} GHz')

    # Data errors (triangles without fill)
    for j, key in enumerate(cross_keys):
        if key in cl_sim_new:
            simerr = cl_sim_new[key]['EE']['ERROR']
            ax.plot(ell, np.abs(simerr), '^', ms=4, fillstyle='none',
                   color=colors[j], alpha=0.8, label=f'Dat {key.replace("_","-")} GHz')

    ax.set_yscale('log')
    ax.set_xlim(20, 210)
    ax.set_ylim(ylims[0])
    ax.set_title('EE Errors, ' + title, fontsize=16)
    ax.set_xlabel(r'$\ell$', fontsize=14)
    ax.set_ylabel(r'$\sigma(C_{\ell})\ [mK^{2}]$', fontsize=14)

    # BB Errors
    ax = axes[1]
    for j, key in enumerate(cross_keys):
        y = error_cross_ii[j, 3]
        ax.plot(ell, np.abs(y), 'o', ms=4, color=colors[j], alpha=0.8, label=f'WS {key.replace("_","-")} GHz')

    # Simulations errors (squares without fill)
    # for j, key in enumerate(cross_keys):
    #     if key in cl_sim_old:
    #         simerr = cl_sim_old[key]['BB']['ERROR']
    #         ax.plot(ell, np.abs(simerr), 's', ms=4, fillstyle='none',
    #                color=colors[j], alpha=0.8, label=f'Sim {key.replace("_","-")} GHz')

    # Data errors (triangles without fill)
    for j, key in enumerate(cross_keys):
        if key in cl_sim_new:
            simerr = cl_sim_new[key]['BB']['ERROR']
            ax.plot(ell, np.abs(simerr), '^', ms=4, fillstyle='none',
                   color=colors[j], alpha=0.8, label=f'Dat {key.replace("_","-")} GHz')

    ax.set_yscale('log')
    ax.set_xlim(20, 210)
    ax.set_ylim(ylims[0])
    ax.set_title('BB Errors, ' + title, fontsize=16)
    ax.set_xlabel(r'$\ell$', fontsize=14)

    axes[1].legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False, fontsize=14)
    plt.tight_layout()
    plt.show()
    # fig.savefig('/home/pablo/Desktop/master/tfm/figures/1-methodology_tests/spectra/cross_errors_11_23_30_WS_Sim.pdf')


# Plot auto errors (EE and BB) 
print("\n" + "="*60)
print("PLOTTING ERROR COMPARISONS")
print("="*60)

for ii in range(len(galcuts)):
    bands = freq_auto
    print(f"\nPlotting auto errors for {titles[ii]}...")
    plot_auto_errors(ell, error_auto[ii], cl_old, cl_new, bands, titles[ii], ylims)

# Plot cross errors
for ii in range(len(galcuts)):
    print(f"\nPlotting cross errors for {titles[ii]}...")
    plot_cross_errors(ell, error_cross[ii], cl_old, cl_new, cross_keys, titles[ii], ylims)

