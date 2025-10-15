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

    # WS data (filled circles)
    for j, b in enumerate(bands):
        y = cl_auto_ii[j, 2]
        yerr = error_auto_ii[j, 2] * 0
        ax.errorbar(ell, y, yerr=np.abs(yerr), fmt='o', ms=4, color=colors[j], alpha=0.8, label=rf'WS $C_{{\ell}}$ {b} GHz')

    # Simulations
    for j, b in enumerate(bands):
        key = f"{b}_{b}"
        if key in cl_sim_old:
            sim = cl_sim_old[key]['EE']['SPECTRUM']
            simerr = cl_sim_old[key]['EE']['ERROR'] * 0
            ax.errorbar(ell, sim, yerr=np.abs(simerr), fmt='s', ms=4, fillstyle='none', 
                       color=colors[j], alpha=0.8, label=rf'Sim $C_{{\ell}}$ {b} GHz')

    # Data (triangles without fill)  
    # for j, b in enumerate(bands):
    #     key = f"{b}_{b}"
    #     if key in cl_sim_new:
    #         sim = cl_sim_new[key]['EE']['SPECTRUM']
    #         simerr = cl_sim_new[key]['EE']['ERROR'] * 0
    #         ax.errorbar(ell, sim, yerr=np.abs(simerr), fmt='^', ms=6, fillstyle='none',
    #                    color=colors[j], alpha=0.8, label=rf'$C_{{\ell}}$ {b} GHz')

    ax.set_yscale('log')
    ax.set_xlim(20, 210)
    ax.set_ylim(1e-10, 1e-4)
    ax.set_ylim(ylims[0])
    ax.set_title('EE, ' + title, fontsize=16)
    ax.set_xlabel(r'$\ell$', fontsize=14)
    ax.set_ylabel(r'$C_{\ell}\ [mK^{2}]$', fontsize=14)

    # BB
    ax = axes[1]

    # WS data (filled circles)
    for j, b in enumerate(bands):
        y = cl_auto_ii[j, 3]
        yerr = error_auto_ii[j, 3] * 0
        ax.errorbar(ell, y, yerr=np.abs(yerr), fmt='o', ms=4, color=colors[j], alpha=0.8, label=rf'WS $C_{{\ell}}$ {b} GHz')

    # Simulations
    for j, b in enumerate(bands):
        key = f"{b}_{b}"
        if key in cl_sim_old:
            sim = cl_sim_old[key]['BB']['SPECTRUM']
            simerr = cl_sim_old[key]['BB']['ERROR'] * 0
            ax.errorbar(ell, sim, yerr=np.abs(simerr), fmt='s', ms=4, fillstyle='none',
                       color=colors[j], alpha=0.8, label=rf'Sim $C_{{\ell}}$ {b} GHz')

    # Data (triangles without fill)
    # for j, b in enumerate(bands):
    #     key = f"{b}_{b}"
    #     if key in cl_sim_new:
    #         sim = cl_sim_new[key]['BB']['SPECTRUM']
    #         simerr = cl_sim_new[key]['BB']['ERROR'] * 0
    #         ax.errorbar(ell, sim, yerr=np.abs(simerr), fmt='^', ms=6, fillstyle='none',
    #                    color=colors[j], alpha=0.8, label=rf'$C_{{\ell}}$ {b} GHz')

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
    fig.savefig('/home/pablo/Desktop/master/tfm/figures/1-methodology_tests/spectra/autos_11_23_30_WS_Sim.pdf')



def plot_cross_sim(ell, cl_cross_ii, error_cross_ii, cl_sim_old, cl_sim_new, cross_keys, title, ylims):
    """Plot crossspectra (EE and BB) comparing WS measurements with old and new simulations.
    Uses error bars instead of shaded regions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    colors = ['steelblue', 'k', 'goldenrod']

    # EE
    ax = axes[0]

    # WS data (filled circles)
    for j, key in enumerate(cross_keys):
        y = cl_cross_ii[j, 2]
        yerr = error_cross_ii[j, 2] * 0
        ax.errorbar(ell, y, yerr=np.abs(yerr), fmt='o', ms=4, color=colors[j], alpha=0.8, label=rf'WS $C_{{\ell}}$ {key.replace("_","-")} GHz')

    # Simulations
    for j, key in enumerate(cross_keys):
        if key in cl_sim_old:
            sim = cl_sim_old[key]['EE']['SPECTRUM']
            simerr = cl_sim_old[key]['EE']['ERROR'] * 0
            ax.errorbar(ell, sim, yerr=np.abs(simerr), fmt='s', ms=4, fillstyle='none',
                       color=colors[j], alpha=0.8, label=rf'Sim $C_{{\ell}}$ {key.replace("_","-")} GHz')

    # Data (triangles without fill)
    # for j, key in enumerate(cross_keys):
    #     if key in cl_sim_new:
    #         sim = cl_sim_new[key]['EE']['SPECTRUM']
    #         simerr = cl_sim_new[key]['EE']['ERROR'] * 0
    #         ax.errorbar(ell, sim, yerr=np.abs(simerr), fmt='^', ms=6, fillstyle='none',
    #                    color=colors[j], alpha=0.8, label=rf'$C_{{\ell}}$ {key.replace("_","-")} GHz')

    ax.set_yscale('log')
    ax.set_xlim(20, 210)
    ax.set_ylim(1e-10, 1e-4)
    ax.set_ylim(ylims[0])
    ax.set_title('EE, ' + title, fontsize=16)
    ax.set_xlabel(r'$\ell$', fontsize=14)
    ax.set_ylabel(r'$C_{\ell}\ [mK^{2}]$', fontsize=14)

    # BB
    ax = axes[1]

    # WS data (filled circles)
    for j, key in enumerate(cross_keys):
        y = cl_cross_ii[j, 3]
        yerr = error_cross_ii[j, 3] * 0
        ax.errorbar(ell, y, yerr=np.abs(yerr), fmt='o', ms=4, color=colors[j], alpha=0.8, label=rf'WS $C_{{\ell}}$ {key.replace("_","-")} GHz')

    # Simulations
    for j, key in enumerate(cross_keys):
        if key in cl_sim_old:
            sim = cl_sim_old[key]['BB']['SPECTRUM']
            simerr = cl_sim_old[key]['BB']['ERROR'] * 0
            ax.errorbar(ell, sim, yerr=np.abs(simerr), fmt='s', ms=4, fillstyle='none',
                       color=colors[j], alpha=0.8, label=rf'Sim $C_{{\ell}}$ {key.replace("_","-")} GHz')

    # Data (triangles without fill)
    # for j, key in enumerate(cross_keys):
    #     if key in cl_sim_new:
    #         sim = cl_sim_new[key]['BB']['SPECTRUM']
    #         simerr = cl_sim_new[key]['BB']['ERROR'] * 0
    #         ax.errorbar(ell, sim, yerr=np.abs(simerr), fmt='^', ms=6, fillstyle='none',
    #                    color=colors[j], alpha=0.8, label=rf'$C_{{\ell}}$ {key.replace("_","-")} GHz')

    ax.set_yscale('log')
    ax.set_xlim(20, 210)
    ax.set_ylim(1e-10, 1e-4)
    ax.set_ylim(ylims[0])
    ax.set_title('BB, ' + title, fontsize=16)
    ax.set_xlabel(r'$\ell$', fontsize=14)

    axes[1].legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False, fontsize=14)
    plt.tight_layout()
    plt.show()
    fig.savefig('/home/pablo/Desktop/master/tfm/figures/1-methodology_tests/spectra/cross_11_23_30_WS_Sim.pdf')



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
# RAW (UNCORRECTED) POWER SPECTRA PLOTS
# =============================================================================

print("\n" + "="*60)
print("PLOTTING RAW (UNCORRECTED) POWER SPECTRA")
print("="*60)

# Load uncorrected power spectra (both old and new)
path_uncorrected_spectra_old = os.path.join(out_path, f'power_spectra_{mask_name}.fits')
path_uncorrected_spectra_new = os.path.join(out_path, f'power_spectra_{mask_name}_11_23_30.fits')

# Read uncorrected spectra
cl_uncorr_old = functions.read_spectra_from_fits(path_uncorrected_spectra_old, band_list_old, use_white_noise=False)
cl_uncorr_new = functions.read_spectra_from_fits(path_uncorrected_spectra_new, band_list_new, use_white_noise=False)

def plot_auto_uncorrected_with_noise(ell, cl_auto_ii, error_auto_ii, cl_uncorr_old, cl_uncorr_new, cl_noise_old, cl_noise_new, bands, title, ylims):
    """Plot uncorrected autos (EE and BB) with noise spectra in the same figure.
    Left: EE, Right: BB. Noise uses dashed lines with same colors.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    colors = ['steelblue', 'k', 'goldenrod']

    # EE
    ax = axes[0]

    # Old uncorrected (solid lines)
    for j, b in enumerate(bands):
        key = f"{b}_{b}"
        if key in cl_uncorr_old:
            if hasattr(cl_uncorr_old[key]['EE'], 'keys'):  # Check if it has MEAN/STD structure
                uncorr = cl_uncorr_old[key]['EE']['MEAN'] if 'MEAN' in cl_uncorr_old[key]['EE'] else cl_uncorr_old[key]['EE']
            else:
                uncorr = cl_uncorr_old[key]['EE']
            ax.plot(ell, uncorr, 's', ms=4, 
                   color=colors[j], alpha=0.8, label=rf'Sim $C_{{\ell}}$ {b} GHz')

    # New uncorrected (triangles)  
    for j, b in enumerate(bands):
        key = f"{b}_{b}"
        if key in cl_uncorr_new:
            if hasattr(cl_uncorr_new[key]['EE'], 'keys'):  # Check if it has MEAN/STD structure
                uncorr = cl_uncorr_new[key]['EE']['MEAN'] if 'MEAN' in cl_uncorr_new[key]['EE'] else cl_uncorr_new[key]['EE']
            else:
                uncorr = cl_uncorr_new[key]['EE']
            ax.plot(ell, uncorr, '^', ms=4, fillstyle='none',
                   color=colors[j], alpha=0.8, label=rf'$C_{{\ell}}$ {b} GHz')

    # Old noise (dashed lines)
    for j, b in enumerate(bands):
        key = f"{b}_{b}"
        if key in cl_noise_old:
            if hasattr(cl_noise_old[key]['EE'], 'keys') and 'MEAN' in cl_noise_old[key]['EE']:
                noise = cl_noise_old[key]['EE']['MEAN']
            else:
                noise = cl_noise_old[key]['EE']
            ax.plot(ell, noise, ls='-', color=colors[j], alpha=0.8, label=rf'Sim $N_{{\ell}}$ {b} GHz')

    # New noise (dashed lines with markers)
    for j, b in enumerate(bands):
        key = f"{b}_{b}"
        if key in cl_noise_new:
            if hasattr(cl_noise_new[key]['EE'], 'keys') and 'MEAN' in cl_noise_new[key]['EE']:
                noise = cl_noise_new[key]['EE']['MEAN']
            else:
                noise = cl_noise_new[key]['EE']
            ax.plot(ell, noise, ls='--', ms=3, fillstyle='none',
                   color=colors[j], alpha=0.8, label=rf'$N_{{\ell}}$ {b} GHz')

    ax.set_yscale('log')
    ax.set_xlim(20, 210)
    ax.set_ylim(1e-9,1e-4)
    ax.set_title('EE, ' + title, fontsize=16)
    ax.set_xlabel(r'$\ell$', fontsize=14)
    ax.set_ylabel(r'$C_{\ell}\ [mK^{2}]$', fontsize=14)

    # BB
    ax = axes[1]

    # Old uncorrected (solid lines)
    for j, b in enumerate(bands):
        key = f"{b}_{b}"
        if key in cl_uncorr_old:
            if hasattr(cl_uncorr_old[key]['BB'], 'keys'):  # Check if it has MEAN/STD structure
                uncorr = cl_uncorr_old[key]['BB']['MEAN'] if 'MEAN' in cl_uncorr_old[key]['BB'] else cl_uncorr_old[key]['BB']
            else:
                uncorr = cl_uncorr_old[key]['BB']
            ax.plot(ell, uncorr, 's', ms=4,
                   color=colors[j], alpha=0.8, label = rf'Sim $C_{{\ell}}$ {b} GHz')

    # New uncorrected (triangles)
    for j, b in enumerate(bands):
        key = f"{b}_{b}"
        if key in cl_uncorr_new:
            if hasattr(cl_uncorr_new[key]['BB'], 'keys'):  # Check if it has MEAN/STD structure
                uncorr = cl_uncorr_new[key]['BB']['MEAN'] if 'MEAN' in cl_uncorr_new[key]['BB'] else cl_uncorr_new[key]['BB']
            else:
                uncorr = cl_uncorr_new[key]['BB']
            ax.plot(ell, uncorr, '^', ms=4, fillstyle='none',
                   color=colors[j], alpha=0.8, label = rf'$C_{{\ell}}$ {b} GHz')

    # Old noise (dashed lines)
    for j, b in enumerate(bands):
        key = f"{b}_{b}"
        if key in cl_noise_old:
            if hasattr(cl_noise_old[key]['BB'], 'keys') and 'MEAN' in cl_noise_old[key]['BB']:
                noise = cl_noise_old[key]['BB']['MEAN']
            else:
                noise = cl_noise_old[key]['BB']
            ax.plot(ell, noise, ls='-', color=colors[j], alpha=0.8, label = rf'Sim $N_{{\ell}}$ {b} GHz')

    # New noise (dashed lines with markers)
    for j, b in enumerate(bands):
        key = f"{b}_{b}"
        if key in cl_noise_new:
            if hasattr(cl_noise_new[key]['BB'], 'keys') and 'MEAN' in cl_noise_new[key]['BB']:
                noise = cl_noise_new[key]['BB']['MEAN']
            else:
                noise = cl_noise_new[key]['BB']
            ax.plot(ell, noise, ls='--', ms=3, fillstyle='none',
                   color=colors[j], alpha=0.8, label = rf'$N_{{\ell}}$ {b} GHz')


    ax.set_yscale('log')
    ax.set_xlim(20, 210)
    ax.set_ylim(1e-9,1e-4)
    ax.set_title('BB, ' + title, fontsize=16)
    ax.set_xlabel(r'$\ell$', fontsize=14)

    # Legend to the right
    axes[1].legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False, fontsize=12)
    plt.tight_layout()
    plt.show()
    fig.savefig('/home/pablo/Desktop/master/tfm/figures/1-methodology_tests/spectra/spectra_noise_sim_new.pdf')



# Load noise spectra (both old and new) - needed for combined plots
n_sim = 100
path_noise_spectra_old = os.path.join(out_path, f'spectra_avg_std_{mask_name}_avg_std{n_sim}_noise_wn.fits')
path_noise_spectra_new = os.path.join(out_path, f'spectra_avg_std_{mask_name}_avg_std99_noise_11_23_30.fits')

# Read noise spectra (these should have MEAN/STD structure)
cl_noise_old = functions.read_spectra_from_fits(path_noise_spectra_old, band_list_old, use_white_noise=False)
cl_noise_new = functions.read_spectra_from_fits(path_noise_spectra_new, band_list_new, use_white_noise=False)

# Plot uncorrected + noise autos (EE and BB) in the same figure
for ii in range(len(galcuts)):
    bands = freq_auto
    print(f"\nPlotting uncorrected + noise autos for {titles[ii]}...")
    plot_auto_uncorrected_with_noise(ell, cl_auto[ii], error_auto[ii], cl_uncorr_old, cl_uncorr_new, cl_noise_old, cl_noise_new, bands, titles[ii], ylims)

