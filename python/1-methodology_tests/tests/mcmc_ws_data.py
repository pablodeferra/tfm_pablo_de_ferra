#%%
import numpy as np
import sys

sys.path.append('../') 
from data import data, path_map, masks, path_masks
import functions

galcuts = ['10']
freq_auto = ['11', '23', '30']
freq_cross = ['11-23', '11-30', '23-30']

cl_auto = np.zeros([len(galcuts), len(freq_auto), 7, 102])
cl_cross = np.zeros([len(galcuts), len(freq_cross), 7, 102])
error_auto = np.zeros([len(galcuts), len(freq_auto), 7, 102])
error_cross = np.zeros([len(galcuts), len(freq_cross), 7, 102])

for ii in range(len(galcuts)):
    for jj in range(len(freq_auto)):
        cl_auto[ii,jj] = np.loadtxt('/home/pablo/Desktop/Paper/1_wide_survey/data/spectra/cl_' + freq_auto[jj] + 'ghz_galcut' + galcuts[ii] + '.txt', skiprows=1)
        error_auto[ii,jj] = np.loadtxt('/home/pablo/Desktop/Paper/1_wide_survey/data/errorbars/errorbar_' + freq_auto[jj] + 'ghz_galcut' + galcuts[ii] + '.txt', skiprows=1)
    
    for jj in range(len(freq_cross)):
        cl_cross[ii,jj] = np.loadtxt('/home/pablo/Desktop/Paper/1_wide_survey/data/spectra/cross_' + freq_cross[jj] + 'ghz_galcut' + galcuts[ii] + '.txt', skiprows=1)
        error_cross[ii,jj] = np.loadtxt('/home/pablo/Desktop/Paper/1_wide_survey/data/errorbars/errorbar_cross_' + freq_cross[jj] + 'ghz_galcut' + galcuts[ii] + '.txt', skiprows=1)

ell_ws = cl_auto[0, 0, 0, :]


galcut_idx = 0

ws_spectra = {}

band_map_auto = {'11': 0, '23': 1, '30': 2}
cross_map = {'11-23': 0, '11-30': 1, '23-30': 2}

for freq in freq_auto:
    freq_idx = band_map_auto[freq]
    key = f"{freq}_{freq}"
    
    # Auto-spectrum
    ell_data = cl_auto[galcut_idx, freq_idx, 0, :]  # ell
    ee_data = cl_auto[galcut_idx, freq_idx, 2, :]   # EE 
    bb_data = cl_auto[galcut_idx, freq_idx, 3, :]   # BB 
    
    # Errors
    ee_err = error_auto[galcut_idx, freq_idx, 2, :]
    bb_err = error_auto[galcut_idx, freq_idx, 3, :]
    
    ws_spectra[key] = {
        'ell_eff': ell_data,
        'ell1': ell_data - 5,
        'ell2': ell_data + 5,
        'EE': {'SPECTRUM': ee_data, 'ERROR': ee_err},
        'BB': {'SPECTRUM': bb_data, 'ERROR': bb_err}
    }

cross_pairs = [('11', '23'), ('11', '30'), ('23', '30')]
for i, (freq1, freq2) in enumerate(cross_pairs):
    key = f"{freq1}_{freq2}"
    
    # Cross-spectrum
    ell_data = cl_cross[galcut_idx, i, 0, :]  # ell
    ee_data = cl_cross[galcut_idx, i, 2, :]   # EE 
    bb_data = cl_cross[galcut_idx, i, 3, :]   # BB
    
    # Errors
    ee_err = error_cross[galcut_idx, i, 2, :]
    bb_err = error_cross[galcut_idx, i, 3, :]
    
    ws_spectra[key] = {
        'ell_eff': ell_data,
        'ell1': ell_data - 5,
        'ell2': ell_data + 5,
        'EE': {'SPECTRUM': ee_data, 'ERROR': ee_err},
        'BB': {'SPECTRUM': bb_data, 'ERROR': bb_err}
    }

ws_band_list = ['11', '23', '30']

fit_mode = 'BB'

# Save path of the figure
# save_path = f'/home/pablo/Desktop/master/tfm/figures/1-methodology_tests/corner/corner_sync_{fit_mode}_WS.pdf'
save_path = None

ws_fit_data = functions.prepare_mcmc_data(
    ws_spectra,
    band_list=ws_band_list,
    modes=[fit_mode],
    ell_min=30,
    ell_max=200,
    band_pairs='all'
)
ws_mcmc_sampler, ws_samples_full, ws_samples_free, ws_param_map, chi2_reduced = functions.run_mcmc(
    ws_fit_data,
    fit_components=('sync'), 
    fit_c_terms=True,
    nwalkers=200,
    ninter=5000,  
    discard_fraction=0.5,
    verbose=True,  # Show progress bar
    fit_mode='power-law',
    color_correction=True,
    cov_matrix=None,
    n_processes=18,
)


# Plot and save the corner plot
fig = functions.plot_corner(ws_samples_free, ws_param_map, save_path=save_path, title=f'WS, {fit_mode} mode')


