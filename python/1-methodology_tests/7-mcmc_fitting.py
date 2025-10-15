#%%
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from data import data, path_map, masks, path_masks
import functions 
import emcee
import corner
import matplotlib.pyplot as plt
from scipy.constants import c,h,k
import multiprocessing as mp
from scipy.stats import gaussian_kde

# Bands to use
quijote_bands = ['11']
wmap_bands = ['23', '33', '41', '61', '94']
planck_bands = ['30', '44', '70', '100', '143', '217', '353']

band_list = quijote_bands + wmap_bands + planck_bands

mask_select = masks['quijote_galcut']['galcut10']
mask_name = mask_select['name']
out_path = '/home/pablo/Desktop/master/tfm/spectra/'
path_corrected_spectra = os.path.join(out_path, f'corrected_power_spectra_{mask_name}.fits')

# Read all spectra
spectra_dict = functions.read_corrected_cls(path_corrected_spectra, band_list)

# -------------------------------
# Fitting configuration
# -------------------------------

# Choose 'EE' or 'BB' here to select which mode to prepare and fit
fit_mode = 'BB'

# Fit ell range
ell_min = 30
ell_max = 200

nwalkers = 100
ninter = 10000
discard_fraction = 0.5

fit_components = (
    'sync', 
    'dust', 
    'cross'
)

band_pairs = [
    '11_11', '23_23', '30_30',
    '11_23', '11_30', '23_30',

    # '100_100', '143_143', '217_217', '353_353',
    # '100_143', '100_217', '100_353', '143_217', '143_353', '217_353',

]

band_pairs = 'all'

fit_c_terms = False

# Save path of the figure
components_str = '_'.join(fit_components)
save_path = f'/home/pablo/Desktop/master/tfm/figures/1-methodology_tests/corner/corner_{components_str}_{fit_mode}.pdf'

# save_path = None


# Prepare data first
fit_data = functions.prepare_mcmc_data(
    spectra_dict,
    band_list=band_list,
    modes=[fit_mode],
    ell_min=ell_min,
    ell_max=ell_max,
    band_pairs=band_pairs
)

# Run MCMC
sampler, samples_full, samples_free, param_map = functions.run_mcmc(
    fit_data=fit_data,
    fit_components=fit_components,
    fit_c_terms=fit_c_terms,
    nwalkers=nwalkers,
    ninter=ninter,
    discard_fraction=discard_fraction
)

# Plot and save the corner plot
fig = functions.plot_corner(samples_free, param_map, save_path=save_path)

