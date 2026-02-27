#%%
import functions
import os
from data_ssh import data, path_map, masks, path_masks, color_corrections
import healpy as hp
import numpy as np
from astropy.io import fits

# Default configuration
nside = 512
n_sim = 100

quijote_bands = ['11', '13', '17', '19']
wmap_bands = ['23', '33', '41', '61', '94']
planck_bands = ['30', '44', '70', '100', '143', '217', '353']


band_list = quijote_bands + wmap_bands + planck_bands

name_suffix = '_full_bin_20-199'

mask_select = masks['QUIJOTE_galcut']['galcut10']
mask_name = mask_select['name']
use_simulated_maps = False
use_white_noise = False
use_noise = False # Use noise simulations instead of the HMDM
out_path = '/home/pdeferra-ext/spectra/'
path_spectra = os.path.join(out_path, f'power_spectra_{mask_name}{name_suffix}.fits')
if use_noise:
    path_hmdm_spectra = os.path.join(out_path, f'power_spectra_{mask_name}_noise_sim{name_suffix}.fits')
else:
    path_hmdm_spectra = os.path.join(out_path, f'power_spectra_{mask_name}_hmdm{name_suffix}.fits')
path_avg_std_skyplusnoise = os.path.join(out_path, f'spectra_avg_std_{mask_name}_avg_std{n_sim}_skyplusnoise{name_suffix}.fits')
path_avg_std_noise = os.path.join(out_path, f'spectra_avg_std_{mask_name}_avg_std{n_sim}_noise{name_suffix}.fits')
# Store per-simulation spectra compressed on disk (Astropy supports .fits.gz transparently)
path_full_skyplusnoise = os.path.join(out_path, f'spectra_full_{mask_name}_{n_sim}_skyplusnoise{name_suffix}.fits.gz')
path_full_noise = os.path.join(out_path, f'spectra_full_{mask_name}_{n_sim}_noise{name_suffix}.fits.gz')
path_corrected_spectra = os.path.join(out_path, f'corrected_power_spectra_{mask_name}{name_suffix}.fits')


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# -------------------------------
# Fitting configuration (power-law with Gaussian priors and cross-only pairs)
# -------------------------------
from functions import set_gaussian_priors

# Gaussian priors:
# - beta_s ~ N(-3.1, 0.18)  synchrotron spectral index (frequency)
# - beta_d ~ N(1.55, 0.05)  dust spectral index (frequency)
# - alpha_s ~ N(-3.0, 0.30) synchrotron ell-slope (same for EE/BB)
# - alpha_d ~ N(-2.48, 0.20) dust ell-slope (average of Planck ~-2.42 EE and ~-2.54 BB)
#   Dust temperature T_d is fixed internally at 19.6 K (effectively a delta prior).

# set_gaussian_priors({
#     'beta_s': (-3.1, 0.18),
#     'beta_d': (1.55, 0.05),
#     'alpha_s': (-3.0, 0.30),
#     'alpha_d': (-2.48, 0.20),
# })

set_gaussian_priors(None)

fitting_mode = 'power-law'

# Multipole range
ell_min = 30
ell_max = 200

# Sampler configuration
# nwalkers = 200
# ninter = 25000

nwalkers = 100
ninter = 1000

discard_fraction = 0.5

# Components to fit in the power-law model
fit_components = (
    'sync',
    'dust',
    'cross'
)

# Bands to include in the fit (auto + cross among these)
# QUIJOTE: 11, 13
# WMAP: K(23), Ka(33)
# Planck: LFI 30, HFI 100/143/217/353
quijote_fit_bands = ['11']
wmap_fit_bands = ['23', '33']
planck_fit_bands = ['30', '100', '143', '217', '353']
band_list_fit = wmap_fit_bands + planck_fit_bands

# # Build cross-only band pairs across all bands (exclude autos)
# band_pairs_cross_all = []
# for i in range(len(band_list)):
#     for j in range(i + 1, len(band_list)):
#         band_pairs_cross_all.append(f"{band_list[i]}_{band_list[j]}")

# # Use cross-only pairs
# band_pairs = band_pairs_cross_all

# Use 'all' to include both autos and crosses within band_list_fit
band_pairs = 'all'

fit_c_terms = False

# Save paths for corner plots
components_str = '_'.join(fit_components)
save_path_EE = f'/home/pdeferra-ext/figures/corner/corner_{mask_name}_{components_str}_EE{name_suffix}.pdf'
save_path_BB = f'/home/pdeferra-ext/figures/corner/corner_{mask_name}_{components_str}_BB{name_suffix}.pdf'
save_path_EE_BB = f'/home/pdeferra-ext/figures/corner/corner_{mask_name}_{components_str}_EE-BB{name_suffix}.pdf'


# Load corrected spectra
spectra_dict = functions.read_corrected_cls(path_corrected_spectra, band_list_fit)


'''
# =====================
# EE
# =====================
'''

# Prepare EE data and run MCMC (power-law with prior and cross-only pairs)
fit_data_EE = functions.prepare_mcmc_data(
    spectra_dict,
    band_list=band_list_fit,
    modes=['EE'],
    ell_min=ell_min,
    ell_max=ell_max,
    band_pairs=band_pairs
)


sampler_EE, samples_full_EE, samples_free_EE, param_map_EE, chi2_reduced_EE = functions.run_mcmc(
    fit_data=fit_data_EE,
    fit_components=fit_components,
    fit_c_terms=fit_c_terms,
    nwalkers=nwalkers,
    ninter=ninter,
    discard_fraction=discard_fraction,
    verbose=True,
    fit_mode=fitting_mode,
    color_correction=True,
    cov_matrix=None,
)

# Plot and save the corner plot
fig_EE = functions.plot_corner(samples_free_EE, param_map_EE, save_path_EE, title=f'EE Mode')


'''
# =====================
# BB
# =====================
'''

# Prepare BB data and run MCMC (power-law with prior and cross-only pairs)
fit_data_BB = functions.prepare_mcmc_data(
    spectra_dict,
    band_list=band_list_fit,
    modes=['BB'],
    ell_min=ell_min,
    ell_max=ell_max,
    band_pairs=band_pairs
)

# Run MCMC with the selected fitting mode
sampler_BB, samples_full_BB, samples_free_BB, param_map_BB, chi2_reduced_BB = functions.run_mcmc(
    fit_data=fit_data_BB,
    fit_components=fit_components,
    fit_c_terms=fit_c_terms,
    nwalkers=nwalkers,
    ninter=ninter,
    discard_fraction=discard_fraction,
    verbose=True,  # Show progress bar
    fit_mode=fitting_mode,
    color_correction=True,
    cov_matrix=None,
)

# Plot and save the corner plot
fig_BB = functions.plot_corner(samples_free_BB, param_map_BB, save_path_BB, title=f'BB Mode')


'''
# =====================
# Joint EE-BB
# =====================
'''

# Prepare EE-BB data and run MCMC
fit_data_EE_BB = functions.prepare_mcmc_data(
    spectra_dict,
    band_list=band_list_fit,
    modes=['EE', 'BB'],
    ell_min=ell_min,
    ell_max=ell_max,
    band_pairs=band_pairs
)

# Run MCMC with the selected fitting mode
sampler_EE_BB, samples_full_EE_BB, samples_free_EE_BB, param_map_EE_BB, chi2_reduced_EE_BB = functions.run_mcmc(
    fit_data=fit_data_EE_BB,
    fit_components=fit_components,
    fit_c_terms=fit_c_terms,
    nwalkers=nwalkers,
    ninter=ninter,
    discard_fraction=discard_fraction,
    verbose=True,  # Show progress bar
    fit_mode=fitting_mode,
    color_correction=True,
    joint_analysis = True,
    cov_matrix=None,
)

# Plot and save the corner plot
fig_joint = functions.plot_corner(
    samples_free_EE_BB, param_map_EE_BB,save_path_EE_BB, title='Joint EE-BB Analysis'
)
