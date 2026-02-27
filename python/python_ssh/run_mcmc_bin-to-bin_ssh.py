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

#Create binning scheme
ell_1 = [20, 40, 60, 80, 100, 120, 140, 160, 180]
ell_2 = [39, 59, 79, 99, 119, 139, 159, 179, 199]

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

# Paths mcmc results
table_save_path_latex = f'/home/pdeferra-ext/tables/bin_to_bin_results_{mask_name}{name_suffix}.tex'
plot_save_path = f'/home/pdeferra-ext/figures/bin-to-bin/bin_to_bin_evolution_{mask_name}{name_suffix}.pdf'
convergence_save_path = f'/home/pdeferra-ext/figures/bin-to-bin/bin_to_bin_convergence_{mask_name}{name_suffix}.pdf'


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Configuration for bin-to-bin fitting (Gaussian priors + cross-only pairs)
from functions import set_gaussian_priors

# Apply the same Gaussian priors:
# - beta_s ~ N(-3.1, 0.18)
# - beta_d ~ N(1.55, 0.05)
# - alpha_s ~ N(-3.0, 0.30)
# - alpha_d ~ N(-2.48, 0.20)  (average of Planck alpha_EE ~ -2.42 and alpha_BB ~ -2.54)
#   T_d fixed at 19.6 K inside the dust scaling (implicit delta prior)
set_gaussian_priors({
    'beta_s': (-3.1, 0.30),
    # 'beta_d': (1.55, 0.05),
    # 'alpha_s': (-3.0, 0.30),
    # 'alpha_d': (-2.48, 0.20),
})

fit_mode_btb = 'bin-to-bin'  # Use 'bin-to-bin' mode

# Multipole range
ell_min_btb = 20
ell_max_btb = 200

# Sampler configuration
nwalkers_btb = 100
ninter_btb = 15000
discard_fraction_btb = 0.5

# Fit synchrotron, dust, and cross components per bin
fit_components_btb = ('sync', 'dust', 'cross')


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

# band_pairs_btb = band_pairs_cross_all

band_pairs_btb = 'all'

# Load corrected spectra
spectra_dict_btb = functions.read_corrected_cls(path_corrected_spectra, band_list_fit)


'''
# =====================
# EE and BB
# =====================
'''
# Prepare EE data and run bin-to-bin MCMC (cross-only pairs)
fit_data_EE = functions.prepare_mcmc_data(
    spectra_dict_btb,
    band_list=band_list_fit,
    modes=['EE'],
    ell_min=ell_min_btb,
    ell_max=ell_max_btb,
    band_pairs=band_pairs_btb
)

# Run bin-to-bin MCMC for EE mode
samplers_EE, samples_full_EE, samples_free_EE, param_names_btb, chi2_reduced_EE = functions.run_mcmc(
    fit_data=fit_data_EE,
    fit_components=fit_components_btb,
    fit_c_terms=False,
    nwalkers=nwalkers_btb,
    ninter=ninter_btb,
    discard_fraction=discard_fraction_btb,
    verbose=True,
    fit_mode=fit_mode_btb,
    color_correction=True
)

# Prepare BB data and run bin-to-bin MCMC (cross-only pairs)
fit_data_BB = functions.prepare_mcmc_data(
    spectra_dict_btb,
    band_list=band_list_fit,
    modes=['BB'],
    ell_min=ell_min_btb,
    ell_max=ell_max_btb,
    band_pairs=band_pairs_btb
)

# Run bin-to-bin MCMC for BB mode
samplers_BB, samples_full_BB, samples_free_BB, param_names_btb, chi2_reduced_BB = functions.run_mcmc(
    fit_data=fit_data_BB,
    fit_components=fit_components_btb,
    fit_c_terms=False,
    nwalkers=nwalkers_btb,
    ninter=ninter_btb,
    discard_fraction=discard_fraction_btb,
    verbose=True,
    fit_mode=fit_mode_btb,
    color_correction=True
)

# Generate LaTeX table
table_latex = functions.create_bin_to_bin_table(
    fit_data_EE=fit_data_EE,
    fit_data_BB=fit_data_BB,
    samples_free_list_EE=samples_free_EE,
    samples_free_list_BB=samples_free_BB,
    param_names=param_names_btb,
    ell1=ell_1,
    ell2=ell_2,
    save_path=table_save_path_latex,
    format='latex'
)

# Generate ASCII table for quick viewing
table_ascii = functions.create_bin_to_bin_table(
    fit_data_EE=fit_data_EE,
    fit_data_BB=fit_data_BB,
    samples_free_list_EE=samples_free_EE,
    samples_free_list_BB=samples_free_BB,
    param_names=param_names_btb,
    ell1=ell_1,
    ell2=ell_2,
    save_path=None,
    format='ascii'
)

print(table_ascii)

# Plot parameter evolution with ell

fig = functions.plot_bin_to_bin_results(
    fit_data_EE=fit_data_EE,
    fit_data_BB=fit_data_BB,
    samples_free_list_EE=samples_free_EE,
    samples_free_list_BB=samples_free_BB,
    param_names=param_names_btb,
    chi2_reduced_EE=chi2_reduced_EE,
    chi2_reduced_BB=chi2_reduced_BB,
    save_path=plot_save_path,
    figsize=(14, 10)
)

fig_convergence = functions.plot_bin_to_bin_convergence(
    samplers_EE=samplers_EE,
    samplers_BB=samplers_BB,
    ell_1=ell_1,
    ell_2=ell_2,
    ninter=ninter_btb,
    discard_fraction=discard_fraction_btb,
    save_path=convergence_save_path,
    figsize=(14, 10)
)