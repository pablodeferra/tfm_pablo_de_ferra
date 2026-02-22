#%%
import os

# CRITICAL: Set to 1 thread per process BEFORE imports!
# MCMC parallelizes across walkers (processes), not threads
# Setting >1 causes massive thread oversubscription and slowdown
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import functions
from data import data, masks
import healpy as hp
import numpy as np
from astropy.io import fits

# Number of parallel processes for MCMC (set to control CPU usage)
# None = auto (uses min(available_cores, nwalkers//2))
# Set to specific number to limit usage, e.g., 20 on a 50-core machine
n_processes = 18  # Adjust this for your 50-core machine


# Default configuration
nside = 512
n_sim = 100

quijote_bands = ['11', '13', '17', '19']
wmap_bands = ['23', '33', '41', '61', '94']
planck_bands = ['30', '44', '70', '100', '143', '217', '353']

band_list = quijote_bands + wmap_bands + planck_bands

name_suffix = '_full_bin_20-199'
# name_suffix = '__TEST__'

mask_select = masks['QUIJOTE_galcut']['galcut10']
mask_name = mask_select['name']
use_simulated_maps = False
use_white_noise = False
use_noise = False # Use noise simulations instead of the HMDM
out_path = '/home/pablo/Desktop/master/tfm/spectra/'
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

# mask = hp.read_map(mask_select['path'])

#Create binning scheme
ell_1 = [20, 40, 60, 80, 100, 120, 140, 160, 180]
ell_2 = [39, 59, 79, 99, 119, 139, 159, 179, 199]


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
ell_max = 120

# Sampler configuration
nwalkers = 200
ninter = 25000

# nwalkers = 200
# ninter = 10000


discard_fraction = 0.5


# Components to fit in the power-law model
fit_components = (
    'sync',
    'dust',
    'cross'
)

# Bands to include in the fit
wmap_fit_bands = ['23', '33']
planck_fit_bands = ['30', '100', '143', '217', '353']

# Two band configurations: with and without QUIJOTE
band_configs = {
    'WMAP+Planck': wmap_fit_bands + planck_fit_bands,
    'QUIJOTE+WMAP+Planck': ['11'] + wmap_fit_bands + planck_fit_bands,
}


# Use 'all' to include both autos and crosses within band_list_fit
band_pairs = 'all'

fit_c_terms = False

# Save paths for corner plots
components_str = '_'.join(fit_components)
c_terms_str = '_c_terms' if fit_c_terms else ''

# Collect all results for the final table
results_list = []

# =====================================================================
# Loop over band configurations and modes
# =====================================================================
for config_label, band_list_fit in band_configs.items():
    config_short = config_label.replace('+', '+')  # keep as-is for file names
    print(f"\n{'='*60}")
    print(f"  Configuration: {config_label}  |  Bands: {band_list_fit}")
    print(f"{'='*60}\n")

    # Load corrected spectra for this band set
    spectra_dict = functions.read_corrected_cls(path_corrected_spectra, band_list_fit)

    for mode in ['EE', 'BB']:
        print(f"\n--- {config_label} / {mode} ---\n")

        # Save path for the corner plot
        save_path_corner = (
            f'/home/pablo/Desktop/master/tfm/figures/corner/'
            f'corner_{mask_name}_{components_str}_{mode}{name_suffix}_{config_short}{c_terms_str}.pdf'
        )

        # Prepare MCMC data
        fit_data = functions.prepare_mcmc_data(
            spectra_dict,
            band_list=band_list_fit,
            modes=[mode],
            ell_min=ell_min,
            ell_max=ell_max,
            band_pairs=band_pairs
        )

        # Run MCMC
        sampler, samples_full, samples_free, param_map, chi2_reduced = functions.run_mcmc(
            fit_data=fit_data,
            fit_components=fit_components,
            fit_c_terms=fit_c_terms,
            nwalkers=nwalkers,
            ninter=ninter,
            discard_fraction=discard_fraction,
            verbose=True,
            fit_mode=fitting_mode,
            color_correction=True,
            cov_matrix=None,
            n_processes=n_processes,
        )

        # Plot corner
        functions.plot_corner(samples_free, param_map, save_path_corner, title=f'{config_label} — {mode}')

        # Store result for table
        results_list.append({
            'data_label': config_label,
            'mode': mode,
            'samples_free': samples_free,
            'param_map': param_map,
            'chi2_reduced': chi2_reduced,
        })

# =====================================================================
# Generate LaTeX table
# =====================================================================
table_save_path = f'/home/pablo/Desktop/master/tfm/tables/table_{mask_name}_ell{ell_min}-{ell_max}_QUIJOTE_mean_noise.tex'
os.makedirs(os.path.dirname(table_save_path), exist_ok=True)

table_latex = functions.create_fitting_results_table(
    results_list,
    save_path=table_save_path,
    label='tab:fit_results_c_terms' if fit_c_terms else 'tab:fit_results',
    ell_range=f'{ell_min}-{ell_max}',
    mask_name=mask_name,
    include_c_terms=fit_c_terms,
)

print("\n" + table_latex)
