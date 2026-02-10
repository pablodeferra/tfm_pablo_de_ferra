#%%

import numpy as np
import functions
from data import data

# ============================================================================
# STEP 1: Configuration
# ============================================================================

# Paths to your simulation files (from save_sims_to_fits output)
path_full_skyplusnoise = '/home/pablo/Desktop/master/tfm/spectra/spectra_full_quijote_galcut10_5_skyplusnoise_full_bin_20-199.fits.gz'
path_full_noise = '/home/pablo/Desktop/master/tfm/spectra/spectra_full_quijote_galcut10_5_noise_full_bin_20-199.fits.gz'

# Your band list
quijote_bands = ['11', '13', '17', '19']
wmap_bands = ['23', '33', '41', '61', '94']
planck_bands = ['30', '44', '70', '100', '143', '217', '353']
band_list = quijote_bands + wmap_bands + planck_bands

# ============================================================================
# STEP 2: Prepare your MCMC data
# ============================================================================

# Load your corrected spectra
path_corrected_spectra = '/home/pablo/Desktop/master/tfm/spectra/corrected_power_spectra_quijote_galcut10_full_bin_20-199.fits'
spectra_dict = functions.read_corrected_cls(path_corrected_spectra, band_list)

# Prepare MCMC data for EE mode
fit_data_EE = functions.prepare_mcmc_data(
    spectra_dict,
    band_list=band_list,
    modes=['EE'],
    ell_min=30,
    ell_max=200,
    band_pairs='all'  # or specify which pairs you want
)

# ============================================================================
# STEP 3: MCMC Configurations
# ============================================================================

nwalkers = 100
ninter = 3000  # fast test
discard_fraction = 0.5
fit_components = ('sync', 'dust', 'cross')
fit_c_terms = False
fit_mode = 'power-law'
components_str = '_'.join(fit_components)

# Directory for figures
save_dir_figures = '/home/pablo/Desktop/master/tfm/figures/corner/'
import os
os.makedirs(save_dir_figures, exist_ok=True)

print(f"Configuration: {nwalkers} walkers, {ninter} steps, components={fit_components}")

# ============================================================================
# SECTION A: EE MODE
# ============================================================================
print("\n" + "="*60)
print("SECTION A: EE MODE Analysis")
print("="*60)

# A.1: Data Preparation is already done above (fit_data_EE)

# A.2: Build Covariance
print("\n[EE] Building block-diagonal inverse covariance...")
cov_info_EE = functions.build_block_diagonal_cov_inv(
    path_sims_fits=path_full_skyplusnoise,
    fit_data=fit_data_EE,
    modes=['EE'],
    quijote_bands_11_13=['11', '13'],
    quijote_bands_17_19=['17', '19'],
    path_noise_sims=path_full_noise  # <-- Added noise sims for total error
)

# A.3: Run MCMC (With Covariance)
print("\n[EE] Running MCMC with block-diagonal covariance...")
sampler_EE, samples_full_EE, samples_free_EE, param_map_EE, chi2_EE = functions.run_mcmc(
    fit_data=fit_data_EE,
    fit_components=fit_components,
    fit_c_terms=fit_c_terms,
    nwalkers=nwalkers,
    ninter=ninter,
    discard_fraction=discard_fraction,
    verbose=True,
    fit_mode=fit_mode,
    color_correction=True,
    cov_matrix=cov_info_EE
)
print(f"[EE] Reduced chi2 (with cov): {chi2_EE:.4f}")

# A.5: Plot Corner
save_path_EE = os.path.join(save_dir_figures, f'example_EE_{components_str}.pdf')
functions.plot_corner(samples_free_EE, param_map_EE, save_path=save_path_EE, title='EE Mode (With Covariance)')
print(f"[EE] Corner plot saved to {save_path_EE}")

#%%
# A.4: Run MCMC (Without Covariance)
print("\n[EE] Running MCMC with diagonal covariance (comparison)...")
sampler_EE_diag, samples_full_EE_diag, samples_free_EE_diag, param_map_EE_diag, chi2_EE_diag = functions.run_mcmc(
    fit_data=fit_data_EE,
    fit_components=fit_components,
    fit_c_terms=fit_c_terms,
    nwalkers=nwalkers,
    ninter=ninter,
    discard_fraction=discard_fraction,
    verbose=True,
    fit_mode=fit_mode,
    color_correction=True,
    cov_matrix=None
)
print(f"[EE] Reduced chi2 (no cov): {chi2_EE_diag:.4f}")



save_path_EE_diag = os.path.join(save_dir_figures, f'example_EE_{components_str}_diag.pdf')
functions.plot_corner(samples_free_EE_diag, param_map_EE_diag, save_path=save_path_EE_diag, title='EE Mode (Diagonal Covariance)')
print(f"[EE-Diag] Corner plot saved to {save_path_EE_diag}")


# ============================================================================
# SECTION B: BB MODE
# ============================================================================
print("\n" + "="*60)
print("SECTION B: BB MODE Analysis")
print("="*60)

# B.1: Prepare MCMC data
print("\n[BB] Preparing data...")
fit_data_BB = functions.prepare_mcmc_data(
    spectra_dict,
    band_list=band_list,
    modes=['BB'],
    ell_min=30,
    ell_max=200,
    band_pairs='all'
)

# B.2: Build Covariance
print("\n[BB] Building block-diagonal inverse covariance...")
cov_info_BB = functions.build_block_diagonal_cov_inv(
    path_sims_fits=path_full_skyplusnoise,
    fit_data=fit_data_BB,
    modes=['BB'], # Explicitly for BB
    quijote_bands_11_13=['11', '13'],
    quijote_bands_17_19=['17', '19'],
    path_noise_sims=path_full_noise  # <-- Added noise sims for total error
)

# B.3: Run MCMC (With Covariance)
print("\n[BB] Running MCMC with block-diagonal covariance...")
sampler_BB, samples_full_BB, samples_free_BB, param_map_BB, chi2_BB = functions.run_mcmc(
    fit_data=fit_data_BB,
    fit_components=fit_components,
    fit_c_terms=fit_c_terms,
    nwalkers=nwalkers,
    ninter=ninter,
    discard_fraction=discard_fraction,
    verbose=True,
    fit_mode=fit_mode,
    color_correction=True,
    cov_matrix=cov_info_BB
)
print(f"[BB] Reduced chi2 (with cov): {chi2_BB:.4f}")

# B.4: Run MCMC (Without Covariance)
print("\n[BB] Running MCMC with diagonal covariance (comparison)...")
sampler_BB_diag, samples_full_BB_diag, samples_free_BB_diag, param_map_BB_diag, chi2_BB_diag = functions.run_mcmc(
    fit_data=fit_data_BB,
    fit_components=fit_components,
    fit_c_terms=fit_c_terms,
    nwalkers=nwalkers,
    ninter=ninter,
    discard_fraction=discard_fraction,
    verbose=True,
    fit_mode=fit_mode,
    color_correction=True,
    cov_matrix=None
)
print(f"[BB] Reduced chi2 (no cov): {chi2_BB_diag:.4f}")

# B.5: Plot Corner
save_path_BB = os.path.join(save_dir_figures, f'example_BB_{components_str}.pdf')
functions.plot_corner(samples_free_BB, param_map_BB, save_path=save_path_BB, title='BB Mode (With Covariance)')
print(f"[BB] Corner plot saved to {save_path_BB}")

save_path_BB_diag = os.path.join(save_dir_figures, f'example_BB_{components_str}_diag.pdf')
functions.plot_corner(samples_free_BB_diag, param_map_BB_diag, save_path=save_path_BB_diag, title='BB Mode (Diagonal Covariance)')
print(f"[BB-Diag] Corner plot saved to {save_path_BB_diag}")


# ============================================================================
# SECTION C: JOINT EE-BB MODE
# ============================================================================
print("\n" + "="*60)
print("SECTION C: JOINT EE-BB Analysis")
print("="*60)

# C.1: Prepare MCMC data
print("\n[Joint] Preparing data...")
fit_data_Joint = functions.prepare_mcmc_data(
    spectra_dict,
    band_list=band_list,
    modes=['EE', 'BB'],
    ell_min=30,
    ell_max=200,
    band_pairs='all'
)

# C.2: Build Covariance
print("\n[Joint] Building block-diagonal inverse covariance...")
# This now works because we fixed functions.py to handle multiple modes!
cov_info_Joint = functions.build_block_diagonal_cov_inv(
    path_sims_fits=path_full_skyplusnoise,
    fit_data=fit_data_Joint,
    modes=['EE', 'BB'], 
    quijote_bands_11_13=['11', '13'],
    quijote_bands_17_19=['17', '19'],
    path_noise_sims=path_full_noise  # <-- Added noise sims for total error
)

# C.3: Run MCMC (With Covariance)
print("\n[Joint] Running MCMC with block-diagonal covariance...")
sampler_Joint, samples_full_Joint, samples_free_Joint, param_map_Joint, chi2_Joint = functions.run_mcmc(
    fit_data=fit_data_Joint,
    fit_components=fit_components,
    fit_c_terms=fit_c_terms,
    nwalkers=nwalkers,
    ninter=ninter,
    discard_fraction=discard_fraction,
    verbose=True,
    fit_mode=fit_mode,
    color_correction=True,
    joint_analysis=True,
    cov_matrix=cov_info_Joint
)
print(f"[Joint] Reduced chi2 (with cov): {chi2_Joint:.4f}")

# C.4: Run MCMC (Without Covariance)
print("\n[Joint] Running MCMC with diagonal covariance (comparison)...")
sampler_Joint_diag, samples_full_Joint_diag, samples_free_Joint_diag, param_map_Joint_diag, chi2_Joint_diag = functions.run_mcmc(
    fit_data=fit_data_Joint,
    fit_components=fit_components,
    fit_c_terms=fit_c_terms,
    nwalkers=nwalkers,
    ninter=ninter,
    discard_fraction=discard_fraction,
    verbose=True,
    fit_mode=fit_mode,
    color_correction=True,
    joint_analysis=True,
    cov_matrix=None
)
print(f"[Joint] Reduced chi2 (no cov): {chi2_Joint_diag:.4f}")

# C.5: Plot Corner
save_path_Joint = os.path.join(save_dir_figures, f'example_Joint_{components_str}.pdf')
functions.plot_corner(samples_free_Joint, param_map_Joint, save_path=save_path_Joint, title='Joint EE-BB (With Covariance)')
print(f"[Joint] Corner plot saved to {save_path_Joint}")

save_path_Joint_diag = os.path.join(save_dir_figures, f'example_Joint_{components_str}_diag.pdf')
functions.plot_corner(samples_free_Joint_diag, param_map_Joint_diag, save_path=save_path_Joint_diag, title='Joint EE-BB (Diagonal Covariance)')
print(f"[Joint-Diag] Corner plot saved to {save_path_Joint_diag}")
