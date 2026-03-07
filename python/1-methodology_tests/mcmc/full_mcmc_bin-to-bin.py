#%%
import os
import gc

# CRITICAL: Set to 1 thread per process BEFORE imports!
# MCMC parallelizes across walkers (processes), not threads
# Setting >1 causes massive thread oversubscription and slowdown
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.append('../')
import functions
from data import data, masks
import numpy as np

# Number of parallel processes for MCMC (set to control CPU usage)
# bin-to-bin runs one per-bin pool sequentially, so this limits each pool
n_processes = 10  # Adjust this for your 50-core machine

# =====================================================================
# Global configuration
# =====================================================================
nside = 512
n_sim = 100

name_suffix = '_full_bin_20-199'

mask_select = masks['QUIJOTE_galcut']['galcut10']
mask_name = mask_select['name']

out_path = '/home/pablo/Desktop/master/tfm/spectra/'
path_corrected_spectra = os.path.join(out_path, f'corrected_power_spectra_{mask_name}{name_suffix}.fits')

# =====================================================================
# Fitting configuration
# =====================================================================
from functions import set_gaussian_priors

set_gaussian_priors({
    'beta_s': (-3.1, 0.30),
})

fitting_mode = 'bin-to-bin'

# Sampler configuration
nwalkers = 100
ninter   = 50
discard_fraction = 0.5

# Components: sync + dust + cross, no constant terms
fit_components = ('sync', 'dust', 'cross')

# =====================================================================
# Band definitions
# =====================================================================
# QUIJOTE: only 11 and 17 GHz
quijote_fit_bands = ['11', '17']
# All WMAP bands
wmap_fit_bands = ['23', '33', '41', '61', '94']
# All Planck LFI + HFI up to 353 GHz
planck_fit_bands = ['30', '44', '70', '100', '143', '217', '353']

# Two band configurations: with and without QUIJOTE
band_configs = {
    # 'WMAP+Planck':         wmap_fit_bands + planck_fit_bands,
    'QUIJOTE+WMAP+Planck': quijote_fit_bands + wmap_fit_bands + planck_fit_bands,
}

# =====================================================================
# Helper: build the explicit list of band pairs for a given strategy
# =====================================================================
def build_band_pairs(band_list_fit, strategy):
    """
    Return the list of pairs to pass to prepare_mcmc_data.

    strategy : str
        'all'          - all auto + cross spectra
        'no_qq_autos'  - all pairs EXCEPT the QUIJOTExQUIJOTE autos
                         (i.e. 11_11 and 17_17 excluded; 11_17 kept)
        'cross_only'   - only off-diagonal pairs (no auto spectra at all)
    """
    all_pairs = [
        f"{a}_{b}"
        for i, a in enumerate(band_list_fit)
        for b in band_list_fit[i:]
    ]
    if strategy == 'all':
        return all_pairs
    elif strategy == 'no_qq_autos':
        qq_autos = {f"{b}_{b}" for b in quijote_fit_bands}
        return [p for p in all_pairs if p not in qq_autos]
    elif strategy == 'cross_only':
        return [p for p in all_pairs if p.split('_')[0] != p.split('_')[1]]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

# =====================================================================
# Table definitions: 6 tables in total
# (3 pair strategies) × (2 ell ranges)
# Each table uses both EE and BB modes and both band configs.
#
# ell_ranges entries: (ell_min, ell_max, ell_1_edges, ell_2_edges)
# The edge lists must match the binning used when computing the spectra.
# =====================================================================
pair_strategies = [
    ('all',         'all_pairs'),
    # ('no_qq_autos', 'no_QQ_autos'),
    # ('cross_only',  'cross_only'),
]

ell_ranges = [
    (20, 200,
     [20, 40, 60, 80, 100, 120, 140, 160, 180],
     [39, 59, 79, 99, 119, 139, 159, 179, 199]),
]

table_dir = '/home/pablo/Desktop/master/tfm/tables/final/'
os.makedirs(table_dir, exist_ok=True)

# =====================================================================
# Main loop: one table at a time to keep RAM usage low.
# For each (ell_range, strategy) combination we run both band configs,
# save a LaTeX + ASCII table per config, then free everything.
# =====================================================================
for ell_min, ell_max, ell_1, ell_2 in ell_ranges:
    for strategy_key, strategy_label in pair_strategies:

        print(f"\n{'#'*70}")
        print(f"  BIN-TO-BIN TABLE: ell={ell_min}-{ell_max}  |  pairs={strategy_label}")
        print(f"{'#'*70}\n")

        # Accumulate per-config results so we can build one table per config
        config_results = {}

        for config_label, band_list_fit in band_configs.items():
            config_short = config_label.replace('+', 'p')

            print(f"\n{'='*60}")
            print(f"  Config: {config_label}  |  Bands: {band_list_fit}")
            print(f"{'='*60}\n")

            # Load spectra for this band set (released at end of config block)
            spectra_dict = functions.read_corrected_cls(
                path_corrected_spectra, band_list_fit
            )

            # Build explicit pair list for this strategy
            pairs_for_fit = build_band_pairs(band_list_fit, strategy_key)

            mode_results = {}
            for mode in ['EE', 'BB']:
                print(f"\n--- {config_label} / {mode} ---\n")

                # Prepare MCMC data
                fit_data = functions.prepare_mcmc_data(
                    spectra_dict,
                    band_list=band_list_fit,
                    modes=[mode],
                    ell_min=ell_min,
                    ell_max=ell_max,
                    band_pairs=pairs_for_fit,
                )

                # Run bin-to-bin MCMC
                samplers, samples_full, samples_free, param_names, chi2_reduced = functions.run_mcmc(
                    fit_data=fit_data,
                    fit_components=fit_components,
                    fit_c_terms=False,
                    nwalkers=nwalkers,
                    ninter=ninter,
                    discard_fraction=discard_fraction,
                    verbose=True,
                    fit_mode=fitting_mode,
                    color_correction=True,
                    n_processes=n_processes,
                )

                # Keep only the lightweight results needed for the table
                mode_results[mode] = {
                    'fit_data':     fit_data,
                    'samples_free': samples_free,
                    'param_names':  param_names,
                    'chi2_reduced': chi2_reduced,   # list of float, one per ell bin
                }

                # Free the heavy sampler objects immediately
                del samplers, samples_full, fit_data
                gc.collect()

            # Free spectra before loading the next config
            del spectra_dict
            gc.collect()

            config_results[config_label] = mode_results

        # ----------------------------------------------------------------
        # Generate and save one LaTeX + ASCII table per band config
        # ----------------------------------------------------------------
        for config_label, mode_results in config_results.items():
            config_short = config_label.replace('+', 'p')

            save_latex = (
                f'{table_dir}btb_{mask_name}_ell{ell_min}-{ell_max}'
                f'_{strategy_label}_{config_short}__TEST__.tex'
            )
            save_ascii = save_latex.replace('.tex', '.txt')

            res_EE = mode_results['EE']
            res_BB = mode_results['BB']

            # LaTeX table
            table_latex = functions.create_bin_to_bin_table(
                fit_data_EE=res_EE['fit_data'],
                fit_data_BB=res_BB['fit_data'],
                samples_free_list_EE=res_EE['samples_free'],
                samples_free_list_BB=res_BB['samples_free'],
                param_names=res_EE['param_names'],
                ell1=ell_1,
                ell2=ell_2,
                save_path=save_latex,
                format='latex',
                chi2_reduced_EE=res_EE['chi2_reduced'],
                chi2_reduced_BB=res_BB['chi2_reduced'],
            )

            # ASCII table (printed + saved as .txt for quick inspection)
            table_ascii = functions.create_bin_to_bin_table(
                fit_data_EE=res_EE['fit_data'],
                fit_data_BB=res_BB['fit_data'],
                samples_free_list_EE=res_EE['samples_free'],
                samples_free_list_BB=res_BB['samples_free'],
                param_names=res_EE['param_names'],
                ell1=ell_1,
                ell2=ell_2,
                save_path=save_ascii,
                format='ascii',
                chi2_reduced_EE=res_EE['chi2_reduced'],
                chi2_reduced_BB=res_BB['chi2_reduced'],
            )

            print(f"\nTable saved to: {save_latex}")
            print(table_ascii)

        # Free everything before the next (ell_range, strategy) iteration
        del config_results
        gc.collect()
