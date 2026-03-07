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

# Number of parallel processes for MCMC.
# emcee parallelises by distributing walkers across processes.
# Each likelihood call is cheap (a few µs), so IPC overhead dominates
# at high process counts → fewer processes = less overhead = faster.
# Rule of thumb: n_processes ≈ nwalkers // 10 to nwalkers // 20.
# With nwalkers=200: try 10–20. If CPU usage per worker is well below
# 100%, reduce further. Start with 10 and benchmark.
n_processes = 10  # tune: try 6, 8, 10 and pick the fastest wall-clock time

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

set_gaussian_priors(None)

fitting_mode = 'power-law'

# Sampler configuration
nwalkers = 100
ninter = 100
discard_fraction = 0.5

# Components: sync + dust + cross, no constant terms
fit_components = ('sync', 'dust', 'cross')
fit_c_terms = False

components_str = '_'.join(fit_components)

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
    'WMAP+Planck':          wmap_fit_bands + planck_fit_bands,
    'QUIJOTE+WMAP+Planck':  quijote_fit_bands + wmap_fit_bands + planck_fit_bands,
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
                         (i.e. 11_11 and 17_17 are excluded; 11_17 is kept)
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
# =====================================================================
pair_strategies = [
    ('all',         'all_pairs'),
    # ('no_qq_autos', 'no_QQ_autos'),
    # ('cross_only',  'cross_only'),
]

ell_ranges = [
    # (30, 120),
    (30, 200),
]

table_dir = '/home/pablo/Desktop/master/tfm/tables/final/'
corner_dir = '/home/pablo/Desktop/master/tfm/figures/corner/final/'
os.makedirs(table_dir, exist_ok=True)
os.makedirs(corner_dir, exist_ok=True)

# =====================================================================
# Main loop: one table at a time to keep RAM usage low
# =====================================================================
for ell_min, ell_max in ell_ranges:
    for strategy_key, strategy_label in pair_strategies:

        print(f"\n{'#'*70}")
        print(f"  TABLE: ell={ell_min}-{ell_max}  |  pairs={strategy_label}")
        print(f"{'#'*70}\n")

        table_save_path = (
            f'{table_dir}table_{mask_name}_ell{ell_min}-{ell_max}_{strategy_label}__TEST__.tex'
        )

        # Collect results for this single table (2 configs × 2 modes = 4 entries)
        results_list = []

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

            for mode in ['EE', 'BB']:
                print(f"\n--- {config_label} / {mode} ---\n")

                save_path_corner = (
                    f'{corner_dir}corner_{mask_name}_{components_str}_{mode}'
                    f'{name_suffix}_{config_short}_ell{ell_min}-{ell_max}'
                    f'_{strategy_label}.pdf'
                )

                # Prepare MCMC data
                fit_data = functions.prepare_mcmc_data(
                    spectra_dict,
                    band_list=band_list_fit,
                    modes=[mode],
                    ell_min=ell_min,
                    ell_max=ell_max,
                    band_pairs=pairs_for_fit,
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

                # Save corner plot
                functions.plot_corner(
                    samples_free, param_map, save_path_corner,
                    title=f'{config_label} — {mode}  (ell {ell_min}-{ell_max}, {strategy_label})'
                )

                # Accumulate results for this table
                results_list.append({
                    'data_label': config_label,
                    'mode': mode,
                    'samples_free': samples_free,
                    'param_map': param_map,
                    'chi2_reduced': chi2_reduced,
                })

                # Free large sampler object immediately
                del sampler, samples_full, fit_data
                gc.collect()

            # Free spectra for this band config before loading next one
            del spectra_dict
            gc.collect()

        # ----------------------------------------------------------------
        # Generate and save the LaTeX table for this (ell_range, strategy)
        # ----------------------------------------------------------------
        table_latex = functions.create_fitting_results_table(
            results_list,
            save_path=table_save_path,
            label=f'tab:fit_{strategy_label}_ell{ell_min}-{ell_max}',
            ell_range=f'{ell_min}-{ell_max}',
            mask_name=mask_name,
            include_c_terms=fit_c_terms,
        )

        print(f"\nTable saved to: {table_save_path}")
        print(table_latex)

        # Free results before next table
        del results_list
        gc.collect()
