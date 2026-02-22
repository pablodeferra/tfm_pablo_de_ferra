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

import functions
from data import data, masks
import healpy as hp
import numpy as np
from astropy.io import fits
from functions import set_gaussian_priors

# =====================================================================
# DEFAULT CONFIGURATION
# =====================================================================
nside = 512
n_sim = 100

name_suffix = '_full_bin_20-199'

lmax = 2 * nside - 1

mask_select = masks['QUIJOTE_galcut']['galcut10']
mask_name = mask_select['name']

out_path = '/home/pablo/Desktop/master/tfm/spectra/'
path_corrected_spectra = os.path.join(out_path, f'corrected_power_spectra_{mask_name}{name_suffix}.fits')

# Multipole range
ell_min = 30
ell_max = 200

# Sampler configuration
nwalkers = 200
ninter = 5000
discard_fraction = 0.5

# Number of parallel processes for MCMC
# None = auto; set to a fixed number to cap CPU usage
n_processes = 20

# Output directories
os.makedirs('/home/pablo/Desktop/master/tfm/figures/tables/', exist_ok=True)
os.makedirs('/home/pablo/Desktop/master/tfm/figures/corner/', exist_ok=True)

# =====================================================================
# BAND CONFIGURATIONS
# =====================================================================
# With 33 GHz  (QUIJOTE 11 GHz + WMAP 23/30/33 + Planck HFI)
bands_with33 = {
    'QUIJOTE+WMAP+Planck':  ['11', '23', '30', '33', '100', '143', '217', '353'],
    'WMAP+Planck':          ['23', '30', '33', '100', '143', '217', '353'],
}

# Without 33 GHz
bands_no33 = {
    'QUIJOTE+WMAP+Planck_no33': ['11', '23', '30', '100', '143', '217', '353'],
    'WMAP+Planck_no33':         ['23', '30', '100', '143', '217', '353'],
}

# Component tuples
components_no_cross   = ('sync', 'dust')
components_with_cross = ('sync', 'dust', 'cross')

# =====================================================================
# CORE RUNNER  –  one config × one EE/BB mode at a time
# =====================================================================

def run_single_fit(config_label, band_list_fit, mode,
                   fit_components, pair_type, use_c_terms,
                   prior_label, table_num):
    """
    Run one MCMC fit for a single (band config, polarisation mode) pair.

    The heavy sampler object is destroyed before returning; only the
    lightweight sample array and metadata are kept.

    Parameters
    ----------
    pair_type : str  'auto' | 'cross' | 'all'

    Returns
    -------
    dict  keys: data_label, mode, samples_free, param_map, chi2_reduced
    """
    print(f"\n{'='*60}")
    print(f"  Table {table_num} | {config_label} | {mode} | {pair_type.upper()}"
          f" | c_terms={use_c_terms} | prior={prior_label or 'flat'}")
    print(f"  Bands: {band_list_fit}")
    print(f"{'='*60}\n")

    spectra_dict = functions.read_corrected_cls(path_corrected_spectra, band_list_fit)

    # Build pair list
    auto_pairs  = [f"{b}_{b}" for b in band_list_fit]
    cross_pairs = [
        f"{a}_{b}"
        for i, a in enumerate(band_list_fit)
        for b in band_list_fit[i + 1:]
    ]
    if pair_type == 'auto':
        pairs = auto_pairs
    elif pair_type == 'cross':
        pairs = cross_pairs
    else:  # 'all'
        pairs = auto_pairs + cross_pairs

    components_str = '_'.join(fit_components)
    config_short   = config_label.replace('+', '+')

    save_corner = (
        f'/home/pablo/Desktop/master/tfm/figures/corner/'
        f'corner_{mask_name}_{components_str}_{mode}{name_suffix}'
        f'_{config_short}_{pair_type}'
        f'{"_c_terms" if use_c_terms else ""}'
        f'{prior_label}.pdf'
    )

    fit_data = functions.prepare_mcmc_data(
        spectra_dict,
        band_list=band_list_fit,
        modes=[mode],
        ell_min=ell_min,
        ell_max=ell_max,
        band_pairs=pairs,
    )

    sampler, samp_full, samp_free, pm, chi2 = functions.run_mcmc(
        fit_data=fit_data,
        fit_components=fit_components,
        fit_c_terms=use_c_terms,
        nwalkers=nwalkers,
        ninter=ninter,
        discard_fraction=discard_fraction,
        verbose=True,
        fit_mode='power-law',
        color_correction=True,
        cov_matrix=None,
        n_processes=n_processes,
    )

    functions.plot_corner(
        samp_free, pm, save_corner,
        title=(f'{config_label} — {mode} — {pair_type.capitalize()}'
               f'{" (c_terms)" if use_c_terms else ""}'
               f'{" — prior β_s" if prior_label else ""}'),
    )

    # NOTE: plot_spectra_with_bestfit requires BOTH EE and BB simultaneously.
    # We only cache fit_data here; the combined plot is made in _try_plot_spectra()
    # after both modes are complete for this config.
    result = {
        'data_label':   f'{config_label} ({pair_type})',
        'mode':         mode,
        'samples_free': samp_free.copy(),
        'param_map':    pm,
        'chi2_reduced': chi2,
        # keep a minimal copy of fit_data needed for the combined EE+BB spectra plot;
        # will be deleted from results_list once the plot is made
        '_fit_data':    fit_data,
        '_band_list':   band_list_fit,
        '_config_short': config_short,
        '_components_str': components_str,
    }

    # ---- Explicit RAM release (sampler is the heavy object) ----
    del sampler, samp_full
    del spectra_dict
    gc.collect()

    return result


def _try_plot_spectra(results_list, config_label, pair_type,
                     fit_components, use_c_terms, prior_label):
    """
    If both EE and BB results for `config_label` are present in results_list,
    call plot_spectra_with_bestfit and then remove the cached fit_data to free RAM.
    """
    ee = next((r for r in results_list
               if r['data_label'] == f'{config_label} ({pair_type})'
               and r['mode'] == 'EE'
               and '_fit_data' in r), None)
    bb = next((r for r in results_list
               if r['data_label'] == f'{config_label} ({pair_type})'
               and r['mode'] == 'BB'
               and '_fit_data' in r), None)

    if ee is None or bb is None:
        return  # not both modes done yet

    band_list_fit  = ee['_band_list']
    config_short   = ee['_config_short']
    components_str = ee['_components_str']

    bands_to_plot = [b for b in ('11', '23', '30') if b in band_list_fit]
    if bands_to_plot:
        save_spectra = (
            f'/home/pablo/Desktop/master/tfm/figures/'
            f'spectra_bestfit_{mask_name}_{components_str}{name_suffix}'
            f'_{config_short}_{pair_type}'
            f'{"_c_terms" if use_c_terms else ""}'
            f'{prior_label}.pdf'
        )
        try:
            functions.plot_spectra_with_bestfit(
                fit_data={'EE': ee['_fit_data'], 'BB': bb['_fit_data']},
                results_entry={
                    'EE': {'samples_free': ee['samples_free'],
                           'param_map':    ee['param_map'],
                           'chi2_reduced': ee['chi2_reduced']},
                    'BB': {'samples_free': bb['samples_free'],
                           'param_map':    bb['param_map'],
                           'chi2_reduced': bb['chi2_reduced']},
                },
                fit_components=fit_components,
                fit_c_terms=use_c_terms,
                bands_to_plot=bands_to_plot,
                color_correction=True,
                save_path=save_spectra,
                title=(f'{config_label} — Synchrotron {pair_type}-spectra + best fit'
                       f'{" — prior β_s" if prior_label else ""}'),
            )
        except Exception as e:
            print(f"[WARNING] plot_spectra_with_bestfit failed: {e}")

    # Remove cached fit_data from both entries to free RAM
    for r in (ee, bb):
        r.pop('_fit_data', None)
        r.pop('_band_list', None)
        r.pop('_config_short', None)
        r.pop('_components_str', None)
    gc.collect()


def _save_table(results_list, save_tex_path, tex_kwargs):
    """Save (or overwrite) the running LaTeX table to disk and return the string."""
    # Strip internal underscore-prefixed keys before passing to create_fitting_results_table
    clean = [{k: v for k, v in r.items() if not k.startswith('_')}
             for r in results_list]
    latex = functions.create_fitting_results_table(
        clean,
        save_path=save_tex_path,
        **tex_kwargs,
    )
    return latex


def run_table(table_num, band_configs, fit_components, pair_type, use_c_terms,
              prior_label, save_tex_path, tex_kwargs):
    """
    Run all (config × mode) fits for one table, writing the .tex after each
    individual fit so progress is never lost, then release all RAM.
    """
    results_list = []

    for config_label, band_list_fit in band_configs.items():
        for mode in ['EE', 'BB']:
            result = run_single_fit(
                config_label=config_label,
                band_list_fit=band_list_fit,
                mode=mode,
                fit_components=fit_components,
                pair_type=pair_type,
                use_c_terms=use_c_terms,
                prior_label=prior_label,
                table_num=table_num,
            )
            results_list.append(result)
            # Partial save so a crash mid-table still leaves useful output
            _save_table(results_list, save_tex_path, tex_kwargs)

        # Both EE and BB done for this config: make the combined spectra plot
        # and immediately free the cached fit_data
        _try_plot_spectra(results_list, config_label, pair_type,
                          fit_components, use_c_terms, prior_label)

    # Final pretty-print
    latex = _save_table(results_list, save_tex_path, tex_kwargs)
    print(f"\n===== TABLE {table_num} =====\n{latex}\n")

    # Release all sample arrays before moving to the next table
    del results_list
    gc.collect()


# =====================================================================
# TABLE DEFINITIONS
# =====================================================================
# 12 tables in total:
#   Tables  1– 6 : including the 33 GHz band
#   Tables  7–12 : excluding the 33 GHz band
#
# For each group:
#   1/7  – auto,       sync+dust,       no prior,   c_terms
#   2/8  – auto,       sync+dust,       beta_s prior, c_terms
#   3/9  – cross,      sync+dust+cross, no prior,   no c_terms
#   4/10 – cross,      sync+dust+cross, beta_s prior, no c_terms
#   5/11 – auto+cross, sync+dust+cross, no prior,   no c_terms
#   6/12 – auto+cross, sync+dust+cross, beta_s prior, no c_terms

table_definitions = [
    # ----------------------------------------------------------------
    # WITH 33 GHz
    # ----------------------------------------------------------------
    dict(
        table_num=1,
        band_configs=bands_with33,
        fit_components=components_no_cross,
        pair_type='auto',
        use_c_terms=True,
        prior_label='',
        priors=None,
        save_tex_path=(
            f'/home/pablo/Desktop/master/tfm/figures/tables/'
            f'fit_results_{mask_name}{name_suffix}_table1_auto_with33_sync_dust.tex'
        ),
        tex_kwargs=dict(
            label='tab:fit_results_auto_with33_sync_dust',
            ell_range='30--200',
            mask_name=mask_name,
            include_c_terms=True,
            caption=(
                "Posterior constraints from auto-spectra fits (sync+dust model, "
                "with constant terms) including the 33\\,GHz band. "
                "Multipole range $\\ell=30$--$200$."
            ),
        ),
    ),
    dict(
        table_num=2,
        band_configs=bands_with33,
        fit_components=components_no_cross,
        pair_type='auto',
        use_c_terms=True,
        prior_label='_prior_betas',
        priors={'beta_s': (-3.1, 0.3)},
        save_tex_path=(
            f'/home/pablo/Desktop/master/tfm/figures/tables/'
            f'fit_results_{mask_name}{name_suffix}_table2_auto_with33_sync_dust_prior_betas.tex'
        ),
        tex_kwargs=dict(
            label='tab:fit_results_auto_with33_sync_dust_prior_betas',
            ell_range='30--200',
            mask_name=mask_name,
            include_c_terms=True,
            caption=(
                "Same as Table~\\ref{tab:fit_results_auto_with33_sync_dust} but with a "
                "Gaussian prior $\\beta_s \\sim \\mathcal{N}(-3.1,\\,0.3)$."
            ),
        ),
    ),
    dict(
        table_num=3,
        band_configs=bands_with33,
        fit_components=components_with_cross,
        pair_type='cross',
        use_c_terms=False,
        prior_label='',
        priors=None,
        save_tex_path=(
            f'/home/pablo/Desktop/master/tfm/figures/tables/'
            f'fit_results_{mask_name}{name_suffix}_table3_cross_with33_sync_dust_cross.tex'
        ),
        tex_kwargs=dict(
            label='tab:fit_results_cross_with33_sync_dust_cross',
            ell_range='30--200',
            mask_name=mask_name,
            include_c_terms=False,
            caption=(
                "Posterior constraints from cross-spectra fits "
                "(sync+dust+corr model, no constant terms) including the 33\\,GHz band. "
                "Multipole range $\\ell=30$--$200$."
            ),
        ),
    ),
    dict(
        table_num=4,
        band_configs=bands_with33,
        fit_components=components_with_cross,
        pair_type='cross',
        use_c_terms=False,
        prior_label='_prior_betas',
        priors={'beta_s': (-3.1, 0.3)},
        save_tex_path=(
            f'/home/pablo/Desktop/master/tfm/figures/tables/'
            f'fit_results_{mask_name}{name_suffix}_table4_cross_with33_sync_dust_cross_prior_betas.tex'
        ),
        tex_kwargs=dict(
            label='tab:fit_results_cross_with33_sync_dust_cross_prior_betas',
            ell_range='30--200',
            mask_name=mask_name,
            include_c_terms=False,
            caption=(
                "Same as Table~\\ref{tab:fit_results_cross_with33_sync_dust_cross} but "
                "with a Gaussian prior $\\beta_s \\sim \\mathcal{N}(-3.1,\\,0.3)$."
            ),
        ),
    ),
    dict(
        table_num=5,
        band_configs=bands_with33,
        fit_components=components_with_cross,
        pair_type='all',
        use_c_terms=False,
        prior_label='',
        priors=None,
        save_tex_path=(
            f'/home/pablo/Desktop/master/tfm/figures/tables/'
            f'fit_results_{mask_name}{name_suffix}_table5_all_with33_sync_dust_cross.tex'
        ),
        tex_kwargs=dict(
            label='tab:fit_results_all_with33_sync_dust_cross',
            ell_range='30--200',
            mask_name=mask_name,
            include_c_terms=False,
            caption=(
                "Posterior constraints from all (auto+cross) spectra fits "
                "(sync+dust+corr model, no constant terms) including the 33\\,GHz band. "
                "Multipole range $\\ell=30$--$200$."
            ),
        ),
    ),
    dict(
        table_num=6,
        band_configs=bands_with33,
        fit_components=components_with_cross,
        pair_type='all',
        use_c_terms=False,
        prior_label='_prior_betas',
        priors={'beta_s': (-3.1, 0.3)},
        save_tex_path=(
            f'/home/pablo/Desktop/master/tfm/figures/tables/'
            f'fit_results_{mask_name}{name_suffix}_table6_all_with33_sync_dust_cross_prior_betas.tex'
        ),
        tex_kwargs=dict(
            label='tab:fit_results_all_with33_sync_dust_cross_prior_betas',
            ell_range='30--200',
            mask_name=mask_name,
            include_c_terms=False,
            caption=(
                "Same as Table~\\ref{tab:fit_results_all_with33_sync_dust_cross} but "
                "with a Gaussian prior $\\beta_s \\sim \\mathcal{N}(-3.1,\\,0.3)$."
            ),
        ),
    ),
    # ----------------------------------------------------------------
    # WITHOUT 33 GHz
    # ----------------------------------------------------------------
    dict(
        table_num=7,
        band_configs=bands_no33,
        fit_components=components_no_cross,
        pair_type='auto',
        use_c_terms=True,
        prior_label='',
        priors=None,
        save_tex_path=(
            f'/home/pablo/Desktop/master/tfm/figures/tables/'
            f'fit_results_{mask_name}{name_suffix}_table7_auto_no33_sync_dust.tex'
        ),
        tex_kwargs=dict(
            label='tab:fit_results_auto_no33_sync_dust',
            ell_range='30--200',
            mask_name=mask_name,
            include_c_terms=True,
            caption=(
                "Posterior constraints from auto-spectra fits (sync+dust model, "
                "with constant terms) excluding the 33\\,GHz band. "
                "Multipole range $\\ell=30$--$200$."
            ),
        ),
    ),
    dict(
        table_num=8,
        band_configs=bands_no33,
        fit_components=components_no_cross,
        pair_type='auto',
        use_c_terms=True,
        prior_label='_prior_betas',
        priors={'beta_s': (-3.1, 0.3)},
        save_tex_path=(
            f'/home/pablo/Desktop/master/tfm/figures/tables/'
            f'fit_results_{mask_name}{name_suffix}_table8_auto_no33_sync_dust_prior_betas.tex'
        ),
        tex_kwargs=dict(
            label='tab:fit_results_auto_no33_sync_dust_prior_betas',
            ell_range='30--200',
            mask_name=mask_name,
            include_c_terms=True,
            caption=(
                "Same as Table~\\ref{tab:fit_results_auto_no33_sync_dust} but with a "
                "Gaussian prior $\\beta_s \\sim \\mathcal{N}(-3.1,\\,0.3)$."
            ),
        ),
    ),
    dict(
        table_num=9,
        band_configs=bands_no33,
        fit_components=components_with_cross,
        pair_type='cross',
        use_c_terms=False,
        prior_label='',
        priors=None,
        save_tex_path=(
            f'/home/pablo/Desktop/master/tfm/figures/tables/'
            f'fit_results_{mask_name}{name_suffix}_table9_cross_no33_sync_dust_cross.tex'
        ),
        tex_kwargs=dict(
            label='tab:fit_results_cross_no33_sync_dust_cross',
            ell_range='30--200',
            mask_name=mask_name,
            include_c_terms=False,
            caption=(
                "Posterior constraints from cross-spectra fits "
                "(sync+dust+corr model, no constant terms) excluding the 33\\,GHz band. "
                "Multipole range $\\ell=30$--$200$."
            ),
        ),
    ),
    dict(
        table_num=10,
        band_configs=bands_no33,
        fit_components=components_with_cross,
        pair_type='cross',
        use_c_terms=False,
        prior_label='_prior_betas',
        priors={'beta_s': (-3.1, 0.3)},
        save_tex_path=(
            f'/home/pablo/Desktop/master/tfm/figures/tables/'
            f'fit_results_{mask_name}{name_suffix}_table10_cross_no33_sync_dust_cross_prior_betas.tex'
        ),
        tex_kwargs=dict(
            label='tab:fit_results_cross_no33_sync_dust_cross_prior_betas',
            ell_range='30--200',
            mask_name=mask_name,
            include_c_terms=False,
            caption=(
                "Same as Table~\\ref{tab:fit_results_cross_no33_sync_dust_cross} but "
                "with a Gaussian prior $\\beta_s \\sim \\mathcal{N}(-3.1,\\,0.3)$."
            ),
        ),
    ),
    dict(
        table_num=11,
        band_configs=bands_no33,
        fit_components=components_with_cross,
        pair_type='all',
        use_c_terms=False,
        prior_label='',
        priors=None,
        save_tex_path=(
            f'/home/pablo/Desktop/master/tfm/figures/tables/'
            f'fit_results_{mask_name}{name_suffix}_table11_all_no33_sync_dust_cross.tex'
        ),
        tex_kwargs=dict(
            label='tab:fit_results_all_no33_sync_dust_cross',
            ell_range='30--200',
            mask_name=mask_name,
            include_c_terms=False,
            caption=(
                "Posterior constraints from all (auto+cross) spectra fits "
                "(sync+dust+corr model, no constant terms) excluding the 33\\,GHz band. "
                "Multipole range $\\ell=30$--$200$."
            ),
        ),
    ),
    dict(
        table_num=12,
        band_configs=bands_no33,
        fit_components=components_with_cross,
        pair_type='all',
        use_c_terms=False,
        prior_label='_prior_betas',
        priors={'beta_s': (-3.1, 0.3)},
        save_tex_path=(
            f'/home/pablo/Desktop/master/tfm/figures/tables/'
            f'fit_results_{mask_name}{name_suffix}_table12_all_no33_sync_dust_cross_prior_betas.tex'
        ),
        tex_kwargs=dict(
            label='tab:fit_results_all_no33_sync_dust_cross_prior_betas',
            ell_range='30--200',
            mask_name=mask_name,
            include_c_terms=False,
            caption=(
                "Same as Table~\\ref{tab:fit_results_all_no33_sync_dust_cross} but "
                "with a Gaussian prior $\\beta_s \\sim \\mathcal{N}(-3.1,\\,0.3)$."
            ),
        ),
    ),
]

# =====================================================================
# MAIN LOOP — one table at a time, freeing RAM between tables
# =====================================================================
for tdef in table_definitions:
    # Apply priors for this table only
    set_gaussian_priors(tdef['priors'])

    run_table(
        table_num=tdef['table_num'],
        band_configs=tdef['band_configs'],
        fit_components=tdef['fit_components'],
        pair_type=tdef['pair_type'],
        use_c_terms=tdef['use_c_terms'],
        prior_label=tdef['prior_label'],
        save_tex_path=tdef['save_tex_path'],
        tex_kwargs=tdef['tex_kwargs'],
    )

    # Reset priors and free memory before next table
    set_gaussian_priors(None)
    gc.collect()

print("\n\nAll 12 tables completed and saved.")
