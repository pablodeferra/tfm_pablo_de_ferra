#%%
import os
os.environ["OMP_NUM_THREADS"]     = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"]     = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import importlib

import functions
importlib.reload(functions)

from functions import set_gaussian_priors
from data import data, masks

nside      = 512
n_sim      = 100
name_suffix = '_full_bin_20-199'
mask_select = masks['QUIJOTE_galcut']['galcut10']
mask_name   = mask_select['name']

out_path = '/home/pablo/Desktop/master/tfm/spectra/'
path_corrected_spectra = os.path.join(out_path, f'corrected_power_spectra_{mask_name}{name_suffix}.fits')

tables_dir = '/home/pablo/Desktop/master/tfm/tables/'
os.makedirs(tables_dir, exist_ok=True)

# ==== MCMC configuration ====
nwalkers         = 100
ninter           = 20000
discard_fraction = 0.5

fit_components = ('sync',)   # synchrotron-only fit

# No Gaussian priors for this per-band fit
set_gaussian_priors(None)

# Bands to fit individually (auto-spectra only)
fit_bands = ['11', '23', '30', '33']

# Two ell ranges to run:
#   'full'  : 20-199 → 9 bins (ell_eff = 29,49,...,189), dof = 6 (no_c) / 5 (with_c)
#   'short' : 30-120 → 4 bins (ell_eff = 49,69,89,109),  dof = 1 (no_c) / 0 → inf (with_c)
# NOTE: ell_eff[0]=29 is stored as integer 29, so ell_min=30 excludes it.
# For 'short', with_c has dof=0 and chi2_reduced will be inf — this is correct behaviour
# (cannot reduce chi2 with 0 degrees of freedom; the fit is effectively unconstrained).
ELL_RANGES = {
    'full':  (30, 200),
    'short': (30, 120),
}

# All bands present in the corrected spectra file
quijote_bands = ['11', '13', '17', '19']
wmap_bands    = ['23', '33', '41', '61', '94']
planck_bands  = ['30', '44', '70', '100', '143', '217', '353']
band_list_full = quijote_bands + wmap_bands + planck_bands

# ==== load corrected spectra ====
print("[INFO] Loading corrected spectra...")
spectra_dict = functions.read_corrected_cls(path_corrected_spectra, band_list_full)


# ==== helper: extract median ± 1-sigma from flat chain ====
def chain_summary(samples_free, param_map):
    """Return {param_name: (median, +err, -err)} for all free parameters."""
    free_names = [name for name, is_free in param_map if is_free]
    summary = {}
    for i, name in enumerate(free_names):
        col   = samples_free[:, i]
        med   = np.median(col)
        lo, hi = np.percentile(col, [16, 84])
        summary[name] = (med, hi - med, med - lo)
    return summary


# ==== main loop ====
# all_results[ell_range_key][band][mode][fit_variant] = summary dict
all_results = {}

for ell_range_key, (ell_min, ell_max) in ELL_RANGES.items():
    print(f"\n{'#'*60}")
    print(f"  ELL RANGE: {ell_min} - {ell_max}  ({ell_range_key})")
    print(f"{'#'*60}")

    results = {b: {m: {} for m in ['EE', 'BB']} for b in fit_bands}

    for band in fit_bands:
        auto_pair = f'{band}_{band}'
        print(f"\n{'='*60}")
        print(f"  Band: {band} GHz   (pair: {auto_pair})")
        print(f"{'='*60}")

        for mode in ['EE', 'BB']:
            print(f"\n  Mode: {mode}")

            # Prepare data with only the single auto-spectrum pair
            fit_data = functions.prepare_mcmc_data(
                spectra_dict,
                band_list=[band],
                modes=[mode],
                ell_min=ell_min,
                ell_max=ell_max,
                band_pairs='all',   # gives [band_band] only
            )

            for fit_variant, fit_c in [('no_c', False), ('with_c', True)]:
                tag = f"{band} GHz | {mode} | {'with c' if fit_c else 'no c'}"
                print(f"\n    Fitting: {tag}")

                _, _, samples_free, param_map, chi2 = functions.run_mcmc(
                    fit_data=fit_data,
                    fit_components=fit_components,
                    fit_c_terms=fit_c,
                    nwalkers=nwalkers,
                    ninter=ninter,
                    discard_fraction=discard_fraction,
                    verbose=True,
                    fit_mode='power-law',
                    color_correction=False,
                    joint_analysis=False,
                    cov_matrix=None,
                    # For a single auto-pair (band_band), the frequency scaling
                    # (f/23)^beta_s appears on both sides and is degenerate with A_s.
                    # Freeze beta_s=0 so the model is exactly:
                    #   Cl = A_s * (ell/80)^alpha_s [+ c]
                    freeze_params={'beta_s': 0.0},
                )

                summary = chain_summary(samples_free, param_map)
                summary['chi2_reduced'] = chi2
                results[band][mode][fit_variant] = summary
                chi2_str = f'{chi2:.3f}' if np.isfinite(chi2) else 'inf (dof=0)'
                print(f"    χ²_red = {chi2_str}")
                for pname, val in summary.items():
                    if pname == 'chi2_reduced':
                        continue
                    med, ep, em = val
                    print(f"      {pname:20s}  {med:+.4e}  +{ep:.2e}  -{em:.2e}")

    all_results[ell_range_key] = results

# ==== table builder ====
# Amplitude (A_s) and constant (c_sync) are in mK²_RJ.
# The table expresses them in units of 10^{-3} µK² = 10^{-9} mK²,
# so multiply by 1e9 before formatting.
AMPLITUDE_SCALE = 1e9   # mK² → 10^{-3} µK²
AMPLITUDE_UNIT  = r'[10^{-3}\,\mu\mathrm{K}^2_\mathrm{RJ}]'

AMPLITUDE_PARAMS = {'A_s', 'c_sync'}   # key prefixes that need rescaling

def _needs_rescaling(key):
    """Return True if this parameter should be rescaled to amplitude units."""
    return key == 'A_s' or key.startswith('c_sync')

def fmt_val(summary, key, rescale=False):
    """Format median +err / -err for a parameter, or '---' if missing.

    Parameters
    ----------
    summary : dict
    key     : str
    rescale : bool
        If True, multiply the value by AMPLITUDE_SCALE before formatting.
    """
    if key not in summary:
        return r'\multicolumn{1}{c}{---}'
    med, ep, em = summary[key]
    if rescale:
        med *= AMPLITUDE_SCALE
        ep  *= AMPLITUDE_SCALE
        em  *= AMPLITUDE_SCALE

    # Format as fixed decimal (no scientific notation). Default: 3 decimal places.
    nd = 3
    fmt = f"{{:.{nd}f}}"

    # Protect against non-finite values
    def safe_fmt(x):
        if not np.isfinite(x):
            return 'nan'
        return fmt.format(x)

    med_s = safe_fmt(med)
    ep_s = safe_fmt(ep)
    em_s = safe_fmt(em)

    return rf'${med_s}^{{+{ep_s}}}_{{-{em_s}}}$'


def build_latex_table(results, fit_variant, with_c, ell_min, ell_max):
    """Build a full LaTeX table for a given fit variant ('no_c' or 'with_c')."""
    caption_tag = 'with constant $c_s$' if with_c else 'without constant $c_s$'
    label_tag   = 'withc' if with_c else 'noc'

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(
        rf'\caption{{Synchrotron power-law fit per band ({caption_tag}), '
        rf'$\ell \in [{ell_min}, {ell_max}]$. '
        r'Model: $C_\ell = A_s\,(\ell/80)^{\alpha_s} [+ c_s]$, '
        r'with $\beta_s$ frozen to 0 (single-band fit). '
        r'Values are median$^{+1\sigma}_{-1\sigma}$ from the MCMC posterior. '
        rf'$A_s$ and $c_s$ are in ${AMPLITUDE_UNIT}$.}}'
    )
    lines.append(rf'\label{{tab:sync_single_band_{label_tag}_ell{ell_min}-{ell_max}}}')
    lines.append(r'\begin{tabular}{lcccc}')
    lines.append(r'\toprule')
    lines.append(
        r'Parameter & 11\,GHz & 23\,GHz & 30\,GHz & 33\,GHz \\'
    )
    lines.append(r'\midrule')

    # ==== EE block ====
    ee_params = [
        (r'$A_s^{EE}$',     'A_s',     True),
        (r'$\alpha_s^{EE}$','alpha_s',  False),
    ]
    if with_c:
        ee_params.append((r'$c_s^{EE}$', None, True))

    for label, key, rescale in ee_params:
        row_vals = []
        for band in fit_bands:
            summary = results[band]['EE'][fit_variant]
            if key is None:
                c_key = next((k for k in summary if k.startswith('c_sync')), None)
                row_vals.append(fmt_val(summary, c_key, rescale=True) if c_key else r'\multicolumn{1}{c}{---}')
            else:
                row_vals.append(fmt_val(summary, key, rescale=rescale))
        lines.append(rf'{label} & ' + ' & '.join(row_vals) + r' \\')

    lines.append(r'\midrule')

    # ==== BB block ====
    bb_params = [
        (r'$A_s^{BB}$',     'A_s',     True),
        (r'$\alpha_s^{BB}$','alpha_s',  False),
    ]
    if with_c:
        bb_params.append((r'$c_s^{BB}$', None, True))

    for label, key, rescale in bb_params:
        row_vals = []
        for band in fit_bands:
            summary = results[band]['BB'][fit_variant]
            if key is None:
                c_key = next((k for k in summary if k.startswith('c_sync')), None)
                row_vals.append(fmt_val(summary, c_key, rescale=True) if c_key else r'\multicolumn{1}{c}{---}')
            else:
                row_vals.append(fmt_val(summary, key, rescale=rescale))
        lines.append(rf'{label} & ' + ' & '.join(row_vals) + r' \\')

    # (chi2 row removed by user request)

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    return '\n'.join(lines)


# ==== generate and save tables ====
for ell_range_key, (ell_min, ell_max) in ELL_RANGES.items():
    results = all_results[ell_range_key]
    for fit_variant, with_c in [('no_c', False), ('with_c', True)]:
        tag = 'withc' if with_c else 'noc'
        tex_path = os.path.join(
            tables_dir,
            f'sync_single_band_{tag}_{mask_name}_ell{ell_min}-{ell_max}{name_suffix}.tex'
        )
        table = build_latex_table(results, fit_variant, with_c, ell_min, ell_max)

        with open(tex_path, 'w') as f:
            f.write(table)
        print(f"\n[INFO] Table saved: {tex_path}")
        print(table)
