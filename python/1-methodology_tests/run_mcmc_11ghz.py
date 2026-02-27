# %%
# ===========================================================================
#  QUIJOTE 11 GHz  –  EE & BB, no_c / with_c, both ell ranges
#  Standalone script (does not depend on run_mcmc_single_band.py).
# ===========================================================================
import os
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["NUMEXPR_NUM_THREADS"]  = "1"

import numpy as np
import importlib

import functions
importlib.reload(functions)

from functions import set_gaussian_priors
from data import data, masks

# ── paths & identifiers ────────────────────────────────────────────────────
nside       = 512
n_sim       = 100
name_suffix = '_full_bin_20-199'
mask_select = masks['QUIJOTE_galcut']['galcut10']
mask_name   = mask_select['name']

out_path               = '/home/pablo/Desktop/master/tfm/spectra/'
path_corrected_spectra = os.path.join(
    out_path, f'corrected_power_spectra_{mask_name}{name_suffix}.fits'
)

tables_dir = '/home/pablo/Desktop/master/tfm/tables/'
os.makedirs(tables_dir, exist_ok=True)

# ── MCMC settings ──────────────────────────────────────────────────────────
nwalkers         = 100
ninter           = 10000
discard_fraction = 0.5

# No Gaussian priors for this single-band fit
set_gaussian_priors(None)

# ── ell ranges ─────────────────────────────────────────────────────────────
#   full  : 30-200  → 9 bins, dof = 6 (no_c) / 5 (with_c)
#   short : 30-120  → 4 bins, dof = 1 (no_c) / 0 → chi2_red = inf (with_c)
ELL_RANGES = {
    'full':  (30, 200),
    'short': (30, 120),
}

# ── bands in the spectra file (needed by read_corrected_cls) ───────────────
quijote_bands  = ['11', '13', '17', '19']
wmap_bands     = ['23', '33', '41', '61', '94']
planck_bands   = ['30', '44', '70', '100', '143', '217', '353']
band_list_full = quijote_bands + wmap_bands + planck_bands

# ── load spectra ───────────────────────────────────────────────────────────
print("[INFO] Loading corrected spectra...")
spectra_dict = functions.read_corrected_cls(path_corrected_spectra, band_list_full)


# ── helper: chain → summary dict ──────────────────────────────────────────
def chain_summary(samples_free, param_map):
    """Return {param_name: (median, +err, -err)} for all free parameters."""
    free_names = [name for name, is_free in param_map if is_free]
    summary = {}
    for i, name in enumerate(free_names):
        col = samples_free[:, i]
        med = np.median(col)
        lo, hi = np.percentile(col, [16, 84])
        summary[name] = (med, hi - med, med - lo)
    return summary


# ── table formatting ───────────────────────────────────────────────────────
AMPLITUDE_SCALE = 1e9   # mK²_RJ  →  10^{-3} µK²_RJ
AMPLITUDE_UNIT  = r'[10^{-3}\,\mu\mathrm{K}^2_\mathrm{RJ}]'


def fmt_val(summary, key, rescale=False):
    """Format median +err / -err for *key*, or '---' if missing."""
    if key not in summary:
        return r'\multicolumn{1}{c}{---}'
    med, ep, em = summary[key]
    if rescale:
        med *= AMPLITUDE_SCALE
        ep  *= AMPLITUDE_SCALE
        em  *= AMPLITUDE_SCALE
    nd  = 3
    fmt = f"{{:.{nd}f}}"

    def safe_fmt(x):
        return fmt.format(x) if np.isfinite(x) else 'nan'

    return rf'${safe_fmt(med)}^{{+{safe_fmt(ep)}}}_{{-{safe_fmt(em)}}}$'


# ── fits ───────────────────────────────────────────────────────────────────
# results_11[mode]['{no_c|with_c}_{full|short}'] = summary dict
band_11    = '11'
results_11 = {'EE': {}, 'BB': {}}

print("\n" + "#"*70)
print("  QUIJOTE 11 GHz  –  EE & BB  |  no_c / with_c  |  full & short ell")
print("#"*70)

for ell_range_key, (ell_min, ell_max) in ELL_RANGES.items():
    print(f"\n{'='*60}")
    print(f"  ell range: {ell_min}–{ell_max}  ({ell_range_key})")
    print(f"{'='*60}")

    for mode in ['EE', 'BB']:
        fit_data_11 = functions.prepare_mcmc_data(
            spectra_dict,
            band_list=[band_11],
            modes=[mode],
            ell_min=ell_min,
            ell_max=ell_max,
            band_pairs='all',   # → single auto-pair 11_11
        )

        for fit_variant, fit_c in [('no_c', False), ('with_c', True)]:
            tag = f"11 GHz | {mode} | {'with c' if fit_c else 'no c'} | ell {ell_min}-{ell_max}"
            print(f"\n  Fitting: {tag}")

            _, _, samples_free_11, param_map_11, chi2_11 = functions.run_mcmc(
                fit_data=fit_data_11,
                fit_components=('sync',),
                fit_c_terms=fit_c,
                nwalkers=nwalkers,
                ninter=ninter,
                discard_fraction=discard_fraction,
                verbose=True,
                fit_mode='power-law',
                color_correction=False,
                joint_analysis=False,
                cov_matrix=None,
                # beta_s is degenerate with A_s in a single auto-spectrum:
                # freeze it to 0 so the model is C_ell = A_s*(ell/80)^alpha_s [+c]
                freeze_params={'beta_s': 0.0},
                print_residuals=True,
            )

            summary_11 = chain_summary(samples_free_11, param_map_11)
            summary_11['chi2_reduced'] = chi2_11
            results_11[mode][f'{fit_variant}_{ell_range_key}'] = summary_11

            chi2_str = f'{chi2_11:.3f}' if np.isfinite(chi2_11) else 'inf (dof=0)'
            print(f"  χ²_red = {chi2_str}")
            for pname, val in summary_11.items():
                if pname == 'chi2_reduced':
                    continue
                med, ep, em = val
                print(f"    {pname:20s}  {med:+.4e}  +{ep:.2e}  -{em:.2e}")


# ── table builder ──────────────────────────────────────────────────────────
def build_latex_table_11ghz(results_11):
    """
    One table, 4 data columns:
      ell-full / no_c  |  ell-full / with_c  |  ell-short / no_c  |  ell-short / with_c
    Rows: EE block (A_s, alpha_s, c_s) then BB block.
    """
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(
        r'\caption{QUIJOTE 11\,GHz synchrotron power-law fit: '
        r'$C_\ell = A_s\,(\ell/80)^{\alpha_s} [+ c_s]$ with $\beta_s$ frozen to 0. '
        r'Values are median$^{+1\sigma}_{-1\sigma}$ from the MCMC posterior. '
        rf'$A_s$ and $c_s$ are in ${AMPLITUDE_UNIT}$.}}'
    )
    lines.append(r'\label{tab:sync_11ghz}')
    lines.append(r'\begin{tabular}{lcccc}')
    lines.append(r'\toprule')
    lines.append(
        r'& \multicolumn{2}{c}{$\ell \in [30,\,200]$}'
        r' & \multicolumn{2}{c}{$\ell \in [30,\,120]$} \\'
    )
    lines.append(r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}')
    lines.append(r'Parameter & no $c_s$ & with $c_s$ & no $c_s$ & with $c_s$ \\')
    lines.append(r'\midrule')

    def _row(mode, param_label, param_key, rescale=False):
        """Build one data row across all 4 (ell_range × fit_variant) columns."""
        cols = []
        for ell_key in ['full', 'short']:
            for fv in ['no_c', 'with_c']:
                summary = results_11[mode].get(f'{fv}_{ell_key}', {})
                if param_key is None:
                    # c_sync key: find dynamically (name encodes band index)
                    c_key = next((k for k in summary if k.startswith('c_sync')), None)
                    cols.append(
                        fmt_val(summary, c_key, rescale=True) if c_key
                        else r'\multicolumn{1}{c}{---}'
                    )
                else:
                    cols.append(fmt_val(summary, param_key, rescale=rescale))
        return rf'{param_label} & ' + ' & '.join(cols) + r' \\'

    # ── EE ──
    lines.append(r'\multicolumn{5}{l}{\textit{EE mode}} \\')
    lines.append(_row('EE', r'$A_s^{EE}$',      'A_s',    rescale=True))
    lines.append(_row('EE', r'$\alpha_s^{EE}$',  'alpha_s'))
    lines.append(_row('EE', r'$c_s^{EE}$',       None,     rescale=True))

    lines.append(r'\midrule')

    # ── BB ──
    lines.append(r'\multicolumn{5}{l}{\textit{BB mode}} \\')
    lines.append(_row('BB', r'$A_s^{BB}$',      'A_s',    rescale=True))
    lines.append(_row('BB', r'$\alpha_s^{BB}$',  'alpha_s'))
    lines.append(_row('BB', r'$c_s^{BB}$',       None,     rescale=True))

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


# ── save table ─────────────────────────────────────────────────────────────
tex_path = os.path.join(tables_dir, f'sync_11ghz_{mask_name}{name_suffix}_30-200_nocc.tex')
table_str = build_latex_table_11ghz(results_11)
with open(tex_path, 'w') as f:
    f.write(table_str)
print(f"\n[INFO] Table saved: {tex_path}")
print(table_str)
