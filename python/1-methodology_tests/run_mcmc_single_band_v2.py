#%%
# ===========================================================================
#  run_mcmc_single_band_v2.py
#
#  Per-band synchrotron fit:  C_ell = A * (ell/80)^alpha  [+ c]
#
#  Why a new script?
#  -----------------
#  The original run_mcmc_single_band.py calls functions.run_mcmc() which:
#    1. Initialises c_sync walkers at 0 ± 1e-2, far from the true scale
#       (spectra are ~1e-6 – 1e-9 mK²).
#    2. Applies only isfinite() as a prior on c_sync — completely flat, so
#       walkers drift to ±∞ and never converge.
#    3. Has no lower bound on A_s, so negative amplitudes are allowed,
#       causing degenerate solutions.
#
#  This script uses a self-contained emcee loop (à la the TFG code) with:
#    • Data loaded through the standard functions.read_corrected_cls pipeline.
#    • Model: A * (ell/80)^alpha  [+ c],  β_s frozen to 0 (single auto-band).
#    • Priors:  A > 0,  -6 < alpha < 0,
#               c unconstrained but with a Gaussian prior N(0, sigma_c)
#               where sigma_c = median(|data|) to keep it close to data scale.
#    • Walker initialisation from a quick least-squares estimate so walkers
#      start in the right ball-park regardless of data magnitude.
# ===========================================================================

import os
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["NUMEXPR_NUM_THREADS"]  = "1"

import numpy as np
import emcee
import importlib
import functions
importlib.reload(functions)
from data import data, masks

# ── configuration ─────────────────────────────────────────────────────────
nside       = 512
n_sim       = 100
name_suffix = '_full_bin_20-199'
mask_select = masks['QUIJOTE_galcut']['galcut10']
mask_name   = mask_select['name']

out_path   = '/home/pablo/Desktop/master/tfm/spectra/'
tables_dir = '/home/pablo/Desktop/master/tfm/tables/'
os.makedirs(tables_dir, exist_ok=True)

path_corrected_spectra = os.path.join(
    out_path, f'corrected_power_spectra_{mask_name}{name_suffix}.fits'
)

# MCMC settings
nwalkers         = 100    # enough for 2-3 parameters
ninter           = 50000
discard_fraction = 0.5

# Bands and ell ranges
fit_bands = ['11', '23', '30', '33']

ELL_RANGES = {
    'full':  (30, 200),   # 9 bins  → dof = 7 (no_c) / 6 (with_c)
    'short': (30, 120),   # 4 bins  → dof = 2 (no_c) / 1 (with_c)
}

quijote_bands = ['11', '13', '17', '19']
wmap_bands    = ['23', '33', '41', '61', '94']
planck_bands  = ['30', '44', '70', '100', '143', '217', '353']
band_list_full = quijote_bands + wmap_bands + planck_bands

AMPLITUDE_SCALE = 1e9           # mK² → 10^{-3} µK²
AMPLITUDE_UNIT  = r'[10^{-3}\,\mu\mathrm{K}^2_\mathrm{RJ}]'

# ── load data ─────────────────────────────────────────────────────────────
print("[INFO] Loading corrected spectra...")
spectra_dict = functions.read_corrected_cls(path_corrected_spectra, band_list_full)


# ===========================================================================
#  Self-contained MCMC  (no functions.run_mcmc)
# ===========================================================================

def _model_noc(theta, ell):
    """C_ell = A * (ell/80)^alpha"""
    A, alpha = theta
    return A * (ell / 80.0) ** alpha


def _model_withc(theta, ell):
    """C_ell = A * (ell/80)^alpha + c"""
    A, alpha, c = theta
    return A * (ell / 80.0) ** alpha + c


# ── priors ────────────────────────────────────────────────────────────────

def _lnprior_noc(theta, sigma_A):
    """Uniform in alpha; log-normal-ish for A (require A > 0)."""
    A, alpha = theta
    if A <= 0:
        return -np.inf
    if not (-6.0 < alpha < 0.0):
        return -np.inf
    # Weak Gaussian prior on log(A) to help with scale, centred at log(sigma_A)
    log_A = np.log(A)
    log_A0 = np.log(sigma_A)
    return -0.5 * ((log_A - log_A0) / 5.0) ** 2   # very wide: ±5 decades


def _lnprior_withc(theta, sigma_A, sigma_c):
    """A > 0, alpha in (-6,0), Gaussian prior on c."""
    A, alpha, c = theta
    # if A <= 0:
    #     return -np.inf
    if not (-6.0 < alpha < 0.0):
        return -np.inf
    log_A  = np.log(A)
    log_A0 = np.log(sigma_A)
    lp_A   = -0.5 * ((log_A - log_A0) / 5.0) ** 2
    # Gaussian prior on c centred at 0 with scale = sigma_c
    lp_c   = -0.5 * (c / sigma_c) ** 2
    return lp_A + lp_c


def _lnlike(y, yerr, y_model):
    return -0.5 * np.sum(((y - y_model) / yerr) ** 2)


def _lnprob_noc(theta, ell, y, yerr, sigma_A):
    lp = _lnprior_noc(theta, sigma_A)
    if not np.isfinite(lp):
        return -np.inf
    return lp + _lnlike(y, yerr, _model_noc(theta, ell))


def _lnprob_withc(theta, ell, y, yerr, sigma_A, sigma_c):
    lp = _lnprior_withc(theta, sigma_A, sigma_c)
    if not np.isfinite(lp):
        return -np.inf
    return lp + _lnlike(y, yerr, _model_withc(theta, ell))


# ── least-squares initialisation ─────────────────────────────────────────

def _ls_init_noc(ell, y, yerr):
    """
    Simple least-squares estimate in log space for (A, alpha).
    Ignores negative data points (can't take log), falls back if needed.
    """
    mask = y > 0
    if mask.sum() < 2:
        # fallback: use typical amplitude
        A0     = float(np.median(np.abs(y)))
        alpha0 = -3.0
    else:
        log_y    = np.log(y[mask])
        log_ell  = np.log(ell[mask] / 80.0)
        # Weighted linear regression: log_y = log(A) + alpha * log_ell
        w = 1.0 / (yerr[mask] / y[mask]) ** 2
        W   = np.sum(w)
        Wx  = np.sum(w * log_ell)
        Wy  = np.sum(w * log_y)
        Wxx = np.sum(w * log_ell ** 2)
        Wxy = np.sum(w * log_ell * log_y)
        denom = W * Wxx - Wx ** 2
        if abs(denom) < 1e-30:
            alpha0 = -3.0
            A0 = float(np.exp(np.median(log_y)))
        else:
            alpha0 = float((W * Wxy - Wx * Wy) / denom)
            A0     = float(np.exp((Wy - alpha0 * Wx) / W))
    # Clip alpha to valid range
    alpha0 = float(np.clip(alpha0, -5.5, -0.5))
    A0     = max(A0, 1e-20)
    return A0, alpha0


def _ls_init_withc(ell, y, yerr):
    """
    Least-squares for (A, alpha, c) by iterative subtraction:
    start with c=0, estimate (A,alpha), then estimate c from residuals.
    """
    A0, alpha0 = _ls_init_noc(ell, y, yerr)
    residuals  = y - _model_noc([A0, alpha0], ell)
    c0 = float(np.average(residuals, weights=1.0 / yerr ** 2))
    return A0, alpha0, c0


def _chain_summary(samples):
    """Return (median, +1σ, -1σ) for each parameter."""
    results = []
    for i in range(samples.shape[1]):
        col = samples[:, i]
        med       = np.median(col)
        lo, hi    = np.percentile(col, [16, 84])
        results.append((med, hi - med, med - lo))
    return results


def run_single_band_mcmc(ell, y, yerr, fit_c, nwalkers, ninter, discard_fraction):
    """
    Run MCMC for the model  C_ell = A*(ell/80)^alpha [+ c].

    Parameters
    ----------
    ell, y, yerr : arrays
    fit_c        : bool – include constant term
    nwalkers, ninter, discard_fraction : MCMC settings

    Returns
    -------
    samples : ndarray  (n_samples × ndim)
    summary : list of (median, +err, -err) per parameter
    chi2_red : float
    """
    # --- characteristic scales for priors / initialisation
    sigma_A = float(np.median(np.abs(y[y > 0]))) if np.any(y > 0) else float(np.median(np.abs(y)))
    sigma_c = float(np.median(np.abs(y)))   # order-of-magnitude prior on c

    if not fit_c:
        ndim = 2
        A0, alpha0    = _ls_init_noc(ell, y, yerr)
        p0_center     = np.array([A0, alpha0])
        # Spread walkers: relative for A, absolute for alpha
        spread        = np.array([0.3 * A0, 0.3])
        lnprob_fn     = _lnprob_noc
        extra_args    = (sigma_A,)
        param_labels  = ['A', 'alpha']
    else:
        ndim = 3
        A0, alpha0, c0 = _ls_init_withc(ell, y, yerr)
        p0_center      = np.array([A0, alpha0, c0])
        spread         = np.array([0.3 * A0, 0.3, 0.3 * sigma_c])
        lnprob_fn      = _lnprob_withc
        extra_args     = (sigma_A, sigma_c)
        param_labels   = ['A', 'alpha', 'c']

    # Initialise walkers around p0_center with Gaussian scatter
    rng = np.random.default_rng(42)
    p0  = p0_center + spread * rng.standard_normal((nwalkers, ndim))
    # Enforce A > 0 in initial positions
    p0[:, 0] = np.abs(p0[:, 0])

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, lnprob_fn,
        args=(ell, y, yerr) + extra_args
    )

    # Burn-in: 20 % of ninter
    burn  = max(100, int(0.2 * ninter))
    p0_b, _, _ = sampler.run_mcmc(p0, burn, progress=False)
    sampler.reset()

    # Production run
    sampler.run_mcmc(p0_b, ninter, progress=False)

    discard  = int(ninter * discard_fraction)
    samples  = sampler.get_chain(discard=discard, flat=True, thin=10)
    summary  = _chain_summary(samples)

    # chi²_red at best fit
    # Use the same thinned+discarded log-prob so the index matches samples
    log_prob_flat = sampler.get_log_prob(discard=discard, flat=True, thin=10)
    best_idx = np.argmax(log_prob_flat)
    best     = samples[best_idx]
    if fit_c:
        y_model = _model_withc(best, ell)
    else:
        y_model = _model_noc(best, ell)
    chi2     = np.sum(((y - y_model) / yerr) ** 2)
    dof      = len(y) - ndim
    chi2_red = chi2 / dof if dof > 0 else np.inf

    return samples, summary, chi2_red


# ===========================================================================
#  Main loop
# ===========================================================================

# all_results[ell_range_key][band][mode][fit_variant] = {
#     'A':  (med, +err, -err),
#     'alpha': ...,
#     'c':     ...  (only if fit_variant='with_c'),
#     'chi2_reduced': float
# }
all_results = {}

for ell_range_key, (ell_min, ell_max) in ELL_RANGES.items():
    print(f"\n{'#'*60}")
    print(f"  ELL RANGE: {ell_min} – {ell_max}  ({ell_range_key})")
    print(f"{'#'*60}")

    results = {b: {m: {} for m in ['EE', 'BB']} for b in fit_bands}

    for band in fit_bands:
        auto_pair = f'{band}_{band}'
        if auto_pair not in spectra_dict:
            print(f"  [WARN] {auto_pair} not found in spectra, skipping.")
            continue

        sp_entry = spectra_dict[auto_pair]
        ell_eff  = np.asarray(sp_entry['ell_eff'])

        for mode in ['EE', 'BB']:
            spec = np.asarray(sp_entry[mode]['SPECTRUM'])
            err  = np.asarray(sp_entry[mode]['ERROR'])

            # Apply ell cut
            mask = (ell_eff >= ell_min) & (ell_eff < ell_max)
            ell_fit  = ell_eff[mask]
            spec_fit = spec[mask]
            err_fit  = err[mask]

            if len(ell_fit) < 2:
                print(f"  [WARN] Not enough ell bins for {band} GHz {mode}, skipping.")
                continue

            for fit_variant, fit_c in [('no_c', False), ('with_c', True)]:
                tag = f"{band} GHz | {mode} | {'with c' if fit_c else 'no c'}"
                print(f"\n  Fitting: {tag}")

                samples, summary, chi2_red = run_single_band_mcmc(
                    ell_fit, spec_fit, err_fit,
                    fit_c=fit_c,
                    nwalkers=nwalkers,
                    ninter=ninter,
                    discard_fraction=discard_fraction,
                )

                res = {}
                param_labels = ['A', 'alpha'] if not fit_c else ['A', 'alpha', 'c']
                for i, lbl in enumerate(param_labels):
                    res[lbl] = summary[i]
                res['chi2_reduced'] = chi2_red

                results[band][mode][fit_variant] = res

                chi2_str = f'{chi2_red:.3f}' if np.isfinite(chi2_red) else 'inf'
                print(f"    χ²_red = {chi2_str}")
                for pname, val in res.items():
                    if pname == 'chi2_reduced':
                        continue
                    med, ep, em = val
                    print(f"      {pname:6s}  {med:+.4e}  +{ep:.2e}  -{em:.2e}")

    all_results[ell_range_key] = results


# ===========================================================================
#  LaTeX table builder
# ===========================================================================

def fmt_val(summary_dict, key):
    """Format (med, +err, -err) as  $med^{+err}_{-err}$  or '---'."""
    if key not in summary_dict:
        return r'\multicolumn{1}{c}{---}'
    med, ep, em = summary_dict[key]
    # choose appropriate unit: A and c in 10^{-3} µK², alpha dimensionless
    if key in ('A', 'c'):
        # rescale mK² → 10^{-3} µK²
        med *= AMPLITUDE_SCALE
        ep  *= AMPLITUDE_SCALE
        em  *= AMPLITUDE_SCALE
    nd = 3
    fmt = f"{{:.{nd}f}}"
    def sf(x):
        return fmt.format(x) if np.isfinite(x) else 'nan'
    return rf'${sf(med)}^{{+{sf(ep)}}}_{{-{sf(em)}}}$'


def build_latex_table(results, fit_variant, with_c, ell_min, ell_max):
    caption_tag = r'with constant $c_s$' if with_c else r'without constant $c_s$'
    label_tag   = 'withc' if with_c else 'noc'

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(
        rf'\caption{{Synchrotron power-law fit per band ({caption_tag}), '
        rf'$\ell \in [{ell_min}, {ell_max}]$. '
        r'Model: $C_\ell = A_s\,(\ell/80)^{{\alpha_s}} [+ c_s]$, '
        r'$\beta_s$ frozen to 0. '
        r'Values are median$^{{+1\sigma}}_{{-1\sigma}}$. '
        rf'$A_s$ and $c_s$ in ${AMPLITUDE_UNIT}$.}}'
    )
    lines.append(rf'\label{{tab:sync_single_band_{label_tag}_ell{ell_min}-{ell_max}}}')
    lines.append(r'\begin{tabular}{lcccc}')
    lines.append(r'\toprule')
    lines.append(r'Parameter & 11\,GHz & 23\,GHz & 30\,GHz & 33\,GHz \\')
    lines.append(r'\midrule')

    # EE block
    ee_params = [(r'$A_s^{EE}$', 'A'), (r'$\alpha_s^{EE}$', 'alpha')]
    if with_c:
        ee_params.append((r'$c_s^{EE}$', 'c'))
    for label, key in ee_params:
        row = [fmt_val(results[b]['EE'][fit_variant], key) for b in fit_bands]
        lines.append(rf'{label} & ' + ' & '.join(row) + r' \\')

    lines.append(r'\midrule')

    # BB block
    bb_params = [(r'$A_s^{BB}$', 'A'), (r'$\alpha_s^{BB}$', 'alpha')]
    if with_c:
        bb_params.append((r'$c_s^{BB}$', 'c'))
    for label, key in bb_params:
        row = [fmt_val(results[b]['BB'][fit_variant], key) for b in fit_bands]
        lines.append(rf'{label} & ' + ' & '.join(row) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


# ===========================================================================
#  Save tables
# ===========================================================================
for ell_range_key, (ell_min, ell_max) in ELL_RANGES.items():
    results = all_results[ell_range_key]
    for fit_variant, with_c in [('no_c', False), ('with_c', True)]:
        tag = 'withc' if with_c else 'noc'
        tex_path = os.path.join(
            tables_dir,
            f'sync_single_band_v2_{tag}_{mask_name}_ell{ell_min}-{ell_max}{name_suffix}.tex'
        )
        table = build_latex_table(results, fit_variant, with_c, ell_min, ell_max)
        with open(tex_path, 'w') as f:
            f.write(table)
        print(f"\n[INFO] Table saved: {tex_path}")
        print(table)


# %%
