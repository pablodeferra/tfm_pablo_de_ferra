#%%
import sys
sys.path.append('../')
import functions
import os
from data import data, path_map, masks, path_masks, color_corrections
import healpy as hp
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.constants import h, k  # Planck constant, Boltzmann constant

# Global plotting style
plt.rcParams.update({
    'axes.titlesize': 18,
    'axes.labelsize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 15,
    'font.size': 15,
})

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
nside       = 512
n_sim       = 100
name_suffix = '_full_bin_20-199'

quijote_bands = ['11', '13', '17', '19']
wmap_bands    = ['23', '33', '41', '61', '94']
planck_bands  = ['30', '44', '70', '100', '143', '217', '353']
band_list     = quijote_bands + wmap_bands + planck_bands

out_path = '/home/pablo/Desktop/master/tfm/spectra/'

freq_ref      = 23.0    # synchrotron reference frequency [GHz]
freq_ref_dust = 353.0   # dust reference frequency [GHz]
T_d           = 19.6    # dust temperature [K]
ell_ref       = 80.0
ell_fine      = np.linspace(20, 200, 300)


FIT_PARAMS = {
    # |b| > 10°  — QJ+WP+Pl, cross-only, ell 20-200
    'quijote_galcut10': {
        'EE': dict(A_s=7.418e-9,  alpha_s=-3.623, beta_s=-3.088,
                   A_d=3.181e-9,  alpha_d=-2.542, beta_d=1.521, rho=0.095),
        'BB': dict(A_s=1.703e-9,  alpha_s=-3.291, beta_s=-2.960,
                   A_d=2.395e-9,  alpha_d=-2.204, beta_d=1.556, rho=0.114),
    },
    # |b| > 15°  — QJ+WP+Pl, cross-only, ell 20-200
    'quijote_galcut15': {
        'EE': dict(A_s=5.998e-9,  alpha_s=-3.764, beta_s=-3.086,
                   A_d=2.066e-9,  alpha_d=-2.514, beta_d=1.534, rho=0.068),
        'BB': dict(A_s=1.363e-9,  alpha_s=-3.405, beta_s=-3.003,
                   A_d=1.380e-9,  alpha_d=-2.323, beta_d=1.541, rho=0.118),
    },
    # |b| > 20°  — QJ+WP+Pl, cross-only, ell 20-200
    'quijote_galcut20': {
        'EE': dict(A_s=5.908e-9,  alpha_s=-3.726, beta_s=-3.129,
                   A_d=1.411e-9,  alpha_d=-2.475, beta_d=1.461, rho=0.022),
        'BB': dict(A_s=1.358e-9,  alpha_s=-3.326, beta_s=-2.978,
                   A_d=0.926e-9,  alpha_d=-2.242, beta_d=1.539, rho=0.123),
    },
}

# Visual style per mask
MASK_STYLES = {
    'quijote_galcut10': dict(color='steelblue',  ls='-',  label=r'$|b|>10\degree$'),
    'quijote_galcut15': dict(color='darkorange', ls='-', label=r'$|b|>15\degree$'),
    'quijote_galcut20': dict(color='firebrick',  ls='-',  label=r'$|b|>20\degree$'),
}

# ---------------------------------------------------------------
# Load corrected spectra for each mask
# ---------------------------------------------------------------
SPECTRA = {}
for mask_key in ('galcut10', 'galcut15', 'galcut20'):
    mask_name = masks['QUIJOTE_galcut'][mask_key]['name']
    path_corrected = os.path.join(
        out_path, f'corrected_power_spectra_{mask_name}{name_suffix}.fits')
    SPECTRA[mask_name] = functions.read_corrected_cls(path_corrected, band_list)

# ---------------------------------------------------------------
# Load colour-correction polynomials (optional)
# ---------------------------------------------------------------
try:
    cc_dict = functions.load_color_correction_polynomials()
except Exception:
    cc_dict = None

# ---------------------------------------------------------------
# MBB SED scaling in K_RJ units
# ---------------------------------------------------------------
def _planck_factor(nu_GHz, T):
    """B_nu proportional factor used in MBB (no RJ normalisation needed)."""
    nu = np.asarray(nu_GHz, dtype=float) * 1e9
    x  = h * nu / (k * T)
    return x**2 * np.exp(x) / np.expm1(x)**2

def mbb_scaling_KRJ(nu_GHz, beta, Td=T_d):
    """MBB SED scaling in K_RJ units relative to freq_ref_dust."""
    nu_GHz = np.asarray(nu_GHz, dtype=float)
    return ((nu_GHz / freq_ref_dust) ** beta
            * _planck_factor(nu_GHz, Td) / _planck_factor(freq_ref_dust, Td))

# ---------------------------------------------------------------
# Colour-correction factor for a single band
# ---------------------------------------------------------------
def _cc(band_str, component, alpha_cc):
    if cc_dict is None:
        return 1.0
    poly = (cc_dict.get(component, {}) or {}).get(band_str)
    if poly is None:
        return 1.0
    return poly[0] + poly[1] * alpha_cc + poly[2] * alpha_cc**2

# ---------------------------------------------------------------
# Full model: synch + dust + cross-correlation
# Mirrors model_synchrotron / model_dust / model_cross in functions.py
# ---------------------------------------------------------------
def full_model(ell_arr, f1, f2, p):
    """
    Parameters
    ----------
    ell_arr : array-like
    f1, f2  : float  — frequencies in GHz
    p       : dict   — {A_s, alpha_s, beta_s, A_d, alpha_d, beta_d, rho}
    """
    A_s, alpha_s, beta_s = p['A_s'], p['alpha_s'], p['beta_s']
    A_d, alpha_d, beta_d = p['A_d'], p['alpha_d'], p['beta_d']
    rho                  = p['rho']

    b1s = str(int(float(f1)))
    b2s = str(int(float(f2)))

    # synchrotron
    alpha_s_cc = 2.0 + beta_s
    cc_s1 = _cc(b1s, 'synch', alpha_s_cc)
    cc_s2 = _cc(b2s, 'synch', alpha_s_cc)
    cl_s  = (A_s
             * (ell_arr / ell_ref) ** alpha_s
             * (f1 / freq_ref) ** beta_s
             * (f2 / freq_ref) ** beta_s
             / (cc_s1 * cc_s2))

    # dust
    alpha_d_cc = 2.0 + beta_d
    cc_d1 = _cc(b1s, 'dust', alpha_d_cc)
    cc_d2 = _cc(b2s, 'dust', alpha_d_cc)
    d1 = float(mbb_scaling_KRJ(f1, beta_d))
    d2 = float(mbb_scaling_KRJ(f2, beta_d))
    cl_d  = (A_d
             * (ell_arr / ell_ref) ** alpha_d
             * d1 * d2
             / (cc_d1 * cc_d2))

    # cross-correlation
    s1 = (f1 / freq_ref) ** beta_s
    s2 = (f2 / freq_ref) ** beta_s
    ell_scale_cross = (ell_arr / ell_ref) ** ((alpha_s + alpha_d) / 2.0)
    mix = (s1 / cc_s1) * (d2 / cc_d2) + (s2 / cc_s2) * (d1 / cc_d1)
    cl_corr = rho * np.sqrt(A_s * A_d) * mix * ell_scale_cross

    return cl_s + cl_d + cl_corr

# ---------------------------------------------------------------
# Transparent-PNG helper (black -> white)
# ---------------------------------------------------------------
def save_figure_transparent_white(fig, out_path, dpi=300):
    """Save fig as a transparent PNG, converting black elements to white."""
    def is_black(c):
        try:
            if isinstance(c, str):
                return c.lower() in ('k', 'black', '#000000')
            if hasattr(c, '__len__'):
                rgba = mpl.colors.to_rgba(c)
                return rgba[0] < 1e-6 and rgba[1] < 1e-6 and rgba[2] < 1e-6
        except Exception:
            pass
        return False

    fig.patch.set_alpha(0.0)
    for ax in fig.axes:
        try:
            ax.set_facecolor('none')
        except Exception:
            pass
        for attr in (ax.title, ax.xaxis.label, ax.yaxis.label):
            try:
                if is_black(attr.get_color()):
                    attr.set_color('white')
            except Exception:
                pass
        try:
            ax.tick_params(colors='white')
            for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                if is_black(lbl.get_color()):
                    lbl.set_color('white')
        except Exception:
            pass
        for spine in ax.spines.values():
            try:
                if is_black(spine.get_edgecolor()):
                    spine.set_edgecolor('white')
            except Exception:
                pass
        leg = ax.get_legend()
        if leg is not None:
            for txt in leg.get_texts():
                try:
                    if is_black(txt.get_color()):
                        txt.set_color('white')
                except Exception:
                    pass
            try:
                frame = leg.get_frame()
                frame.set_alpha(0.0)
                if is_black(frame.get_edgecolor()):
                    frame.set_edgecolor('white')
            except Exception:
                pass
        for line in ax.get_lines():
            try:
                if is_black(line.get_color()):
                    line.set_color('white')
            except Exception:
                pass
        for coll in ax.collections:
            try:
                fc = coll.get_facecolors()
                if fc is not None and len(fc) and is_black(tuple(fc[0])):
                    new = np.array(fc); new[:, :3] = 1.0; coll.set_facecolors(new)
                ec = coll.get_edgecolors()
                if ec is not None and len(ec) and is_black(tuple(ec[0])):
                    new = np.array(ec); new[:, :3] = 1.0; coll.set_edgecolors(new)
            except Exception:
                pass
    for txt in fig.findobj(mpl.text.Text):
        try:
            if is_black(txt.get_color()):
                txt.set_color('white')
        except Exception:
            pass
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, transparent=True, facecolor='none',
                bbox_inches='tight', dpi=dpi)
    print('Saved transparent PNG to', out_path)


# =================================================================
# FIGURES — one synch figure + one dust figure per mask (6 total)
# Cross-only QJ+WP+Pl fit
# =================================================================
synch_auto_bands  = ['11', '23', '30']
synch_cross_pairs = [('11', '23'), ('11', '30'), ('23', '30')]
synch_band_colors  = {'11': 'steelblue', '23': 'k', '30': 'goldenrod'}
synch_cross_colors = list(synch_band_colors.values())
synch_plot_freqs   = {b: float(b) for b in synch_band_colors}

dust_auto_bands  = ['100', '217', '353']
dust_cross_pairs = [('100', '217'), ('100', '353'), ('217', '353')]
dust_band_colors  = {'100': 'steelblue', '217': 'k', '353': 'goldenrod'}
dust_cross_colors = list(dust_band_colors.values())
dust_plot_freqs   = {b: float(b) for b in dust_band_colors}

for mask_name, spectra_plot in SPECTRA.items():
    mstyle = MASK_STYLES[mask_name]
    params = FIT_PARAMS[mask_name]
    mask_label = mstyle['label']   # e.g. r'$|b|>10\degree$'

    # -----------------------------------------------------------------
    # FIGURE 1 — Synchrotron-dominated bands: 11, 23, 30 GHz
    # -----------------------------------------------------------------
    fig_s, axes_s = plt.subplots(2, 2, figsize=(14, 10))
    # fig_s.suptitle(f'Synchrotron bands — cross-only fit ({mask_label})', fontsize=16)
    ax_sEE_auto,  ax_sBB_auto  = axes_s[0]
    ax_sEE_cross, ax_sBB_cross = axes_s[1]
    ax_sEE_auto.set_title('Auto-spectra — EE')
    ax_sBB_auto.set_title('Auto-spectra — BB')
    ax_sEE_cross.set_title('Cross-spectra — EE')
    ax_sBB_cross.set_title('Cross-spectra — BB')

    # auto spectra
    for b in synch_auto_bands:
        key = f'{b}_{b}'
        if key not in spectra_plot:
            continue
        ell_eff = spectra_plot[key]['ell_eff']
        f   = synch_plot_freqs[b]
        col = synch_band_colors[b]
        for ax, mode in [(ax_sEE_auto, 'EE'), (ax_sBB_auto, 'BB')]:
            cl  = spectra_plot[key][mode]['SPECTRUM'] * 1e6
            err = spectra_plot[key][mode]['ERROR']    * 1e6
            ax.errorbar(ell_eff, cl, yerr=err, fmt='o', color=col,
                        capsize=2, ms=4, label=f'{b}x{b} GHz')
            ax.plot(ell_fine,
                    full_model(ell_fine, f, f, params[mode]) * 1e6,
                    color=col, ls=mstyle['ls'], lw=1.8,
                    label=f'{b}x{b} – fit' if ax == ax_sEE_auto else None)

    # cross spectra
    for i, (b1, b2) in enumerate(synch_cross_pairs):
        key = f'{b1}_{b2}'
        if key not in spectra_plot:
            key = f'{b2}_{b1}'
        if key not in spectra_plot:
            continue
        ell_eff = spectra_plot[key]['ell_eff']
        f1, f2  = synch_plot_freqs[b1], synch_plot_freqs[b2]
        col     = synch_cross_colors[i % len(synch_cross_colors)]
        for ax, mode in [(ax_sEE_cross, 'EE'), (ax_sBB_cross, 'BB')]:
            cl  = spectra_plot[key][mode]['SPECTRUM'] * 1e6
            err = spectra_plot[key][mode]['ERROR']    * 1e6
            ax.errorbar(ell_eff, cl, yerr=err, fmt='o', color=col,
                        capsize=2, ms=4, label=f'{b1}x{b2} GHz')
            ax.plot(ell_fine,
                    full_model(ell_fine, f1, f2, params[mode]) * 1e6,
                    color=col, ls=mstyle['ls'], lw=1.8)

    for ax in axes_s.flat:
        ax.set_yscale('log')
        ax.set_xlabel(r'$\ell$')
        ax.set_ylabel(r'$C_\ell \; [\mu\mathrm{K}^2]$')
        ax.legend(frameon=False, fontsize=11)

    plt.tight_layout()
    plt.show()

    fig_s.savefig(
        f'/home/pablo/Desktop/master/tfm/figures/spectra/spectra_synch_bands_full_model_fit_cross_only_{mask_name}.pdf')
    save_figure_transparent_white(
        fig_s,
        f'/home/pablo/Desktop/master/tfm/figures_ppt/spectra/spectra_synch_bands_full_model_fit_cross_only_{mask_name}_transparent.png',
        dpi=300)
    plt.close(fig_s)

    # -----------------------------------------------------------------
    # FIGURE 2 — Dust-dominated bands: 100, 217, 353 GHz
    # -----------------------------------------------------------------
    fig_d, axes_d = plt.subplots(2, 2, figsize=(14, 10))
    # fig_d.suptitle(f'Dust bands — cross-only fit ({mask_label})', fontsize=16)
    ax_dEE_auto,  ax_dBB_auto  = axes_d[0]
    ax_dEE_cross, ax_dBB_cross = axes_d[1]
    ax_dEE_auto.set_title('Auto-spectra — EE')
    ax_dBB_auto.set_title('Auto-spectra — BB')
    ax_dEE_cross.set_title('Cross-spectra — EE')
    ax_dBB_cross.set_title('Cross-spectra — BB')

    # auto spectra
    for b in dust_auto_bands:
        key = f'{b}_{b}'
        if key not in spectra_plot:
            continue
        ell_eff = spectra_plot[key]['ell_eff']
        f   = dust_plot_freqs[b]
        col = dust_band_colors[b]
        for ax, mode in [(ax_dEE_auto, 'EE'), (ax_dBB_auto, 'BB')]:
            cl  = spectra_plot[key][mode]['SPECTRUM'] * 1e6
            err = spectra_plot[key][mode]['ERROR']    * 1e6
            ax.errorbar(ell_eff, cl, yerr=err, fmt='o', color=col,
                        capsize=2, ms=4, label=f'{b}x{b} GHz')
            ax.plot(ell_fine,
                    full_model(ell_fine, f, f, params[mode]) * 1e6,
                    color=col, ls=mstyle['ls'], lw=1.8,
                    label=f'{b}x{b} – fit' if ax == ax_dEE_auto else None)

    # cross spectra
    for i, (b1, b2) in enumerate(dust_cross_pairs):
        key = f'{b1}_{b2}'
        if key not in spectra_plot:
            key = f'{b2}_{b1}'
        if key not in spectra_plot:
            continue
        ell_eff = spectra_plot[key]['ell_eff']
        f1, f2  = dust_plot_freqs[b1], dust_plot_freqs[b2]
        col     = dust_cross_colors[i % len(dust_cross_colors)]
        for ax, mode in [(ax_dEE_cross, 'EE'), (ax_dBB_cross, 'BB')]:
            cl  = spectra_plot[key][mode]['SPECTRUM'] * 1e6
            err = spectra_plot[key][mode]['ERROR']    * 1e6
            ax.errorbar(ell_eff, cl, yerr=err, fmt='o', color=col,
                        capsize=2, ms=4, label=f'{b1}x{b2} GHz')
            ax.plot(ell_fine,
                    full_model(ell_fine, f1, f2, params[mode]) * 1e6,
                    color=col, ls=mstyle['ls'], lw=1.8)

    for ax in axes_d.flat:
        ax.set_yscale('log')
        ax.set_xlabel(r'$\ell$')
        ax.set_ylabel(r'$C_\ell \; [\mu\mathrm{K}^2]$')
        ax.legend(frameon=False, fontsize=11)

    plt.tight_layout()
    plt.show()

    fig_d.savefig(
        f'/home/pablo/Desktop/master/tfm/figures/spectra/spectra_dust_bands_full_model_fit_cross_only_{mask_name}.pdf')
    save_figure_transparent_white(
        fig_d,
        f'/home/pablo/Desktop/master/tfm/figures_ppt/spectra/spectra_dust_bands_full_model_fit_cross_only_{mask_name}_transparent.png',
        dpi=300)
    plt.close(fig_d)
