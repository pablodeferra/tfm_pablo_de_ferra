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
nside      = 512
n_sim      = 100
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

# ---------------------------------------------------------------
# Model parameters — all-pairs fit (auto+cross), full model
# ---------------------------------------------------------------
# Format: 'quijote_galcutXX': { 'EE': dict(...), 'BB': dict(...) }
# Amplitudes in K^2 (plots multiply by 1e6 to get µK^2).

FIT_PARAMS = {
    'quijote_galcut10': {
        'EE': dict(A_s=7.398e-9, alpha_s=-3.619, beta_s=-3.085,
                   A_d=3.216e-9, alpha_d=-2.547, beta_d=1.530, rho=0.096),
        'BB': dict(A_s=1.657e-9, alpha_s=-3.334, beta_s=-2.999,
                   A_d=2.359e-9, alpha_d=-2.203, beta_d=1.530, rho=0.111),
    },
    'quijote_galcut15': {
        'EE': dict(A_s=5.990e-9, alpha_s=-3.767, beta_s=-3.079,
                   A_d=2.092e-9, alpha_d=-2.535, beta_d=1.521, rho=0.068),
        'BB': dict(A_s=1.367e-9, alpha_s=-3.410, beta_s=-3.017,
                   A_d=1.379e-9, alpha_d=-2.317, beta_d=1.531, rho=0.116),
    },
    'quijote_galcut20': {
        'EE': dict(A_s=5.880e-9, alpha_s=-3.732, beta_s=-3.113,
                   A_d=1.439e-9, alpha_d=-2.478, beta_d=1.486, rho=0.023),
        'BB': dict(A_s=1.347e-9, alpha_s=-3.371, beta_s=-2.984,
                   A_d=0.927e-9, alpha_d=-2.238, beta_d=1.533, rho=0.121),
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
    # fig_s.suptitle(f'Synchrotron bands — full model fit ({mask_label})', fontsize=16)
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
                    color=col, ls=mstyle['ls'], lw=1.8)

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
            ax.errorbar(ell_eff, cl, yerr=err, fmt='^', color=col,
                        capsize=2, ms=6, label=f'{b1}x{b2} GHz')
            ax.plot(ell_fine,
                    full_model(ell_fine, f1, f2, params[mode]) * 1e6,
                    color=col, ls=mstyle['ls'], lw=1.8)

    for ax in axes_s.flat:
        ax.set_yscale('log')
        ax.set_xlabel(r'$\ell$')
        ax.set_ylabel(r'$C_\ell \; [\mu\mathrm{K}^2]$')
        ax.legend(frameon=False, fontsize=15, loc='upper right')

    plt.tight_layout()
    plt.show()

    fig_s.savefig(
        f'/home/pablo/Desktop/master/tfm/figures/spectra/spectra_synch_bands_full_model_fit_{mask_name}.pdf')
    save_figure_transparent_white(
        fig_s,
        f'/home/pablo/Desktop/master/tfm/figures_ppt/spectra/spectra_synch_bands_full_model_fit_{mask_name}_transparent.png',
        dpi=300)
    plt.close(fig_s)

    # -----------------------------------------------------------------
    # FIGURE 2 — Dust-dominated bands: 100, 217, 353 GHz
    # -----------------------------------------------------------------
    fig_d, axes_d = plt.subplots(2, 2, figsize=(14, 10))
    # fig_d.suptitle(f'Dust bands — full model fit ({mask_label})', fontsize=16)
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
                    color=col, ls=mstyle['ls'], lw=1.8)

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
            ax.errorbar(ell_eff, cl, yerr=err, fmt='^', color=col,
                        capsize=2, ms=6, label=f'{b1}x{b2} GHz')
            ax.plot(ell_fine,
                    full_model(ell_fine, f1, f2, params[mode]) * 1e6,
                    color=col, ls=mstyle['ls'], lw=1.8)

    for ax in axes_d.flat:
        ax.set_yscale('log')
        ax.set_xlabel(r'$\ell$')
        ax.set_ylabel(r'$C_\ell \; [\mu\mathrm{K}^2]$')
        ax.legend(frameon=False, fontsize=15, loc='upper right')

    plt.tight_layout()
    plt.show()

    fig_d.savefig(
        f'/home/pablo/Desktop/master/tfm/figures/spectra/spectra_dust_bands_full_model_fit_{mask_name}.pdf')
    save_figure_transparent_white(
        fig_d,
        f'/home/pablo/Desktop/master/tfm/figures_ppt/spectra/spectra_dust_bands_full_model_fit_{mask_name}_transparent.png',
        dpi=300)
    plt.close(fig_d)

# =================================================================
# COMBINED FIGURES — 3 rows (masks) × 4 columns
#   Figure A: synchrotron bands  11, 23, 30 GHz
#   Figure B: dust bands        100, 217, 353 GHz
# Each panel shows exactly 3 curves (auto or cross), with data
# points and the corresponding model fit line.
# =================================================================

#%%
mask_order = ['quijote_galcut10', 'quijote_galcut15', 'quijote_galcut20']
row_labels = [r'$10°$', r'$15°$', r'$20°$']
col_titles = ['Auto-spectra — EE', 'Auto-spectra — BB',
              'Cross-spectra — EE', 'Cross-spectra — BB']

def _make_combined(band_cfgs, cross_cfgs, fig_label):
    """
    band_cfgs  : list of dict(b, f, color, label)  — 3 auto bands
    cross_cfgs : list of dict(b1, b2, f1, f2, color, label)  — 3 cross pairs
    fig_label  : str used in output filenames
    """
    fig, axes = plt.subplots(
        3, 4, figsize=(27, 14),
        sharex=True,
        gridspec_kw={'hspace': 0.08, 'wspace': 0.28},
    )

    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=15)

    for row_idx, mask_name in enumerate(mask_order):
        spectra_plot = SPECTRA[mask_name]
        params       = FIT_PARAMS[mask_name]

        # left column: only show the numeric y-axis label (no mask name)
        axes[row_idx, 0].set_ylabel(
            r'$C_\ell\;[\mu\mathrm{K}^2]$',
            fontsize=15,
        )

        for col_idx, mode in [(0, 'EE'), (1, 'BB'), (2, 'EE'), (3, 'BB')]:
            ax       = axes[row_idx, col_idx]
            is_cross = col_idx >= 2
            cfgs     = cross_cfgs if is_cross else band_cfgs

            for cfg in cfgs:
                if is_cross:
                    key = f"{cfg['b1']}_{cfg['b2']}"
                    if key not in spectra_plot:
                        key = f"{cfg['b2']}_{cfg['b1']}"
                    if key not in spectra_plot:
                        continue
                    f1, f2 = cfg['f1'], cfg['f2']
                else:
                    key = f"{cfg['b']}_{cfg['b']}"
                    if key not in spectra_plot:
                        continue
                    f1 = f2 = cfg['f']

                ell_eff = spectra_plot[key]['ell_eff']
                cl  = spectra_plot[key][mode]['SPECTRUM'] * 1e6
                err = spectra_plot[key][mode]['ERROR']    * 1e6

                marker = '^' if is_cross else 'o'
                msize = 6 if is_cross else 4
                ax.errorbar(ell_eff, cl, yerr=err,
                            fmt=marker, color=cfg['color'], ms=msize, capsize=2,
                            lw=1.0, label=cfg['label'])
                ax.plot(ell_fine,
                        full_model(ell_fine, f1, f2, params[mode]) * 1e6,
                        color=cfg['color'], ls='-', lw=1.8)

            ax.set_yscale('log')
            # ax.set_xscale('log')
            if col_idx != 0:
                ax.set_ylabel('')
            if row_idx == 2:
                ax.set_xlabel(r'$\ell$', fontsize=15)

        # legend in the first column only (keep as before)
        handles, labels = axes[0, 1].get_legend_handles_labels()
        axes[0, 1].legend(handles, labels, frameon=False,
                                fontsize=15, loc='upper right')
        
        handles, labels = axes[0, 3].get_legend_handles_labels()
        axes[0, 3].legend(handles, labels, frameon=False,
                                fontsize=15, loc='upper right')

        # place the mask label at the right of the row (outside the last subplot)
        try:
            right_ax = axes[row_idx, -1]
            # position in axes coordinates (x slightly >1 places it to the right)
            right_ax.text(1.03, 0.5, row_labels[row_idx], transform=right_ax.transAxes,
                          va='center', ha='left', fontsize=18)
        except Exception:
            # fallback: put the text in figure coordinates near the right edge
            fig.text(0.98, 0.66 - row_idx * 0.33, row_labels[row_idx],
                     va='center', ha='right', fontsize=18)

    plt.tight_layout()
    plt.show()

    fig.savefig(
        f'/home/pablo/Desktop/master/tfm/figures/spectra/spectra_combined_3masks_{fig_label}.pdf',
        bbox_inches='tight')
    save_figure_transparent_white(
        fig,
        f'/home/pablo/Desktop/master/tfm/figures_ppt/spectra/spectra_combined_3masks_{fig_label}_transparent.png',
        dpi=300)
    plt.close(fig)


# --- Synchrotron combined figure: 11, 23, 30 GHz ---
_make_combined(
    band_cfgs=[
        dict(b='11', f=11.0,  color='steelblue', label='11×11 GHz'),
        dict(b='23', f=23.0,  color='k',         label='23×23 GHz'),
        dict(b='30', f=30.0,  color='goldenrod',  label='30×30 GHz'),
    ],
    cross_cfgs=[
        dict(b1='11', b2='23', f1=11.0, f2=23.0, color='steelblue', label='11×23 GHz'),
        dict(b1='11', b2='30', f1=11.0, f2=30.0, color='k',         label='11×30 GHz'),
        dict(b1='23', b2='30', f1=23.0, f2=30.0, color='goldenrod',  label='23×30 GHz'),
    ],
    fig_label='synch_bands',
)

# --- Dust combined figure: 100, 217, 353 GHz ---
_make_combined(
    band_cfgs=[
        dict(b='100', f=100.0, color='steelblue', label='100×100 GHz'),
        dict(b='217', f=217.0, color='k',         label='217×217 GHz'),
        dict(b='353', f=353.0, color='goldenrod',  label='353×353 GHz'),
    ],
    cross_cfgs=[
        dict(b1='100', b2='217', f1=100.0, f2=217.0, color='steelblue', label='100×217 GHz'),
        dict(b1='100', b2='353', f1=100.0, f2=353.0, color='k',         label='100×353 GHz'),
        dict(b1='217', b2='353', f1=217.0, f2=353.0, color='goldenrod',  label='217×353 GHz'),
    ],
    fig_label='dust_bands',
)
