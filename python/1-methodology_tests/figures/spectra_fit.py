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
from astropy.io import fits as astropy_fits

# Global plotting style (use rcParams so sizes are consistent across figures)
plt.rcParams.update({
    'axes.titlesize': 18,
    'axes.labelsize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 15,
    'font.size': 15,
})

# Default configuration
nside = 512
n_sim = 100
path_save = path_map + 'PYSM/'

quijote_bands = ['11', '13', '17', '19']
wmap_bands = ['23', '33', '41', '61', '94']
planck_bands = ['30', '44', '70', '100', '143', '217', '353']


band_list = quijote_bands + wmap_bands + planck_bands

name_suffix = '_full_bin_20-199'

# Differential Assemblies (DAs) per frequency band
BANDS = {
    'K': ['K1'],
    'Ka': ['Ka1'],
    'Q': ['Q1', 'Q2'],
    'V': ['V1', 'V2'],
    'W': ['W1', 'W2', 'W3', 'W4'],
}

lmax = 2 * nside - 1
dl = 10

mask_select = masks['QUIJOTE_galcut']['galcut10']
mask_name = mask_select['name']
use_noise = False # Use noise simulations instead of the HMDM for QUIJOTE autos
out_path = '/home/pablo/Desktop/master/tfm/spectra/'
path_spectra = os.path.join(out_path, f'power_spectra_{mask_name}{name_suffix}.fits')
path_hmdm_spectra = os.path.join(out_path, f'power_spectra_{mask_name}_hmdm{name_suffix}.fits')
path_avg_std_skyplusnoise = os.path.join(out_path, f'spectra_avg_std_{mask_name}_avg_std{n_sim}_skyplusnoise{name_suffix}.fits')
path_avg_std_noise = os.path.join(out_path, f'spectra_avg_std_{mask_name}_avg_std{n_sim}_noise{name_suffix}.fits')
path_corrected_spectra = os.path.join(out_path, f'corrected_power_spectra_{mask_name}{name_suffix}.fits')

# ------------------------------
# Synchrotron model parameters
# ------------------------------
# EE mode
A_s_EE     = 7.418e-9
alpha_s_EE = -3.623
beta_s_EE  = -3.291

# BB mode
A_s_BB     = 1.703e-9
alpha_s_BB = -3.088
beta_s_BB  = -2.960

# ====================


freq_ref = 23.0
ell_ref  = 80.0

auto_bands = ['11', '23', '30']

cross_pairs = [
    ('11', '23'),
    ('11', '30'),
    ('23', '30'),
]

band_colors = {
    '11': 'steelblue',
    '23': 'k',
    '30': 'goldenrod',
    '33': 'purple',
}

# Colors for cross pairs — same order as band_colors
cross_colors = list(band_colors.values())

plot_freqs   = {b: float(b) for b in band_colors}
spectra_plot = functions.read_corrected_cls(path_corrected_spectra, band_list)

# Load color-correction polynomials (used by the MCMC model). If the FITS
# is not available, fall back to no color corrections (cc_dict = None).
try:
    cc_dict = functions.load_color_correction_polynomials()
except Exception:
    cc_dict = None

def sync_model(ell_arr, f1, f2, A_s, alpha_s, beta_s):
    base = (A_s
            * (ell_arr / ell_ref) ** alpha_s
            * (f1 / freq_ref) ** beta_s
            * (f2 / freq_ref) ** beta_s)
    # Apply color-correction factors if available (polynomial evaluated at alpha=2+beta_s)
    # Convention: divide by cc (same as functions.py model_synchrotron / model_synchrotron_joint)
    if cc_dict is None:
        return base
    alpha_s_cc = 2.0 + float(beta_s)
    b1 = str(int(float(f1)))
    b2 = str(int(float(f2)))
    poly1 = (cc_dict.get('synch', {}) or {}).get(b1)
    poly2 = (cc_dict.get('synch', {}) or {}).get(b2)
    cc_s1 = (poly1[0] + poly1[1]*alpha_s_cc + poly1[2]*(alpha_s_cc**2)) if poly1 is not None else 1.0
    cc_s2 = (poly2[0] + poly2[1]*alpha_s_cc + poly2[2]*(alpha_s_cc**2)) if poly2 is not None else 1.0
    return base / (cc_s1 * cc_s2)

mode_params = {
    'EE': (A_s_EE, alpha_s_EE, beta_s_EE),
    'BB': (A_s_BB, alpha_s_BB, beta_s_BB),
}

ell_fine = np.linspace(20, 200, 300)

# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False)
ax_EE_auto, ax_BB_auto   = axes[0]
ax_EE_cross, ax_BB_cross = axes[1]

# ---- AUTO spectra ----
for b in auto_bands:
    key = f'{b}_{b}'
    if key not in spectra_plot:
        continue
    ell_eff = spectra_plot[key]['ell_eff']
    f = plot_freqs[b]
    col = band_colors[b]

    for ax, mode in [(ax_EE_auto, 'EE'), (ax_BB_auto, 'BB')]:
        A_s, alpha_s, beta_s = mode_params[mode]
        cl  = spectra_plot[key][mode]['SPECTRUM'] * 1e6
        err = spectra_plot[key][mode]['ERROR'] * 1e6
        ax.errorbar(ell_eff, cl, yerr=err, fmt='o', color=col,
                    label=f'{b}x{b} GHz', capsize=2, ms=4)
        ax.plot(ell_fine, sync_model(ell_fine, f, f, A_s, alpha_s, beta_s) * 1e6,
                '-', color=col, lw=1.5)

ax_EE_auto.set_title('Auto-spectra — EE')
ax_BB_auto.set_title('Auto-spectra — BB')

# ---- CROSS spectra ----
for i, (b1, b2) in enumerate(cross_pairs):
    key = f'{b1}_{b2}'
    if key not in spectra_plot:
        key = f'{b2}_{b1}'
    if key not in spectra_plot:
        print(f'Warning: pair {b1}x{b2} not found in spectra, skipping.')
        continue
    ell_eff = spectra_plot[key]['ell_eff']
    f1, f2  = plot_freqs[b1], plot_freqs[b2]
    col     = cross_colors[i % len(cross_colors)]
    label   = f'{b1}x{b2} GHz'

    for ax, mode in [(ax_EE_cross, 'EE'), (ax_BB_cross, 'BB')]:
        A_s, alpha_s, beta_s = mode_params[mode]
        cl  = spectra_plot[key][mode]['SPECTRUM'] * 1e6
        err = spectra_plot[key][mode]['ERROR'] * 1e6
        ax.errorbar(ell_eff, cl, yerr=err, fmt='o', color=col,
                    label=label, capsize=2, ms=4)
        ax.plot(ell_fine, sync_model(ell_fine, f1, f2, A_s, alpha_s, beta_s) * 1e6,
                '-', color=col, lw=1.5)

ax_EE_cross.set_title('Cross-spectra — EE')
ax_BB_cross.set_title('Cross-spectra — BB')

# ---- Shared formatting ----
for ax in axes.flat:
    ax.set_yscale('log')
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$C_\ell \; [\mu\mathrm{K}^2]$')
    ax.legend(frameon=False)

plt.tight_layout()
plt.show()

fig.savefig(f'/home/pablo/Desktop/master/tfm/figures/spectra/spectra_{mask_name}_synch_model_fit.pdf')

# ---------------------------------------------------------------
# Dust figure — same layout, same bands 100/217/353 GHz
# ---------------------------------------------------------------
from scipy.constants import h, k  # Planck constant, Boltzmann constant

def planck_func(nu_GHz, T):
    nu = nu_GHz * 1e9
    x = h * nu / (k * T)
    return x**2 * np.exp(x) / (np.expm1(x))**2   # proportional to B_nu / (RJ factor)

def g_RJ(nu_GHz):
    nu = nu_GHz * 1e9
    x = h * nu / (k * 2.725)
    return (np.expm1(x))**2 / (x**2 * np.exp(x))

def mbb_scaling_KRJ(nu_GHz, nu0_GHz=353.0, beta=1.59, T_d=19.6):
    """
    Scaling of the dust MBB SED in K_RJ units relative to the reference
    frequency nu0_GHz.  In K_RJ the Rayleigh-Jeans nu^2 factor is already
    absorbed, so the only frequency dependence is:
        f(nu) = (nu/nu0)^beta * B_nu(T_d) / B_nu0(T_d)
    No g_RJ conversion factor is needed here.
    """
    nu_GHz = np.asarray(nu_GHz, dtype=float)
    power = (nu_GHz / nu0_GHz) ** beta
    planck_ratio = planck_func(nu_GHz, T_d) / planck_func(nu0_GHz, T_d)
    return power * planck_ratio

# ------------------------------
# Dust model parameters
# ------------------------------
freq_ref_dust = 353.0
T_d           = 19.6
ell_ref_dust  = 80.0

# EE mode  (WMAP+Planck posterior medians, Table 9)
A_d_EE     = 3.181e-9
alpha_d_EE = -2.542
beta_d_EE  = 1.521

# BB mode  (WMAP+Planck posterior medians, Table 9)
A_d_BB     = 2.395e-9
alpha_d_BB = -2.204
beta_d_BB  = 1.556

# ====================

dust_auto_bands   = ['100', '217', '353']
dust_cross_pairs  = [
    ('100', '217'),
    ('100', '353'),
    ('217', '353'),
]

dust_band_colors = {
    '100': 'steelblue',
    '217': 'k',
    '353': 'goldenrod',
}
dust_cross_colors = list(dust_band_colors.values())
dust_plot_freqs   = {b: float(b) for b in dust_band_colors}

dust_mode_params = {
    'EE': (A_d_EE, alpha_d_EE, beta_d_EE),
    'BB': (A_d_BB, alpha_d_BB, beta_d_BB),
}

def dust_model(ell_arr, f1, f2, A_d, alpha_d, beta_d):
    scale_f1 = mbb_scaling_KRJ(np.array([f1]), nu0_GHz=freq_ref_dust, beta=beta_d, T_d=T_d)[0]
    scale_f2 = mbb_scaling_KRJ(np.array([f2]), nu0_GHz=freq_ref_dust, beta=beta_d, T_d=T_d)[0]
    base = A_d * (ell_arr / ell_ref_dust) ** alpha_d * scale_f1 * scale_f2
    # Apply color corrections: alpha_cc = 2 + beta_d (frequency spectral index)
    if cc_dict is None:
        return base
    alpha_d_cc = 2.0 + float(beta_d)
    b1 = str(int(float(f1)))
    b2 = str(int(float(f2)))
    poly1 = (cc_dict.get('dust', {}) or {}).get(b1)
    poly2 = (cc_dict.get('dust', {}) or {}).get(b2)
    cc_d1 = (poly1[0] + poly1[1]*alpha_d_cc + poly1[2]*(alpha_d_cc**2)) if poly1 is not None else 1.0
    cc_d2 = (poly2[0] + poly2[1]*alpha_d_cc + poly2[2]*(alpha_d_cc**2)) if poly2 is not None else 1.0
    return base / (cc_d1 * cc_d2)

# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------
fig_d, axes_d = plt.subplots(2, 2, figsize=(14, 10), sharex=False)
ax_dEE_auto, ax_dBB_auto   = axes_d[0]
ax_dEE_cross, ax_dBB_cross = axes_d[1]

# ---- AUTO spectra ----
for b in dust_auto_bands:
    key = f'{b}_{b}'
    if key not in spectra_plot:
        continue
    ell_eff = spectra_plot[key]['ell_eff']
    f = dust_plot_freqs[b]
    col = dust_band_colors[b]

    for ax, mode in [(ax_dEE_auto, 'EE'), (ax_dBB_auto, 'BB')]:
        A_d, alpha_d, beta_d = dust_mode_params[mode]
        cl  = spectra_plot[key][mode]['SPECTRUM'] * 1e6
        err = spectra_plot[key][mode]['ERROR'] * 1e6
        ax.errorbar(ell_eff, cl, yerr=err, fmt='o', color=col,
                    label=f'{b}x{b} GHz', capsize=2, ms=4)
        ax.plot(ell_fine, dust_model(ell_fine, f, f, A_d, alpha_d, beta_d) * 1e6,
                '-', color=col, lw=1.5)

ax_dEE_auto.set_title('Auto-spectra — EE')
ax_dBB_auto.set_title('Auto-spectra — BB')

# ---- CROSS spectra ----
for i, (b1, b2) in enumerate(dust_cross_pairs):
    key = f'{b1}_{b2}'
    if key not in spectra_plot:
        key = f'{b2}_{b1}'
    if key not in spectra_plot:
        print(f'Warning: pair {b1}x{b2} not found in spectra, skipping.')
        continue
    ell_eff = spectra_plot[key]['ell_eff']
    f1, f2  = dust_plot_freqs[b1], dust_plot_freqs[b2]
    col     = dust_cross_colors[i % len(dust_cross_colors)]
    label   = f'{b1}x{b2} GHz'

    for ax, mode in [(ax_dEE_cross, 'EE'), (ax_dBB_cross, 'BB')]:
        A_d, alpha_d, beta_d = dust_mode_params[mode]
        cl  = spectra_plot[key][mode]['SPECTRUM'] * 1e6
        err = spectra_plot[key][mode]['ERROR'] * 1e6
        ax.errorbar(ell_eff, cl, yerr=err, fmt='o', color=col,
                    label=label, capsize=2, ms=4)
        ax.plot(ell_fine, dust_model(ell_fine, f1, f2, A_d, alpha_d, beta_d) * 1e6,
                '-', color=col, lw=1.5)

ax_dEE_cross.set_title('Cross-spectra — EE')
ax_dBB_cross.set_title('Cross-spectra — BB')

# ---- Shared formatting ----
for ax in axes_d.flat:
    ax.set_yscale('log')
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$C_\ell \; [\mu\mathrm{K}^2]$')
    ax.legend(frameon=False)

# fig_d.suptitle('Dust emission — MBB model', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()

fig_d.savefig(f'/home/pablo/Desktop/master/tfm/figures/spectra/spectra_{mask_name}_dust_model_fit.pdf')

# ---------------------------------------------------------------
# Full-model figure — synch + dust + correlation term
# Same layout as the dust figure, bands 100 / 217 / 353 GHz
# ---------------------------------------------------------------
# The total model for a pair (f1, f2) is:
#   C_ell(f1,f2) = C_ell^synch(f1,f2) + C_ell^dust(f1,f2)
#                + C_ell^corr(f1,f2)
# where the synch-dust correlation term follows the geometric mean:
#   C_ell^corr(f1,f2) = rho * sqrt(C_ell^synch(f1,f2) * C_ell^dust(f1,f2))
#   with rho the correlation coefficient (assumed 0 here — set to a non-zero
#   value if you want to include it).
# ---------------------------------------------------------------

# Synch-dust correlation coefficient (0 = no correlation)
rho_sd_EE = 0.095
rho_sd_BB = 0.114

def full_model(ell_arr, f1, f2, mode):
    """
    Return synch + dust + synch-dust correlation model for a frequency pair.

    Matches the convention of functions.model_cross:
      C_corr = rho * sqrt(A_s * A_d) * ell_scale_cross
               * [ (s1/cc_s1)*(d2/cc_d2) + (s2/cc_s2)*(d1/cc_d1) ]
    where s_i = (fi/freq_ref)^beta_s  and  d_i = mbb_scaling(fi).
    """
    A_s, alpha_s, beta_s = mode_params[mode]
    A_d, alpha_d, beta_d = dust_mode_params[mode]

    cl_s = sync_model(ell_arr, f1, f2, A_s, alpha_s, beta_s)
    cl_d = dust_model(ell_arr, f1, f2, A_d, alpha_d, beta_d)

    # Correlation term — mirrors model_cross in functions.py exactly
    s1 = (f1 / freq_ref) ** beta_s
    s2 = (f2 / freq_ref) ** beta_s
    d1 = float(mbb_scaling_KRJ(np.array([f1]), nu0_GHz=freq_ref_dust, beta=beta_d, T_d=T_d))
    d2 = float(mbb_scaling_KRJ(np.array([f2]), nu0_GHz=freq_ref_dust, beta=beta_d, T_d=T_d))
    ell_scale_cross = (ell_arr / ell_ref) ** ((alpha_s + alpha_d) / 2.0)

    # Per-band cc factors
    b1 = str(int(float(f1)))
    b2 = str(int(float(f2)))
    if cc_dict is not None:
        alpha_s_cc = 2.0 + float(beta_s)
        alpha_d_cc = 2.0 + float(beta_d)
        syn1 = (cc_dict.get('synch', {}) or {}).get(b1)
        syn2 = (cc_dict.get('synch', {}) or {}).get(b2)
        dus1 = (cc_dict.get('dust',  {}) or {}).get(b1)
        dus2 = (cc_dict.get('dust',  {}) or {}).get(b2)
        cc_s1 = (syn1[0] + syn1[1]*alpha_s_cc + syn1[2]*(alpha_s_cc**2)) if syn1 is not None else 1.0
        cc_s2 = (syn2[0] + syn2[1]*alpha_s_cc + syn2[2]*(alpha_s_cc**2)) if syn2 is not None else 1.0
        cc_d1 = (dus1[0] + dus1[1]*alpha_d_cc + dus1[2]*(alpha_d_cc**2)) if dus1 is not None else 1.0
        cc_d2 = (dus2[0] + dus2[1]*alpha_d_cc + dus2[2]*(alpha_d_cc**2)) if dus2 is not None else 1.0
    else:
        cc_s1 = cc_s2 = cc_d1 = cc_d2 = 1.0

    mix = (s1 / cc_s1) * (d2 / cc_d2) + (s2 / cc_s2) * (d1 / cc_d1)
    # Select per-mode synchrotron-dust correlation coefficient
    rho = rho_sd_EE if mode == 'EE' else rho_sd_BB
    cl_corr = rho * np.sqrt(A_s * A_d) * mix * ell_scale_cross

    return cl_s + cl_d + cl_corr

# Reuse the same band/color/pair config from the dust figure
full_auto_bands  = dust_auto_bands   # ['100', '217', '353']
full_cross_pairs = dust_cross_pairs  # [('100','217'), ('100','353'), ('217','353')]
full_band_colors = dust_band_colors
full_cross_colors = dust_cross_colors
full_plot_freqs  = dust_plot_freqs

# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------
fig_f, axes_f = plt.subplots(2, 2, figsize=(14, 10), sharex=False)
ax_fEE_auto, ax_fBB_auto   = axes_f[0]
ax_fEE_cross, ax_fBB_cross = axes_f[1]

# ---- AUTO spectra ----
for b in full_auto_bands:
    key = f'{b}_{b}'
    if key not in spectra_plot:
        continue
    ell_eff = spectra_plot[key]['ell_eff']
    f = full_plot_freqs[b]
    col = full_band_colors[b]

    for ax, mode in [(ax_fEE_auto, 'EE'), (ax_fBB_auto, 'BB')]:
        cl  = spectra_plot[key][mode]['SPECTRUM'] * 1e6
        err = spectra_plot[key][mode]['ERROR'] * 1e6
        ax.errorbar(ell_eff, cl, yerr=err, fmt='o', color=col,
                    label=f'{b}x{b} GHz', capsize=2, ms=4)
        # Individual components (dashed)
        A_s, alpha_s, beta_s = mode_params[mode]
        A_d, alpha_d, beta_d = dust_mode_params[mode]
        # ax.plot(ell_fine, sync_model(ell_fine, f, f, A_s, alpha_s, beta_s) * 1e6,
        #         '--', color=col, lw=1.0, alpha=0.5)
        # ax.plot(ell_fine, dust_model(ell_fine, f, f, A_d, alpha_d, beta_d) * 1e6,
        #         ':', color=col, lw=1.0, alpha=0.5)
        # Total (solid)
        ax.plot(ell_fine, full_model(ell_fine, f, f, mode) * 1e6,
                '-', color=col, lw=1.8)

ax_fEE_auto.set_title('Auto-spectra — EE')
ax_fBB_auto.set_title('Auto-spectra — BB')

# ---- CROSS spectra ----
for i, (b1, b2) in enumerate(full_cross_pairs):
    key = f'{b1}_{b2}'
    if key not in spectra_plot:
        key = f'{b2}_{b1}'
    if key not in spectra_plot:
        print(f'Warning: pair {b1}x{b2} not found in spectra, skipping.')
        continue
    ell_eff = spectra_plot[key]['ell_eff']
    f1, f2  = full_plot_freqs[b1], full_plot_freqs[b2]
    col     = full_cross_colors[i % len(full_cross_colors)]
    label   = f'{b1}x{b2} GHz'

    for ax, mode in [(ax_fEE_cross, 'EE'), (ax_fBB_cross, 'BB')]:
        cl  = spectra_plot[key][mode]['SPECTRUM'] * 1e6
        err = spectra_plot[key][mode]['ERROR'] * 1e6
        ax.errorbar(ell_eff, cl, yerr=err, fmt='o', color=col,
                    label=label, capsize=2, ms=4)
        # Individual components (dashed / dotted)
        A_s, alpha_s, beta_s = mode_params[mode]
        A_d, alpha_d, beta_d = dust_mode_params[mode]
        # ax.plot(ell_fine, sync_model(ell_fine, f1, f2, A_s, alpha_s, beta_s) * 1e6,
        #         '--', color=col, lw=1.0, alpha=0.5)
        # ax.plot(ell_fine, dust_model(ell_fine, f1, f2, A_d, alpha_d, beta_d) * 1e6,
        #         ':', color=col, lw=1.0, alpha=0.5)
        # Total (solid)
        ax.plot(ell_fine, full_model(ell_fine, f1, f2, mode) * 1e6,
                '-', color=col, lw=1.8)

ax_fEE_cross.set_title('Cross-spectra — EE')
ax_fBB_cross.set_title('Cross-spectra — BB')

# ---- Shared formatting ----
for ax in axes_f.flat:
    ax.set_yscale('log')
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$C_\ell \; [\mu\mathrm{K}^2]$')
    ax.legend(frameon=False)

# fig_f.suptitle('Full model — synchrotron + dust (100 / 217 / 353 GHz)', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()

fig_f.savefig(f'/home/pablo/Desktop/master/tfm/figures/spectra/spectra_{mask_name}_full_model_fit.pdf')


#%%
# ---------------------------------------------------------------
# Save transparent PNGs with black -> white conversion
# ---------------------------------------------------------------
import matplotlib as mpl

def save_figure_transparent_white(fig, out_path, dpi=300):
    """
    Save `fig` as a transparent PNG and convert black elements to white.
    - fig: matplotlib.figure.Figure
    - out_path: output PNG path
    """
    def is_black_color(c):
        # Accept color string 'k'/'black', hex '#000000', or RGBA tuple close to black
        try:
            # If it's a string like 'k' or 'black' or a hex string
            if isinstance(c, str):
                return c.lower() in ('k', 'black', '#000000')
            # If it's a tuple/list/ndarray RGBA-ish
            if hasattr(c, '__len__'):
                # Convert to rgba tuple of floats
                rgba = mpl.colors.to_rgba(c)
                # consider black if RGB components all near zero (alpha ignored)
                return (rgba[0] < 1e-6) and (rgba[1] < 1e-6) and (rgba[2] < 1e-6)
        except Exception:
            pass
        return False

    # Make figure background transparent
    try:
        fig.patch.set_alpha(0.0)
    except Exception:
        pass

    # Iterate axes and change colors for black elements -> white
    for ax in fig.axes:
        # Transparent axes background
        try:
            ax.set_facecolor('none')
        except Exception:
            pass

        # Titles and axis labels to white if they are black
        try:
            if is_black_color(ax.title.get_color()):
                ax.title.set_color('white')
        except Exception:
            pass
        try:
            if is_black_color(ax.xaxis.label.get_color()):
                ax.xaxis.label.set_color('white')
            if is_black_color(ax.yaxis.label.get_color()):
                ax.yaxis.label.set_color('white')
        except Exception:
            pass

        # Tick labels and tick lines to white
        try:
            ax.tick_params(colors='white')
            # Also change ticklabel text objects specifically if needed
            for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                try:
                    if is_black_color(lbl.get_color()):
                        lbl.set_color('white')
                except Exception:
                    pass
        except Exception:
            pass

        # Spines (border) to white if black
        try:
            for spine in ax.spines.values():
                try:
                    if is_black_color(spine.get_edgecolor()):
                        spine.set_edgecolor('white')
                except Exception:
                    # spine.get_edgecolor may return RGBA; check via to_rgba
                    try:
                        ec = spine.get_edgecolor()
                        if is_black_color(ec):
                            spine.set_edgecolor('white')
                    except Exception:
                        pass
        except Exception:
            pass

        # Legend text and frame
        try:
            leg = ax.get_legend()
            if leg is not None:
                for text in leg.get_texts():
                    try:
                        if is_black_color(text.get_color()):
                            text.set_color('white')
                    except Exception:
                        pass
                try:
                    # make legend frame transparent but edge in white if black
                    frame = leg.get_frame()
                    frame.set_alpha(0.0)
                    if is_black_color(frame.get_edgecolor()):
                        frame.set_edgecolor('white')
                except Exception:
                    pass
        except Exception:
            pass

        # Lines: convert explicitly-black lines to white (keeps colored lines intact)
        try:
            for line in ax.get_lines():
                try:
                    c = line.get_color()
                    if is_black_color(c):
                        line.set_color('white')
                except Exception:
                    pass
        except Exception:
            pass

        # Collections (e.g., errorbars, scatter): attempt to recolor edge/face colors if black
        try:
            for coll in ax.collections:
                try:
                    # PathCollection (scatter)
                    fc = coll.get_facecolors()
                    if fc is not None and len(fc):
                        # fc may be an array Nx4; check first color
                        first = fc[0]
                        if is_black_color(tuple(first)):
                            # set to white (keep alpha)
                            new = np.array(fc)
                            new[:, :3] = 1.0
                            coll.set_facecolors(new)
                    ec = coll.get_edgecolors()
                    if ec is not None and len(ec):
                        first_e = ec[0]
                        if is_black_color(tuple(first_e)):
                            new_e = np.array(ec)
                            new_e[:, :3] = 1.0
                            coll.set_edgecolors(new_e)
                except Exception:
                    # LineCollection or other: try a generic to_rgba check on default color
                    try:
                        c = getattr(coll, 'get_color', None)
                        if callable(c):
                            col = coll.get_color()
                            if is_black_color(col):
                                coll.set_color('white')
                    except Exception:
                        pass
        except Exception:
            pass

    # Also change any Text objects on the figure that are black
    try:
        for text in fig.findobj(mpl.text.Text):
            try:
                if is_black_color(text.get_color()):
                    text.set_color('white')
            except Exception:
                pass
    except Exception:
        pass

    # Finally save transparent PNG
    fig.savefig(out_path, transparent=True, facecolor='none', bbox_inches='tight', dpi=dpi)
    print('Saved transparent PNG to', out_path)


# Save the three figures created above as transparent PNGs with black->white conversion
png_synch = f'/home/pablo/Desktop/master/tfm/figures_ppt/spectra/spectra_{mask_name}_synch_model_fit_transparent.png'
png_dust  = f'/home/pablo/Desktop/master/tfm/figures_ppt/spectra/spectra_{mask_name}_dust_model_fit_transparent.png'
png_full  = f'/home/pablo/Desktop/master/tfm/figures_ppt/spectra/spectra_{mask_name}_full_model_fit_transparent.png'

save_figure_transparent_white(fig, png_synch, dpi=300)
save_figure_transparent_white(fig_d, png_dust, dpi=300)
save_figure_transparent_white(fig_f, png_full, dpi=300)
