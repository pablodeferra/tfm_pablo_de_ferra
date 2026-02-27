#%%
import numpy as np
from astropy.io import fits
import sys
sys.path.append('../') 
import functions as functions
from data import data, masks
import matplotlib.pyplot as plt
import healpy as hp
from astropy.io import fits as astropy_fits
import os

mask_path = masks['QUIJOTE_galcut']['galcut10']['path']
mask = hp.read_map(mask_path)

# =================================================

alberto_path = '/home/pablo/Downloads/spectra311_galcut10_noise_Nmt_pure.txt'

alberto_nl = np.loadtxt(alberto_path, skiprows=1).T

ell_alb_2 = alberto_nl[0]
nl_EE_alb_2 = alberto_nl[2]
nl_BB_alb_2 = alberto_nl[3]

# =================================================


# Default configuration
nside = 512
n_sim = 100

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
use_simulated_maps = False
use_white_noise = False
use_noise = False # Use noise simulations instead of the HMDM for QUIJOTE autos
out_path = '/home/pablo/Desktop/master/tfm/spectra/'
path_spectra = os.path.join(out_path, f'power_spectra_{mask_name}{name_suffix}.fits')
path_hmdm_spectra = os.path.join(out_path, f'power_spectra_{mask_name}_hmdm{name_suffix}.fits')
path_avg_std_skyplusnoise = os.path.join(out_path, f'spectra_avg_std_{mask_name}_avg_std{n_sim}_skyplusnoise{name_suffix}.fits')
path_avg_std_noise = os.path.join(out_path, f'spectra_avg_std_{mask_name}_avg_std{n_sim}_noise{name_suffix}.fits')
# Store per-simulation spectra compressed on disk (Astropy supports .fits.gz transparently)
path_full_skyplusnoise = os.path.join(out_path, f'spectra_full_{mask_name}_{n_sim}_skyplusnoise{name_suffix}.fits.gz')
path_full_noise = os.path.join(out_path, f'spectra_full_{mask_name}_{n_sim}_noise{name_suffix}.fits.gz')
path_corrected_spectra = os.path.join(out_path, f'corrected_power_spectra_{mask_name}{name_suffix}.fits')


#Create binning scheme
ell_1 = [20, 40, 60, 80, 100, 120, 140, 160, 180]
ell_2 = [39, 59, 79, 99, 119, 139, 159, 179, 199]

binning_params = {
    'type': 'edges',  #'linear' or 'edges'
    'lmax': lmax,
    'dl': dl,
    # For edges
    'ell1': ell_1,
    'ell2': ell_2
}

mask_path = '/home/pablo/Downloads/mask_lowdec_satband_galcut10_0mk_apodC2_5.fits'
map_path = '/home/pablo/Downloads/quijote_mfi_skymap_11ghz_512_dr1.fits'
mask = hp.read_map(mask_path)
map = hp.read_map(map_path, field=[0,1,2])

binning_params_lin = {
    'type': 'linear',
    'lmax': lmax,
    'dl': dl,
    'ell1': ell_1,
    'ell2': ell_2,
}

b_lin = functions.create_binning(binning_params_lin)
workspaces_lin = functions.prepare_workspaces(mask, b_lin, nside, lmax=lmax, purify_e=True, purify_b=True)

spectra_lin = functions.compute_all_power_spectra(
    data, ['11'], mask, b_lin,
    use_simulated_maps=use_simulated_maps,
    use_white_noise=use_white_noise,
    noise_realization=1,
    only_noise=False,
    workspaces=workspaces_lin,
    lmax=lmax
)

hmdm_lin = functions.compute_hmdm_power_spectra(
    data, ['11'], mask, b_lin,
    workspaces=workspaces_lin,
    lmax=lmax,
    use_noise=use_noise
)

ell_eff = spectra_lin[0, 0]['ell_eff']
cl_EE_lin   = spectra_lin[0, 0]['EE']
cl_BB_lin   = spectra_lin[0, 0]['BB']
nl_EE_lin   = hmdm_lin[0, 0]['EE']
nl_BB_lin   = hmdm_lin[0, 0]['BB']

error_311_10 = np.loadtxt('/home/pablo/Desktop/Fisica/TFG/python/quijote_spectra/final_data/corrected_errorbars_311_galcut10.txt', skiprows=1)


#%%
# ==========================================================

ell_min = 3
ell_max = 20

beam_11 = np.loadtxt('/home/pablo/Desktop/Fisica/TFG/txts/beam_311.txt')[1][ell_min:ell_max]
wp = np.loadtxt('/home/pablo/Desktop/Fisica/TFG/txts/wp.txt')[1][ell_min:ell_max]

ell_eff = spectra_lin[0, 0]['ell_eff'][ell_min:ell_max]

# ======================== Alberto ========================
corr_EE_alb = (cl_EE_lin[ell_min:ell_max] - nl_EE_alb_2[ell_min:ell_max]) / (beam_11**2 * wp**2)
corr_BB_alb = (cl_BB_lin[ell_min:ell_max] - nl_BB_alb_2[ell_min:ell_max]) / (beam_11**2 * wp**2)

# ======================== Pablo ========================
corr_EE_pab = (cl_EE_lin[ell_min:ell_max] - nl_EE_lin[ell_min:ell_max]) / (beam_11**2 * wp**2)
corr_BB_pab = (cl_BB_lin[ell_min:ell_max] - nl_BB_lin[ell_min:ell_max]) / (beam_11**2 * wp**2)

error_EE = error_311_10[2,3:20]
error_BB = error_311_10[2,3:20]

# =============================================================================
# Save spectra to txt files
# =============================================================================

txt_dir = '/home/pablo/Desktop/master/tfm/spectra/txt/'
os.makedirs(txt_dir, exist_ok=True)

ell_eff_full = spectra_lin[0, 0]['ell_eff']   # all bins, for raw & noise

np.savetxt(
    os.path.join(txt_dir, 'raw_spectra_11ghz_dl10.txt'),
    np.column_stack([ell_eff_full, cl_EE_lin, cl_BB_lin]),
    header='ell_eff  cl_EE  cl_BB',
    fmt='%.6e'
)

np.savetxt(
    os.path.join(txt_dir, 'noise_spectra_11ghz_dl10.txt'),
    np.column_stack([ell_eff_full, nl_EE_lin, nl_BB_lin]),
    header='ell_eff  nl_EE  nl_BB',
    fmt='%.6e'
)

np.savetxt(
    os.path.join(txt_dir, 'corrected_spectra_11ghz_dl10.txt'),
    np.column_stack([ell_eff, corr_EE_pab, corr_BB_pab]),
    header='ell_eff  corr_EE  corr_BB',
    fmt='%.6e'
)

np.savetxt(
    os.path.join(txt_dir, 'errors_11ghz_dl10.txt'),
    np.column_stack([ell_eff, error_EE, error_BB]),
    header='ell_eff  error_EE  error_BB',
    fmt='%.6e'
)

print(f"[INFO] txt files saved to {txt_dir}")

#%%

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ======================
# EE
# ======================
ax = axes[0]

ax.errorbar(
    ell_eff,
    corr_EE_alb,
    yerr=error_EE,
    fmt='o',
    label=r'$C_\ell^{EE} - N_\ell^{EE}$',
    color='steelblue'
)

ax.errorbar(
    ell_eff,
    cl_EE_lin[ell_min:ell_max] / (beam_11**2 * wp**2),
    yerr=error_EE,
    fmt='^',
    ls=':',
    label=r'$C_\ell^{EE}$ (raw)',
    color='goldenrod'
)

ax.plot(
    ell_eff,
    nl_EE_alb_2[ell_min:ell_max] / (beam_11**2 * wp**2),
    '--',
    label=r'$N_\ell^{EE}$ (HMDM)',
    color='k'
)

ax.set_yscale('log')
ax.set_xlabel(r'$\ell$', fontsize=13)
ax.set_ylabel(r'$C_\ell \; [\mathrm{mK}^2]$', fontsize=13)
ax.set_title('QUIJOTE 11 GHz — EE', fontsize=13)
ax.legend(frameon=False)
ax.set_xlim(20, 210)
ax.set_ylim(1e-10, 1e-4)

# ======================
# BB
# ======================
ax = axes[1]

ax.errorbar(
    ell_eff,
    corr_BB_alb,
    yerr=error_BB,
    fmt='o',
    label=r'$C_\ell^{BB}$ - $N_\ell^{BB}$',
    color='steelblue'
)

ax.errorbar(
    ell_eff,
    cl_BB_lin[ell_min:ell_max] / (beam_11**2 * wp**2),
    yerr=error_EE,
    fmt='^',
    ls=':',
    label=r'$C_\ell^{BB}$ (raw)',
    color='goldenrod'
)

ax.plot(
    ell_eff,
    nl_BB_alb_2[ell_min:ell_max] / (beam_11**2 * wp**2),
    '--',
    label=r'$N_\ell^{BB}$ (HMDM)',
    color='black'
)

ax.set_yscale('log')
ax.set_xlabel(r'$\ell$', fontsize=13)
ax.set_ylabel(r'$C_\ell \; [\mathrm{mK}^2]$', fontsize=13)
ax.set_title('QUIJOTE 11 GHz — BB', fontsize=13)
ax.legend(frameon=False)
ax.set_xlim(20, 210)
ax.set_ylim(1e-10, 1e-4)

plt.tight_layout()
plt.show()

fig.savefig('/home/pablo/Desktop/master/tfm/figures/spectra/Cl_Nl_alberto.pdf')


# ======================== Pablo ========================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ======================
# EE
# ======================
ax = axes[0]

ax.errorbar(
    ell_eff,
    corr_EE_pab,
    yerr=error_EE,
    fmt='o',
    label=r'$C_\ell^{EE} - N_\ell^{EE}$',
    color='steelblue'
)

ax.errorbar(
    ell_eff,
    cl_EE_lin[ell_min:ell_max] / (beam_11**2 * wp**2),
    yerr=error_EE,
    fmt='^',
    ls=':',
    label=r'$C_\ell^{EE}$ (raw)',
    color='goldenrod'
)

ax.plot(
    ell_eff,
    nl_EE_alb_2[ell_min:ell_max] / (beam_11**2 * wp**2),
    '--',
    label=r'$N_\ell^{EE}$ (HMDM)',
    color='k'
)

ax.set_yscale('log')
ax.set_xlabel(r'$\ell$', fontsize=13)
ax.set_ylabel(r'$C_\ell \; [\mathrm{mK}^2]$', fontsize=13)
ax.set_title('QUIJOTE 11 GHz — EE', fontsize=13)
ax.legend(frameon=False)
ax.set_xlim(20, 210)
ax.set_ylim(1e-10, 1e-4)

# ======================
# BB
# ======================
ax = axes[1]

ax.errorbar(
    ell_eff,
    corr_BB_pab,
    yerr=error_BB,
    fmt='o',
    label=r'$C_\ell^{BB}$ - $N_\ell^{BB}$',
    color='steelblue'
)

ax.errorbar(
    ell_eff,
    cl_BB_lin[ell_min:ell_max] / (beam_11**2 * wp**2),
    yerr=error_EE,
    fmt='^',
    ls=':',
    label=r'$C_\ell^{BB}$ (raw)',
    color='goldenrod'
)

ax.plot(
    ell_eff,
    nl_BB_alb_2[ell_min:ell_max] / (beam_11**2 * wp**2),
    '--',
    label=r'$N_\ell^{BB}$ (HMDM)',
    color='black'
)

ax.set_yscale('log')
ax.set_xlabel(r'$\ell$', fontsize=13)
ax.set_ylabel(r'$C_\ell \; [\mathrm{mK}^2]$', fontsize=13)
ax.set_title('QUIJOTE 11 GHz — BB', fontsize=13)
ax.legend(frameon=False)
ax.set_xlim(20, 210)
ax.set_ylim(1e-10, 1e-4)

plt.tight_layout()
plt.show()

fig.savefig('/home/pablo/Desktop/master/tfm/figures/spectra/Cl_Nl_pablo.pdf')

#%%
# ===========================================================================
#  MCMC fit: C_ell = A * (ell/80)^alpha + c   for EE and BB
# ===========================================================================
import emcee
import corner

# =============================================================================
# Model, likelihood, prior, posterior
# =============================================================================

def model(theta, x):
    A, alpha, c = theta
    return A * 1e-6 * (x / 80.0) ** alpha + c * 1e-9

def lnlike(theta, x, y, yerr):
    return -0.5 * np.sum(((y - model(theta, x)) / yerr)**2)

def lnprior(theta):
    A, alpha, c = theta
    if -6 < alpha < -0.5: # and c > 0:
        return 0.0
    return -np.inf

def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp): # If the parameter is not within the priors, return -infinite
        return -np.inf
    return lp + lnlike(theta, x, y, yerr) # if theta fulfills the priors, then lp = 0

def main(p0, nwalkers, niter, ndim, lnprob, data):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)

    print("Running burn-in...")
    p0, _, _ = sampler.run_mcmc(p0, 100, progress=True)
    sampler.reset()

    print("Running production...")
    pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)

    return sampler, pos, prob, state

# =============================================================================
# Walkers, iterations and initial parameters
# =============================================================================

nwalkers    = 100
niter       = 20000
ndim        = 3          # A, alpha, c
discard_frac = 0.5
initial     = np.array([1e-6, -3., 0.])
variations  = np.array([1e-7, 1e-1, 1e-7])

p0_EE = [initial + variations * np.random.randn(ndim) for _ in range(nwalkers)]
p0_BB = [initial + variations * np.random.randn(ndim) for _ in range(nwalkers)]

# =============================================================================
# Run MCMC — EE and BB
# =============================================================================

data_EE_alb = (ell_eff, corr_EE_alb, error_EE)  # (x, y, yerr)
data_BB_alb = (ell_eff, corr_BB_alb, error_BB)

data_EE_pab = (ell_eff, corr_EE_pab, error_EE)
data_BB_pab = (ell_eff, corr_BB_pab, error_BB)


print("Running MCMC for EE ...")
sampler_EE_alb, pos_EE_alb, prob_EE_alb, state_E_albE = main(p0_EE, nwalkers, niter, ndim, lnprob, data_EE_alb)
sampler_EE_pab, pos_EE_pab, prob_EE_pab, state_E_pabE = main(p0_EE, nwalkers, niter, ndim, lnprob, data_EE_pab)

print("Running MCMC for BB ...")
sampler_BB_alb, pos_BB_alb, prob_BB_alb, state_BB_alb = main(p0_BB, nwalkers, niter, ndim, lnprob, data_BB_alb)
sampler_BB_pab, pos_BB_pab, prob_BB_pab, state_BB_pab = main(p0_BB, nwalkers, niter, ndim, lnprob, data_BB_pab)

# =============================================================================
# Extract samples
# =============================================================================

samples_EE_alb = sampler_EE_alb.get_chain(flat=True, thin=10, discard=int(discard_frac * niter))
samples_BB_alb = sampler_BB_alb.get_chain(flat=True, thin=10, discard=int(discard_frac * niter))

samples_EE_pab = sampler_EE_pab.get_chain(flat=True, thin=10, discard=int(discard_frac * niter))
samples_BB_pab = sampler_BB_pab.get_chain(flat=True, thin=10, discard=int(discard_frac * niter))

# =============================================================================
# Corner plots
# =============================================================================

# Label units: A in microkelvin^2, c shown in 10^-3 microkelvin^2 for clarity
labels = [r'$A\;[\mu\mathrm{K}^2]$', r'$\alpha$', r'$c\;[10^{-3}\,\mu\mathrm{K}^2]$']

fig_EE_alb = corner.corner(
    samples_EE_alb,
    show_titles=True,
    labels=labels,
    plot_datapoints=True,
    quantiles=[0.16, 0.5, 0.84],
    label_kwargs={"fontsize": 14},
)
fig_EE_alb.suptitle(r'EE mode', fontsize=12)
plt.show()
fig_EE_alb.savefig('/home/pablo/Desktop/master/tfm/figures/spectra/corner_EE_mcmc_alberto.pdf')

fig_BB_alb = corner.corner(
    samples_BB_alb,
    show_titles=True,
    labels=labels,
    plot_datapoints=True,
    quantiles=[0.16, 0.5, 0.84],
    label_kwargs={"fontsize": 14},
)
fig_BB_alb.suptitle(r'BB mode', fontsize=12)
plt.show()
fig_BB_alb.savefig('/home/pablo/Desktop/master/tfm/figures/spectra/corner_BB_mcmc_alberto.pdf')



fig_EE_pab = corner.corner(
    samples_EE_pab,
    show_titles=True,
    labels=labels,
    plot_datapoints=True,
    quantiles=[0.16, 0.5, 0.84],
    label_kwargs={"fontsize": 14},
)
fig_EE_pab.suptitle(r'EE mode', fontsize=12)
plt.show()
fig_EE_pab.savefig('/home/pablo/Desktop/master/tfm/figures/spectra/corner_EE_mcmc_pablo.pdf')

fig_BB_pab = corner.corner(
    samples_BB_pab,
    show_titles=True,
    labels=labels,
    plot_datapoints=True,
    quantiles=[0.16, 0.5, 0.84],
    label_kwargs={"fontsize": 14},
)
fig_BB_pab.suptitle(r'BB mode', fontsize=12)
plt.show()
fig_BB_pab.savefig('/home/pablo/Desktop/master/tfm/figures/spectra/corner_BB_mcmc_pablo.pdf')




#%%
# =============================================================================
# Best-fit values and plot
# =============================================================================

A_EE, alpha_EE, c_EE = np.median(samples_EE_alb, axis=0)
A_BB, alpha_BB, c_BB = np.median(samples_BB_alb, axis=0)

ell_plot = np.linspace(ell_eff.min(), ell_eff.max(), 300)

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

for ax, corr, error, A, alpha, c, mode, color in [
    (axes2[0], corr_EE, error_EE, A_EE, alpha_EE, c_EE, 'EE', 'steelblue'),
    (axes2[1], corr_BB, error_BB, A_BB, alpha_BB, c_BB, 'BB', 'tomato'),
]:
    ax.errorbar(ell_eff, corr, yerr=error, fmt='o', color=color, capsize=3,
                label=rf'$C_\ell^{{{mode}}} - N_\ell^{{{mode}}}$')
    ax.plot(ell_plot, model([A, alpha, c], ell_plot), 'k-',
            label=rf'$A(\ell/80)^\alpha + c$   $A$={A:.2e}, $\alpha$={alpha:.2f}, $c$={c:.2e}')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\ell$', fontsize=13)
    ax.set_ylabel(r'$C_\ell\;[\mathrm{mK}^2]$', fontsize=13)
    ax.set_title(f'QUIJOTE 11 GHz — {mode}', fontsize=13)
    ax.legend(frameon=False, fontsize=10)
    ax.set_xlim(ell_eff.min() - 5, ell_eff.max() + 5)

plt.tight_layout()
plt.show()
fig2.savefig('/home/pablo/Desktop/master/tfm/figures/spectra/Cl_mcmc_fit_alberto.pdf')


#%%
# =============================================================================
# Load saved dl10 txt data and plot
# =============================================================================

txt_dir = '/home/pablo/Desktop/master/tfm/spectra/txt/'

_raw  = np.loadtxt(os.path.join(txt_dir, 'raw_spectra_11ghz_dl10.txt'))
_nl   = np.loadtxt(os.path.join(txt_dir, 'noise_spectra_11ghz_dl10.txt'))
_corr = np.loadtxt(os.path.join(txt_dir, 'corrected_spectra_11ghz_dl10.txt'))
_err  = np.loadtxt(os.path.join(txt_dir, 'errors_11ghz_dl10.txt'))

# col1=ell_eff  col2=cl_EE  col3=cl_BB
ell_eff_raw   = _raw[:, 0];  cl_EE_t = _raw[:, 1];  cl_BB_t = _raw[:, 2]
# col1=ell_eff  col2=nl_EE  col3=nl_BB
ell_eff_nl    = _nl[:, 0];   nl_EE_t = _nl[:, 1];   nl_BB_t = _nl[:, 2]
# col1=ell_eff  col2=corr_EE  col3=corr_BB
ell_eff_corr  = _corr[:, 0]; corr_EE_t = _corr[:, 1]; corr_BB_t = _corr[:, 2]
# col1=ell_eff  col2=error_EE  col3=error_BB
ell_eff_err   = _err[:, 0];  err_EE_t = _err[:, 1];   err_BB_t = _err[:, 2]

# ---- Plot 1: corrected spectra with errors ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.errorbar(ell_eff_corr, corr_EE_t, yerr=err_EE_t, fmt='o',
            label=r'$C_\ell^{EE} - N_\ell^{EE}$', color='steelblue')
ax.set_yscale('log')
ax.set_xlabel(r'$\ell$', fontsize=13)
ax.set_ylabel(r'$C_\ell \; [\mathrm{mK}^2]$', fontsize=13)
ax.set_title('QUIJOTE 11 GHz — EE (dl10)', fontsize=13)
ax.legend(frameon=False)
ax.set_xlim(20, 210)
ax.set_ylim(1e-10, 1e-4)

ax = axes[1]
ax.errorbar(ell_eff_corr, corr_BB_t, yerr=err_BB_t, fmt='o',
            label=r'$C_\ell^{BB} - N_\ell^{BB}$', color='steelblue')
ax.set_yscale('log')
ax.set_xlabel(r'$\ell$', fontsize=13)
ax.set_ylabel(r'$C_\ell \; [\mathrm{mK}^2]$', fontsize=13)
ax.set_title('QUIJOTE 11 GHz — BB (dl10)', fontsize=13)
ax.legend(frameon=False)
ax.set_xlim(20, 210)
ax.set_ylim(1e-10, 1e-4)

plt.tight_layout()
plt.show()

# ---- Plot 2: raw Cl, Nl, and Cl - Nl ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.errorbar(ell_eff_raw, cl_EE_t - nl_EE_t, fmt='o', ls='-',
            label=r'$C_\ell^{EE} - N_\ell^{EE}$', color='steelblue')
ax.errorbar(ell_eff_raw, cl_EE_t, fmt='^', ls=':',
            label=r'$C_\ell^{EE}$ (raw)', color='goldenrod')
ax.plot(ell_eff_nl, nl_EE_t, '--',
        label=r'$N_\ell^{EE}$ (HMDM)', color='k')
ax.set_yscale('log')
ax.set_xlabel(r'$\ell$', fontsize=13)
ax.set_ylabel(r'$C_\ell \; [\mathrm{mK}^2]$', fontsize=13)
ax.set_title('QUIJOTE 11 GHz — EE (dl10)', fontsize=13)
ax.legend(frameon=False)
ax.set_xlim(20, 210)
ax.set_ylim(1e-10, 1e-4)

ax = axes[1]
ax.errorbar(ell_eff_raw, cl_BB_t - nl_BB_t, fmt='o', ls='-',
            label=r'$C_\ell^{BB} - N_\ell^{BB}$', color='steelblue')
ax.errorbar(ell_eff_raw, cl_BB_t, fmt='^', ls=':',
            label=r'$C_\ell^{BB}$ (raw)', color='goldenrod')
ax.plot(ell_eff_nl, nl_BB_t, '--',
        label=r'$N_\ell^{BB}$ (HMDM)', color='black')
ax.set_yscale('log')
ax.set_xlabel(r'$\ell$', fontsize=13)
ax.set_ylabel(r'$C_\ell \; [\mathrm{mK}^2]$', fontsize=13)
ax.set_title('QUIJOTE 11 GHz — BB (dl10)', fontsize=13)
ax.legend(frameon=False)
ax.set_xlim(20, 210)
ax.set_ylim(1e-10, 1e-4)

plt.tight_layout()
plt.show()





