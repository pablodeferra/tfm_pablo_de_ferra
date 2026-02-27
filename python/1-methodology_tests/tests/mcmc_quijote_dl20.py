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

# =============================================================================
# Load corrected 11 GHz spectra (same data as plotted in run_all_code.ipynb)
# =============================================================================

mask_select = masks['QUIJOTE_galcut']['galcut10']
mask_name = mask_select['name']
name_suffix = '_full_bin_20-199'
out_path = '/home/pablo/Desktop/master/tfm/spectra/'
path_spectra = os.path.join(out_path, f'power_spectra_{mask_name}{name_suffix}.fits')
path_hmdm_spectra = os.path.join(out_path, f'power_spectra_{mask_name}_hmdm{name_suffix}.fits')

path_corrected_spectra = '/home/pablo/Desktop/master/tfm/spectra/corrected_power_spectra_quijote_galcut10_full_bin_20-199.fits'
band_list = ['11', '13', '17', '19', '23', '33', '41', '61', '94', '30', '44', '70', '100', '143', '217', '353']

raw_spectra = functions.read_spectra_from_fits(path_spectra, band_list)
spectra_plot = functions.read_corrected_cls(path_corrected_spectra, band_list)
noise_spectra = functions.read_spectra_from_fits(path_hmdm_spectra, band_list)

ell_eff   = spectra_plot['11_11']['ell_eff']
corr_EE_pab = spectra_plot['11_11']['EE']['SPECTRUM']
corr_BB_pab = spectra_plot['11_11']['BB']['SPECTRUM']
error_EE  = spectra_plot['11_11']['EE']['ERROR']
error_BB  = spectra_plot['11_11']['BB']['ERROR']
cl_EE = raw_spectra['11_11']['EE']
cl_BB = raw_spectra['11_11']['BB']
nl_EE = noise_spectra['11_11']['EE']
nl_BB = noise_spectra['11_11']['BB']

# ell range for the fit
ell_min = 0
ell_max = len(ell_eff)   # use all bins; adjust if you want a narrower range

# =============================================================================
# Save spectra to txt files
# =============================================================================

ell_1 = raw_spectra['11_11']['ell1']
ell_2 = raw_spectra['11_11']['ell2']

txt_dir = '/home/pablo/Desktop/master/tfm/spectra/txt/'
os.makedirs(txt_dir, exist_ok=True)

np.savetxt(
    os.path.join(txt_dir, 'raw_spectra_11ghz_dl20.txt'),
    np.column_stack([ell_1, ell_2, ell_eff, cl_EE, cl_BB]),
    header='ell_1  ell_2  ell_eff  cl_EE  cl_BB',
    fmt='%.6e'
)

np.savetxt(
    os.path.join(txt_dir, 'noise_spectra_11ghz_dl20.txt'),
    np.column_stack([ell_1, ell_2, ell_eff, nl_EE, nl_BB]),
    header='ell_1  ell_2  ell_eff  nl_EE  nl_BB',
    fmt='%.6e'
)

np.savetxt(
    os.path.join(txt_dir, 'corrected_spectra_11ghz_dl20.txt'),
    np.column_stack([ell_1, ell_2, ell_eff, corr_EE_pab, corr_BB_pab]),
    header='ell_1  ell_2  ell_eff  corr_EE  corr_BB',
    fmt='%.6e'
)

np.savetxt(
    os.path.join(txt_dir, 'errors_11ghz_dl20.txt'),
    np.column_stack([ell_1, ell_2, ell_eff, error_EE, error_BB]),
    header='ell_1  ell_2  ell_eff  error_EE  error_BB',
    fmt='%.6e'
)

print(f"[INFO] txt files saved to {txt_dir}")


#%%
# ======================== Pablo ========================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ======================
# EE
# ======================
ax = axes[0]

# ax.errorbar(
#     ell_eff,
#     cl_EE - nl_EE,
#     # yerr=error_EE,
#     fmt='o',
#     ls='-',
#     label=r'$C_\ell^{EE} - N_\ell^{EE}$',
#     color='steelblue'
# )

# ax.errorbar(
#     ell_eff,
#     cl_EE,
#     # yerr=error_EE,
#     fmt='^',
#     ls=':',
#     label=r'$C_\ell^{EE}$ (raw)',
#     color='goldenrod'
# )

# ax.plot(
#     ell_eff,
#     nl_EE,
#     '--',
#     label=r'$N_\ell^{EE}$ (HMDM)',
#     color='k'
# )

ax.errorbar(
    ell_eff,
    corr_EE_pab,
    yerr=error_EE,
    fmt='o',
    ls='-',
    label=r'$C_\ell^{EE}$ (corr)',
    color='steelblue'
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

# ax.errorbar(
#     ell_eff,
#     cl_BB - nl_BB,
#     # yerr=error_BB,
#     fmt='o',
#     ls='-',
#     label=r'$C_\ell^{BB} - N_\ell^{BB}$',
#     color='steelblue'
# )

# ax.errorbar(
#     ell_eff,
#     cl_BB,
#     # yerr=error_BB,
#     fmt='^',
#     ls=':',
#     label=r'$C_\ell^{BB}$ (raw)',
#     color='goldenrod'
# )

# ax.plot(
#     ell_eff,
#     nl_BB,
#     '--',
#     label=r'$N_\ell^{BB}$ (HMDM)',
#     color='k'
# )

ax.errorbar(
    ell_eff,
    corr_BB_pab,
    yerr=error_BB,
    fmt='o',
    ls='-',
    label=r'$C_\ell^{BB}$ (corr)',
    color='steelblue'
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

# fig.savefig('/home/pablo/Desktop/master/tfm/figures/spectra/Cl_Nl_pablo_dl20.pdf')

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
    if -15 < alpha < -0.5: # and c > 0:
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
initial     = np.array([1, -3., 0.])
variations  = np.array([1e-1, 1e-1, 1])

p0_EE = [initial + variations * np.random.randn(ndim) for _ in range(nwalkers)]
p0_BB = [initial + variations * np.random.randn(ndim) for _ in range(nwalkers)]

# =============================================================================
# Run MCMC — EE and BB
# =============================================================================


data_EE_pab = (ell_eff[:], corr_EE_pab[:], error_EE[:])
data_BB_pab = (ell_eff[:], corr_BB_pab[:], error_BB[:])


print("Running MCMC for EE ...")
sampler_EE_pab, pos_EE_pab, prob_EE_pab, state_E_pabE = main(p0_EE, nwalkers, niter, ndim, lnprob, data_EE_pab)

print("Running MCMC for BB ...")
sampler_BB_pab, pos_BB_pab, prob_BB_pab, state_BB_pab = main(p0_BB, nwalkers, niter, ndim, lnprob, data_BB_pab)

# =============================================================================
# Extract samples
# =============================================================================

samples_EE_pab = sampler_EE_pab.get_chain(flat=True, thin=10, discard=int(discard_frac * niter))
samples_BB_pab = sampler_BB_pab.get_chain(flat=True, thin=10, discard=int(discard_frac * niter))

# =============================================================================
# Corner plots
# =============================================================================

# Label units: A in microkelvin^2, c shown in 10^-3 microkelvin^2 for clarity
labels = [r'$A\;[\mu\mathrm{K}^2]$', r'$\alpha$', r'$c\;[10^{-3}\,\mu\mathrm{K}^2]$']

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
fig_EE_pab.savefig('/home/pablo/Desktop/master/tfm/figures/spectra/corner_EE_mcmc_pablo_dl20_ell20-200.pdf')

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
fig_BB_pab.savefig('/home/pablo/Desktop/master/tfm/figures/spectra/corner_BB_mcmc_pablo_dl20_ell20-200.pdf')


#%%
# =============================================================================
# Load saved dl20 txt data and plot
# =============================================================================

txt_dir = '/home/pablo/Desktop/master/tfm/spectra/txt/'

_raw  = np.loadtxt(os.path.join(txt_dir, 'raw_spectra_11ghz_dl20.txt'))
_nl   = np.loadtxt(os.path.join(txt_dir, 'noise_spectra_11ghz_dl20.txt'))
_corr = np.loadtxt(os.path.join(txt_dir, 'corrected_spectra_11ghz_dl20.txt'))
_err  = np.loadtxt(os.path.join(txt_dir, 'errors_11ghz_dl20.txt'))

# col1=ell_1  col2=ell_2  col3=ell_eff  col4=cl_EE  col5=cl_BB
ell_eff_raw  = _raw[:, 2];  cl_EE_t = _raw[:, 3];  cl_BB_t = _raw[:, 4]
# col1=ell_1  col2=ell_2  col3=ell_eff  col4=nl_EE  col5=nl_BB
ell_eff_nl   = _nl[:, 2];   nl_EE_t = _nl[:, 3];   nl_BB_t = _nl[:, 4]
# col1=ell_1  col2=ell_2  col3=ell_eff  col4=corr_EE  col5=corr_BB
ell_eff_corr = _corr[:, 2]; corr_EE_t = _corr[:, 3]; corr_BB_t = _corr[:, 4]
# col1=ell_1  col2=ell_2  col3=ell_eff  col4=error_EE  col5=error_BB
ell_eff_err  = _err[:, 2];  err_EE_t = _err[:, 3];   err_BB_t = _err[:, 4]

# ---- Plot 1: corrected spectra with errors ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.errorbar(ell_eff_corr, corr_EE_t, yerr=err_EE_t, fmt='o',
            label=r'$C_\ell^{EE} - N_\ell^{EE}$', color='steelblue')
ax.set_yscale('log')
ax.set_xlabel(r'$\ell$', fontsize=13)
ax.set_ylabel(r'$C_\ell \; [\mathrm{mK}^2]$', fontsize=13)
ax.set_title('QUIJOTE 11 GHz — EE (dl20)', fontsize=13)
ax.legend(frameon=False)
ax.set_xlim(20, 210)
ax.set_ylim(1e-10, 1e-4)

ax = axes[1]
ax.errorbar(ell_eff_corr, corr_BB_t, yerr=err_BB_t, fmt='o',
            label=r'$C_\ell^{BB} - N_\ell^{BB}$', color='steelblue')
ax.set_yscale('log')
ax.set_xlabel(r'$\ell$', fontsize=13)
ax.set_ylabel(r'$C_\ell \; [\mathrm{mK}^2]$', fontsize=13)
ax.set_title('QUIJOTE 11 GHz — BB (dl20)', fontsize=13)
ax.legend(frameon=False)
ax.set_xlim(20, 210)
ax.set_ylim(1e-10, 1e-4)

plt.tight_layout()
plt.show()

# ---- Plot 2: raw Cl, Nl, and Cl - Nl ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.errorbar(ell_eff_corr, corr_EE_t, yerr=err_EE_t, fmt='o',
            label=r'$C_\ell^{EE} - N_\ell^{EE}$', color='steelblue')
ax.errorbar(ell_eff_raw, cl_EE_t, fmt='^', ls=':',
            label=r'$C_\ell^{EE}$ (raw)', color='goldenrod')
ax.plot(ell_eff_nl, nl_EE_t, '--',
        label=r'$N_\ell^{EE}$ (HMDM)', color='k')
ax.set_yscale('log')
ax.set_xlabel(r'$\ell$', fontsize=13)
ax.set_ylabel(r'$C_\ell \; [\mathrm{mK}^2]$', fontsize=13)
ax.set_title('QUIJOTE 11 GHz — EE (dl20)', fontsize=13)
ax.legend(frameon=False)
ax.set_xlim(20, 210)
ax.set_ylim(1e-10, 1e-4)

ax = axes[1]
ax.errorbar(ell_eff_corr, corr_BB_t, yerr=err_BB_t, fmt='o',
            label=r'$C_\ell^{BB} - N_\ell^{BB}$', color='steelblue')
ax.errorbar(ell_eff_raw, cl_BB_t, fmt='^', ls=':',
            label=r'$C_\ell^{BB}$ (raw)', color='goldenrod')
ax.plot(ell_eff_nl, nl_BB_t, '--',
        label=r'$N_\ell^{BB}$ (HMDM)', color='black')
ax.set_yscale('log')
ax.set_xlabel(r'$\ell$', fontsize=13)
ax.set_ylabel(r'$C_\ell \; [\mathrm{mK}^2]$', fontsize=13)
ax.set_title('QUIJOTE 11 GHz — BB (dl20)', fontsize=13)
ax.legend(frameon=False)
ax.set_xlim(20, 210)
ax.set_ylim(1e-10, 1e-4)

plt.tight_layout()
plt.show()



