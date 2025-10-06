#%%
import numpy as np
import os
from data import data, path_map, masks, path_masks
import functions 
import emcee
import corner
import matplotlib.pyplot as plt
from scipy.constants import c,h,k

# Bands to use
quijote_bands = ['11']
wmap_bands = ['23', '33', '41', '61', '94']
planck_bands = ['30', '44', '70', '100', '143', '217', '353']

band_list = quijote_bands + wmap_bands + planck_bands

mask_select = masks['quijote_galcut']['galcut10']
mask_name = mask_select['name']
out_path = '/home/pablo/Desktop/master/tfm/spectra/'
path_corrected_spectra = os.path.join(out_path, f'corrected_power_spectra_{mask_name}.fits')

spectra_dict = functions.read_corrected_cls(path_corrected_spectra, band_list)

ell_min = 30
ell_max = 200


def prepare_mcmc_data(
    spectra,
    band_list=None,
    modes=['EE'],
    ell_min=30,
    ell_max=200,
    band_pairs='all'
):
    """
    Prepare arrays for MCMC fitting from the dictionary returned by read_corrected_cls.

    Parameters
    ----------
    spectra : dict
        Dictionary with structure like spectra['band1_band2']['EE']['SPECTRUM'], etc.
    band_list : list of str, optional
        List of available frequency bands (e.g. ['11','23','30']). 
        Required if band_pairs='all'.
    modes : list of str, optional
        Modes to use: any combination of 'EE' and/or 'BB'. Default ['EE'].
    ell_min, ell_max : int, optional
        Multipole range (inclusive). Default 30-200.
    band_pairs : 'all' or list of str, optional
        - If 'all': build all possible pairs band_i_band_j from band_list (auto and cross).
        - If list: explicit list of pairs to use (e.g. ['11_23','11_30']).
        - If None: no automatic guessing, only use keys found in spectra (not recommended).

    Returns
    -------
    result : dict
        {
            'ell_eff': np.ndarray
                Common multipole array (ell_eff) used for all datasets, filtered by ell_min/ell_max.
            'y_all': np.ndarray
                1D stacked array of all selected spectra (concatenated).
            'yerr_all': np.ndarray
                1D stacked array of all selected error bars, aligned with y_all.
            'datasets': list of dict
                One entry per (pair, mode). Each dict contains:
                    {
                        'pair': '11_23',
                        'mode': 'EE',
                        'spectrum': array,
                        'error': array,
                        'freqs': (11.0, 23.0) or None,
                        'slice': (start, stop) indices into y_all/yerr_all
                    }
            'index_map': list of tuple
                Same order as datasets. Each (start, stop) gives the slice in y_all/yerr_all.
            'modes': list of str
                Modes requested (same as input).
            'pairs_used': list of str
                Actual band pairs successfully used (filtered from band_pairs).
        }
    """
    # --- Build list of pairs
    if band_pairs == 'all':
        if band_list is None:
            raise ValueError("band_list must be provided when band_pairs='all'")
        pairs = [f"{i}_{j}" for i in band_list for j in band_list]
    elif isinstance(band_pairs, list):
        pairs = band_pairs
    elif band_pairs is None:
        pairs = list(spectra.keys())
    else:
        raise ValueError("band_pairs must be 'all', a list of strings, or None")

    # Filter pairs that actually exist in spectra
    valid_pairs = [p for p in pairs if p in spectra]
    missing = [p for p in pairs if p not in spectra]
    if missing:
        print(f"Warning: {len(missing)} pairs not found in spectra and will be ignored: {missing}")
    if not valid_pairs:
        raise ValueError("No valid pairs found in spectra.")

    # Build ell_eff common grid
    ell_sets = []
    idx_map = {}
    for p in valid_pairs:
        ell_eff = np.array(spectra[p]['ell_eff'])
        mask = (ell_eff >= ell_min) & (ell_eff <= ell_max)
        ell_sel = ell_eff[mask]
        if ell_sel.size == 0:
            continue
        rounded = np.round(ell_sel).astype(int)
        ell_sets.append(set(rounded))
        idx_map[p] = np.where(mask)[0]
    if not ell_sets:
        raise ValueError("No multipoles in requested range.")

    ell_common_int = sorted(set.intersection(*ell_sets))
    if not ell_common_int:
        raise ValueError("No common multipoles across all pairs.")

    # Take float ell values from the first pair as representative
    first_pair = valid_pairs[0]
    ell_common = np.array([
        spectra[first_pair]['ell_eff'][idx]
        for idx in idx_map[first_pair]
        if int(round(spectra[first_pair]['ell_eff'][idx])) in ell_common_int
    ])

    # --- Build datasets
    datasets = []
    y_list, yerr_list, index_map = [], [], []
    cursor = 0

    for mode in modes:
        for p in valid_pairs:
            if mode not in spectra[p]:
                print(f"Warning: mode {mode} not in {p}, skipped.")
                continue
            ell_eff = np.array(spectra[p]['ell_eff'])
            # indices matching ell_common_int
            mask = (ell_eff >= ell_min) & (ell_eff <= ell_max)
            idxs = np.where(mask)[0]
            keep = [i for i in idxs if int(round(ell_eff[i])) in ell_common_int]
            if not keep:
                continue
            spec = spectra[p][mode]['SPECTRUM'][keep]
            err = spectra[p][mode]['ERROR'][keep]

            freqs = None
            if "_" in p:
                a, b = p.split("_", 1)
                try:
                    freqs = (float(a), float(b))
                except ValueError:
                    pass

            d = {
                'pair': p,
                'mode': mode,
                'spectrum': np.array(spec),
                'error': np.array(err),
                'freqs': freqs,
                'slice': (cursor, cursor + len(spec))
            }
            datasets.append(d)

            y_list.append(spec)
            yerr_list.append(err)
            index_map.append(d['slice'])
            cursor += len(spec)

    if not datasets:
        raise ValueError("No datasets prepared (check modes and ell range).")

    y_all = np.concatenate(y_list)
    yerr_all = np.concatenate(yerr_list)

    result = {
        'ell_eff': np.array(ell_common),
        'y_all': np.array(y_all),
        'yerr_all': np.array(yerr_all),
        'datasets': datasets,
        'index_map': index_map,
        'modes': modes,
        'pairs_used': valid_pairs
    }
    return result

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def planck(nu_GHz, T):
    # asegurar tipo numérico (scalar o array)
    nu_GHz = np.asarray(nu_GHz, dtype=float)
    nu = nu_GHz * 1e9
    x = h * nu / (k * T)
    return (2.0 * h * nu**3 / c**2) / np.expm1(x)

def g_RJ(nu_GHz):
    nu_GHz = np.asarray(nu_GHz, dtype=float)
    nu = nu_GHz * 1e9
    return 2.0 * k * nu**2 / c**2

def mbb_scaling_KRJ(nu_GHz, nu0_GHz=353.0, beta=1.59, T_d=19.6):
    nu_GHz = np.asarray(nu_GHz, dtype=float)
    nu0_GHz = float(nu0_GHz)
    power = (nu_GHz / nu0_GHz)**beta
    planck_ratio = planck(nu_GHz, T_d) / planck(nu0_GHz, T_d)
    rj_ratio = g_RJ(nu0_GHz) / g_RJ(nu_GHz)
    return power * planck_ratio * rj_ratio


# ============================================================================
# SYNCHROTRON MODEL
# ============================================================================

def model_synchrotron(theta, datasets, ell, fit_c_terms=False,
                      freq_ref=11.1, ell_ref=80.0):
    """
    Synchrotron power-spectrum model with (ell/ell_ref)^α scaling.
    Optionally includes constant c_i offsets for auto-spectra.

    Parameters
    ----------
    theta : array-like
        Model parameters:
          If fit_c_terms = False:
              [A_ref, alpha, beta]
          If fit_c_terms = True:
              [A_ref, alpha, beta, c_1, ..., c_N]
          where N = number of unique frequency bands in `datasets`.
        - A_ref : amplitude at reference frequency (freq_ref) and ell = ell_ref
        - alpha : multipole spectral slope
        - beta  : frequency spectral index
        - c_i   : constant noise/offset term for each auto-spectrum band
    datasets : list of dict
        Each element must contain:
          { 'pair': '23_33', 'freqs': (f1, f2), 'slice': (start, stop), ... }
    ell : array-like
        Multipole values (effective ell of each bin).
    fit_c_terms : bool, optional
        If True, include constant offset terms for autos (default False).
    freq_ref : float, optional
        Reference frequency in GHz (default 23.0 GHz).
    ell_ref : float, optional
        Reference multipole (default 80.0).

    Returns
    -------
    model_all : ndarray
        1D stacked array of predicted synchrotron power spectra, aligned with datasets order.
    """
    A_ref, alpha, beta = theta[:3]
    unique_freqs = sorted({f for d in datasets for f in d['freqs']})
    freq_to_idx = {f: i for i, f in enumerate(unique_freqs)}
    N = len(unique_freqs)

    c_terms = np.zeros(N)
    if fit_c_terms:
        if len(theta) != 3 + N:
            raise ValueError(f"Expected {3+N} parameters, got {len(theta)}")
        c_terms = theta[3:]

    model_list = []
    for d in datasets:
        f1, f2 = d['freqs']
        freq_factor = ((f1 * f2) / freq_ref**2)**beta
        c_i = c_terms[freq_to_idx[f1]] if (fit_c_terms and f1 == f2) else 0.0
        cl = A_ref * freq_factor * (ell / ell_ref)**alpha + c_i
        model_list.append(cl)

    return np.concatenate(model_list)


# ============================================================================
# DUST MODEL
# ============================================================================

def model_dust(theta, datasets, ell, fit_c_terms=False,
               freq_ref=353.0, T_d=19.6, ell_ref=80.0):
    """
    Dust power-spectrum model with modified blackbody (MBB) frequency scaling.
    Optionally includes constant c_i offsets for auto-spectra.

    Parameters
    ----------
    theta : array-like
        Model parameters:
          If fit_c_terms = False:
              [A_ref, alpha, beta_d]
          If fit_c_terms = True:
              [A_ref, alpha, beta_d, c_1, ..., c_N]
          where N = number of unique frequency bands in `datasets`.
        - A_ref : amplitude at reference frequency (freq_ref) and ell = ell_ref
        - alpha : multipole spectral slope
        - beta_d: dust emissivity index
        - c_i   : constant noise/offset term for each auto-spectrum band
    datasets : list of dict
        Same format as in model_synchrotron().
    ell : array-like
        Multipole values (effective ell of each bin).
    fit_c_terms : bool, optional
        If True, include constant offset terms for autos (default False).
    freq_ref : float, optional
        Reference frequency in GHz (default 353.0 GHz).
    T_d : float, optional
        Dust temperature in Kelvin (default 19.6 K).
    ell_ref : float, optional
        Reference multipole (default 80.0).

    Returns
    -------
    model_all : ndarray
        1D stacked array of predicted dust power spectra, aligned with datasets order.
    """
    A_ref, alpha, beta_d = theta[:3]
    unique_freqs = sorted({f for d in datasets for f in d['freqs']})
    freq_to_idx = {f: i for i, f in enumerate(unique_freqs)}
    N = len(unique_freqs)

    c_terms = np.zeros(N)
    if fit_c_terms:
        if len(theta) != 3 + N:
            raise ValueError(f"Expected {3+N} parameters, got {len(theta)}")
        c_terms = theta[3:]

    model_list = []
    for d in datasets:
        f1, f2 = d['freqs']
        fscale1 = mbb_scaling_KRJ(f1, nu0_GHz=freq_ref, beta=beta_d, T_d=T_d)
        fscale2 = mbb_scaling_KRJ(f2, nu0_GHz=freq_ref, beta=beta_d, T_d=T_d)
        freq_factor = fscale1 * fscale2
        c_i = c_terms[freq_to_idx[f1]] if (fit_c_terms and f1 == f2) else 0.0
        cl = A_ref * freq_factor * (ell / ell_ref)**alpha + c_i
        model_list.append(cl)

    return np.concatenate(model_list)


# ============================================================================
# SYNCHROTRON-DUST CROSS MODEL
# ============================================================================

def model_cross(theta, datasets, ell,
                ref_sync=23.0, ref_dust=353.0, T_d=19.6, ell_ref=80.0):
    """
    Synchrotron-dust cross-correlation power-spectrum model.

    Parameters
    ----------
    theta : array-like
        [rho_sd, A_sync, A_dust, alpha_s, alpha_d, beta_s, beta_d]
        - rho_sd : correlation coefficient between dust and synchrotron (|rho| ≤ 1)
        - A_sync : synchrotron amplitude at ref_sync and ell = ell_ref
        - A_dust : dust amplitude at ref_dust and ell = ell_ref
        - alpha_s, alpha_d : multipole slopes for synchrotron and dust
        - beta_s, beta_d   : frequency spectral indices
    datasets : list of dict
        Same format as before, with frequency pairs and metadata.
    ell : array-like
        Multipole values (effective ell of each bin).
    ref_sync : float, optional
        Reference frequency for synchrotron in GHz (default 23.0 GHz).
    ref_dust : float, optional
        Reference frequency for dust in GHz (default 353.0 GHz).
    T_d : float, optional
        Dust temperature in Kelvin (default 19.6 K).
    ell_ref : float, optional
        Reference multipole (default 80.0).

    Returns
    -------
    model_all : ndarray
        1D stacked array of predicted synchrotron-dust cross-spectra,
        aligned with datasets order.
    """
    rho_sd, A_sync, A_dust, alpha_s, alpha_d, beta_s, beta_d = theta

    model_list = []
    for d in datasets:
        f1, f2 = d['freqs']

        f_s1 = (f1 / ref_sync)**beta_s
        f_s2 = (f2 / ref_sync)**beta_s
        f_d1 = mbb_scaling_KRJ(f1, nu0_GHz=ref_dust, beta=beta_d, T_d=T_d)
        f_d2 = mbb_scaling_KRJ(f2, nu0_GHz=ref_dust, beta=beta_d, T_d=T_d)

        ell_factor = (ell / ell_ref)**((alpha_s + alpha_d) / 2.)
        cl = rho_sd * np.sqrt(A_sync * A_dust) * \
             (f_s1 * f_d2 + f_s2 * f_d1) * ell_factor

        model_list.append(cl)

    return np.concatenate(model_list)


'''
# ==========================================================================
'''

def lnlike(theta, datasets, ell, y_all, yerr_all,
           fit_c_terms=False,
           fit_components=('sync', 'dust', 'cross')):
    """
    Log-likelihood for synchrotron, dust, and cross models combined.

    Parameters
    ----------
    theta : array
        Model parameters concatenated for each fitted component.
        Expected structure depends on fit_components:
          If ('sync',):
              [A_s, alpha_s, beta_s, (c_i...)]
          If ('sync','dust'):
              [A_s, alpha_s, beta_s, (c_i_s...),
               A_d, alpha_d, beta_d, (c_i_d...)]
          If ('sync','dust','cross'):
              [A_s, alpha_s, beta_s, (c_i_s...),
               A_d, alpha_d, beta_d, (c_i_d...),
               rho_sd, A_s, A_d, alpha_s, alpha_d, beta_s, beta_d]
    datasets : list of dict
        From prepare_mcmc_data()['datasets'].
    ell : array
        Multipole values (effective ell per bin).
    y_all : array
        Observed data (1D stacked).
    yerr_all : array
        Observational errors (1D stacked).
    fit_c_terms : bool
        Whether constant terms per auto band are included.
    fit_components : tuple of str
        Components to include: any combination of
        ('sync', 'dust', 'cross').

    Returns
    -------
    lnlike : float
        Log-likelihood value.
    """
    # running index for parameter slicing
    idx = 0
    model_total = np.zeros_like(y_all)

    # --- SYNCHROTRON ---------------------------------------------------------
    if 'sync' in fit_components:
        unique_freqs = sorted({f for d in datasets for f in d['freqs']})
        n_c = len(unique_freqs) if fit_c_terms else 0
        n_params = 3 + n_c  # A_ref, alpha, beta, (+ constants)
        theta_sync = theta[idx:idx+n_params]
        idx += n_params

        model_total += model_synchrotron(
            theta_sync, datasets, ell,
            fit_c_terms=fit_c_terms
        )

    # --- DUST ----------------------------------------------------------------
    if 'dust' in fit_components:
        unique_freqs = sorted({f for d in datasets for f in d['freqs']})
        n_c = len(unique_freqs) if fit_c_terms else 0
        n_params = 3 + n_c
        theta_dust = theta[idx:idx+n_params]
        idx += n_params

        model_total += model_dust(
            theta_dust, datasets, ell,
            fit_c_terms=fit_c_terms
        )

    # --- CROSS TERM ----------------------------------------------------------
    if 'cross' in fit_components:
        # fixed 7 parameters: [rho_sd, A_sync, A_dust, alpha_s, alpha_d, beta_s, beta_d]
        theta_cross = theta[idx:idx+7]
        idx += 7

        model_total += model_cross(theta_cross, datasets, ell)

    # --- COMPUTE LIKELIHOOD --------------------------------------------------
    chi2 = np.sum(((y_all - model_total) / yerr_all)**2)
    return -0.5 * chi2


# ============================================================================
# PRIORS
# ============================================================================

def lnprior(theta, datasets, fit_c_terms=False,
            fit_components=('sync', 'dust', 'cross')):
    """
    Prior for multi-component (sync + dust + cross) Galactic model.

    Parameters
    ----------
    theta : array
        Model parameters, same ordering as in lnlike().
    datasets : list of dict
        From prepare_mcmc_data()['datasets'].
    fit_c_terms : bool
        Whether constant terms are included.
    fit_components : tuple of str
        Which components are active.

    Returns
    -------
    lnprior : float
        Log-prior value (0 if valid, -inf if invalid).
    """
    idx = 0

    # --- SYNCHROTRON ---------------------------------------------------------
    if 'sync' in fit_components:
        unique_freqs = sorted({f for d in datasets for f in d['freqs']})
        n_c = len(unique_freqs) if fit_c_terms else 0
        n_params = 3 + n_c
        A_ref, alpha, beta = theta[idx:idx+3]
        idx += n_params

        if not (A_ref > 0 and -6 < alpha < -0.5 and -5 < beta < -1):
            return -np.inf
        if fit_c_terms:
            c_terms = theta[idx-n_c:idx]
            if np.any(c_terms < 0):
                return -np.inf

    # --- DUST ----------------------------------------------------------------
    if 'dust' in fit_components:
        unique_freqs = sorted({f for d in datasets for f in d['freqs']})
        n_c = len(unique_freqs) if fit_c_terms else 0
        n_params = 3 + n_c
        A_ref, alpha, beta_d = theta[idx:idx+3]
        idx += n_params

        if not (A_ref > 0 and -6 < alpha < -0.5 and 0.5 < beta_d):
            return -np.inf
        if fit_c_terms:
            c_terms = theta[idx-n_c:idx]
            if np.any(c_terms < 0):
                return -np.inf

    # --- CROSS TERM ----------------------------------------------------------
    if 'cross' in fit_components:
        rho_sd, A_s, A_d, alpha_s, alpha_d, beta_s, beta_d = theta[idx:idx+7]
        idx += 7
        if not (-1 <= rho_sd <= 1):
            return -np.inf
        if not (A_s > 0 and A_d > 0):
            return -np.inf
        if not (-6 < alpha_s < -0.5 and -6 < alpha_d < -0.5):
            return -np.inf
        if not (-5 < beta_s < -1 and 0.5 < beta_d):
            return -np.inf

    return 0.0


# ============================================================================
# POSTERIOR
# ============================================================================

def lnprob(theta, datasets, ell, y_all, yerr_all,
           fit_c_terms=False,
           fit_components=('sync', 'dust', 'cross')):
    """
    Combined log-posterior = lnprior + lnlike for the Galactic emission model.

    Parameters
    ----------
    theta : array
        Parameter vector.
    datasets, ell, y_all, yerr_all : see lnlike().
    fit_c_terms : bool
        Whether constant terms are included.
    fit_components : tuple of str
        Components to include: any of ('sync','dust','cross').

    Returns
    -------
    lnprob : float
        Log posterior value.
    """
    lp = lnprior(theta, datasets, fit_c_terms=fit_c_terms,
                 fit_components=fit_components)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, datasets, ell, y_all, yerr_all,
                       fit_c_terms=fit_c_terms,
                       fit_components=fit_components)



band_pairs = [
            #   '11_11', '11_23', '23_23', '11_30', '30_30', '23_30',
              '94_94', '94_143', '143_143', '143_353', '353_353', '94_353', '217_217', '143_217', '94_217', '217_353',
            #   '143_143', '217_217', '143_217', '100_100', '100_143', '100_217'
              ]

fit_data_ee = prepare_mcmc_data(spectra_dict, band_list=band_list, modes=['EE'], band_pairs=band_pairs)


# -------------------------------
# Fitting configuration
# -------------------------------

nwalkers = 50
ninter = 10000
discard_fraction = 0.5  # Burn-in fraction

fit_components = (
    # 'sync', 
    'dust', 
    # 'cross'
)
fit_c_terms = False

datasets = fit_data_ee['datasets']
ell = fit_data_ee['ell_eff']
y_all = fit_data_ee['y_all']
yerr_all = fit_data_ee['yerr_all']

# -------------------------------
# Determine number of parameters
# -------------------------------

unique_freqs = sorted({f for d in datasets for f in d['freqs']})
n_c = len(unique_freqs) if fit_c_terms else 0

ndim = 0
if 'sync' in fit_components:
    ndim += 3 + n_c
if 'dust' in fit_components:
    ndim += 3 + n_c
if 'cross' in fit_components:
    ndim += 7

# -------------------------------
# Initial positions (p0)
# -------------------------------

p0 = []

# --- SYNCHROTRON ---
if 'sync' in fit_components:
    # rough guesses for amplitude, ell slope, frequency index
    A_sync_guess = 1.5
    alpha_sync_guess = -3.0
    beta_sync_guess = -3.0
    if fit_c_terms:
        c_guess = np.zeros(n_c)
        p0 += [A_sync_guess, alpha_sync_guess, beta_sync_guess] + c_guess.tolist()
    else:
        p0 += [A_sync_guess, alpha_sync_guess, beta_sync_guess]

# --- DUST ---
if 'dust' in fit_components:
    # rough guesses
    A_dust_guess = 1e-1
    alpha_dust_guess = -2.5
    beta_dust_guess = 1.59
    if fit_c_terms:
        c_guess = np.zeros(n_c)
        p0 += [A_dust_guess, alpha_dust_guess, beta_dust_guess] + c_guess.tolist()
    else:
        p0 += [A_dust_guess, alpha_dust_guess, beta_dust_guess]

# --- CROSS ---
if 'cross' in fit_components:
    # use the same values as above for consistency
    rho_guess = 0.5
    p0 += [rho_guess, A_sync_guess, A_dust_guess,
           alpha_sync_guess, alpha_dust_guess,
           beta_sync_guess, beta_dust_guess]


# Convert p0 into initial positions for all walkers by adding small random noise
p0_walkers = np.array(p0) + 1e-1 * np.random.randn(nwalkers, ndim)

# -------------------------------
# Run emcee
# -------------------------------

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, lnprob,
    args=(datasets, ell, y_all, yerr_all, fit_c_terms, fit_components)
)

print("Starting MCMC...")
sampler.run_mcmc(p0_walkers, ninter, progress=True)

# -------------------------------
# Discard burn-in and flatten
# -------------------------------

discard = int(ninter * discard_fraction)
samples = sampler.get_chain(discard=discard, flat=True)

print(f"Shape of flattened samples: {samples.shape}")

# -------------------------------
# Corner plot
# -------------------------------


# Determine parameter indices for scaling
idx = 0
scaling = np.ones(ndim)  # default 1

labels = []

# --- Synchrotron ---
if 'sync' in fit_components:
    labels += [r'$A_{\rm sync}\ (\mu{\rm K}^2)$', r'$\alpha_{\rm sync}$', r'$\beta_{\rm sync}$']
    scaling[idx] = 1e6  # scale amplitude
    idx += 3
    if fit_c_terms:
        for f in unique_freqs:
            labels.append(fr'$c_{{\rm sync,{int(f)}GHz}}$')
            idx += 1

# --- Dust ---
if 'dust' in fit_components:
    labels += [r'$A_{\rm dust}\ (\mu{\rm K}^2)$', r'$\alpha_{\rm dust}$', r'$\beta_{\rm dust}$']
    scaling[idx] = 1e6
    idx += 3
    if fit_c_terms:
        for f in unique_freqs:
            labels.append(fr'$c_{{\rm dust,{int(f)}GHz}}$')
            idx += 1

# --- Cross ---
if 'cross' in fit_components:
    labels += [r'$\rho_{\rm sd}$',
               r'$A_{\rm sync}^{\rm cross}\ (\mu{\rm K}^2)$',
               r'$A_{\rm dust}^{\rm cross}\ (\mu{\rm K}^2)$',
               r'$\alpha_{\rm sync}^{\rm cross}$',
               r'$\alpha_{\rm dust}^{\rm cross}$',
               r'$\beta_{\rm sync}^{\rm cross}$',
               r'$\beta_{\rm dust}^{\rm cross}$']
    scaling[idx+1] = 1e6  # A_sync_cross
    scaling[idx+2] = 1e6  # A_dust_cross
    idx += 7

# Apply scaling
samples_scaled = samples.copy()
for i, s in enumerate(scaling):
    samples_scaled[:, i] *= s

# Create corner plot
fig = corner.corner(samples_scaled, labels=labels,
                    quantiles=[0.16, 0.5, 0.84], show_titles=True,
                    title_fmt=".2f", label_kwargs={"fontsize":12})
plt.show()

#%%

import matplotlib.cm as cm

# ==============================
# COMPARISON PLOT: DATA vs DUST (solo autos)
# ==============================

# --- Define fixed dust parameters ---
A_dust = 1e-6      # en K^2
alpha_dust = -2.5
beta_dust = 1.59
T_d = 19.6        # temperatura polvo
freq_ref = 353.0
ell_ref = 80.0

theta_dust_fixed = [A_dust, alpha_dust, beta_dust]

plt.figure(figsize=(10,6))

# Generar colores distintos según el número de frecuencias
unique_freqs = sorted({d['freqs'][0] for d in datasets if d['freqs'][0] == d['freqs'][1]})
colors = cm.viridis(np.linspace(0,1,len(unique_freqs)))
freq_to_color = {f: c for f,c in zip(unique_freqs, colors)}

for d in datasets:
    f1, f2 = d['freqs']
    if f1 != f2:
        continue  # solo autos

    color = freq_to_color[f1]

    # Datos
    plt.errorbar(ell, d['spectrum']*1e6, yerr=d['error']*1e6, fmt='o', label=f"Data {int(f1)} GHz", alpha=0.6, color=color)

    # Modelo de polvo
    fscale1 = mbb_scaling_KRJ(f1, nu0_GHz=freq_ref, beta=beta_dust, T_d=T_d)
    cl_model = A_dust * fscale1**2 * (ell / ell_ref)**alpha_dust
    plt.plot(ell, cl_model*1e6, linestyle='--', color=color, label=f"Dust model {int(f1)} GHz")

plt.xlabel(r"$\ell$", fontsize=14)
plt.ylabel(r"$C_\ell^{EE}\ [\mu K^2]$", fontsize=14)
plt.title("Comparison: Auto-spectra Data vs Dust Model")
plt.yscale('log')
plt.legend(fontsize=10, frameon=False, ncol=2)
plt.show()

