#%%
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from data import data, path_map, masks, path_masks
import functions 
import emcee
import corner
import matplotlib.pyplot as plt
from scipy.constants import c,h,k
import multiprocessing as mp
from scipy.stats import gaussian_kde

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
        Dictionary of spectra data.
    band_list : list of str
        List of frequency bands.
    modes : list of str
        Spectral modes to include ('EE', 'BB', etc.).
    ell_min : int
        Minimum multipole.
    ell_max : int
        Maximum multipole.
    band_pairs : 'all', list, or None
        Which frequency pairs to use.

    Returns
    -------
    dict
        Contains 'ell_eff', 'y_all', 'yerr_all', 'datasets', 'index_map', 'modes', and 'pairs_used'.
    """
    # --- Build list of pairs
    if band_pairs == 'all':
        if band_list is None:
            raise ValueError("band_list is required when band_pairs='all'.")
        pairs = [f"{a}_{b}" for i, a in enumerate(band_list) for b in band_list[i:]]
    elif isinstance(band_pairs, list):
        pairs = list(band_pairs)
    elif band_pairs is None:
        pairs = list(spectra.keys())
    else:
        raise ValueError("band_pairs must be 'all', list, or None")

    # Filter valid pairs
    valid_pairs = [p for p in pairs if p in spectra]
    missing = [p for p in pairs if p not in spectra]
    if missing:
        print(f"Warning: missing pairs in spectra: {missing}")
    if not valid_pairs:
        raise ValueError("No valid pairs found in spectra with the given selection.")

    # Helper: obtain ell array (float) for a given pair (try several keys)
    def _get_ell_array(p, mode_hint=None):
        entry = spectra[p]
        # priority: 'ell_eff' at top-level, then mode-specific 'ell_eff', then 'ell1'/'ell2' mean
        if 'ell_eff' in entry:
            return np.asarray(entry['ell_eff'])
        if mode_hint is not None and mode_hint in entry and 'ell_eff' in entry[mode_hint]:
            return np.asarray(entry[mode_hint]['ell_eff'])
        # try per-pair ell1/ell2
        if 'ell1' in entry and 'ell2' in entry:
            e1 = np.asarray(entry['ell1'])
            e2 = np.asarray(entry['ell2'])
            # if lengths equal, use midpoint; else try to use ell2 (more common for bandning)
            if e1.shape == e2.shape:
                return 0.5 * (e1 + e2)
            return np.asarray(entry.get('ell2', e2))
        # try mode-specific ell1/ell2
        if mode_hint is not None and mode_hint in entry:
            m = entry[mode_hint]
            if 'ell_eff' in m:
                return np.asarray(m['ell_eff'])
            if 'ell1' in m and 'ell2' in m:
                e1 = np.asarray(m['ell1']); e2 = np.asarray(m['ell2'])
                if e1.shape == e2.shape:
                    return 0.5 * (e1 + e2)
                return e2
        raise KeyError(f"No ell information found for pair '{p}'")

    # --- Determine common ell bins
    ell_sets = []
    idx_map = {}
    for p in valid_pairs:
        ell_eff_pair = _get_ell_array(p, mode_hint=modes[0] if modes else None)
        ell_int = np.array([int(round(x)) for x in ell_eff_pair])
        # Apply ell range
        use = (ell_int >= ell_min) & (ell_int <= ell_max)
        idx = np.where(use)[0]
        idx_map[p] = idx
        ell_sets.append(set(ell_int[use]))

    if not ell_sets:
        raise ValueError("No ell bins after applying ell range.")

    ell_common_int = sorted(set.intersection(*ell_sets))
    if not ell_common_int:
        raise ValueError("Empty intersection of ell bins across selected pairs.")

    # Map intersection back to float ells using first pair (safe retrieval)
    first_pair = valid_pairs[0]
    ell_first = _get_ell_array(first_pair, mode_hint=modes[0] if modes else None)
    ell_first_int = np.array([int(round(x)) for x in ell_first])
    mask_common = np.isin(ell_first_int, ell_common_int)
    ell_common = ell_first[mask_common]

    # --- Build datasets and stacked vectors
    datasets = []
    y_list, yerr_list, index_map = [], [], []
    cursor = 0

    for mode in modes:
        for p in valid_pairs:
            if mode not in spectra[p]:
                continue
            # per-mode ell (safe)
            ell_eff_pair = _get_ell_array(p, mode_hint=mode)
            ell_int = np.array([int(round(x)) for x in ell_eff_pair])
            keep = np.where(np.isin(ell_int, ell_common_int))[0]
            if keep.size == 0:
                continue

            # required keys under mode
            mode_entry = spectra[p][mode]
            if 'SPECTRUM' not in mode_entry or 'ERROR' not in mode_entry:
                raise KeyError(f"Missing SPECTRUM/ERROR for pair '{p}', mode '{mode}'")
            spec = np.asarray(mode_entry['SPECTRUM'])[keep]
            err = np.asarray(mode_entry['ERROR'])[keep]

            # parse frequencies from pair name (expect 'f1_f2' numeric)
            try:
                f1, f2 = map(float, p.split('_'))
            except Exception:
                # fallback: if stored in dict under 'freqs' use that
                freqs = spectra[p].get('freqs') or mode_entry.get('freqs')
                if freqs is None:
                    raise ValueError(f"Cannot parse frequencies from pair name '{p}' and no 'freqs' field found.")
                f1, f2 = float(freqs[0]), float(freqs[1])

            start, stop = cursor, cursor + keep.size

            datasets.append({
                'pair': p,
                'mode': mode,
                'spectrum': spec,
                'error': err,
                'freqs': (f1, f2),
                'slice': (start, stop)
            })
            y_list.append(spec)
            yerr_list.append(err)
            index_map.append((start, stop))
            cursor = stop

    if not datasets:
        raise ValueError("No datasets constructed for the given selection.")

    y_all = np.concatenate(y_list)
    yerr_all = np.concatenate(yerr_list)

    return {
        'ell_eff': np.array(ell_common),
        'y_all': np.array(y_all),
        'yerr_all': np.array(yerr_all),
        'datasets': datasets,
        'index_map': index_map,
        'modes': modes,
        'pairs_used': valid_pairs
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def planck(nu_GHz, T):
    """
    Compute the Planck function B_nu(T) in SI units [W/m^2/Hz/sr].

    Parameters
    ----------
    nu_GHz : float or array
        Frequency in GHz.
    T : float
        Temperature in K.

    Returns
    -------
    B_nu : float or array
    """
    nu = np.asarray(nu_GHz, dtype=float) * 1e9
    x = h * nu / (k * T)
    return (2.0 * h * nu**3 / c**2) / np.expm1(x)

def g_RJ(nu_GHz):
    """
    Rayleigh-Jeans conversion factor in K_RJ units.

    Parameters
    ----------
    nu_GHz : float or array
        Frequency in GHz.

    Returns
    -------
    g_RJ : float or array
    """
    nu = np.asarray(nu_GHz, dtype=float) * 1e9
    return 2.0 * k * nu**2 / c**2

def mbb_scaling_KRJ(nu_GHz, nu0_GHz=353.0, beta=1.59, T_d=19.6):
    """
    Compute MBB scaling in K_RJ units relative to reference frequency.

    Parameters
    ----------
    nu_GHz : float or array
        Frequency in GHz.
    nu0_GHz : float
        Reference frequency.
    beta : float
        Dust spectral index.
    T_d : float
        Dust temperature in K.

    Returns
    -------
    scale : float or array
        Scaling factor for dust emission.
    """
    nu_GHz = np.asarray(nu_GHz, dtype=float)
    power = (nu_GHz / nu0_GHz) ** beta
    planck_ratio = planck(nu_GHz, T_d) / planck(nu0_GHz, T_d)
    rj_ratio = g_RJ(nu0_GHz) / g_RJ(nu_GHz)
    return power * planck_ratio * rj_ratio

# ============================================================================
# MODEL
# ============================================================================

def model_synchrotron(theta, datasets, ell, fit_c_terms=False,
                      freq_ref=11.1, ell_ref=80.0):
    """
    Synchrotron angular power spectrum model.
    
    Parameters
    ----------
    theta : list or array
        Parameters [A_s, alpha_s, beta_s] (+ c_sync[band] if fit_c_terms).
    datasets : list of dict
        Prepared datasets with frequency pairs, spectra, and errors.
    ell : array
        Multipoles.
    fit_c_terms : bool
        Whether to fit constant terms for auto-spectra.
    freq_ref : float
        Reference frequency in GHz.
    ell_ref : float
        Reference multipole.

    Returns
    -------
    model_vector : array
        Concatenated modeled spectra for all datasets.
    """
    A_s, alpha_s, beta_s = theta[:3]
    unique_freqs = sorted({f for d in datasets for f in d['freqs']})
    freq_to_idx = {f: i for i, f in enumerate(unique_freqs)}
    N = len(unique_freqs)

    c_terms = np.zeros(N)
    if fit_c_terms:
        if len(theta) != 3 + N:
            raise ValueError("theta length mismatch for synch c_terms")
        c_terms = np.asarray(theta[3:3+N])

    model_list = []
    for d in datasets:
        f1, f2 = d['freqs']
        scale_f1 = (f1 / freq_ref) ** beta_s
        scale_f2 = (f2 / freq_ref) ** beta_s
        ell_scale = (ell / ell_ref) ** alpha_s
        Cl = A_s * ell_scale * scale_f1 * scale_f2
        if fit_c_terms and (f1 == f2):
            i = freq_to_idx[f1]
            Cl = Cl + c_terms[i]
        model_list.append(Cl)
    return np.concatenate(model_list)

def model_dust(theta, datasets, ell, fit_c_terms=False,
               freq_ref=353.0, T_d=19.6, ell_ref=80.0):
    """
    Dust angular power spectrum model with modified blackbody scaling.

    Parameters
    ----------
    theta : list or array
        Parameters [A_d, alpha_d, beta_d] (+ c_dust[band] if fit_c_terms).
    datasets : list of dict
        Prepared datasets.
    ell : array
        Multipoles.
    fit_c_terms : bool
        Whether to fit constant terms for auto-spectra.
    freq_ref : float
        Reference frequency in GHz.
    T_d : float
        Dust temperature in K.
    ell_ref : float
        Reference multipole.

    Returns
    -------
    model_vector : array
        Concatenated modeled spectra for all datasets.
    """
    A_d, alpha_d, beta_d = theta[:3]
    unique_freqs = sorted({f for d in datasets for f in d['freqs']})
    freq_to_idx = {f: i for i, f in enumerate(unique_freqs)}
    N = len(unique_freqs)

    c_terms = np.zeros(N)
    if fit_c_terms:
        if len(theta) != 3 + N:
            raise ValueError("theta length mismatch for dust c_terms")
        c_terms = np.asarray(theta[3:3+N])

    # Precompute per-frequency dust scaling
    freqs_all = np.array(unique_freqs)
    S = mbb_scaling_KRJ(freqs_all, nu0_GHz=freq_ref, beta=beta_d, T_d=T_d)

    model_list = []
    for d in datasets:
        f1, f2 = d['freqs']
        s1 = S[freq_to_idx[f1]]
        s2 = S[freq_to_idx[f2]]
        ell_scale = (ell / ell_ref) ** alpha_d
        Cl = A_d * ell_scale * s1 * s2
        if fit_c_terms and (f1 == f2):
            i = freq_to_idx[f1]
            Cl = Cl + c_terms[i]
        model_list.append(Cl)
    return np.concatenate(model_list)

def model_cross(theta, datasets, ell,
                ref_sync=11.1, ref_dust=353.0, T_d=19.6, ell_ref=80.0):
    """
    Cross-correlation between synchrotron and dust components.

    Parameters
    ----------
    theta : list or array
        Parameters [rho, A_s, A_d, alpha_s, alpha_d, beta_s, beta_d].
    datasets : list of dict
        Prepared datasets.
    ell : array
        Multipoles.
    ref_sync : float
        Reference synchrotron frequency in GHz.
    ref_dust : float
        Reference dust frequency in GHz.
    T_d : float
        Dust temperature in K.
    ell_ref : float
        Reference multipole.

    Returns
    -------
    model_vector : array
        Concatenated modeled spectra for all datasets.
    """
    rho, A_s, A_d, alpha_s, alpha_d, beta_s, beta_d = theta
    unique_freqs = sorted({f for d in datasets for f in d['freqs']})
    freq_to_idx = {f: i for i, f in enumerate(unique_freqs)}

    freqs_all = np.array(unique_freqs)
    Sd = mbb_scaling_KRJ(freqs_all, nu0_GHz=ref_dust, beta=beta_d, T_d=T_d)

    model_list = []
    for d in datasets:
        f1, f2 = d['freqs']
        s1 = (f1 / ref_sync) ** beta_s
        s2 = (f2 / ref_sync) ** beta_s
        ell_scale_s = (ell / ell_ref) ** alpha_s
        C_s_ij = A_s * ell_scale_s * s1 * s2

        d1 = Sd[freq_to_idx[f1]]
        d2 = Sd[freq_to_idx[f2]]
        ell_scale_d = (ell / ell_ref) ** alpha_d
        C_d_ij = A_d * ell_scale_d * d1 * d2

        # cross term (rho * sqrt(C_s * C_d))
        ell_scale_cross = (ell / ell_ref) ** ((alpha_s + alpha_d) / 2)
        C_sd_ij = rho * np.sqrt(A_s * A_d) * (s1 * d2 + s2 * d1) * ell_scale_cross

        model_list.append(C_sd_ij)
    return np.concatenate(model_list)



def lnlike(theta_full, datasets, ell, y_all, yerr_all,
           fit_c_terms=False, fit_components=('sync', 'dust', 'cross')):
    """
    Compute the log-likelihood (-0.5 chi^2) for the model given data.

    Parameters
    ----------
    theta_full : array
        Full parameter vector.
    datasets : list of dict
        Prepared datasets.
    ell : array
        Multipoles.
    y_all : array
        Observed spectra concatenated.
    yerr_all : array
        Errors associated with y_all.
    fit_c_terms : bool
        Whether to fit constant c terms.
    fit_components : tuple
        Components to include ('sync','dust','cross').

    Returns
    -------
    lnL : float
        Log-likelihood value.
    """
    unique_freqs = sorted({f for d in datasets for f in d['freqs']})
    N = len(unique_freqs)

    A_s, alpha_s, beta_s, A_d, alpha_d, beta_d, rho = theta_full[:7]
    c_sync = np.zeros(N)
    c_dust = np.zeros(N)
    offset = 7
    if fit_c_terms:
        c_sync = np.asarray(theta_full[offset:offset+N]); offset += N
        c_dust = np.asarray(theta_full[offset:offset+N]); offset += N

    y_model = np.zeros_like(y_all)

    if 'sync' in fit_components:
        y_model += model_synchrotron([A_s, alpha_s, beta_s, *c_sync] if fit_c_terms else
                                     [A_s, alpha_s, beta_s],
                                     datasets, ell, fit_c_terms=fit_c_terms)

    if 'dust' in fit_components:
        y_model += model_dust([A_d, alpha_d, beta_d, *c_dust] if fit_c_terms else
                              [A_d, alpha_d, beta_d],
                              datasets, ell, fit_c_terms=fit_c_terms)

    if 'cross' in fit_components:
        y_model += model_cross([rho, A_s, A_d, alpha_s, alpha_d, beta_s, beta_d],
                               datasets, ell)

    chi2 = np.sum(((y_all - y_model) / yerr_all) ** 2)
    return -0.5 * chi2

def lnprior(theta_full, datasets, fit_c_terms=False, fit_components=('sync', 'dust', 'cross')):
    """
    Apply priors on parameters.

    Returns -np.inf if any prior is violated.
    """
    unique_freqs = sorted({f for d in datasets for f in d['freqs']})
    N = len(unique_freqs)

    A_s, alpha_s, beta_s, A_d, alpha_d, beta_d, rho = theta_full[:7]

    if 'sync' in fit_components:
        if not (A_s >= 0.0): return -np.inf
        if not (-6 <= alpha_s <= 0): return -np.inf
        if not (-6 <= beta_s <= 0): return -np.inf

    if 'dust' in fit_components:
        if not (A_d >= 0.0): return -np.inf
        if not (-6 <= alpha_d <= 0): return -np.inf
        if not (0 <= beta_d <= 6): return -np.inf

    if 'cross' in fit_components:
        if not (-1 <= rho <= 1): return -np.inf

    if fit_c_terms:
        offset = 7
        if 'sync' in fit_components:
            c_sync = np.asarray(theta_full[offset:offset+N]); offset += N
            if not (np.all(np.isfinite(c_sync)) and np.all(np.abs(c_sync) <= 1e6)):
                return -np.inf
        if 'dust' in fit_components:
            c_dust = np.asarray(theta_full[offset:offset+N]); offset += N
            if not (np.all(np.isfinite(c_dust)) and np.all(np.abs(c_dust) <= 1e6)):
                return -np.inf

    return 0.0

def lnprob(theta_free, datasets, ell, y_all, yerr_all,
           fit_c_terms=False, fit_components=('sync', 'dust', 'cross'),
           param_map=None, fixed_values=None):
    """
    Reconstruct full parameter vector from free parameters and fixed values,
    then compute log-posterior (lnprior + lnlike).

    Parameters
    ----------
    theta_free : array
        Free parameters only.
    param_map : list of tuples
        List of (name, is_free) for each parameter in full vector.
    fixed_values : dict
        Values for parameters that are fixed.

    Returns
    -------
    lnpost : float
        Log-posterior probability.
    """
    theta_full = np.zeros(len(param_map))
    free_cursor = 0
    for i, (name, is_free) in enumerate(param_map):
        if is_free:
            theta_full[i] = theta_free[free_cursor]
            free_cursor += 1
        else:
            theta_full[i] = fixed_values.get(name, 0.0)

    lp = lnprior(theta_full, datasets, fit_c_terms=fit_c_terms, fit_components=fit_components)
    if not np.isfinite(lp):
        return -np.inf
    ll = lnlike(theta_full, datasets, ell, y_all, yerr_all,
                fit_c_terms=fit_c_terms, fit_components=fit_components)
    return lp + ll


fit_data_ee = prepare_mcmc_data(spectra_dict, band_list=band_list, modes=['EE'], band_pairs='all')
fit_data_bb = prepare_mcmc_data(spectra_dict, band_list=band_list, modes=['BB'], band_pairs='all')

# -------------------------------
# Fitting configuration
# -------------------------------

fit_components = (
    'sync', 
    'dust', 
    'cross'
)

fit_c_terms = False

datasets = fit_data_ee['datasets']
ell = fit_data_ee['ell_eff']
y_all = fit_data_ee['y_all']
yerr_all = fit_data_ee['yerr_all']

# -------------------------------
# Determine parameter mapping
# -------------------------------
unique_freqs = sorted({f for d in datasets for f in d['freqs']})
N = len(unique_freqs)

# full parameter order
param_names = ['A_s', 'alpha_s', 'beta_s', 'A_d', 'alpha_d', 'beta_d', 'rho']
if fit_c_terms:
    param_names += [f'c_sync[{int(f)}]' for f in unique_freqs]
    param_names += [f'c_dust[{int(f)}]' for f in unique_freqs]

free_mask = {
    'A_s':    ('sync' in fit_components),
    'alpha_s':('sync' in fit_components),
    'beta_s': ('sync' in fit_components),
    'A_d':    ('dust' in fit_components),
    'alpha_d':('dust' in fit_components),
    'beta_d': ('dust' in fit_components),
    'rho':    ('cross' in fit_components)
}
param_map = []
for name in ['A_s','alpha_s','beta_s','A_d','alpha_d','beta_d','rho']:
    param_map.append((name, free_mask[name]))
if fit_c_terms:
    for f in unique_freqs:
        param_map.append((f'c_sync[{int(f)}]', True if 'sync' in fit_components else False))
    for f in unique_freqs:
        param_map.append((f'c_dust[{int(f)}]', True if 'dust' in fit_components else False))

# fixed values for inactive params (show as 0 in corner)
fixed_values = {name: 0.0 for name, is_free in param_map if not is_free}

ndim = sum(1 for _, is_free in param_map if is_free)

# -------------------------------
# Initial positions (p0)
# -------------------------------
rng = np.random.default_rng()
p0_center = []
for name, is_free in param_map:
    if not is_free:
        continue
    if name == 'A_s':
        p0_center.append(1)
    elif name == 'alpha_s':
        p0_center.append(-3.)
    elif name == 'beta_s':
        p0_center.append(-3.)
    elif name == 'A_d':
        p0_center.append(1.)
    elif name == 'alpha_d':
        p0_center.append(-2.3)
    elif name == 'beta_d':
        p0_center.append(1.59)
    elif name == 'rho':
        p0_center.append(0.05)
    elif name.startswith('c_sync') or name.startswith('c_dust'):
        p0_center.append(0.0)
    else:
        p0_center.append(0.0)

p0_center = np.array(p0_center, dtype=float)
nwalkers = 200
ninter = 10000
discard_fraction = 0.5

# small Gaussian ball
p0_walkers = p0_center + 1e-2 * rng.standard_normal((nwalkers, ndim))

# -------------------------------
# Run emcee
# -------------------------------

if __name__ == "__main__":
    # Número de procesos: no más que CPUs disponibles ni que nwalkers//2 (emcee actualiza por mitades)
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    n_procs = max(1, min(available, max(1, nwalkers // 2)))

    print(f"Using {n_procs} processes (available={available}, nwalkers={nwalkers})")

    # Use a process pool for parallel log-probability evaluations
    with mp.get_context("fork").Pool(processes=n_procs, maxtasksperchild=200) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, lnprob,
            args=(datasets, ell, y_all, yerr_all, fit_c_terms, fit_components, param_map, fixed_values),
            pool=pool
        )
        print("Starting MCMC (parallel)...")
        sampler.run_mcmc(p0_walkers, ninter, progress=True)
    

# -------------------------------
# Discard burn-in and flatten
# -------------------------------
discard = int(ninter * discard_fraction)
samples_free = sampler.get_chain(discard=discard, flat=True)

# Expand to full samples (insert zeros for fixed)
n_full = len(param_map)
samples_full = np.zeros((samples_free.shape[0], n_full))
free_cols = [i for i, (_, is_free) in enumerate(param_map) if is_free]
fixed_cols = [i for i, (_, is_free) in enumerate(param_map) if not is_free]
samples_full[:, free_cols] = samples_free
for i in fixed_cols:
    samples_full[:, i] = fixed_values[param_names[i]]

print(f"Shape of flattened samples (free): {samples_free.shape}, (full): {samples_full.shape}")

# -------------------------------
# Corner labels / scaling
# -------------------------------

def apply_corner_scales(samples, labels, scale_map):
    """
    Scale MCMC samples for plotting in corner plots.
    
    Parameters
    ----------
    samples : array-like
        MCMC samples (n_samples, n_params).
    labels : list of str
        Parameter names corresponding to columns of samples.
    scale_map : dict
        Dictionary mapping parameter names to (scale_factor, unit_tex_str_or_None).

    Returns
    -------
    scaled_samples : array
        Samples scaled according to scale_map.
    """
    X = np.array(samples, copy=True)
    for j, name in enumerate(labels):
        factor, _ = scale_map.get(name, (1.0, None))
        if factor != 1.0:
            X[:, j] *= factor
    return X

# -------------------------------
# Prepare labels and scales
# -------------------------------

labels_free = [name for name, is_free in param_map if is_free]

# Scaling factors and units
scale_map = {
    'A_s': (1e6, r'$\mu\mathrm{K}^2$'),
    'A_d': (1e9, r'$10^{3}\,\mu\mathrm{K}^2$'),
}

# Also scale c_terms if present
for name in labels_free:
    if name.startswith('c_sync') or name.startswith('c_dust'):
        scale_map[name] = (1e9, r'$10^{3}\,\mu\mathrm{K}^2$')

samples_plot = apply_corner_scales(samples_free, labels_free, scale_map)

# Labels for corner plot
latex_labels = {
    'A_s': r'$A_{\mathrm{s}}\,[\mu\mathrm{K}^2]$',
    'alpha_s': r'$\alpha_{\mathrm{s}}$',
    'beta_s': r'$\beta_{\mathrm{s}}$',
    'A_d': r'$A_{\mathrm{d}}\,[10^{3}\,\mu\mathrm{K}^2]$',
    'alpha_d': r'$\alpha_{\mathrm{d}}$',
    'beta_d': r'$\beta_{\mathrm{d}}$',
    'rho': r'$\rho$',
}

# Dynamic labels for c_sync and c_dust
for name in labels_free:
    if name.startswith('c_sync'):
        freq = name.split('[')[-1].strip(']')
        latex_labels[name] = rf'$c_{{\mathrm{{sync}},\,{freq}}}\,[10^3\,\mu\mathrm{{K}}^2]$'
    elif name.startswith('c_dust'):
        freq = name.split('[')[-1].strip(']')
        latex_labels[name] = rf'$c_{{\mathrm{{dust}},\,{freq}}}\,[10^3\,\mu\mathrm{{K}}^2]$'

labels_plot = [latex_labels.get(name, name) for name in labels_free]


# -------------------------------
# Corner plot settings
# -------------------------------
corner_kwargs = dict(
    labels=labels_plot,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_fmt=".3f",
    label_kwargs={"fontsize": 13},
    title_kwargs={"fontsize": 11, "color": "k"},
    smooth=1.3,          # smoothing for 2D contours
    smooth1d=1.,        # smoothing for 1D histograms
    plot_datapoints=False,
    fill_contours=True,   
    plot_density=True,    
    levels=(0.16, 0.5, 0.84, 0.99),
    color="steelblue",
    hist_kwargs={"color": "steelblue", "alpha": 0.35, "linewidth": 0},
    contour_kwargs={"linewidths": 1.5},
    max_n_ticks=4,
)

# -------------------------------
# Generate corner plot
# -------------------------------

fig = corner.corner(samples_plot, **corner_kwargs)
fig.set_facecolor("white")

plt.subplots_adjust(
    left=0.1, right=0.95, bottom=0.1, top=0.95, wspace=0.15, hspace=0.15
)

axes = np.array(fig.axes).reshape(len(labels_plot), len(labels_plot))
for i in range(len(labels_plot)):
    ax = axes[i, i]
    # Try to fill existing KDE line if present (corner may draw a Line2D)
    filled = False
    for line in ax.get_lines():
        x = getattr(line, "get_xdata", lambda: None)()
        y = getattr(line, "get_ydata", lambda: None)()
        if x is None or y is None or len(x) < 2:
            continue
        if np.nanmax(y) > 0:
            ax.fill_between(x, 0, y, color="steelblue", alpha=0.5, linewidth=0)
            filled = True
            break
    # If no sensible line was found, compute KDE and fill manually
    if not filled:
        data = samples_plot[:, i]
        try:
            kde = gaussian_kde(data)
            x_vals = np.linspace(np.min(data), np.max(data), 400)
            y_vals = kde(x_vals)
            ax.cla()  # clear axis contents and draw filled KDE for consistency
            ax.fill_between(x_vals, 0, y_vals, color="steelblue", alpha=0.5, linewidth=0)
        except Exception:
            # fallback: draw a simple filled histogram
            ax.cla()
            ax.hist(samples_plot[:, i], bins=50, color="steelblue", alpha=0.5)

    # median vertical line
    median_val = np.median(samples_plot[:, i])
    ax.axvline(median_val, color="k", lw=1.2, ls="--")

plt.show()

fig.savefig('/home/pablo/Desktop/master/tfm/figures/corner_synch_dust_corr_bb.pdf')

#%%

# -------------------------------
# Guardado de resultados y tablas LaTeX (completo)
# -------------------------------
import os, json, csv
import numpy as np

def save_fit_results(filename_json, samples_flat, param_names,
                     param_unit='mK2', metadata=None, overwrite=True):
    """
    Guarda resultados del ajuste en JSON (y CSV) con mediana, 16/84 percentiles y metadatos.
    - filename_json: path destino (se creará .json y .csv con mismo prefijo)
    - samples_flat: array (n_samples, n_params) de la cadena a usar (sin burn-in, ya aplanada)
    - param_names: lista de nombres de parámetros en el mismo orden de samples_flat
    - param_unit: unidad de los parámetros en la cadena ('K2','mK2','uK2')
    - metadata: dict con info adicional (fit_components, ell_ref, freq_ref, comment...)
    """
    if metadata is None:
        metadata = {}
    os.makedirs(os.path.dirname(filename_json), exist_ok=True)
    base = os.path.splitext(filename_json)[0]
    jfn = base + '.json'
    cfn = base + '.csv'

    samples = np.asarray(samples_flat)
    n_params = samples.shape[1]
    assert len(param_names) == n_params, "param_names length mismatch"

    results = {'metadata': metadata.copy(), 'params': []}
    for i, name in enumerate(param_names):
        arr = samples[:, i]
        med = float(np.nanmedian(arr))
        p16 = float(np.nanpercentile(arr, 16))
        p84 = float(np.nanpercentile(arr, 84))
        mean = float(np.nanmean(arr))
        std = float(np.nanstd(arr, ddof=1))
        results['params'].append({
            'name': name,
            'median': med,
            'p16': p16,
            'p84': p84,
            'mean': mean,
            'std': std,
            'unit': param_unit
        })

    # JSON
    if overwrite or not os.path.exists(jfn):
        with open(jfn, 'w') as f:
            json.dump(results, f, indent=2)
    # CSV
    with open(cfn, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'median', 'p16', 'p84', 'mean', 'std', 'unit'])
        for p in results['params']:
            writer.writerow([p['name'], p['median'], p['p16'], p['p84'], p['mean'], p['std'], p['unit']])
    return jfn, cfn


def results_json_to_tex(json_file, tex_file, caption='Fit results', label='tab:fit',
                        display_unit=('uK2', 1.0), sigfigs=3):
    """
    Lee el JSON escrito por save_fit_results y produce una tabla LaTeX.
    - display_unit: tuple (unit_str, display_scale) where unit_str is 'uK2','mK2','K2'
      y display_scale e.g. 1.0 (mostrar en μK^2) o 1e-3 (mostrar 10^-3 μK^2).
    - sigfigs: cifras significativas.
    """
    to_muK2 = {'K2': 1e12, 'mK2': 1e6, 'uK2': 1.0}
    with open(json_file, 'r') as f:
        data = json.load(f)

    meta = data.get('metadata', {})
    params = data['params']

    desired_unit_str, display_unit_scale = display_unit

    def _fmt_latex(x):
        if x == 0:
            return "0"
        s = f"{x:.{sigfigs}g}"
        if "e" in s:
            mant, exp = s.split("e")
            exp = int(exp)
            return rf"${mant}\times10^{{{exp}}}$"
        else:
            return rf"${s}$"

    rows = []
    for p in params:
        name = p['name']
        unit_from = p.get('unit', 'mK2')
        med = p['median']
        p16 = p['p16']
        p84 = p['p84']

        factor = to_muK2.get(unit_from, 1.0) / to_muK2.get(desired_unit_str, 1.0)
        scale = factor / display_unit_scale

        med_d = med * scale
        down = med_d - (p16 * scale)
        up = (p84 * scale) - med_d

        val_str = _fmt_latex(med_d)
        err_minus = _fmt_latex(down)
        err_plus = _fmt_latex(up)
        row_entry = rf"{name} & {val_str}^{{+{err_plus}}}_{{-{err_minus}}} \\"
        rows.append((name, row_entry))

    with open(tex_file, 'w') as f:
        f.write(r"\begin{table}[htbp]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{" + caption + "}\n")
        f.write(r"\label{" + label + "}\n")
        f.write(r"\begin{tabular}{lc}" + "\n")
        f.write(r"\hline" + "\n")
        f.write(r"Parameter & Value (" + rf"{desired_unit_str}" + r") \\" + "\n")
        f.write(r"\hline" + "\n")
        for _, row in rows:
            f.write(row + "\n")
        f.write(r"\hline" + "\n")
        f.write(r"\end{tabular}" + "\n")
        if meta:
            f.write(r"\vspace{0.3cm}" + "\n")
            f.write(r"\begin{flushleft}" + "\n")
            for k, v in meta.items():
                f.write(r"\footnotesize " + f"{k}: {v} \\\\" + "\n")
            f.write(r"\end{flushleft}" + "\n")
        f.write(r"\end{table}" + "\n")
    return tex_file


# Metadatos
meta = {
    'mask': mask_name,
    'fit_components': tuple(fit_components),
    'modes': fit_data_ee['modes'],
    'pairs_used': fit_data_ee['pairs_used'],
    'ell_ref': 80.0,
    'freq_ref_sync': 11.1,
    'freq_ref_dust': 353.0,
    'ell_range': (int(np.min(ell)), int(np.max(ell))),
    'nwalkers': nwalkers,
    'nsteps': ninter,
    'discard_fraction': discard_fraction,
}
try:
    meta['acceptance_fraction'] = float(np.mean(sampler.acceptance_fraction))
except Exception:
    pass

results_dir = os.path.join(out_path, 'mcmc_results', mask_name)
os.makedirs(results_dir, exist_ok=True)

# Guardar: conjunto completo (incluye parámetros fijos)
json_full, csv_full = save_fit_results(
    os.path.join(results_dir, 'results_fit_full.json'),
    samples_full,
    param_names,
    param_unit='mK2',
    metadata=meta
)

# Guardar: solo parámetros libres
labels_free = [name for name, is_free in param_map if is_free]
json_free, csv_free = save_fit_results(
    os.path.join(results_dir, 'results_fit_free.json'),
    samples_free,
    labels_free,
    param_unit='mK2',
    metadata=meta
)

# Tablas LaTeX
tex_full = results_json_to_tex(
    json_full,
    os.path.join(results_dir, 'results_fit_full.tex'),
    caption='Resultados del ajuste (todos los parámetros).',
    label='tab:fit_full',
    display_unit=('uK2', 1e-3),
    sigfigs=3
)
tex_free = results_json_to_tex(
    json_free,
    os.path.join(results_dir, 'results_fit_free.tex'),
    caption='Resultados del ajuste (parámetros libres).',
    label='tab:fit_free',
    display_unit=('uK2', 1e-3),
    sigfigs=3
)

print('Saved files:')
print('  ', json_full)
print('  ', csv_full)
print('  ', tex_full)
print('  ', json_free)
print('  ', csv_free)
print('  ', tex_free)