#%%
import numpy as np
import os
from data import data, path_map, masks, path_masks
import functions 
import emcee
import corner
import matplotlib.pyplot as plt

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


def model_synchrotron(theta, datasets, ell, fit_c_terms=False):
    """
    General synchrotron model for any number of bands, covering both auto and cross spectra.
    Autos and cross use the same frequency formula, constant terms c_i are only added for autos.

    Parameters
    ----------
    theta : array-like
        Model parameters:
          If fit_c_terms = False:
              [A_ref, alpha, beta]
          If fit_c_terms = True:
              [A_ref, alpha, beta, c_1, ..., c_N]
          where N = number of unique bands in datasets.
          - A_ref : amplitude at reference freq (11.1 GHz) and ell=80
          - alpha : multipole spectral index
          - beta  : frequency spectral index
          - c_i   : constant noise term for band i (only for autos)
    datasets : list of dict
        List from prepare_mcmc_data['datasets'], each dict has:
          { 'pair': '11_23', 'mode':'EE', 'freqs': (f1,f2), 'slice': (start,stop) }
    ell : array
        Multipole values (ell_eff).
    fit_c_terms : bool, default False
        Whether to include constant c_i terms in auto-spectra.

    Returns
    -------
    model_all : ndarray
        1D stacked array of model values, aligned with datasets order.
    """
    # --- parameters
    A_ref = theta[0]
    alpha = theta[1]
    beta  = theta[2]

    # Unique frequencies in datasets (needed for mapping c_i terms)
    unique_freqs = sorted({f for d in datasets for f in d['freqs']})
    freq_to_idx = {f: i for i, f in enumerate(unique_freqs)}
    N = len(unique_freqs)

    # Constant terms (if included)
    c_terms = np.zeros(N)
    if fit_c_terms:
        if len(theta) != 3 + N:
            raise ValueError(f"Expected {3+N} parameters, got {len(theta)}")
        c_terms = theta[3:]

    # --- build model
    model_list = []
    for d in datasets:
        f1, f2 = d['freqs']

        # frequency scaling
        freq_factor = ((f1 * f2) / 11.1**2)**beta

        # constant term only for autos
        if f1 == f2 and fit_c_terms:
            i = freq_to_idx[f1]
            c_i = c_terms[i]
        else:
            c_i = 0.0

        cl = (A_ref *
              freq_factor *
              (ell/80.)**alpha +
              c_i)

        model_list.append(cl)

    # stack into 1D array in the same order as y_all
    model_all = np.concatenate(model_list)
    return model_all



def lnlike(theta, datasets, ell, y_all, yerr_all, fit_c_terms=False):
    """
    General log-likelihood for synchrotron model with arbitrary number of bands.

    Parameters
    ----------
    theta : array
        Model parameters (depends on fit_c_terms).
    datasets : list
        From prepare_mcmc_data['datasets'].
    ell : array
        Multipole values.
    y_all : array
        Observed data (stacked).
    yerr_all : array
        Errors of observed data (stacked).
    fit_c_terms : bool
        Whether constants c_i are included.

    Returns
    -------
    lnlike : float
    """
    model_vals = model_synchrotron(theta, datasets, ell, fit_c_terms=fit_c_terms)
    chi2 = np.sum(((y_all - model_vals) / yerr_all)**2)
    return -0.5 * chi2

def lnprior(theta, datasets, fit_c_terms=False):
    """
    General prior for synchrotron parameters.

    Parameters
    ----------
    theta : array
        Model parameters.
    datasets : list
        From prepare_mcmc_data['datasets'] (used only if fit_c_terms=True).
    fit_c_terms : bool
        Whether constants c_i are included.

    Returns
    -------
    lnprior : float
    """
    A_ref, alpha, beta = theta[:3]

    # Priors on spectral indices
    if not (-6 < alpha < -0.5 and -5 < beta < -1):
        return -np.inf

    # Optional priors on c_i > 0
    if fit_c_terms:
        c_terms = theta[3:]
        if np.any(c_terms < 0):
            return -np.inf

    return 0.0


def lnprob(theta, datasets, ell, y_all, yerr_all, fit_c_terms=False):
    lp = lnprior(theta, datasets, fit_c_terms=fit_c_terms)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, datasets, ell, y_all, yerr_all, fit_c_terms=fit_c_terms)


# ======================================================================0


def run_mcmc_fit(p0, nwalkers, niter, ndim, lnprob, lnlike, data, burnin_frac=0.5, thin=10, labels=None, corner_title=""):
    """
    Run MCMC sampler and return flattened samples along with a corner plot.
    
    Parameters
    ----------
    p0 : list of np.array
        Initial positions of walkers.
    nwalkers : int
        Number of walkers.
    niter : int
        Number of iterations for production run.
    ndim : int
        Number of dimensions in the parameter space.
    lnprob : function
        Log-probability function.
    lnlike : function
        Log-likelihood function.
    data : tuple
        Arguments to pass to lnprob.
    burnin_frac : float
        Fraction of iterations to discard as burn-in.
    thin : int
        Thinning factor for chain.
    labels : list of str
        Labels for corner plot.
    corner_title : str
        Title for the corner plot.

    Returns
    -------
    samples : np.ndarray
        Flattened MCMC chain.
    sampler : emcee.EnsembleSampler
        The sampler object.
    """
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)

    # Burn-in
    print("Running burn-in...")
    p0, _, _ = sampler.run_mcmc(p0, 100, progress=True, store=True)
    sampler.reset()

    # Production run
    print("Running production...")
    pos, prob, state = sampler.run_mcmc(p0, niter, progress=True, store=True)

    # Flatten chain and discard burn-in
    burnin = int(burnin_frac * niter)
    samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)

    # Corner plot
    if labels is None:
        labels = [f"param_{i}" for i in range(ndim)]
    fig = corner.corner(samples, labels=labels, show_titles=True, plot_datapoints=True, quantiles=[0.16,0.5,0.84])
    fig.suptitle(corner_title, fontsize=14)
    plt.show()

    # Best-fit parameters and reduced chi^2
    best_index = np.argmax(prob)
    best_params = pos[best_index]
    chi_squared = -2 * lnlike(best_params, *data)
    reduced_chi2 = chi_squared / (len(data[2]) - ndim)  # data[2] assumed to be ell array

    return samples, sampler, best_params, reduced_chi2


fit_data_ee = prepare_mcmc_data(spectra_dict, band_list=band_list, modes=['EE'], band_pairs=['11_11', '11_23', '23_23', '11_30', '30_30', '23_30'])

#%%

nwalkers = 100
niter = 5000
ndim = 5  # or len(theta) for number of free parameters
p0 = [np.array([1.5, -3., -3., 0., 0.]) + 1e-1*np.random.randn(ndim) for _ in range(nwalkers)]
labels = ['A_ref', 'alpha', 'beta', 'c1', 'c2']

samples, sampler, best_params, reduced_chi2 = run_mcmc_fit(
    p0, nwalkers, niter, ndim,
    lnprob, lnlike,
    data=(fit_data_ee['freqs'][0], None, fit_data_ee['ell_eff'], fit_data_ee['cl_data'][0], fit_data_ee['cl_data'][1], fit_data_ee['cl_data'][3],
          fit_data_ee['cl_err'][0], fit_data_ee['cl_err'][1], fit_data_ee['cl_err'][3]),
    labels=labels,
    corner_title="EE 11-23 GHz"
)




def plot_corner(sampler, ndim, labels=None, burnin=0, thin=1, truths=None):
    """
    Make a corner plot from an emcee sampler.

    Parameters
    ----------
    sampler : emcee.EnsembleSampler
        The MCMC sampler after running.
    ndim : int
        Number of parameters.
    labels : list of str, optional
        Names of parameters to show on axes.
    burnin : int, default 0
        Number of initial steps to discard.
    thin : int, default 1
        Thinning factor for the chain.
    truths : list, optional
        Reference "true" parameter values for vertical/horizontal lines.
    """
    # Flatten the chains (discard burn-in, apply thinning)
    samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)

    # If no labels provided, use generic names
    if labels is None:
        labels = [f"$\\theta_{i}$" for i in range(ndim)]

    fig = corner.corner(
        samples,
        labels=labels,
        truths=truths,
        show_titles=True,
        title_fmt=".3f",
        title_kwargs={"fontsize": 12}
    )

    plt.show()
    return fig

# Supón que tienes ndim=3 (A_ref, alpha, beta)
labels = [r"$A_\mathrm{ref}$", r"$\alpha$", r"$\beta$"]

fig = plot_corner(sampler, ndim=3, labels=labels, burnin=200, thin=10)


#%%
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner

plt.rcParams['figure.figsize'] = [15, 8]

masks = ['north', 'south']
freq_auto = ['11', '23', '30']
freq_cross = ['11-23', '11-30', '23-30']

cl_auto = np.zeros([len(masks), len(freq_auto), 7, 102])
cl_cross = np.zeros([len(masks), len(freq_cross), 7, 102])
error_auto = np.zeros([len(masks), len(freq_auto), 7, 102])
error_cross = np.zeros([len(masks), len(freq_cross), 7, 102])

for ii in range(len(masks)):
    for jj in range(len(freq_auto)):
        cl_auto[ii,jj] = np.loadtxt('/home/pablo/Desktop/Paper/3_north_south/data/spectra/cl_' + freq_auto[jj] + 'ghz_' + masks[ii] + '.txt', skiprows=1)
        cl_cross[ii,jj] = np.loadtxt('/home/pablo/Desktop/Paper/3_north_south/data/spectra/cross_' + freq_cross[jj] + 'ghz_' + masks[ii] + '.txt', skiprows=1)
        error_auto[ii,jj] = np.loadtxt('/home/pablo/Desktop/Paper/3_north_south/data/errorbars/errorbar_' + freq_auto[jj] + 'ghz_' + masks[ii] + '.txt', skiprows=1)
        error_cross[ii,jj] = np.loadtxt('/home/pablo/Desktop/Paper/3_north_south/data/errorbars/errorbar_cross_' + freq_cross[jj] + 'ghz_' + masks[ii] + '.txt', skiprows=1)

ccorr_ee = np.ones([len(masks), 4, 3])
ccorr_bb = np.ones([len(masks), 4, 3])
ccorr_ee_bb = np.ones([len(masks), 4, 3])

for ii in range(len(masks)):
    ccorr_ee[ii] = np.loadtxt('/home/pablo/Desktop/Paper/3_north_south/data/colour_corrections/north_south_colour_corrections_ee_' + masks[ii] + '.txt', skiprows=1)
    ccorr_bb[ii] = np.loadtxt('/home/pablo/Desktop/Paper/3_north_south/data/colour_corrections/north_south_colour_corrections_bb_' + masks[ii] + '.txt', skiprows=1)
    ccorr_ee_bb[ii] = np.loadtxt('/home/pablo/Desktop/Paper/3_north_south/data/colour_corrections/north_south_colour_corrections_ee_bb_' + masks[ii] + '.txt', skiprows=1)
    
ell = cl_auto[0, 0, 0, 3:30] # 30 < ell < 300

cl_ee_11_north = cl_auto[0, 0, 2, 3:30]
cl_ee_23_north = cl_auto[0, 1, 2, 3:30]
cl_ee_30_north = cl_auto[0, 2, 2, 3:30]

cl_ee_11_south = cl_auto[1, 0, 2, 3:30]
cl_ee_23_south = cl_auto[1, 1, 2, 3:30]
cl_ee_30_south = cl_auto[1, 2, 2, 3:30]

cl_ee_11 = np.array([cl_ee_11_north, cl_ee_11_south])
cl_ee_23 = np.array([cl_ee_23_north, cl_ee_23_south])
cl_ee_30 = np.array([cl_ee_30_north, cl_ee_30_south])

error_ee_11_north = error_auto[0, 0, 2, 3:30]
error_ee_23_north = error_auto[0, 1, 2, 3:30]
error_ee_30_north = error_auto[0, 2, 2, 3:30]

error_ee_11_south = error_auto[1, 0, 2, 3:30]
error_ee_23_south = error_auto[1, 1, 2, 3:30]
error_ee_30_south = error_auto[1, 2, 2, 3:30]


error_ee_11 = np.array([error_ee_11_north, error_ee_11_south])
error_ee_23 = np.array([error_ee_23_north, error_ee_23_south])
error_ee_30 = np.array([error_ee_30_north, error_ee_30_south])

cl_bb_11_north = cl_auto[0, 0, 3, 3:30]
cl_bb_23_north = cl_auto[0, 1, 3, 3:30]
cl_bb_30_north = cl_auto[0, 2, 3, 3:30]

cl_bb_11_south = cl_auto[1, 0, 3, 3:30]
cl_bb_23_south = cl_auto[1, 1, 3, 3:30]
cl_bb_30_south = cl_auto[1, 2, 3, 3:30]


cl_bb_11 = np.array([cl_bb_11_north, cl_bb_11_south])
cl_bb_23 = np.array([cl_bb_23_north, cl_bb_23_south])
cl_bb_30 = np.array([cl_bb_30_north, cl_bb_30_south])

error_bb_11_north = error_auto[0, 0, 3, 3:30]
error_bb_23_north = error_auto[0, 1, 3, 3:30]
error_bb_30_north = error_auto[0, 2, 3, 3:30]

error_bb_11_south = error_auto[1, 0, 3, 3:30]
error_bb_23_south = error_auto[1, 1, 3, 3:30]
error_bb_30_south = error_auto[1, 2, 3, 3:30]


error_bb_11 = np.array([error_bb_11_north, error_bb_11_south])
error_bb_23 = np.array([error_bb_23_north, error_bb_23_south])
error_bb_30 = np.array([error_bb_30_north, error_bb_30_south])

# ===============================================================

cl_ee_11_23_north = cl_cross[0, 0, 2, 3:30]
cl_ee_11_30_north = cl_cross[0, 1, 2, 3:30]
cl_ee_23_30_north = cl_cross[0, 2, 2, 3:30]

cl_ee_11_23_south = cl_cross[1, 0, 2, 3:30]
cl_ee_11_30_south = cl_cross[1, 1, 2, 3:30]
cl_ee_23_30_south = cl_cross[1, 2, 2, 3:30]

cl_ee_11_23 = np.array([cl_ee_11_23_north, cl_ee_11_23_south])
cl_ee_11_30 = np.array([cl_ee_11_30_north, cl_ee_23_30_south])
cl_ee_23_30 = np.array([cl_ee_23_30_north, cl_ee_23_30_south])

error_ee_11_23_north = error_cross[0, 0, 2, 3:30]
error_ee_11_30_north = error_cross[0, 1, 2, 3:30]
error_ee_23_30_north = error_cross[0, 2, 2, 3:30]

error_ee_11_23_south = error_cross[1, 0, 2, 3:30]
error_ee_11_30_south = error_cross[1, 1, 2, 3:30]
error_ee_23_30_south = error_cross[1, 2, 2, 3:30]

error_ee_11_23 = np.array([error_ee_11_23_north, error_ee_11_23_south])
error_ee_11_30 = np.array([error_ee_11_30_north, error_ee_11_30_south])
error_ee_23_30 = np.array([error_ee_23_30_north, error_ee_23_30_south])


cl_bb_11_23_north = cl_cross[0, 0, 3, 3:30]
cl_bb_11_30_north = cl_cross[0, 1, 3, 3:30]
cl_bb_23_30_north = cl_cross[0, 2, 3, 3:30]

cl_bb_11_23_south = cl_cross[1, 0, 3, 3:30]
cl_bb_11_30_south = cl_cross[1, 1, 3, 3:30]
cl_bb_23_30_south = cl_cross[1, 2, 3, 3:30]

cl_bb_11_23 = np.array([cl_bb_11_23_north, cl_bb_11_23_south])
cl_bb_11_30 = np.array([cl_bb_11_30_north, cl_bb_23_30_south])
cl_bb_23_30 = np.array([cl_bb_23_30_north, cl_bb_23_30_south])

error_bb_11_23_north = error_cross[0, 0, 3, 3:30]
error_bb_11_30_north = error_cross[0, 1, 3, 3:30]
error_bb_23_30_north = error_cross[0, 2, 3, 3:30]

error_bb_11_23_south = error_cross[1, 0, 3, 3:30]
error_bb_11_30_south = error_cross[1, 1, 3, 3:30]
error_bb_23_30_south = error_cross[1, 2, 3, 3:30]

error_bb_11_23 = np.array([error_bb_11_23_north, error_bb_11_23_south])
error_bb_11_30 = np.array([error_bb_11_30_north, error_bb_11_30_south])
error_bb_23_30 = np.array([error_bb_23_30_north, error_bb_23_30_south])

"""
# ==================================================================================
# Models
# ==================================================================================
"""

def model_f1_f2(theta, freq, cc, ell=ell):
    A_f1, alpha, betha, c_f1, c_f2 = theta
    f1, f2 = freq
    cc_f1, cc_f2 = cc[0], cc[1]
    cl_f1 = A_f1 * 1e-6 * (f1/11.1)**(2*betha) * (ell/80.)**alpha + c_f1 * 1e-9
    cl_f2 = A_f1 * 1e-6 * (f2/11.1)**(2*betha) * (ell/80.)**alpha + c_f2 * 1e-9
    return np.array([cl_f1/cc_f1**2, cl_f2/cc_f2**2])

def model_f1_f2_f3(theta, freq, cc, ell=ell):
    A_f1, alpha, betha, c_f1, c_f2, c_f3 = theta
    f1, f2, f3 = freq
    cc_f1, cc_f2, cc_f3 = cc[0], cc[1], cc[2]
    cl_f1 = A_f1 * 1e-6 * (f1/11.1)**(2*betha) * (ell/80.)**alpha + c_f1 * 1e-9
    cl_f2 = A_f1 * 1e-6 * (f2/11.1)**(2*betha) * (ell/80.)**alpha + c_f2 * 1e-9
    cl_f3 = A_f1 * 1e-6 * (f3/11.1)**(2*betha) * (ell/80.)**alpha + c_f3 * 1e-9
    return np.array([cl_f1/cc_f1**2, cl_f2/cc_f2**2, cl_f3/cc_f3**2])

def cross_model_f1_f2(theta, freq, cc, ell=ell):
    A_f1, alpha, betha, c_f1, c_f2 = theta
    f1, f2 = freq
    cl = A_f1 * 1e-6 * (f1*f2/11.1**2)**(betha) * (ell/80.)**alpha 
    return cl / (cc[0] * cc[1])

def cross_model_f1_f2_f3(theta, freq, cc, ell=ell):
    A_f1, alpha, betha, c_f1, c_f2, c_f3 = theta
    f1, f2 = freq
    cl = A_f1 * 1e-6 * (f1*f2/11.1**2)**(betha) * (ell/80.)**alpha
    return cl / (cc[0] * cc[1])

# =============================================================================
# Likeness function, priors setting, probability function and MCMC main 
# =============================================================================

def lnlike_f1_f2(theta, freq, cc, x, y_f1, y_f2, y_f1_f2, yerr_f1, yerr_f2, yerr_f1_f2):
    return -0.5 * (np.sum(((y_f1 - model_f1_f2(theta, freq, cc)[0]) / yerr_f1)**2) + 
                   np.sum(((y_f2 - model_f1_f2(theta, freq, cc)[1]) / yerr_f2)**2) + 
                   np.sum(((y_f1_f2 - cross_model_f1_f2(theta, freq, cc)) / yerr_f1_f2)**2))

def lnlike_f1_f2_f3(theta, freq, cc, x, y_f1, y_f2, y_f3, y_f1_f2, y_f1_f3, y_f2_f3, yerr_f1, yerr_f2, yerr_f3, yerr_f1_f2, yerr_f1_f3, yerr_f2_f3):
    return -0.5 * (np.sum(((y_f1 - model_f1_f2_f3(theta, freq, cc)[0]) / yerr_f1)**2) + 
                   np.sum(((y_f2 - model_f1_f2_f3(theta, freq, cc)[1]) / yerr_f2)**2) + 
                   np.sum(((y_f3 - model_f1_f2_f3(theta, freq, cc)[2]) / yerr_f3)**2) + 
                   np.sum(((y_f1_f2 - cross_model_f1_f2_f3(theta, [freq[0],freq[1]], [cc[0],cc[1]])) / yerr_f1_f2)**2) + 
                   np.sum(((y_f1_f3 - cross_model_f1_f2_f3(theta, [freq[0],freq[2]], [cc[0],cc[2]])) / yerr_f1_f3)**2) + 
                   np.sum(((y_f2_f3 - cross_model_f1_f2_f3(theta, [freq[1],freq[2]], [cc[1],cc[2]])) / yerr_f2_f3)**2) )

def lnprior_f1_f2(theta):
    A_f1, alpha, betha, c_f1, c_f2 = theta
    if -6 < alpha < -0.5 and -5 < betha < -1: # and c_f1 > 0 and c_f2 > 0:
        return 0.0
    return -np.inf

def lnprior_f1_f2_f3(theta):
    A_f1, alpha, betha, c_f1, c_f2, c_f3 = theta
    if -6 < alpha < -0.5 and -5 < betha < -1: # and c_f1 > 0 and c_f2 > 0 and c_f3 > 0:
        return 0.0
    return -np.inf

def lnprob_f1_f2(theta, freq, cc, x, y_f1, y_f2, y_f1_f2, yerr_f1, yerr_f2, yerr_f1_f2):
    lp = lnprior_f1_f2(theta)
    if not np.isfinite(lp): # If the parameter is not within the priors, return a -infinite
        return -np.inf
    return lp + lnlike_f1_f2(theta, freq, cc, x, y_f1, y_f2,  y_f1_f2, yerr_f1, yerr_f2, yerr_f1_f2) # if theta fulfills the priors, then lp = 0

def lnprob_f1_f2_f3(theta, freq, cc, x, y_f1, y_f2, y_f3, y_f1_f2, y_f1_f3, y_f2_f3, yerr_f1, yerr_f2, yerr_f3, yerr_f1_f2, yerr_f1_f3, yerr_f2_f3):
    lp = lnprior_f1_f2_f3(theta)
    if not np.isfinite(lp): # If the parameter is not within the priors, return a -infinite
        return -np.inf
    return lp + lnlike_f1_f2_f3(theta, freq, cc, x, y_f1, y_f2, y_f3, y_f1_f2, y_f1_f3, y_f2_f3, yerr_f1, yerr_f2, yerr_f3, yerr_f1_f2, yerr_f1_f3, yerr_f2_f3)

def main(p0, nwalkers, niter, ndim, lnprob, lnlike, data):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)

    print("Running burn-in...")
    p0, _, _ = sampler.run_mcmc(p0, 100, progress=True) # Run 100 iterations for each initial position
    sampler.reset()

    print("Running production...")
    pos, prob, state = sampler.run_mcmc(p0, niter, progress=True) # Run niter iterations with the new initial positions

    best_index = np.argmax(prob)
    best_params = pos[best_index]
    chi_squared = -2 * lnlike(best_params, *data)
    reduced_chi = chi_squared / (ell.size - ndim)

    return sampler, pos, prob, state, reduced_chi

# =============================================================================
# Walkers, iteartions and initial parameters
# =============================================================================

nwalkers = 100
niter = 10000
ndim_f1_f2 = 5
ndim_f1_f2_f3 = 6
discard_frac = 0.5
initial_f1_f2 = [1.5, -3., -3., 0., 0.]
initial_f1_f2_f3 = [1.5, -3., -3., 0., 0., 0.]

variations_f1_f2 = np.array([1e-1, 1e-1, 1e-1, 1e-1, 1e-1])
variations_f1_f2_f3 = np.array([1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1])

p0_f1_f2 = [np.array(initial_f1_f2) + variations_f1_f2 * np.random.randn(ndim_f1_f2) for i in range(nwalkers)]
p0_f1_f2_f3 = [np.array(initial_f1_f2_f3) + variations_f1_f2_f3 * np.random.randn(ndim_f1_f2_f3) for i in range(nwalkers)]

labels = [['A_11', 'alpha', 'betha', 'c_11', 'c_23'], ['A_11', 'alpha', 'betha', 'c_11', 'c_30'], ['A_23', 'alpha', 'betha', 'c_23', 'c_30'], ['A_11', 'alpha', 'betha', 'c_11', 'c_23', 'c_30']]

#%%
# ========================
# EE-mode
# ========================

names = [['A' , 'alpha', 'betha', 'c_1', 'c_2'], ['A' , 'alpha', 'betha', 'c_1', 'c_2', 'c_3']]
tables = ['11-23', '11-30', '23-30', '11-23-30']


all_samples_ee_11_23 = np.zeros([len(masks), int(discard_frac*niter*nwalkers/10), ndim_f1_f2]) # Mask, samples
all_samples_ee_11_30 = np.zeros_like(all_samples_ee_11_23)
all_samples_ee_23_30 = np.zeros_like(all_samples_ee_11_23)
all_samples_ee_11_23_30 = np.zeros([len(masks), int(discard_frac*niter*nwalkers/10), ndim_f1_f2_f3])

chi_squared_ee = np.zeros([len(masks), 4])

for ii in range(len(masks)):

    data_11_23 = ([11.1, 22.82], ccorr_ee[ii, 0, :2], ell, cl_ee_11[ii], cl_ee_23[ii], cl_ee_11_23[ii], error_ee_11[ii], error_ee_23[ii], error_ee_11_23[ii])
    data_11_30 = ([11.1, 28.40], ccorr_ee[ii, 1, :2], ell, cl_ee_11[ii], cl_ee_30[ii], cl_ee_11_30[ii], error_ee_11[ii], error_ee_30[ii], error_ee_11_30[ii])
    data_23_30 = ([22.82, 28.4], ccorr_ee[ii, 2, :2], ell, cl_ee_23[ii], cl_ee_30[ii], cl_ee_23_30[ii], error_ee_23[ii], error_ee_30[ii], error_ee_23_30[ii])
    data_11_23_30 = ([11.1, 22.82, 28.40], ccorr_ee[ii, 3, :], ell, cl_ee_11[ii], cl_ee_23[ii], cl_ee_30[ii], cl_ee_11_23[ii], cl_ee_11_30[ii], cl_ee_23_30[ii], error_ee_11[ii], error_ee_23[ii], error_ee_30[ii], error_ee_11_23[ii], error_ee_11_30[ii], error_ee_23_30[ii])

    sampler_11_23, _, _, _, chi2_11_23   = main(p0_f1_f2, nwalkers, niter, ndim_f1_f2, lnprob_f1_f2, lnlike_f1_f2, data_11_23)
    sampler_11_30, _, _, _, chi2_11_30 = main(p0_f1_f2, nwalkers, niter, ndim_f1_f2, lnprob_f1_f2, lnlike_f1_f2, data_11_30)
    sampler_23_30, _, _, _, chi2_23_30 = main(p0_f1_f2, nwalkers, niter, ndim_f1_f2, lnprob_f1_f2, lnlike_f1_f2, data_23_30)
    sampler_11_23_30, _, _, _, chi2_11_23_30 = main(p0_f1_f2_f3, nwalkers, niter, ndim_f1_f2_f3, lnprob_f1_f2_f3, lnlike_f1_f2_f3, data_11_23_30)
    
    samples_11_23 = sampler_11_23.get_chain(flat=True, thin=10, discard = int(discard_frac * niter))
    samples_11_30 = sampler_11_30.get_chain(flat=True, thin=10, discard = int(discard_frac * niter))
    samples_23_30 = sampler_23_30.get_chain(flat=True, thin=10, discard = int(discard_frac * niter))
    samples_11_23_30 = sampler_11_23_30.get_chain(flat=True, thin=10, discard = int(discard_frac * niter))


    fig_11_23 = corner.corner(samples_11_23, show_titles=True, labels=labels[0], plot_datapoints=True, quantiles=[0.16, 0.5, 0.84])
    fig_11_23.suptitle('11-23 GHz, ' + masks[ii] , fontsize=13)
    plt.show()

    fig_11_30 = corner.corner(samples_11_30, show_titles=True, labels=labels[1], plot_datapoints=True, quantiles=[0.16, 0.5, 0.84])
    fig_11_30.suptitle('11-30 GHz, ' + masks[ii] , fontsize=13)
    plt.show()

    fig_23_30 = corner.corner(samples_23_30, show_titles=True, labels=labels[2], plot_datapoints=True, quantiles=[0.16, 0.5, 0.84])
    fig_23_30.suptitle('23-30 GHz, ' + masks[ii] , fontsize=13)
    plt.show()

    fig_11_23_30 = corner.corner(samples_11_23_30, show_titles=True, labels=labels[3], plot_datapoints=True, quantiles=[0.16, 0.5, 0.84])
    fig_11_23_30.suptitle('11-23-30 GHz, ' + masks[ii] , fontsize=13)
    plt.show()

    all_samples_ee_11_23[ii] = samples_11_23
    all_samples_ee_11_30[ii] = samples_11_30
    all_samples_ee_23_30[ii] = samples_23_30
    all_samples_ee_11_23_30[ii] = samples_11_23_30

    chi_squared_ee[ii] = np.array([chi2_11_23, chi2_11_30, chi2_23_30, chi2_11_23_30])

all_samples_ee_f1_f2 = np.array([all_samples_ee_11_23, all_samples_ee_11_30, all_samples_ee_23_30])
all_samples_ee_f1_f2_f3 = np.array([all_samples_ee_11_23_30])

# ============================
# Saving data
# ============================

values_ee_f1_f2 = np.zeros([ndim_f1_f2, len(masks)])
values_ee_f1_f2_f3 = np.zeros([ndim_f1_f2_f3, len(masks)])
errors_ee_f1_f2 = np.zeros([ndim_f1_f2, len(masks)])
errors_ee_f1_f2_f3 = np.zeros([ndim_f1_f2_f3, len(masks)])

for ii in range(len(all_samples_ee_f1_f2)):
    for jj in range(len(masks)):
        A_ee, alpha_ee, betha_ee, c_f1, c_f2 = np.percentile(all_samples_ee_f1_f2[ii,jj], [16, 50, 84], axis=0).T

        A_value, A_error = A_ee[1], np.max([np.abs(A_ee[1] - A_ee[0]), np.abs(A_ee[1] - A_ee[2])])
        alpha_value, alpha_error = alpha_ee[1], np.max([np.abs(alpha_ee[1] - alpha_ee[0]), np.abs(alpha_ee[1] - alpha_ee[2])])
        betha_value, betha_error = betha_ee[1], np.max([np.abs(betha_ee[1] - betha_ee[0]), np.abs(betha_ee[1] - betha_ee[2])])
        c_f1_value, c_f1_error = c_f1[1], np.max([np.abs(c_f1[1] - c_f1[0]), np.abs(c_f1[1] - c_f1[2])])
        c_f2_value, c_f2_error = c_f2[1], np.max([np.abs(c_f2[1] - c_f2[0]), np.abs(c_f2[1] - c_f2[2])])

        values_ee_f1_f2[:, jj] = A_value, alpha_value, betha_value, c_f1_value, c_f2_value
        errors_ee_f1_f2[:, jj] = A_error, alpha_error, betha_error, c_f1_error, c_f2_error

    with open('/home/pablo/Desktop/Paper/3_north_south/data/tables/values/values_ee_north_south_' + tables[ii] + '.txt', 'w') as f:
        f.write('\t'.join(names[0]) + '\n')
        np.savetxt(f, values_ee_f1_f2, fmt='%.12e', delimiter='\t')

    with open('/home/pablo/Desktop/Paper/3_north_south/data/tables/errors/errors_ee_north_south_' + tables[ii] + '.txt', 'w') as f:
        f.write('\t'.join(names[1]) + '\n')
        np.savetxt(f, errors_ee_f1_f2, fmt='%.12e', delimiter='\t')


for ii in range(len(all_samples_ee_f1_f2_f3)):
    for jj in range(len(masks)):
        A_ee, alpha_ee, betha_ee, c_f1, c_f2, c_f3 = np.percentile(all_samples_ee_f1_f2_f3[ii,jj], [16, 50, 84], axis=0).T

        A_value_ee, A_error_ee = A_ee[1], np.max([np.abs(A_ee[1] - A_ee[0]), np.abs(A_ee[1] - A_ee[2])])
        alpha_value, alpha_error = alpha_ee[1], np.max([np.abs(alpha_ee[1] - alpha_ee[0]), np.abs(alpha_ee[1] - alpha_ee[2])])
        betha_value, betha_error = betha_ee[1], np.max([np.abs(betha_ee[1] - betha_ee[0]), np.abs(betha_ee[1] - betha_ee[2])])
        c_f1_value, c_f1_error = c_f1[1], np.max([np.abs(c_f1[1] - c_f1[0]), np.abs(c_f1[1] - c_f1[2])])
        c_f2_value, c_f2_error = c_f2[1], np.max([np.abs(c_f2[1] - c_f2[0]), np.abs(c_f2[1] - c_f2[2])])
        c_f3_value, c_f3_error = c_f3[1], np.max([np.abs(c_f3[1] - c_f3[0]), np.abs(c_f3[1] - c_f3[2])])

        values_ee_f1_f2_f3[:, jj] = A_value_ee, alpha_value, betha_value, c_f1_value, c_f2_value, c_f3_value
        errors_ee_f1_f2_f3[:, jj] = A_error_ee, alpha_error, betha_error, c_f1_error, c_f2_error, c_f3_error

    with open('/home/pablo/Desktop/Paper/3_north_south/data/tables/values/values_ee_north_south_' + tables[ii+3] + '.txt', 'w') as f:
        f.write('\t'.join(names[0]) + '\n')
        np.savetxt(f, values_ee_f1_f2_f3, fmt='%.12e', delimiter='\t')

    with open('/home/pablo/Desktop/Paper/3_north_south/data/tables/errors/errors_ee_north_south_' + tables[ii+3] + '.txt', 'w') as f:
        f.write('\t'.join(names[1]) + '\n')
        np.savetxt(f, errors_ee_f1_f2_f3, fmt='%.12e', delimiter='\t')
