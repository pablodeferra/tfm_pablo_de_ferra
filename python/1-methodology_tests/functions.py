#%%
import numpy as np
try:
    import pysm3
    import pysm3.units as u_pysm
except Exception:
    pysm3 = None
    u_pysm = None
import healpy as hp
from astropy import units as u
import os
from data import data, color_corrections
from tqdm import tqdm 
import pymaster as nmt
from astropy.io import fits
import re
from scipy.constants import c,h,k
import matplotlib.pyplot as plt
import sys
# u_pysm may already be set above depending on pysm3 import
import emcee
import corner
import multiprocessing as mp
from scipy.stats import gaussian_kde
from scipy import interpolate
from math import log, pi
from math import lgamma as _lgamma
from scipy.special import gammaln

# ------------------------------------------------------------------
# Optional Gaussian priors (global, lightweight mechanism)
# Usage:
#   set_gaussian_priors({'beta_s': (-3.1, 0.2)})
# The lnprior() and bin-to-bin priors will add a Gaussian term for any
# parameter present in this dict, on top of the existing top-hat bounds.
# ------------------------------------------------------------------
_GAUSSIAN_PRIORS = {}

def set_gaussian_priors(priors_dict):
    """
    Set optional Gaussian priors for parameters.

    Parameters
    ----------
    priors_dict : dict
        Mapping from parameter name to (mu, sigma). For example:
        {'beta_s': (-3.1, 0.2), 'beta_d': (1.6, 0.2)}
    """
    global _GAUSSIAN_PRIORS
    # Defensive copy to avoid accidental external mutation
    _GAUSSIAN_PRIORS = dict(priors_dict or {})

'''
# ====================================
# 1
# ====================================
# '''

def _load_instrument_beam_TEB(exp, band_name, data_dict, lmax):
    """
    Load and interpolate the beam window function for a given experiment/band.

    Parameters
    ----------
    exp : str
        Experiment name (e.g., 'WMAP', 'PLANCK', 'QUIJOTE').
    band_name : str
        Band identifier (e.g., '30', '143', '11').
    data_dict : dict
        Metadata dictionary containing beam file paths and frequencies.
    lmax : int
        Maximum multipole to which the beam is interpolated.

    Returns
    -------
    tuple
        (ells, Bl_T, Bl_E, Bl_B): arrays of beam transfer functions
        for temperature and polarization components, interpolated up to `lmax`.

    Notes
    -----
    - Handles QUIJOTE, WMAP, and PLANCK beam formats automatically.
    - If experiment is unknown, assumes ASCII file with [ell, Bl] or [Bl].
    - Columns containing strings (e.g., LFI detector names) are safely ignored.
    """
    if exp not in data_dict or band_name not in data_dict[exp]:
        raise KeyError(f"No entry in data for {exp} {band_name}")
    if 'beam' not in data_dict[exp][band_name]:
        raise KeyError(f"Missing 'beam' path in data[{exp}][{band_name}]")

    beam_path = data_dict[exp][band_name]['beam']
    if not os.path.exists(beam_path):
        raise FileNotFoundError(f"Beam file not found: {beam_path}")

    Lout = np.arange(lmax + 1, dtype=float)

    def _interp_from_xy(x_in, y_in):
        """Safely interpolate to full 0..lmax range, defaulting to 1 outside."""
        x_in = np.asarray(x_in, dtype=float).ravel()
        y_in = np.asarray(y_in, dtype=float).ravel()
        return np.interp(Lout, x_in, y_in, left=1.0, right=1.0)

    # QUIJOTE beams
    if exp.upper() == 'QUIJOTE':
        with fits.open(beam_path) as hdul:
            hdu = hdul[1]
            col_map = {"11": "Bl_311", "13": "Bl_313", "17": "Bl_417", "19": "Bl_419"}
            colname = col_map.get(str(band_name))
            if colname is None:
                for name in hdu.columns.names:
                    if str(name).lower().startswith('bl_'):
                        colname = name
                        break
            if colname is None:
                raise ValueError(f"No Bl_* column found for QUIJOTE {band_name} in {beam_path}")
            beam_arr = np.asarray(hdu.data[colname][0], dtype=float)
            x_in = np.arange(len(beam_arr), dtype=float)
            Bl = _interp_from_xy(x_in, beam_arr)
            return (Lout, Bl, Bl, Bl)

    # WMAP beams
    if exp.upper() == 'WMAP':
        arr = np.loadtxt(beam_path)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[1] == 1:
            y = arr[:, 0]
            x = np.arange(len(y), dtype=float)
        else:
            x = np.arange(len(arr[:, 1]), dtype=float)
            y = arr[:, 1]
        Bl = _interp_from_xy(x, y)
        return (Lout, Bl, Bl, Bl)

    # PLANCK beams
    if exp.upper() == 'PLANCK':
        band_int = int(band_name)
        with fits.open(beam_path) as hdul:
            if band_int <= 70:  # LFI
                beam_hdu = None
                extname = f'BEAMWF_0{band_int}X0{band_int}'
                try:
                    beam_hdu = hdul[extname]
                except Exception:
                    for h in hdul:
                        if not hasattr(h, "columns") or h.columns is None:
                            continue
                        for nm in h.columns.names:
                            col = h.data[nm]
                            if np.issubdtype(col.dtype, np.number) and nm.upper() == 'BL':
                                beam_hdu = h
                                break
                        if beam_hdu is not None:
                            break
                    if beam_hdu is None and len(hdul) > 1:
                        beam_hdu = hdul[1]

                names = list(beam_hdu.columns.names)
                NAMES = [n.upper() for n in names]

                ell_col = next((names[NAMES.index(c)] for c in ('ELL', 'L', 'EL') if c in NAMES), None)

                val_col = None
                for cand in ('BL', 'TT', 'T'):
                    if cand in NAMES:
                        col = beam_hdu.data[names[NAMES.index(cand)]]
                        if np.issubdtype(col.dtype, np.number):
                            val_col = names[NAMES.index(cand)]
                            break

                if val_col is None:
                    for n, uN in zip(names, NAMES):
                        if uN in ('ELL', 'L', 'EL') or uN.startswith('DET'):
                            continue
                        col = beam_hdu.data[n]
                        if np.issubdtype(col.dtype, np.number):
                            val_col = n
                            break
                if val_col is None:
                    raise ValueError(f"No numeric BL/TT/T column found in {beam_path} ({names})")

                y_in = np.asarray(beam_hdu.data[val_col])
                if y_in.ndim == 2:
                    y_in = np.nanmean(y_in, axis=1)
                x_in = np.asarray(beam_hdu.data[ell_col], dtype=float) if ell_col else np.arange(len(y_in), dtype=float)
                Bl = _interp_from_xy(x_in, y_in)
                return (Lout, Bl, Bl, Bl)
            else:
                # HFI beams
                window_hdu = None
                for name in ('WINDOW FUNCTIONS', 'WINDOW_FUNCTIONS'):
                    try:
                        window_hdu = hdul[name]
                        break
                    except Exception:
                        pass
                if window_hdu is None:
                    window_hdu = hdul[1]

                T = np.asarray(window_hdu.data['T'], dtype=float)
                E = np.asarray(window_hdu.data['E'], dtype=float) if 'E' in window_hdu.columns.names else T
                B = np.asarray(window_hdu.data['B'], dtype=float) if 'B' in window_hdu.columns.names else E
                x_in = np.arange(len(T), dtype=float)
                return (Lout, _interp_from_xy(x_in, T), _interp_from_xy(x_in, E), _interp_from_xy(x_in, B))

    # Generic fallback
    arr = np.loadtxt(beam_path)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[1] == 1:
        y = arr[:, 0]
        x = np.arange(len(y), dtype=float)
    else:
        x = arr[:, 0]
        y = arr[:, 1]
    Bl = _interp_from_xy(x, y)
    return (Lout, Bl, Bl, Bl)

 
# ---------------------------------------------------------------------
def _convolve_IQU_with_beam(iqu_map, exp, band_name, data_dict, lmax=None):
    """
    Convolve an IQU sky map with the instrument beam window function.

    Parameters
    ----------
    iqu_map : ndarray, shape (3, npix)
        Input intensity (I), and polarization (Q, U) maps in μK_CMB.
    exp : str
        Experiment name.
    band_name : str
        Band identifier.
    data_dict : dict
        Metadata dictionary (must include beam path).
    lmax : int, optional
        Maximum multipole for harmonic transform (default = 3*nside - 1).

    Returns
    -------
    ndarray
        Beam-convolved IQU map with same shape as input.
    """
    if iqu_map is None or len(iqu_map) != 3:
        raise ValueError("iqu_map must be an array [I, Q, U] of shape (3, npix)")

    nside_map = hp.get_nside(iqu_map[0])
    if lmax is None:
        lmax = 3 * nside_map - 1

    iqu64 = np.asarray(iqu_map, dtype=np.float64, order="C")
    almT, almE, almB = hp.map2alm(iqu64, lmax=lmax, pol=True, iter=3)
    _, Bl_T, Bl_E, Bl_B = _load_instrument_beam_TEB(exp, band_name, data_dict, lmax)

    almT = hp.almxfl(almT, Bl_T)
    almE = hp.almxfl(almE, Bl_E)
    almB = hp.almxfl(almB, Bl_B)

    return hp.alm2map([almT, almE, almB], nside=nside_map, pol=True, lmax=lmax, verbose=False)


# ---------------------------------------------------------------------
def generate_sky_maps(nside, path_save, experiment_select="all", band_select="all"):
    """
    Generate and save beam-convolved PySM IQU sky maps.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter.
    path_save : str
        Directory where the maps will be saved.
    experiment_select : str, optional
        Experiment to simulate ('all' for all in data dict).
    band_select : str, list, optional
        Specific bands to simulate ('all' for all available).

    Notes
    -----
    - Simulates synchrotron ('s1'), dust ('d1'), AME ('a1), 
      free-free ('f1') and CMB ('c1') components.
    - Saves FITS files with metadata in headers.
    """
    os.makedirs(path_save, exist_ok=True)
    sky = pysm3.Sky(nside=nside, preset_strings=['s1', 'd1', 'a1', 'f1', 'c1']) 

    experiments = list(data.keys()) if experiment_select == "all" else [experiment_select]
    for exp in experiments:
        if exp not in data:
            print(f"[WARN] Experiment {exp} not in data. Skipping.", file=sys.stderr)
            continue

        bands_all = list(data[exp].keys())
        if band_select == "all":
            bands = bands_all
        elif isinstance(band_select, (list, tuple, set)):
            bands = [b for b in band_select if b in data[exp]]
        else:
            bands = [band_select] if band_select in data[exp] else []

        if not bands:
            print(f"[WARN] No valid bands for {exp}. Available: {bands_all}", file=sys.stderr)
            continue

        for band in bands:
            try:
                nu = data[exp][band]['freq']
                nu_GHz = nu.to_value(u.GHz)

                iqu = sky.get_emission(nu).to(
                    u_pysm.uK_CMB, equivalencies=u_pysm.cmb_equivalencies(nu)
                ).value

                iqu_conv = _convolve_IQU_with_beam(iqu, exp, band, data, lmax=3 * nside - 1)

                # Convert units based on experiment
                if exp.upper() == "PLANCK":
                    # Convert from uK to K for Planck
                    iqu_conv = iqu_conv / 1e6  # uK to K
                    unit_str = "K_CMB"
                else:
                    # Convert from uK to mK for other experiments
                    iqu_conv = iqu_conv / 1e3  # uK to mK
                    unit_str = "mK_CMB"

                out_dir = path_save
                os.makedirs(out_dir, exist_ok=True)
                out_f = os.path.join(out_dir, f"map_{exp}_{band}_nside{nside}_beamconv.fits")

                header = [
                    ("FREQ_GHZ", float(nu_GHz), "Band center frequency [GHz]"),
                    ("NSIDE", int(nside), "HEALPix NSIDE"),
                    ("UNIT", unit_str, "Units"),
                    ("EXPERIM", str(exp), "Experiment"),
                    ("BAND", str(band), "Band name"),
                ]
                hp.write_map(
                    out_f, iqu_conv, dtype=np.float32, overwrite=True,
                    column_names=["I", "Q", "U"], extra_header=header
                )
                print(f"[OK] Saved: {out_f}")
            except Exception as e:
                print(f"[ERROR] {exp} {band}: {e}", file=sys.stderr)


'''
# ====================================
# 2
# ====================================
# '''


def white_noise_maps(data, nside, experiment_select="all", band_select="all", n_sim=100, path_map="./"):
    """Generate white noise realizations for experiments and bands in data.

    Parameters
    ----------
    data : dict
        Dictionary containing experiment data.
    nside : int
        Healpy nside for maps.
    experiment_select : str, default "all"
        If "all", generate maps for all experiments. Otherwise, only for the given experiment.
    band_select : str, list of str, or "all", default "all"
        If "all", generate maps for all bands. Otherwise, only for the given band(s).
    n_sim : int, default 100
        Number of white noise realizations to generate.
    path_map : str, default "./"
        Root path to store the output maps.
    """

    npix = hp.nside2npix(nside)

    for experiment, bands in data.items():
        if experiment_select != "all" and experiment != experiment_select:
            continue

        print(f"\nRunning {experiment}...")

        for band, band_info in bands.items():
            # --- band filter ---
            if band_select != "all":
                if isinstance(band_select, (list, tuple, set)):
                    if band not in band_select:
                        continue
                else:
                    if band != band_select:
                        continue

            try:
                band_path = band_info['path']
                print(f" [{experiment}] Band {band} GHz -> {band_path}")

                # --- QUIJOTE ---
                if experiment == "QUIJOTE":
                    # Load weights and Q-U covariance maps
                    wei_i, wei_q, wei_u = hp.read_map(band_path, field=[5,6,7])
                    cov_qu = hp.read_map(band_path, field=8)

                    # Replace UNSEEN with 0
                    wei_i = np.where(wei_i == hp.UNSEEN, 0., wei_i)
                    wei_q = np.where(wei_q == hp.UNSEEN, 0., wei_q)
                    wei_u = np.where(wei_u == hp.UNSEEN, 0., wei_u)
                    cov_qu = np.where(cov_qu == hp.UNSEEN, 0., cov_qu)

                    # Compute variances from weights
                    var_i = np.zeros(npix)
                    var_q = np.zeros(npix)
                    var_u = np.zeros(npix)

                    mask_i = wei_i > 0
                    mask_q = wei_q > 0
                    mask_u = wei_u > 0

                    var_i[mask_i] = 1.0 / wei_i[mask_i]
                    var_q[mask_q] = 1.0 / wei_q[mask_q]
                    var_u[mask_u] = 1.0 / wei_u[mask_u]

                    sigma_i = np.sqrt(var_i)

                    # --- Prepare 2x2 covariance matrices per pixel ---
                    # Clip COV_QU to ensure positive-definite covariance
                    cov_qu = np.clip(cov_qu, -np.sqrt(var_q*var_u), np.sqrt(var_q*var_u))
                    
                    # Initialize Cholesky matrix
                    L = np.zeros((npix, 2, 2))

                    # Only compute for pixels with valid variance
                    valid_pix = (var_q > 0) & (var_u > 0)

                    # Safe Cholesky decomposition
                    L[valid_pix, 0, 0] = np.sqrt(var_q[valid_pix])
                    L[valid_pix, 1, 0] = cov_qu[valid_pix] / np.sqrt(var_q[valid_pix])
                    L[valid_pix, 1, 1] = np.sqrt(var_u[valid_pix] - L[valid_pix, 1, 0]**2)

                    # Ensure no NaNs
                    L = np.nan_to_num(L, nan=0.0, posinf=0.0, neginf=0.0)

                # --- WMAP ---
                elif experiment == "WMAP":
                    nobs = hp.read_map(band_path, hdu=2, field=[0,1,2,3])
                    nobs_i, nobs_q, nobs_qu, nobs_u = nobs
                    nobs_q_eff = nobs_q - nobs_qu**2/nobs_u
                    nobs_u_eff = nobs_u - nobs_qu**2/nobs_q
                    sigma_i = band_info['noise_I'].value / np.sqrt(nobs_i)
                    sigma_q = band_info['noise_QU'].value / np.sqrt(nobs_q_eff)
                    sigma_u = band_info['noise_QU'].value / np.sqrt(nobs_u_eff)

                # --- Planck ---
                elif experiment == "Planck":
                    # Covariance fields in the FITS: II, IQ, IU, QQ, QU, UU
                    fields = [4, 5, 6, 7, 8, 9]
                    cov_maps = [hp.read_map(band_path, field=f) for f in fields]

                    # Resample to target NSIDE if needed
                    nside_in = hp.get_nside(cov_maps[0])
                    if nside_in != nside:
                        factor = (nside_in / nside)**2
                        cov_maps = [hp.ud_grade(m, nside_out=nside) / factor for m in cov_maps]

                    # Construct 3x3 covariance matrix per pixel
                    cov_matrix = np.zeros((npix, 3, 3))
                    cov_matrix[:,0,0], cov_matrix[:,0,1], cov_matrix[:,0,2] = cov_maps[0], cov_maps[1], cov_maps[2]
                    cov_matrix[:,1,0], cov_matrix[:,1,1], cov_matrix[:,1,2] = cov_maps[1], cov_maps[3], cov_maps[4]
                    cov_matrix[:,2,0], cov_matrix[:,2,1], cov_matrix[:,2,2] = cov_maps[2], cov_maps[4], cov_maps[5]

                    # Initialize Cholesky matrices
                    L = np.zeros_like(cov_matrix)
                    valid_pix = np.zeros(npix, dtype=bool)

                    # Compute Cholesky per pixel with eigenvalue clipping for robustness
                    for i in range(npix):
                        try:
                            w, v = np.linalg.eigh(cov_matrix[i])
                            w = np.clip(w, 0, None)  # ensure positive semi-definite
                            cov_matrix[i] = v @ np.diag(w) @ v.T
                            L[i] = np.linalg.cholesky(cov_matrix[i])
                            valid_pix[i] = True
                        except np.linalg.LinAlgError:
                            L[i] = np.zeros((3,3))

                # --- Generate n_sim white noise realizations ---
                for ii in tqdm(range(n_sim), desc='generating maps'):
                    noise_map = np.zeros([3, npix])

                    if experiment == "QUIJOTE":
                        # I: independent Gaussian noise
                        noise_map[0] = np.random.normal(0, sigma_i, npix)
                        # Q,U: correlated Gaussian using Cholesky
                        z = np.random.normal(size=(npix,2))
                        noise_map[1:3,:] = np.einsum('ijk,ik->ij', L, z).T
                        noise_map[1, ~valid_pix] = 0.0
                        noise_map[2, ~valid_pix] = 0.0

                    elif experiment == "Planck":
                        # I,Q,U: correlated Gaussian using 3x3 Cholesky
                        z = np.random.normal(size=(npix,3))
                        noise_map[:, :] = np.einsum('ijk,ik->ij', L, z).T
                        noise_map[:, ~valid_pix] = 0.0

                    else:
                        # WMAP: independent Gaussian noise per component
                        noise_map[0] = np.random.normal(0, sigma_i, npix)
                        noise_map[1] = np.random.normal(0, sigma_q, npix)
                        noise_map[2] = np.random.normal(0, sigma_u, npix)

                    # --- Save map ---
                    out_dir = os.path.join(path_map, experiment, "noise_simulations", band)
                    os.makedirs(out_dir, exist_ok=True)
                    out_file = os.path.join(out_dir, f"white_noise_{band}ghz_{str(ii+1).zfill(4)}.fits")
                    hp.write_map(out_file, noise_map, dtype=np.float64, overwrite=True)

            except KeyError:
                print(f" ! Band {band} not found in {experiment}")




'''
# ====================================
# 3
# ====================================
# '''

# Differential Assemblies (DAs) per frequency band
BANDS = {
    'K': ['K1'],
    'Ka': ['Ka1'],
    'Q': ['Q1', 'Q2'],
    'V': ['V1', 'V2'],
    'W': ['W1', 'W2', 'W3', 'W4'],
}

def coadd_da(files):
    '''
    Coadd several Differential Assemblies (DAs) from the same year,
    weighted by their number of observations (N_obs).

    Parameters
    ----------
    files : list of str
        List of FITS file paths corresponding to the DAs for a given year.

    Returns
    -------
    band_map : numpy.ndarray
        Coadded Stokes maps (3, Npix) for the given band/year.
    sum_nobs : numpy.ndarray
        Total number of observations per pixel used in the coaddition.
    '''
    sum_map = None
    sum_nobs = None
    for f in files:
        # Read I, Q, U, N_obs from HDU 1
        I, Q, U, nobs = hp.read_map(f, field=[0, 1, 2, 3], hdu=1)
        m = np.array([I, Q, U])
        if sum_map is None:
            sum_map = m * nobs
            sum_nobs = nobs.copy()
        else:
            sum_map += m * nobs
            sum_nobs += nobs
    band_map = sum_map / sum_nobs
    return band_map, sum_nobs

def coadd_years(band_maps, nobs_maps):
    '''
    Coadd maps from multiple years, weighted by their number of observations.

    Parameters
    ----------
    band_maps : list of numpy.ndarray
        List of Stokes maps (3, Npix), one for each year.
    nobs_maps : list of numpy.ndarray
        List of N_obs arrays (Npix,), one for each year.

    Returns
    -------
    coadded_map : numpy.ndarray
        Final coadded Stokes maps (3, Npix) across all selected years.
    '''
    sum_map = None
    sum_nobs = None
    for m, nobs in zip(band_maps, nobs_maps):
        if sum_map is None:
            sum_map = m * nobs
            sum_nobs = nobs.copy()
        else:
            sum_map += m * nobs
            sum_nobs += nobs
    return sum_map / sum_nobs

def coadd_year_range(base_dir, bands='all', year_1=1, year_2=9, save=False, save_path='./'):
    '''
    Generate band maps per year, coadd years, and optionally save the results.

    Parameters
    ----------
    base_dir : str
        Path to the directory containing WMAP FITS files.
    bands : list of str or 'all', optional
        List of bands to process. If 'all', process all bands.
    year_1 : int
        First year to include in coaddition.
    year_2 : int
        Last year to include in coaddition.
    save : bool, default False
        Whether to save the coadded maps to FITS files.
    save_path : str, default './'
        Directory path where the maps will be saved if save=True.

    Returns
    -------
    combined_maps : dict
        Dictionary with structure {band: coadded_map} for the selected year range.
    '''
    # If bands is 'all', process all available bands
    if bands == 'all':
        bands = list(BANDS.keys())

    # Step 1: Generate band maps per year
    band_year_maps = {band: {} for band in bands}
    for band in bands:
        for year in range(1, 10):
            das = BANDS[band]
            files = [os.path.join(base_dir, f'wmap_iqumap_r9_yr{year}_{da}_v5.fits') for da in das]
            m, nobs = coadd_da(files)
            band_year_maps[band][year] = (m, nobs)

    # Step 2: Coadd selected year range
    combined_maps = {}
    for band in bands:
        maps_to_coadd = [band_year_maps[band][y][0] for y in range(year_1, year_2 + 1)]
        nobs_to_coadd = [band_year_maps[band][y][1] for y in range(year_1, year_2 + 1)]
        combined_map = coadd_years(maps_to_coadd, nobs_to_coadd)
        combined_maps[band] = combined_map

        if save:
            filename = f'wmap_iqumap_r9_yr{year_1}to{year_2}_{band}_v5.fits'
            full_path = os.path.join(save_path, filename)
            hp.write_map(full_path, combined_map, overwrite=True)
            print(f'Saved {full_path}')

    return combined_maps


def make_hmdm(data, bands, save=False):
    """
    Compute Half Mission Difference Maps (HMDM) for the given bands.
    Automatically detects the experiment (QUIJOTE, WMAP, PLANCK).

    Parameters
    ----------
    data : dict
        Dictionary with experiments, bands, and file paths.
    bands : list of str
        List of bands to process (e.g. ['11','23','30']).
    save : bool, default False
        Whether to save the HMDM maps to disk, in the path specified by
        data[exp][band]['hmdm'].
    Returns
    -------
    hmdm_maps : dict
        Dictionary with experiment -> band -> HMDM (numpy array).
    saved_files : dict
        Dictionary with experiment -> band -> output file path (if save=True).
    """
    hmdm_maps = {}
    saved_files = {}

    for exp in data:
        for band in bands:
            if band not in data[exp]:
                continue  # skip if band not in this experiment

            path_half1 = data[exp][band]['half_1']
            path_half2 = data[exp][band]['half_2']

            # --- QUIJOTE ---
            if exp.upper() == "QUIJOTE":
                comp = "IQU"
                h1 = hp.read_map(path_half1, [c + "_STOKES" for c in comp], nest=False)
                h2 = hp.read_map(path_half2, [c + "_STOKES" for c in comp], nest=False)

                w1 = hp.read_map(path_half1, ["WEI_" + c for c in comp], nest=False)
                w2 = hp.read_map(path_half2, ["WEI_" + c for c in comp], nest=False)

                w1[np.isnan(w1)] = 0
                w2[np.isnan(w2)] = 0
                w1[w1 < 0] = 0
                w2[w2 < 0] = 0

                w = np.sqrt((w1 + w2) * (1. / w1 + 1. / w2))
                hmdm = (h1 - h2) / w
                hmdm[w1 * w2 == 0] = 0

            # --- WMAP ---
            elif exp.upper() == "WMAP":
                h1 = hp.read_map(path_half1, field=[0, 1, 2])
                h2 = hp.read_map(path_half2, field=[0, 1, 2])

                # Factor for 1to4-5to9 
                hmdm = (h1 - h2) * np.sqrt(20. / 81.)

            # --- PLANCK ---
            elif exp.upper() == "PLANCK":
                h1 = hp.read_map(path_half1, field=[0, 1, 2])
                h2 = hp.read_map(path_half2, field=[0, 1, 2])

                # Get bad data value from header
                with fits.open(path_half1) as hdul:
                    bad_data = hdul[1].header.get('BAD_DATA', -1.6375e+30)
                
                # Detect LFI vs HFI based on frequency band
                try:
                    freq_num = int(band)
                    is_hfi = freq_num >= 100
                except:
                    is_hfi = False
                
                if is_hfi:
                    # HFI processing: robust BAD_DATA handling with covariance weighting
                    print(f"[{exp} {band}] Processing HFI band with robust BAD_DATA handling")
                    
                    # Read covariance fields for HFI (same fields as LFI)
                    cov1 = hp.read_map(path_half1, field=[4, 7, 9])
                    cov2 = hp.read_map(path_half2, field=[4, 7, 9])
                    
                    # Convert to arrays for manipulation
                    h1 = np.array(h1)
                    h2 = np.array(h2)
                    cov1 = np.array(cov1)
                    cov2 = np.array(cov2)
                    
                    # Create robust bad data masks
                    # Use more restrictive threshold for BAD_DATA detection
                    bad_threshold = np.abs(bad_data) * 0.01  # More restrictive
                    
                    mask1 = np.abs(h1 - bad_data) < bad_threshold
                    mask2 = np.abs(h2 - bad_data) < bad_threshold
                    
                    # Also check for extreme values that might not be exactly BAD_DATA
                    extreme_mask1 = np.abs(h1) > 1e10  # Extremely large values
                    extreme_mask2 = np.abs(h2) > 1e10
                    
                    # Combined bad pixel mask
                    bad_mask = mask1 | mask2 | extreme_mask1 | extreme_mask2
                    
                    # Clean the maps: set bad pixels to zero
                    h1_clean = h1.copy()
                    h2_clean = h2.copy()
                    h1_clean[bad_mask] = 0.0
                    h2_clean[bad_mask] = 0.0
                    
                    # Handle covariances: check if they contain bad values and clean them
                    cov_bad_mask1 = np.abs(cov1 - bad_data) < bad_threshold
                    cov_bad_mask2 = np.abs(cov2 - bad_data) < bad_threshold
                    
                    # Also add pixels that are bad in maps to covariance bad mask
                    cov_bad_mask1 = cov_bad_mask1 | bad_mask
                    cov_bad_mask2 = cov_bad_mask2 | bad_mask
                    
                    # For bad pixels in covariances, use a large variance (small weight)
                    cov1_clean = cov1.copy()
                    cov2_clean = cov2.copy()
                    
                    cov1_clean[cov_bad_mask1] = 1e20  # Large variance for bad pixels
                    cov2_clean[cov_bad_mask2] = 1e20
                    
                    # Ensure all covariances are positive (take absolute value)
                    cov1_clean = np.abs(cov1_clean)
                    cov2_clean = np.abs(cov2_clean)
                    
                    # Add small regularization to avoid numerical issues
                    cov1_clean = cov1_clean + 1e-20
                    cov2_clean = cov2_clean + 1e-20
                    
                    # Calculate sigmas
                    sigma1 = np.sqrt(cov1_clean)
                    sigma2 = np.sqrt(cov2_clean)
                    
                    # Calculate weights using same formula as LFI but with numerical protection
                    sigma1_sq = sigma1**2
                    sigma2_sq = sigma2**2
                    
                    # Protect against division by zero
                    inv_sigma1_sq = np.where(sigma1_sq > 1e-30, 1.0/sigma1_sq, 0.0)
                    inv_sigma2_sq = np.where(sigma2_sq > 1e-30, 1.0/sigma2_sq, 0.0)
                    
                    w = np.sqrt((inv_sigma1_sq + inv_sigma2_sq) * (sigma1_sq + sigma2_sq))
                    
                    # Protect against division by zero in final calculation
                    w = np.where(w > 1e-30, w, 1e30)  # If weight is tiny, make it huge (effectively zero contribution)
                    
                    # Calculate HMDM
                    hmdm = (h1_clean - h2_clean) / w
                    
                    # Final masking: set all bad pixels to zero
                    final_bad_mask = bad_mask | cov_bad_mask1 | cov_bad_mask2
                    hmdm[final_bad_mask] = 0.0
                    
                    n_bad_total = np.sum(final_bad_mask[0])  # Count for I map
                    print(f"[{exp} {band}] Masked {n_bad_total} bad pixels total")
                    print(f"[{exp} {band}] Applied robust covariance weighting")
                    
                else:
                    # LFI processing: use covariance weighting (original method)
                    print(f"[{exp} {band}] Processing LFI band")
                    
                    # Read covariance fields
                    cov1 = hp.read_map(path_half1, field=[4, 7, 9])
                    cov2 = hp.read_map(path_half2, field=[4, 7, 9])
                    
                    # Calculate sigmas
                    sigma1 = np.sqrt(cov1)
                    sigma2 = np.sqrt(cov2)
                    
                    # Calculate weights using original formula
                    w = np.sqrt((1 / sigma1**2 + 1 / sigma2**2) * (sigma1**2 + sigma2**2))
                    hmdm = (h1 - h2) / w



            else:
                raise ValueError(f"Experiment {exp} not implemented")

            if save:
                out_file = data[exp][band]['hmdm']
                hp.write_map(out_file, hmdm, overwrite=True)
                saved_files.setdefault(exp, {})[band] = out_file
                print(f"[{exp} {band}] HMDM saved to {out_file}")

            hmdm_maps.setdefault(exp, {})[band] = hmdm

    return hmdm_maps, saved_files


'''
# ====================================
# 4
# ====================================
# '''


def load_beam_file(file_path):
    '''
    Load B_l data from a WMAP beam file, keeping all columns.

    Returns
    -------
    header_lines : list of str
        List of header lines (starting with '#').
    data : numpy.ndarray
        Array with shape (N,3) containing l, B_l, fractional error.
    '''
    header_lines = []
    data_lines = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                header_lines.append(line.rstrip())
            else:
                data_lines.append([float(x) for x in line.split()])
    data = np.array(data_lines)
    return header_lines, data


def save_band_beam(band_name, header_template, data, save_path):
    '''
    Save the averaged beam for a frequency band to a txt file,
    preserving the original WMAP header format and modifying the
    description to indicate the frequency band instead of the DA.

    Parameters
    ----------
    band_name : str
        Name of the frequency band (e.g., 'K', 'Ka', 'Q', 'V', 'W').
    header_template : list of str
        List of header lines from the original DA beam file (lines starting with '#').
        The function will replace the line mentioning 'differencing assembly'
        with a line indicating the frequency band.
    data : numpy.ndarray
        Array with shape (N,3), where:
            - column 0: multipole moment l
            - column 1: averaged beam transfer function B_l
            - column 2: averaged fractional error (delta B_l / B_l)
    save_path : str
        Path to the directory where the output file will be saved.

    Returns
    -------
    None
        The function writes a txt file to disk and prints a confirmation message.
    '''
    new_header = []
    for line in header_template:
        # Replace DA description with frequency band
        if 'differencing assembly' in line:
            line = f'# Beam Transfer Function (amplitude) for frequency band {band_name},'
        new_header.append(line)
    
    file_name = f'wmap_ampl_bl_{band_name}_9yr_v5p1.txt'
    file_path = os.path.join(save_path, file_name)
    
    with open(file_path, 'w') as f:
        for line in new_header:
            f.write(line + '\n')
        for row in data:
            f.write(f'{int(row[0]):>5} {row[1]:>14.8f} {row[2]:>14.8f}\n')
    
    print(f'Saved band beam file: {file_path}')


def generate_band_beams(BANDS, beam_path, save_path, data_dict=None, band_code_map=None, use_qu=True, mask_path=None, normalize_weights=True):
    '''
    Generate effective band beams by combining DA beams, automatically computing DA weights from N_obs and DA noise if data_dict is provided.

    Parameters
    ----------
    BANDS : dict
        Dictionary of bands and their DAs, e.g. {'W': ['W1','W2','W3','W4'], ...}.
    beam_path : str
        Directory containing original DA beam txt files (wmap_ampl_bl_<DA>_9yr_v5p1.txt).
    save_path : str
        Directory where the band-averaged beam files will be written.
    data_dict : dict or None, optional
        If provided, will be used to read DA map FITS and compute weights from N_obs and DA noise.
    band_code_map : dict or None, optional
        Mapping from band letter to WMAP band code (e.g., {'K': '23', 'Ka': '33', ...}). If None, uses default mapping.
    use_qu : bool, default True
        Use 'noise_QU' for sigma0 if True, else 'noise_I'.
    mask_path : str or None, optional
        Optional Healpy mask FITS path; if provided, weights use N_obs within the unmasked region (mask>0).
    normalize_weights : bool, default True
        Whether to normalize weights so they sum to 1 before averaging.

    Returns
    -------
    None
        Writes one beam file per band and prints a confirmation message.
    '''

    def _compute_band_weights(band, da_list):
        # If data_dict is provided, compute weights from N_obs and DA noise
        if data_dict is not None:
            # Map band letter to WMAP band code
            default_code_map = {'K': '23', 'Ka': '33', 'Q': '41', 'V': '61', 'W': '94'}
            code_map = band_code_map if band_code_map is not None else default_code_map
            band_code = code_map.get(band, None)
            if band_code is None:
                print(f"WARNING: No band code for {band}; using equal weights.")
                return np.ones(len(da_list), dtype=float) / len(da_list)
            # Use helper to compute weights
            w_dict = compute_wmap_da_weights(data_dict, band_code, {band: da_list}, use_qu=use_qu, mask_path=mask_path)
            w = np.array([w_dict.get(da, 1.0) for da in da_list], dtype=float)
            if normalize_weights:
                s = np.sum(w)
                if s > 0:
                    w = w / s
            return w
        else:
            # Default: equal weights
            w = np.ones(len(da_list), dtype=float)
            if normalize_weights:
                w = w / len(da_list)
            return w

    for band, das in BANDS.items():
        all_data = []
        header_template = None
        for da in das:
            file_name = f'wmap_ampl_bl_{da}_9yr_v5p1.txt'
            file_path = os.path.join(beam_path, file_name)
            if not os.path.exists(file_path):
                print(f'WARNING: Beam file not found: {file_path}')
                continue
            header, data = load_beam_file(file_path)
            if header_template is None:
                header_template = header
            all_data.append((da, data))
        if len(all_data) == 0:
            print(f'No beams found for band {band}')
            continue

        # Prepare weights aligned with loaded DA order
        da_loaded = [da for da, _ in all_data]
        w = _compute_band_weights(band, da_loaded)

        # Initialize output array with ell column from the first DA
        first_data = all_data[0][1]
        avg_data = np.zeros_like(first_data)
        avg_data[:, 0] = first_data[:, 0]

        # Sanity check: all ell columns match; if not, we use interpolation
        ell_ref = first_data[:, 0]
        mismatch = False
        for _, data in all_data[1:]:
            if not np.array_equal(data[:, 0], ell_ref):
                mismatch = True
                break

        if mismatch:
            # Interpolate each DA onto the reference ell grid before averaging
            def _interp_to_ref(data_arr):
                ell = data_arr[:, 0]
                bl = data_arr[:, 1]
                ferr = data_arr[:, 2]
                bl_i = np.interp(ell_ref, ell, bl, left=1.0, right=1.0)
                ferr_i = np.interp(ell_ref, ell, ferr, left=ferr[0], right=ferr[-1])
                return bl_i, ferr_i

            bl_sum = np.zeros_like(ell_ref, dtype=float)
            ferr_sum = np.zeros_like(ell_ref, dtype=float)
            for (da, data), wi in zip(all_data, w):
                bl_i, ferr_i = _interp_to_ref(data)
                bl_sum += wi * bl_i
                ferr_sum += wi * ferr_i
            avg_data[:, 1] = bl_sum
            avg_data[:, 2] = ferr_sum
        else:
            # Weighted average on matching ell grid
            bl_sum = np.zeros_like(ell_ref, dtype=float)
            ferr_sum = np.zeros_like(ell_ref, dtype=float)
            for (da, data), wi in zip(all_data, w):
                bl_sum += wi * data[:, 1]
                ferr_sum += wi * data[:, 2]
            avg_data[:, 1] = bl_sum
            avg_data[:, 2] = ferr_sum

        save_band_beam(band, header_template, avg_data, save_path)


def compute_wmap_da_weights(data_dict, band_code, BANDS, use_qu=True, mask_path=None):
    '''
    Compute per-DA weights for a WMAP band using N_obs and DA noise.

    Parameters
    ----------
    data_dict : dict
        The `data` dictionary from data.py containing paths and DA noise.
    band_code : str
        WMAP band frequency code as used in `data_dict['WMAP']` (e.g., '41' for Q, '61' for V, '94' for W).
    BANDS : dict
        Mapping of nominal band letters to DA lists, e.g. {'Q': ['Q1','Q2'], 'W': ['W1','W2','W3','W4']}.
        This function infers the band letter from `band_code`.
    use_qu : bool, default True
        If True, use 'noise_QU' for sigma0; otherwise use 'noise_I'.
    mask_path : str or None
        Optional Healpy mask FITS path; if provided, weights use N_obs within the unmasked region (mask>0).

    Returns
    -------
    dict
        Dictionary mapping DA name to normalized weight, e.g. {'W1': w1, 'W2': w2, ...}.

    Notes
    -----
    - Weight definition: w_i ∝ sigma_p N_obs_i(p) / sigma0_i^2 over pixels p (optionally masked).
    - If sigma0_i is unavailable, falls back to w_i ∝ sigma_p N_obs_i(p).
    - Uses 9-year DA maps (paths like wmap_iqumap_r9_9yr_<DA>_v5.fits) specified in data.py.
    '''
    # Map numeric band code to letter used in DA names
    code_to_letter = {
        '23': 'K', '33': 'Ka', '41': 'Q', '61': 'V', '94': 'W'
    }
    if band_code not in code_to_letter:
        raise ValueError(f"Unsupported WMAP band code: {band_code}")
    band_letter = code_to_letter[band_code]
    da_list = BANDS.get(band_letter, [])
    if not da_list:
        raise ValueError(f"No DAs found for band {band_letter} in BANDS")

    mask = None
    if mask_path is not None and os.path.exists(mask_path):
        try:
            mask = hp.read_map(mask_path)
        except Exception:
            mask = None

    weights = {}
    for da in da_list:
        # Find DA key in data_dict for this band (e.g., '94_1' corresponds to W1)
        # Build expected DA index from last char of DA name
        try:
            da_idx = int(da[-1])
        except Exception:
            da_idx = 1
        da_key = f"{band_code}_{da_idx}"
        if 'WMAP' not in data_dict or da_key not in data_dict['WMAP']:
            print(f"WARNING: Missing WMAP DA entry for {da_key} in data dict; skipping")
            continue

        da_entry = data_dict['WMAP'][da_key]
        da_path = da_entry.get('path')
        if not da_path or not os.path.exists(da_path):
            print(f"WARNING: DA map not found for {da_key}: {da_path}")
            continue

        # Read N_obs from HDU 1, field index 3 (I,Q,U,N_obs)
        try:
            nobs = hp.read_map(da_path, field=3, hdu=1)
        except Exception:
            print(f"WARNING: Could not read N_obs for {da_key}; skipping")
            continue

        if mask is not None:
            sel = (mask > 0)
            nobs_sum = float(np.sum(nobs[sel]))
        else:
            nobs_sum = float(np.sum(nobs))

        # sigma0 from data dict; prefer QU for polarization beams
        sigma0 = da_entry.get('noise_QU' if use_qu else 'noise_I', None)
        if hasattr(sigma0, 'value'):
            sigma0 = float(sigma0.value)
        elif sigma0 is not None:
            sigma0 = float(sigma0)

        if sigma0 and sigma0 > 0:
            w = nobs_sum / (sigma0 ** 2)
        else:
            w = nobs_sum
        weights[da] = w

    # Normalize
    s = sum(weights.values())
    if s > 0:
        for da in list(weights.keys()):
            weights[da] = weights[da] / s
    return weights


'''
# ====================================
# 5
# ====================================
# '''


def create_binning(binning_params):
    """Create an NmtBin object from flexible parameters."""
    if binning_params["type"] == "linear":
        return nmt.NmtBin.from_lmax_linear(
            binning_params["lmax"], binning_params["dl"]
        )
    elif binning_params["type"] == "edges":
        return nmt.NmtBin.from_edges(
            binning_params["ell1"], binning_params["ell2"]
        )
    else:
        raise ValueError("Unknown binning type: choose 'linear' or 'edges'")


def compute_master(f_a, f_b, wsp):
    """Compute the decoupled power spectrum from coupled spectra using a workspace.

    Parameters
    ----------
    f_a : NmtField
        First NaMaster field (spin-0 or spin-2).
    f_b : NmtField
        Second NaMaster field (spin-0 or spin-2).
    wsp : NmtWorkspace
        NaMaster workspace used to decouple the coupled spectra.

    Returns
    -------
    cl_decoupled : array
        Decoupled power spectrum array.
    """
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    cl_decoupled = wsp.decouple_cell(cl_coupled)
    return cl_decoupled

def prepare_workspaces(mask, b, nside, lmax=None, purify_e=True, purify_b=True):
    """
    Precompute NaMaster workspaces for spin-0 and spin-2 fields.
    Uses the same method as the original code.

    Parameters
    ----------
    mask : array
        Healpy mask (0=masked, 1=unmasked)
    b : nmt.NmtBin
        Binning scheme object
    nside : int
        Healpix nside
    lmax : int or None
        Maximum multipole for lmax_sht parameter
    purify_e, purify_b : bool
        Whether to purify E/B modes

    Returns
    -------
    dict
        Dictionary with precomputed workspaces: {'w00': ..., 'w02': ..., 'w22': ...}
    """
    npix = hp.nside2npix(nside)
    
    # Create fields exactly like in the original code (with lmax_sht)
    f0 = nmt.NmtField(mask, [np.zeros(npix)], lmax_sht=lmax)
    f2 = nmt.NmtField(mask, [np.zeros(npix), np.zeros(npix)], lmax_sht=lmax, 
                     purify_e=purify_e, purify_b=purify_b)
    
    # Generate workspace objects exactly like original code
    w00 = nmt.NmtWorkspace()
    w00.compute_coupling_matrix(f0, f0, b)

    w02 = nmt.NmtWorkspace()
    w02.compute_coupling_matrix(f0, f2, b)

    w22 = nmt.NmtWorkspace()
    w22.compute_coupling_matrix(f2, f2, b)
    
    return {'w00': w00, 'w02': w02, 'w22': w22}

def cross_spectrum(mask, map_1, map_2, b, workspaces, purify_e=True, purify_b=True, beam=None, lmax=None):
    """
    Compute cross-spectrum using precomputed workspaces.
    Uses the same field creation method as the original code.

    Parameters
    ----------
    mask : array
        Healpy mask.
    map_1, map_2 : array
        Maps of shape (3, npix) with I,Q,U.
    b : NmtBin
        Binning object.
    workspaces : dict
        Precomputed workspaces: {'w00': w00, 'w02': w02, 'w22': w22}.
    purify_e, purify_b : bool
        Whether to purify E/B modes.
    beam : array or None
        Optional beam window function.
    lmax : int or None
        Maximum multipole for lmax_sht parameter.

    Returns
    -------
    dict with keys ['ell1','ell2','ell_eff','TT','EE','BB','TE','TB','EB']
    """
    ell_min_list = [b.get_ell_min(i) for i in range(b.get_n_bands())]
    ell_max_list = [b.get_ell_max(i) for i in range(b.get_n_bands())]
    ell_eff = b.get_effective_ells()

    # Create fields exactly like in the original cross_spectrum code (with lmax_sht)
    f0_1 = nmt.NmtField(mask, [map_1[0,:]], lmax_sht=lmax, beam=beam)
    f2_1 = nmt.NmtField(mask, [map_1[1,:], map_1[2,:]], lmax_sht=lmax, purify_e=purify_e, purify_b=purify_b, beam=beam)
    
    f0_2 = nmt.NmtField(mask, [map_2[0,:]], lmax_sht=lmax, beam=beam)
    f2_2 = nmt.NmtField(mask, [map_2[1,:], map_2[2,:]], lmax_sht=lmax, purify_e=purify_e, purify_b=purify_b, beam=beam)

    w00, w02, w22 = workspaces['w00'], workspaces['w02'], workspaces['w22']

    # Compute cross-spectrum exactly like original code
    cl_master_tt = compute_master(f0_1, f0_2, w00)  # TT
    cl_master_tetb = compute_master(f0_1, f2_2, w02)  # TE TB
    cl_master_eb = compute_master(f2_1, f2_2, w22)  # EE EB BE BB
    
    # Extract components exactly like original code
    cl_tt = cl_master_tt[0]
    cl_te = cl_master_tetb[0] 
    cl_tb = cl_master_tetb[1]
    cl_ee = cl_master_eb[0]
    cl_eb = cl_master_eb[1]
    cl_bb = cl_master_eb[3]

    return {
        'ell1': ell_min_list,
        'ell2': ell_max_list,
        'ell_eff': ell_eff,
        'TT': cl_tt,
        'EE': cl_ee,
        'BB': cl_bb,
        'TE': cl_te,
        'TB': cl_tb,
        'EB': cl_eb
    }


def compute_hmdm_power_spectra(data, band_list, mask, b, workspaces=None, lmax=None, use_noise=False):
    """
    Compute auto- and cross-power spectra for HMDM maps or simulated noise maps.

    Parameters
    ----------
    data : dict
        Dictionary containing experiment and band information.
    band_list : list of str
        Ordered list of bands to compute spectra for.
    mask : array
        Healpy mask (0=masked, 1=unmasked) to apply.
    b : nmt.NmtBin
        NaMaster binning object.
    workspaces : dict
        Precomputed NaMaster workspaces: {'w00','w02','w22'}.
    lmax : int or None
        Maximum multipole for lmax_sht parameter.
    use_noise : bool
        If True, use simulated noise maps (realization 1) instead of real HMDM maps.

    Returns
    -------
    spectra_matrix : ndarray
        N_band x N_band matrix of dictionaries containing the spectra for each band pair.
        Each dictionary has keys: ['ell1','ell2','ell_eff','TT','EE','BB','TE','TB','EB'].
    """
    N_band = len(band_list)
    spectra_matrix = np.empty((N_band, N_band), dtype=object)

    for i, band_i in enumerate(band_list):
        # Identify experiment
        exp_i = next(exp for exp, bands in data.items() if band_i in bands)
        
        if use_noise:
            # Load simulated noise map (realization 1) instead of HMDM
            map_i = load_map(data, exp_i, band_i, use_simulated_maps=True, 
                           use_white_noise=False, noise_realization=1, only_noise=True)
        else:
            # Load HMDM map i
            hmdm_path_i = data[exp_i][band_i]['hmdm']
            map_i = hp.read_map(hmdm_path_i, field=[0, 1, 2])
            map_i = np.where(map_i == hp.UNSEEN, 0, map_i)
            
            # If Planck, convert to mK and downgrade to nside=512
            if exp_i == 'Planck':
                map_i *= 1e3
                nside_in = hp.get_nside(map_i[0])
                if nside_in != 512:
                    map_i = np.array([hp.ud_grade(m, nside_out=512) for m in map_i])

        for j, band_j in enumerate(band_list):
            if j < i:
                spectra_matrix[i, j] = spectra_matrix[j, i]  # Use symmetry
                continue

            exp_j = next(exp for exp, bands in data.items() if band_j in bands)
            
            if use_noise:
                # Load simulated noise map (realization 1) instead of HMDM
                map_j = load_map(data, exp_j, band_j, use_simulated_maps=True, 
                               use_white_noise=False, noise_realization=1, only_noise=True)
            else:
                # Load HMDM map j
                hmdm_path_j = data[exp_j][band_j]['hmdm']
                map_j = hp.read_map(hmdm_path_j, field=[0, 1, 2])
                map_j = np.where(map_j == hp.UNSEEN, 0, map_j)
                
                # If Planck, convert to mK and downgrade to nside=512
                if exp_j == 'Planck':
                    map_j *= 1e3
                    nside_in = hp.get_nside(map_j[0])
                    if nside_in != 512:
                        map_j = np.array([hp.ud_grade(m, nside_out=512) for m in map_j])

            # Compute cross-spectrum
            cl = cross_spectrum(mask, map_i, map_j, b, workspaces, purify_e=True, purify_b=True, lmax=lmax)
            spectra_matrix[i, j] = cl

    return spectra_matrix


def compute_pure_theoretical_spectra(data, band_list, mask, b, workspaces=None, lmax=None):
    """
    Compute auto- and cross-power spectra for pure simulated maps (no noise added).

    Parameters
    ----------
    data : dict
        Dictionary containing experiment and band information.
    band_list : list of str
        Ordered list of bands to compute spectra for.
    mask : array
        Healpy mask (0=masked, 1=unmasked) to apply.
    b : nmt.NmtBin
        NaMaster binning object.
    workspaces : dict
        Precomputed NaMaster workspaces: {'w00','w02','w22'}.
    lmax : int or None
        Maximum multipole for lmax_sht parameter.

    Returns
    -------
    spectra_matrix : ndarray
        N_band x N_band matrix of dictionaries containing the spectra for each band pair.
        Each dictionary has keys: ['ell1','ell2','ell_eff','TT','EE','BB','TE','TB','EB'].
    """
    N_band = len(band_list)
    spectra_matrix = np.empty((N_band, N_band), dtype=object)

    for i, band_i in enumerate(band_list):
        # Identify experiment
        exp_i = next(exp for exp, bands in data.items() if band_i in bands)
        # Load pure simulated map i (no noise)
        sky_map_i = load_pure_simulated_map(data, exp_i, band_i)

        for j, band_j in enumerate(band_list):
            if j < i:
                spectra_matrix[i, j] = spectra_matrix[j, i]  # Use symmetry
                continue

            exp_j = next(exp for exp, bands in data.items() if band_j in bands)
            # Load pure simulated map j (no noise)
            sky_map_j = load_pure_simulated_map(data, exp_j, band_j)

            # Compute cross-spectrum
            cl = cross_spectrum(mask, sky_map_i, sky_map_j, b, workspaces, purify_e=True, purify_b=True, lmax=lmax)
            spectra_matrix[i, j] = cl

    return spectra_matrix


def compute_all_power_spectra(
    data, band_list, mask, b,
    use_simulated_maps=True,
    use_white_noise=False,
    noise_realization=1,
    only_noise=False,
    workspaces=None,
    lmax=None
):
    """
    Compute auto- and cross-power spectra for a list of bands using precomputed NaMaster workspaces.
    Uses load_map() to handle sky + noise map loading.
    
    IMPORTANT: When use_simulated_maps=True and use_white_noise=False, this function
    loads simulated sky maps + noise simulation (NOT HMDM). HMDM is NEVER added.

    Parameters
    ----------
    data : dict
        Dictionary containing experiment and band information.
    band_list : list of str
        Ordered list of bands to compute spectra for.
    mask : array
        Healpy mask (0=masked, 1=unmasked) to apply.
    b : nmt.NmtBin
        NaMaster binning object.
    use_simulated_maps : bool
        If True, use simulated sky + noise maps; if False, use real maps.
    use_white_noise : bool
        If True, use white noise simulations; otherwise use full noise simulations.
    noise_realization : int
        Noise realization number (1-based).
    only_noise : bool
        If True, ignore sky and use only noise maps.
    workspaces : dict
        Precomputed NaMaster workspaces: {'w00','w02','w22'}.
    lmax : int or None
        Maximum multipole for lmax_sht parameter.

    Returns
    -------
    spectra_matrix : ndarray
        N_band x N_band matrix of dictionaries containing the spectra for each band pair.
        Each dictionary has keys: ['ell1','ell2','ell_eff','TT','EE','BB','TE','TB','EB'].
    """
    N_band = len(band_list)
    spectra_matrix = np.empty((N_band, N_band), dtype=object)

    for i, band_i in enumerate(band_list):
        # Identify experiment
        exp_i = next(exp for exp, bands in data.items() if band_i in bands)
        # Load map ipur
        sky_map_i = load_map(data, exp_i, band_i, use_simulated_maps, use_white_noise, noise_realization, only_noise)

        for j, band_j in enumerate(band_list):
            if j < i:
                spectra_matrix[i, j] = spectra_matrix[j, i]  # Use symmetry
                continue

            exp_j = next(exp for exp, bands in data.items() if band_j in bands)
            # Load map j
            sky_map_j = load_map(data, exp_j, band_j, use_simulated_maps, use_white_noise, noise_realization, only_noise)

            # Compute cross-spectrum
            cl = cross_spectrum(mask, sky_map_i, sky_map_j, b, workspaces, purify_e=True, purify_b=True, lmax=lmax)
            spectra_matrix[i, j] = cl

    return spectra_matrix


def load_pure_simulated_map(data, exp, band):
    """
    Load a pure simulated map (no noise added) for a given experiment and band.

    Parameters
    ----------
    data : dict
        Experiment and band info
    exp : str
        Experiment name
    band : str
        Band name

    Returns
    -------
    np.ndarray
        Pure simulated sky map (3, npix) with no noise
    """
    sky_path = data[exp][band]['path_simulated']
    sky_map = hp.read_map(sky_path, field=[0, 1, 2])
    sky_map = np.where(sky_map == hp.UNSEEN, 0, sky_map)
    
    # Convert Planck maps from K to mK and downgrade if needed
    if exp == 'Planck':
        sky_map *= 1e3
        nside_in = hp.get_nside(sky_map[0])
        if nside_in != 512:
            sky_map = np.array([hp.ud_grade(m, nside_out=512) for m in sky_map])
    
    return sky_map


def load_map(data, exp, band, use_simulated_maps, use_white_noise, noise_realization, only_noise):
    """
    Load a sky+noise map for a given experiment and band.
    
    IMPORTANT: This function NEVER adds HMDM. When use_simulated_maps=True,
    it loads simulated sky + noise simulations only.

    Parameters
    ----------
    data : dict
        Experiment and band info
    exp : str
        Experiment name
    band : str
        Band name
    use_simulated_maps : bool
        Whether to use simulated maps
    use_white_noise : bool
        Whether to use white noise
    noise_realization : int
        Noise simulation number
    only_noise : bool
        If True, use only noise map

    Returns
    -------
    np.ndarray
        Combined sky+noise map (3, npix)
    """
    if use_simulated_maps:
        sky_map = 0
        if not only_noise:
            sky_path = data[exp][band]['path_simulated']
            sky_map = hp.read_map(sky_path, field=[0, 1, 2])
            sky_map = np.where(sky_map == hp.UNSEEN, 0, sky_map)
            if exp == 'Planck':
                sky_map *= 1e3
                # Downgrade to nside=512 if needed
                nside_in = hp.get_nside(sky_map[0])
                if nside_in != 512:
                    sky_map = np.array([hp.ud_grade(m, nside_out=512) for m in sky_map])
        
        # Choose noise directory and base file
        if use_white_noise:
            noise_dir = data[exp][band]['path_white_noise_simulations']
            base_name = data[exp][band]['white_noise_simulation_1']
        else:
            noise_dir = data[exp][band]['path_noise_simulations']
            base_name = data[exp][band]['noise_simulation_1']
        
        # Build noise file path
        noise_fname = get_noise_filename(base_name, noise_realization)
        noise_path = os.path.join(noise_dir, noise_fname)
        noise_map = hp.read_map(noise_path, field=[0, 1, 2])
        noise_map = np.where(noise_map == hp.UNSEEN, 0, noise_map)
        if exp == 'Planck':
            noise_map *= 1e3
            # Downgrade to nside=512 if needed
            nside_in = hp.get_nside(noise_map[0])
            if nside_in != 512:
                noise_map = np.array([hp.ud_grade(m, nside_out=512) for m in noise_map])
        
        # If no sky map loaded, use zeros
        if isinstance(sky_map, int):
            sky_map = np.zeros_like(noise_map)
        
        return sky_map + noise_map
    
    else:
        # Load real map
        path = data[exp][band]['path']
        sky_map = hp.read_map(path, field=[0, 1, 2])
        sky_map = np.where(sky_map == hp.UNSEEN, 0, sky_map)
        
        # If Planck, convert to mK and downgrade to nside=512
        if exp == 'Planck':
            sky_map *= 1e3
            nside_in = hp.get_nside(sky_map[0])
            if nside_in != 512:
                sky_map = np.array([hp.ud_grade(m, nside_out=512) for m in sky_map])
        
        return sky_map


def save_spectra_to_fits(spectra_matrix, band_list, out_file):
    """
    Save a matrix of auto and cross power spectra into a FITS file.

    Parameters
    ----------
    spectra_matrix : ndarray
        Matrix of shape (N_band, N_band) with dictionaries containing the spectra.
        Each dictionary should have keys: ['ell1','ell2','ell_eff','TT','EE','BB','TE','TB','EB'].
        Auto-spectra are stored on the diagonal, cross-spectra in the upper triangle,
        with symmetry applied to the lower triangle.
    band_list : list of str
        Ordered list of bands corresponding to the rows/columns of spectra_matrix.
    out_file : str
        Full path including the filename where the FITS file will be saved.

    Returns
    -------
    None
        The function writes the spectra into a FITS file at the specified path.
    """
    N_band = len(band_list)
    hdu_list = fits.HDUList()
    hdu_list.append(fits.PrimaryHDU())  # Primary HDU

    for i in range(N_band):
        for j in range(N_band):
            cl_dict = spectra_matrix[i, j]
            cols = [fits.Column(name=key, format='D', array=cl_dict[key]) 
                    for key in ['ell1','ell2','ell_eff','TT','EE','BB','TE','TB','EB']]
            hdu = fits.BinTableHDU.from_columns(cols)
            hdu.header['BAND_I'] = band_list[i]
            hdu.header['BAND_J'] = band_list[j]
            hdu.name = f'{band_list[i]}_{band_list[j]}'
            hdu_list.append(hdu)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    hdu_list.writeto(out_file, overwrite=True)
    print(f"Saved power spectra matrix to {out_file}")


def get_noise_filename(base_name, sim_number):
    """
    Generate the noise filename for a given simulation number based on the base filename.
    
    Parameters
    ----------
    base_name : str
        Example filename for simulation #1 (stored in data['...']['noise_simulation_1']).
    sim_number : int
        Simulation number (1-based).
    
    Returns
    -------
    str
        Filename for the requested simulation.
    """
    # Find the last number with at least 3 digits
    matches = list(re.finditer(r"\d{3,}", base_name))
    if not matches:
        raise ValueError(f"No suitable number found in {base_name}")
    
    match = matches[-1]  # Last numbers
    num_str = match.group(0)
    width = len(num_str)

    # Reference number
    base_num = int(num_str)
    new_num = base_num + (sim_number - 1)
    
    new_num_str = str(new_num).zfill(width)
    return base_name[:match.start()] + new_num_str + base_name[match.end():]


def average_and_std_spectra(data, spectra_dict, band_list, mask, b,
                                 use_white_noise=False, n_sim=100, only_noise=False,
                                 workspaces=None, lmax=None,
                                 capture_sims=False, capture_keys=None):
    """
    Compute the average and standard deviation of auto- and cross-spectra
    over multiple noise realizations using precomputed NaMaster workspaces.

    This function returns a unified dictionary structure:
        spectra['band_i_band_j']['TT']['MEAN']
        spectra['band_i_band_j']['TT']['STD']
    If `capture_sims=True`, it also stores per-simulation arrays with shape
    (n_sim, n_bins) under a subkey (default 'SIMS') for selected cl keys.

    Parameters
    ----------
    data : dict
        Dictionary with experiment and band information.
    spectra_dict : dict
        Initial spectra already computed (e.g. from read_spectra_from_fits_dict).
        Keys are strings "band_i_band_j".
    band_list : list of str
        List of frequency bands to compute spectra for.
    mask : array
        Healpy mask array (0=masked, 1=unmasked).
    b : nmt.NmtBin
        NaMaster binning object.
    use_white_noise : bool, optional
        If True, use white noise simulations. Default is False (full noise).
    n_sim : int, optional
        Number of noise realizations to average over. Default is 100.
    only_noise : bool, optional
        If True, compute spectra using only noise maps.
    workspaces : dict or None
        Dictionary of precomputed NaMaster workspaces for each field combination.
    lmax : int or None
        Maximum multipole for lmax_sht parameter.
    capture_sims : bool, optional
        If True, also capture and return per-simulation arrays under non-MEAN/STD
        subkeys so they can be written by `save_sims_to_fits`. Default False.
    capture_keys : list[str] or None
        Which spectra keys to capture per-sim for. Defaults to ['TT','EE','BB','TE','TB','EB'].

    Returns
    -------
    avg_std_dict : dict
        Dictionary with averaged spectra and standard deviations in the format:
        avg_std_dict['band_i_band_j']['TT']['MEAN'/'STD'].
    """
    # Initialize accumulators for sums and squared sums
    sum_dict = {}
    sumsq_dict = {}
    for key, cl_dict in spectra_dict.items():
        sum_dict[key] = {k: np.zeros_like(v, dtype=float) for k, v in cl_dict.items()}
        sumsq_dict[key] = {k: np.zeros_like(v, dtype=float) for k, v in cl_dict.items()}

    # Optional per-simulation storage
    sims_store = None
    sim_subkey = 'SIMS'
    if capture_sims:
        if capture_keys is None:
            capture_keys = ['TT', 'EE', 'BB', 'TE', 'TB', 'EB']
        # Determine n_bins from any available key in spectra_dict
        # Assume common binning for all pairs and cl keys
        first_pair_key = next(iter(spectra_dict.keys()))
        first_cl = spectra_dict[first_pair_key]
        # prefer EE if present, else take any of capture_keys that exists
        ref_key = next((k for k in ['EE','BB','TT','TE','TB','EB'] if k in first_cl), None)
        if ref_key is None:
            # fallback to 'ell_eff'
            ref_key = 'ell_eff'
        n_bins = int(np.asarray(first_cl[ref_key]).size)

        sims_store = {}
        for i, band_i in enumerate(band_list):
            for j, band_j in enumerate(band_list):
                band_pair = f"{band_i}_{band_j}"
                sims_store[band_pair] = {k: np.zeros((n_sim, n_bins), dtype=float) for k in capture_keys}

    # Loop over noise realizations
    for sim in tqdm(range(1, n_sim + 1), desc="Simulations"):
        spectra_sim = compute_all_power_spectra(
            data, band_list, mask, b,
            use_simulated_maps=True,
            use_white_noise=use_white_noise,
            noise_realization=sim,
            only_noise=only_noise,
            workspaces=workspaces,
            lmax=lmax
        )

        # Accumulate for each band pair
        for i, band_i in enumerate(band_list):
            for j, band_j in enumerate(band_list):
                key = f"{band_i}_{band_j}"
                for cl_key, arr in spectra_sim[i, j].items():
                    arr = np.array(arr, dtype=float)
                    sum_dict[key][cl_key] += arr
                    sumsq_dict[key][cl_key] += arr**2
                    # Optionally capture per-simulation arrays for selected keys
                    if capture_sims and sims_store is not None and cl_key in sims_store[key]:
                        sims_store[key][cl_key][sim - 1, :] = arr


    # Build final dictionary with MEAN and STD
    avg_std_dict = {}
    for key in sum_dict:
        avg_std_dict[key] = {}
        for cl_key in sum_dict[key]:
            mean = sum_dict[key][cl_key] / n_sim
            var = (sumsq_dict[key][cl_key] / n_sim) - mean**2
            var = np.where(var < 0, 0, var)  # Avoid negative variance due to numerical errors
            std = np.sqrt(var)
            avg_std_dict[key][cl_key] = {"MEAN": mean, "STD": std}
        # Attach per-simulation arrays only for CL keys (avoid ell arrays)
        if capture_sims and sims_store is not None:
            for cl_key in sims_store[key].keys():
                if cl_key not in avg_std_dict[key]:
                    avg_std_dict[key][cl_key] = {}
                avg_std_dict[key][cl_key][sim_subkey] = sims_store[key][cl_key]

    return avg_std_dict


def save_avg_std_to_fits(avg_std_dict, band_list, out_file, use_white_noise=False):
    """
    Save average and standard deviation spectra (dict format) into a FITS file.

    Parameters
    ----------
    avg_std_dict : dict
        Dict with structure:
          spectra['band_i_band_j']['TT']['MEAN']
          spectra['band_i_band_j']['TT']['STD']
    band_list : list of str
        Ordered list of frequency bands.
    out_file : str
        Full path including filename where the FITS will be saved.
    use_white_noise : bool, optional
        If True, '_wn' will be appended before the '.fits' extension in the file name.
    """
    hdu_list = fits.HDUList()
    hdu_list.append(fits.PrimaryHDU())

    # Loop over all band pairs in band_list
    for band_i in band_list:
        for band_j in band_list:
            key = f"{band_i}_{band_j}"
            spec_dict = avg_std_dict[key]

            cols = []
            for cl_key in ['ell1', 'ell2', 'ell_eff', 'TT', 'EE', 'BB', 'TE', 'TB', 'EB']:
                cols.append(fits.Column(name=f"{cl_key}_MEAN", format="D", array=spec_dict[cl_key]['MEAN']))
                cols.append(fits.Column(name=f"{cl_key}_STD", format="D", array=spec_dict[cl_key]['STD']))

            # Create HDU for this band pair
            hdu = fits.BinTableHDU.from_columns(cols)
            hdu.header['BAND_I'] = band_i
            hdu.header['BAND_J'] = band_j
            hdu.name = key
            hdu_list.append(hdu)

    # Ajustar nombre si use_white_noise=True
    if use_white_noise:
        out_file = _with_suffix_before_fits(out_file, '_wn')

    # Crear directorio si no existe
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    hdu_list.writeto(out_file, overwrite=True)
    print(f"Saved avg+std spectra to {out_file}")


def read_spectra_from_fits(path_fits, band_list, use_white_noise=False):
    """
    Read power spectra from a FITS file into a dictionary.

    The function automatically detects whether the FITS file contains:
    - Simple spectra (columns: 'ell1','ell2','ell_eff','TT','EE','BB','TE','TB','EB')
    - Averaged spectra with errors (columns: 'TT_MEAN','TT_STD', ...)

    Depending on the case, the returned dictionary has one of the following forms:

    Case 1: simple spectra
        spectra['band_i_band_j']['TT'] -> array

    Case 2: average+std spectra
        spectra['band_i_band_j']['TT']['MEAN'] -> array
        spectra['band_i_band_j']['TT']['STD']  -> array

    Parameters
    ----------
    path_fits : str
        Path to the FITS file containing the spectra (without '_wn' suffix).
    band_list : list of str
        Ordered list of frequency bands.
    use_white_noise : bool, optional
        If True, '_wn' will be appended to the filename **only** if the file
        contains average+std spectra.

    Returns
    -------
    spectra_dict : dict
        Dictionary with spectra for all band pairs.
    """
    # Try to open file — if use_white_noise=True, assume it's an avg+std file
    if use_white_noise:
        path_fits = _with_suffix_before_fits(path_fits, '_wn')

    if not os.path.exists(path_fits):
        raise FileNotFoundError(f"FITS file not found: {path_fits}")

    spectra_dict = {}

    with fits.open(path_fits) as hdul:
        for band_i in band_list:
            for band_j in band_list:
                key = f"{band_i}_{band_j}"
                hdu = next((h for h in hdul[1:] if h.name == key), None)
                if hdu is None:
                    raise ValueError(f"HDU {key} not found in {path_fits}")

                colnames = [c.upper() for c in hdu.data.names]
                spec_dict = {}

                # Detect case automatically
                if any(name.endswith("_MEAN") for name in colnames):
                    # Case 2: avg+std spectra
                    for cl_key in ['ell1','ell2','ell_eff','TT','EE','BB','TE','TB','EB']:
                        spec_dict[cl_key] = {
                            "MEAN": hdu.data[f"{cl_key}_MEAN"],
                            "STD":  hdu.data[f"{cl_key}_STD"],
                        }
                else:
                    # Case 1: simple spectra (no _wn suffix logic)
                    for cl_key in ['ell1','ell2','ell_eff','TT','EE','BB','TE','TB','EB']:
                        spec_dict[cl_key] = hdu.data[cl_key]

                spectra_dict[key] = spec_dict

    return spectra_dict

def read_spectra_from_fits(path_fits, band_list, use_white_noise=False):
    """
    Read power spectra from a FITS file into a dictionary.

    The function automatically detects whether the FITS file contains:
    - Simple spectra (columns: 'ell1','ell2','ell_eff','TT','EE','BB','TE','TB','EB')
    - Averaged spectra with errors (columns: 'TT_MEAN','TT_STD', ...)

    Depending on the case, the returned dictionary has one of the following forms:

    Case 1: simple spectra
        spectra['band_i_band_j']['TT'] -> array

    Case 2: average+std spectra
        spectra['band_i_band_j']['TT']['MEAN'] -> array
        spectra['band_i_band_j']['TT']['STD']  -> array

    Parameters
    ----------
    path_fits : str
        Path to the FITS file containing the spectra (without '_wn' suffix).
    band_list : list of str
        Ordered list of frequency bands.
    use_white_noise : bool, optional
        If True, '_wn' will be appended to the filename **only** if the file
        contains average+std spectra.

    Returns
    -------
    spectra_dict : dict
        Dictionary with spectra for all band pairs.
    """
    # Try to open file — if use_white_noise=True, assume it's an avg+std file
    if use_white_noise:
        path_fits = _with_suffix_before_fits(path_fits, '_wn')

    if not os.path.exists(path_fits):
        raise FileNotFoundError(f"FITS file not found: {path_fits}")

    spectra_dict = {}

    with fits.open(path_fits) as hdul:
        for band_i in band_list:
            for band_j in band_list:
                key = f"{band_i}_{band_j}"
                hdu = next((h for h in hdul[1:] if h.name == key), None)
                if hdu is None:
                    raise ValueError(f"HDU {key} not found in {path_fits}")

                colnames = [c.upper() for c in hdu.data.names]
                spec_dict = {}

                # Detect case automatically
                if any(name.endswith("_MEAN") for name in colnames):
                    # Case 2: avg+std spectra
                    for cl_key in ['ell1','ell2','ell_eff','TT','EE','BB','TE','TB','EB']:
                        spec_dict[cl_key] = {
                            "MEAN": hdu.data[f"{cl_key}_MEAN"],
                            "STD":  hdu.data[f"{cl_key}_STD"],
                        }
                else:
                    # Case 1: simple spectra (no _wn suffix logic)
                    for cl_key in ['ell1','ell2','ell_eff','TT','EE','BB','TE','TB','EB']:
                        spec_dict[cl_key] = hdu.data[cl_key]

                spectra_dict[key] = spec_dict

    return spectra_dict


def read_sims_from_fits(path_fits):
    """
    Read per-simulation spectra HDUs appended to a FITS file by
    `save_avg_std_to_fits()` and return a structured dictionary.

    The function looks for HDUs whose name includes the suffix '_SIMS'
    (or which include headers 'BAND_I'/'BAND_J' and 'CLKEY'/'SIMKEY') and
    expects columns named 'SIM_1', 'SIM_2', ..., each column being an array
    over bins. It returns a dictionary with the structure:

        sims[band_pair]["{cl_key}__{subkey}"] = ndarray shape (n_sim, n_bins)

    where `band_pair` is like '11_11', `cl_key` is e.g. 'EE', and `subkey`
    is the name used when saving (e.g. 'SIMS').

    Parameters
    ----------
    path_fits : str
        Path to the FITS file containing appended simulation HDUs.

    Returns
    -------
    sims_dict : dict
        Nested dictionary with per-simulation arrays.
    """
    if not os.path.exists(path_fits):
        raise FileNotFoundError(f"FITS file not found: {path_fits}")

    sims = {}
    with fits.open(path_fits) as hdul:
        for h in hdul[1:]:
            # Heuristic: HDU names that end with '_SIMS' are per-sim tables
            name = h.name if hasattr(h, 'name') else ''
            is_sims_hdu = False
            if isinstance(name, str) and name.endswith('_SIMS'):
                is_sims_hdu = True
            # Also consider header markers
            hdr = h.header
            if not is_sims_hdu:
                if 'SIMKEY' in hdr or ('BAND_I' in hdr and 'CLKEY' in hdr):
                    is_sims_hdu = True

            if not is_sims_hdu:
                continue

            # Attempt to parse band_pair, cl_key and subkey from the HDU name
            band_pair = None
            cl_key = None
            subkey = None
            if isinstance(name, str) and '__' in name:
                parts = name.split('__')
                if len(parts) >= 3:
                    band_pair = parts[0]
                    cl_key = parts[1]
                    subpart = parts[2]
                    subkey = subpart.replace('_SIMS', '')

            # Fallback to header values if parsing failed
            if band_pair is None or cl_key is None:
                bi = hdr.get('BAND_I', None)
                bj = hdr.get('BAND_J', None)
                if bi is not None and bj is not None:
                    band_pair = f"{bi}_{bj}"
                cl_key = cl_key or hdr.get('CLKEY', 'UNKNOWN')
                subkey = subkey or hdr.get('SIMKEY', 'SIMS')

            if not hasattr(h, 'data') or h.data is None:
                continue
            colnames = list(h.data.names) if h.data.names is not None else []
            colnames_upper = [c.upper() for c in colnames]

            # Grouped layout: one HDU per (band_pair, subkey), with columns TT/EE/... each (n_bins, n_sim)
            known_cls = ['TT', 'EE', 'BB', 'TE', 'TB', 'EB']
            if any(k in colnames_upper for k in known_cls):
                for cl in known_cls:
                    if cl not in colnames_upper:
                        continue
                    try:
                        colname = colnames[colnames_upper.index(cl)]
                        sims_by_bin = np.asarray(h.data[colname])  # (n_bins, n_sim)
                    except Exception:
                        continue
                    if sims_by_bin.ndim == 2:
                        sims.setdefault(band_pair, {})[f"{cl}__{subkey}"] = sims_by_bin.T
                continue

            # Per-(band_pair, cl_key) compact layout: a single vector column named 'SIMS'
            if 'SIMS' in colnames_upper:
                try:
                    sims_by_bin = np.asarray(h.data[colnames[colnames_upper.index('SIMS')]])  # (n_bins, n_sim)
                except Exception:
                    continue
                if sims_by_bin.ndim == 2:
                    sims.setdefault(band_pair, {})[f"{cl_key}__{subkey}"] = sims_by_bin.T
                continue

            # Legacy layout: one column per simulation (SIM_1 ... SIM_N)
            sim_cols = [c for c in colnames if c.upper().startswith('SIM_')]
            if len(sim_cols) == 0:
                continue
            try:
                sims_arr = np.vstack([h.data[c] for c in sim_cols])
            except Exception:
                continue
            sims.setdefault(band_pair, {})[f"{cl_key}__{subkey}"] = sims_arr

    return sims


def build_covariance_matrix_from_sims(
    path_sims_fits,
    band_list,
    modes=['EE', 'BB'],
    ell_min=30,
    ell_max=200,
    band_pairs='all',
    correlated_pairs=None
):
    """
    Build a full covariance matrix from simulation FITS files, accounting for
    correlations between specified band pairs.

    Parameters
    ----------
    path_sims_fits : str
        Path to FITS file containing per-simulation spectra (from save_sims_to_fits).
    band_list : list of str
        List of band names (e.g., ['11', '13', '17', '19', '23', ...]).
    modes : list of str
        List of modes to include (e.g., ['EE', 'BB']).
    ell_min : int
        Minimum multipole to include.
    ell_max : int
        Maximum multipole to include.
    band_pairs : str or list
        Either 'all' or a list of specific band pairs (e.g., ['11_13', '17_19']).
    correlated_pairs : list of tuples, optional
        List of band pair tuples that should have cross-covariance computed.
        E.g., [('11', '13'), ('17', '19')] for QUIJOTE correlations.
        If None, only diagonal covariances are computed.

    Returns
    -------
    cov_matrix : ndarray
        Full covariance matrix of shape (n_data, n_data) where
        n_data = n_bins × n_band_pairs × n_modes.
    data_index : dict
        Dictionary mapping (band_pair, mode, bin_idx) to the index in the covariance matrix.
    """
    # Read simulations
    sims_dict = read_sims_from_fits(path_sims_fits)
    
    # Build list of band pairs
    if band_pairs == 'all':
        bp_list = []
        for i, bi in enumerate(band_list):
            for j, bj in enumerate(band_list):
                if i <= j:
                    bp_list.append(f"{bi}_{bj}")
    else:
        bp_list = band_pairs
    
    # Get ell values and determine bin range from first available spectrum
    sample_bp = bp_list[0]
    sample_mode = modes[0]
    sample_key = f"{sample_mode}__SIMS"
    
    if sample_bp not in sims_dict or sample_key not in sims_dict[sample_bp]:
        raise ValueError(f"Cannot find simulations for {sample_bp}, {sample_mode}")
    
    # Get ell_eff from the corresponding spectrum file
    # We need to read the ell values from somewhere - typically from the spectra file
    # For now, we'll infer bin indices from the simulation array shape
    sims_sample = sims_dict[sample_bp][sample_key]  # shape: (n_sim, n_bins)
    n_bins_total = sims_sample.shape[1]
    
    # We need ell values to filter by ell_min/ell_max
    # This requires reading from the original spectra file
    # For simplicity, we'll use all bins and let the user filter externally
    # Or we can add ell_eff as a parameter
    
    # Build index mapping
    data_index = {}
    idx = 0
    for mode in modes:
        for bp in bp_list:
            for bin_idx in range(n_bins_total):
                data_index[(bp, mode, bin_idx)] = idx
                idx += 1
    
    n_data = idx
    cov_matrix = np.zeros((n_data, n_data))
    
    # Compute diagonal and auto-covariances
    for mode in modes:
        for bp in bp_list:
            sim_key = f"{mode}__SIMS"
            if bp not in sims_dict or sim_key not in sims_dict[bp]:
                continue
            
            sims = sims_dict[bp][sim_key]  # shape: (n_sim, n_bins)
            
            # Compute covariance across simulations for this band-pair and mode
            # cov[i,j] = < (sim_i - mean_i) * (sim_j - mean_j) >
            for bin_i in range(n_bins_total):
                for bin_j in range(n_bins_total):
                    idx_i = data_index[(bp, mode, bin_i)]
                    idx_j = data_index[(bp, mode, bin_j)]
                    
                    # Covariance between bins within same band-pair
                    cov_matrix[idx_i, idx_j] = np.cov(sims[:, bin_i], sims[:, bin_j])[0, 1]
    
    # Compute cross-covariances for correlated pairs
    if correlated_pairs is not None:
        for (band_i, band_j) in correlated_pairs:
            # Find all band pairs involving these bands
            for bp1 in bp_list:
                for bp2 in bp_list:
                    # Check if this pair involves the correlated bands
                    bp1_bands = bp1.split('_')
                    bp2_bands = bp2.split('_')
                    
                    # Correlate if they share one of the correlated bands
                    should_correlate = False
                    if (band_i in bp1_bands and band_j in bp2_bands) or \
                       (band_j in bp1_bands and band_i in bp2_bands) or \
                       (band_i in bp1_bands and band_i in bp2_bands) or \
                       (band_j in bp1_bands and band_j in bp2_bands):
                        should_correlate = True
                    
                    if not should_correlate or bp1 == bp2:
                        continue
                    
                    # Compute cross-covariance
                    for mode in modes:
                        sim_key = f"{mode}__SIMS"
                        
                        if bp1 not in sims_dict or sim_key not in sims_dict[bp1]:
                            continue
                        if bp2 not in sims_dict or sim_key not in sims_dict[bp2]:
                            continue
                        
                        sims1 = sims_dict[bp1][sim_key]
                        sims2 = sims_dict[bp2][sim_key]
                        
                        for bin_i in range(n_bins_total):
                            for bin_j in range(n_bins_total):
                                idx_i = data_index[(bp1, mode, bin_i)]
                                idx_j = data_index[(bp2, mode, bin_j)]
                                
                                # Cross-covariance between different band-pairs
                                cov_val = np.cov(sims1[:, bin_i], sims2[:, bin_j])[0, 1]
                                cov_matrix[idx_i, idx_j] = cov_val
                                cov_matrix[idx_j, idx_i] = cov_val  # Symmetric
    
    return cov_matrix, data_index


def extract_covariance_for_mcmc(
    cov_matrix,
    data_index,
    fit_data,
    modes=['EE']
):
    """
    Extract the relevant sub-matrix of the covariance for MCMC fitting.

    Parameters
    ----------
    cov_matrix : ndarray
        Full covariance matrix from build_covariance_matrix_from_sims.
    data_index : dict
        Index mapping from build_covariance_matrix_from_sims.
    fit_data : dict
        Dictionary from prepare_mcmc_data containing 'datasets', 'ell', etc.
    modes : list of str
        Modes being fitted (e.g., ['EE'] or ['EE', 'BB']).

    Returns
    -------
    cov_mcmc : ndarray
        Covariance matrix for the MCMC data (n_mcmc_data, n_mcmc_data).
    """
    datasets = fit_data['datasets']
    ell = fit_data.get('ell_eff', fit_data.get('ell', None))
    
    if ell is None:
        raise KeyError("fit_data must contain either 'ell_eff' or 'ell' key")
    
    # Build list of indices corresponding to the MCMC data
    mcmc_indices = []
    
    for dataset in datasets:
        # Handle both old and new dataset key naming conventions
        if 'pair' in dataset:
            band_pair = dataset['pair']  # New format: 'pair' key
        elif 'band_i' in dataset and 'band_j' in dataset:
            band_pair = f"{dataset['band_i']}_{dataset['band_j']}"  # Old format
        else:
            raise KeyError("Dataset must contain either 'pair' or 'band_i'/'band_j' keys")
        
        # Get mode
        if 'mode' in dataset:
            mode = dataset['mode']  # New format: 'mode' key
        elif 'cl_key' in dataset:
            mode = dataset['cl_key']  # Old format
        else:
            raise KeyError("Dataset must contain either 'mode' or 'cl_key' key")
        
        if mode not in modes:
            continue
        
        # Each dataset contributes len(ell) data points
        for bin_idx in range(len(ell)):
            key = (band_pair, mode, bin_idx)
            if key in data_index:
                mcmc_indices.append(data_index[key])
    
    # Extract sub-matrix
    n_mcmc = len(mcmc_indices)
    cov_mcmc = np.zeros((n_mcmc, n_mcmc))
    
    for i, idx_i in enumerate(mcmc_indices):
        for j, idx_j in enumerate(mcmc_indices):
            cov_mcmc[i, j] = cov_matrix[idx_i, idx_j]
    
    return cov_mcmc


def _compute_physical_correction_factor(pair, cl_key, ell_eff, data_dict, nside):
    """
    Compute the per-ell-bin physical correction factor that maps a raw
    simulation C_l (beam-convolved, mK²_CMB) into the corrected domain
    (beam-deconvolved, K²_RJ) used by the MCMC.

    factor = unit_factor / phys_factor

    where phys_factor = beam1 * beam2 * wpix1 * wpix2
    and   unit_factor = uc1 * uc2  (K_CMB -> K_RJ conversion per band).

    Parameters
    ----------
    pair : str
        Band pair string, e.g. '11_13'.
    cl_key : str
        Spectrum type, e.g. 'EE', 'BB'.
    ell_eff : array
        Effective multipoles.
    data_dict : dict
        The full data dictionary with beam/freq information per experiment/band.
    nside : int
        HEALPix NSIDE of the maps.

    Returns
    -------
    factor : ndarray
        Correction factor array with the same length as ell_eff.
    """
    band1, band2 = pair.split('_')

    # Beam transfer functions
    beam1 = get_beam_for_band(band1, data_dict, ell_eff)
    beam2 = get_beam_for_band(band2, data_dict, ell_eff)

    comp_map = {
        'TT': ('T', 'T'), 'EE': ('E', 'E'), 'BB': ('B', 'B'),
        'TE': ('T', 'E'), 'TB': ('T', 'B'), 'EB': ('E', 'B'),
    }
    c1, c2 = comp_map.get(cl_key, ('T', 'T'))
    bl1 = np.asarray(beam1[c1], dtype=float)
    bl2 = np.asarray(beam2[c2], dtype=float)

    # Pixel window
    wpix = hp.pixwin(nside)
    wp = np.interp(ell_eff, np.arange(len(wpix)), wpix)

    phys_factor = bl1 * bl2 * wp * wp
    # Protect against zeros / negatives
    phys_factor = np.where(phys_factor > 0, phys_factor, np.nan)

    # Unit conversion (K_CMB -> K_RJ) per band
    def _get_uc(band):
        for exp in data_dict:
            if band in data_dict[exp]:
                freq = data_dict[exp][band].get('freq')
                try:
                    nuGHz = freq.to('GHz').value
                except Exception:
                    nuGHz = float(freq)
                # Use HFI-specific conversion for Planck HFI bands
                is_planck = str(exp).lower() == 'planck'
                hfi_band_set = {'100', '143', '217', '353', '545', '857'}
                if is_planck and str(band) in hfi_band_set:
                    try:
                        uc_hfi = planck_uc_hfi(use_bps=True)
                        hfi_order = [100, 143, 217, 353, 545, 857]
                        idx = hfi_order.index(int(float(band)))
                        return float(uc_hfi[idx])
                    except Exception:
                        return float(cmb_unit_conversion(nuGHz, 'KCMB2KRJ'))
                else:
                    return float(cmb_unit_conversion(nuGHz, 'KCMB2KRJ'))
        raise ValueError(f"Band '{band}' not found in data_dict.")

    uc1 = _get_uc(band1)
    uc2 = _get_uc(band2)
    unit_factor = uc1 * uc2

    return unit_factor / phys_factor


def build_block_diagonal_cov_inv(
    path_sims_fits,
    fit_data,
    modes=['EE'],
    quijote_bands_11_13=['11', '13'],
    quijote_bands_17_19=['17', '19'],
    n_sims=100,
    path_noise_sims=None,
    data_dict=None,
    nside=512
):
    """
    Build block-diagonal inverse covariance matrix for MCMC.

    Only QUIJOTE 11-13 and 17-19 get full covariance blocks (off-diagonal
    correlations from shared-horn noise).  Everything else uses diagonal
    errors (1/error²).

    The per-simulation spectra stored on disk are *raw* (beam-convolved,
    mK²_CMB).  Before computing the covariance, each simulation is
    transformed to the corrected domain (beam-deconvolved, K²_RJ) using
    the same physical factors applied in ``correct_power_spectra``.  This
    ensures the off-diagonal correlation coefficients are correct.

    Parameters
    ----------
    path_sims_fits : str
        Path to simulation FITS file (Sky + Noise).
    fit_data : dict
        Output from prepare_mcmc_data.
    modes : list of str
        Modes to include (e.g., ['EE']).
    quijote_bands_11_13 : list
        QUIJOTE bands to correlate (default: ['11', '13']).
    quijote_bands_17_19 : list
        QUIJOTE bands to correlate (default: ['17', '19']).
    n_sims : int
        Number of simulations (default: 100).
    path_noise_sims : str, optional
        Path to noise-only simulation FITS file.
        If provided, noise covariance will be added to the sky+noise covariance:
        Cov_total = Cov(Sky+Noise) + Cov(Noise)
        This matches the error propagation in correct_power_spectra for
        auto-spectra and correlated cross-spectra.
    data_dict : dict, optional
        The full data dictionary with beam/freq info per experiment/band.
        Required for applying physical corrections to each simulation.
        If None, falls back to the (less accurate) scalar rescaling.
    nside : int, optional
        HEALPix NSIDE of the maps (default: 512).

    Returns
    -------
    block_info : dict
        Dictionary with keys:
        - 'indices_11_13', 'indices_17_19': global indices in y_all
        - 'cov_inv_11_13', 'cov_inv_17_19': pre-extracted inverse covariance blocks
        Can be passed directly to run_mcmc(..., cov_matrix=block_info).
    """
    # Read simulations
    print("Reading simulations (Sky + Noise)...")
    sims_dict = read_sims_from_fits(path_sims_fits)

    noise_sims_dict = None
    if path_noise_sims is not None:
        print("Reading simulations (Noise only)...")
        noise_sims_dict = read_sims_from_fits(path_noise_sims)

    datasets = fit_data['datasets']
    y_all = fit_data['y_all']
    yerr_all = fit_data['yerr_all']
    ell_eff = fit_data['ell_eff']
    n_data = len(y_all)

    # Initialize as diagonal (1/error²)
    print(f"Building block-diagonal inverse covariance matrix ({n_data} x {n_data})...")
    cov_inv = np.diag(1.0 / yerr_all**2)

    # Helper: get indices for band pairs where BOTH bands are in band_list
    def get_indices_for_bands(band_list, allowed_modes):
        """Get data indices for pairs where BOTH bands are in band_list and mode is allowed."""
        indices = []
        dataset_info = []

        if isinstance(allowed_modes, str):
            allowed_modes = [allowed_modes]

        for i, dataset in enumerate(datasets):
            pair = dataset.get('pair', f"{dataset.get('band_i', '')}_{dataset.get('band_j', '')}")
            ds_mode = dataset.get('mode', dataset.get('cl_key', ''))

            if ds_mode not in allowed_modes:
                continue

            try:
                b1, b2 = pair.split('_')
                if b1 in band_list and b2 in band_list:
                    start, stop = dataset['slice']
                    idx_range = list(range(start, stop))
                    indices.extend(idx_range)
                    dataset_info.append({
                        'pair': pair,
                        'indices': idx_range,
                        'dataset_idx': i,
                        'mode': ds_mode
                    })
            except Exception:
                continue

        return indices, dataset_info

    # ------------------------------------------------------------------
    # Helper: build and invert one covariance block
    # ------------------------------------------------------------------
    def _process_block(block_label, quijote_bands):
        """Build covariance block, invert it, and return (indices, cov_inv_block)."""
        idx_list, info_list = get_indices_for_bands(quijote_bands, modes)

        if len(idx_list) == 0:
            print(f"\n  {block_label}: no matching data points found.")
            return np.array([], dtype=int), None

        print(f"\nProcessing {block_label} covariance block...")
        print(f"  Found {len(idx_list)} data points involving bands {quijote_bands}")
        print(f"  Pairs: {[d['pair'] for d in info_list]}")

        n_block = len(idx_list)

        # Create mapping: global index -> (dataset_info, local_bin_index)
        idx_to_dataset_map = {}
        for ds_info in info_list:
            for local_bin, global_idx in enumerate(ds_info['indices']):
                idx_to_dataset_map[global_idx] = (ds_info, local_bin)

        # ----- Gather raw simulation data for this block -----
        n_sims_actual = None
        block_sims = np.zeros((n_block, n_sims))
        block_noise_sims = None
        if noise_sims_dict is not None:
            block_noise_sims = np.zeros((n_block, n_sims))

        for i in range(n_block):
            global_idx_i = idx_list[i]
            info_i, local_bin_i = idx_to_dataset_map[global_idx_i]
            pair_i = info_i['pair']
            mode_i = info_i['mode']
            sim_key = f"{mode_i}__SIMS"

            if pair_i in sims_dict and sim_key in sims_dict[pair_i]:
                sims_data = sims_dict[pair_i][sim_key]
                if n_sims_actual is None:
                    n_sims_actual = sims_data.shape[0]
                    if n_sims_actual != n_sims:
                        block_sims = np.zeros((n_block, n_sims_actual))
                        if block_noise_sims is not None:
                            block_noise_sims = np.zeros((n_block, n_sims_actual))

                block_sims[i, :] = sims_data[:, local_bin_i]

                if (block_noise_sims is not None
                        and pair_i in noise_sims_dict
                        and sim_key in noise_sims_dict[pair_i]):
                    noise_data = noise_sims_dict[pair_i][sim_key]
                    n_noise_sims = noise_data.shape[0]
                    limit = min(n_sims_actual, n_noise_sims)
                    block_noise_sims[i, :limit] = noise_data[:limit, local_bin_i]
            else:
                if n_sims_actual is None:
                    n_sims_actual = n_sims
                    block_sims = np.zeros((n_block, n_sims_actual))
                    if block_noise_sims is not None:
                        block_noise_sims = np.zeros((n_block, n_sims_actual))

        if n_sims_actual is None or n_sims_actual < 2:
            print(f"  Not enough simulations ({n_sims_actual}), keeping diagonal.")
            return np.array(idx_list, dtype=int), None

        # ----- Apply physical corrections to each simulation -----
        # The raw sims are in mK²_CMB, beam-convolved.  We need to
        # transform them to K²_RJ, beam-deconvolved — the same domain
        # as y_all and yerr_all.
        #
        # For each data point i belonging to (pair, mode, ell_bin), the
        # correction factor is:
        #     f_i = (uc_band1 * uc_band2) / (beam1 * beam2 * wpix1 * wpix2)
        # evaluated at that ell_bin.
        #
        # Then corrected_sim[i, s] = raw_sim[i, s] * f_i
        # and   Cov_corrected = diag(f) @ Cov_raw @ diag(f)

        if data_dict is not None:
            print("  Applying physical corrections (beam, pixel, unit) to simulations...")
            # Pre-compute correction factors per (pair, mode) to avoid redundant I/O
            _phys_cache = {}
            correction_vector = np.ones(n_block)
            for i in range(n_block):
                global_idx_i = idx_list[i]
                info_i, local_bin_i = idx_to_dataset_map[global_idx_i]
                pair_i = info_i['pair']
                mode_i = info_i['mode']
                cache_key = (pair_i, mode_i)
                if cache_key not in _phys_cache:
                    _phys_cache[cache_key] = _compute_physical_correction_factor(
                        pair_i, mode_i, ell_eff, data_dict, nside
                    )
                correction_vector[i] = _phys_cache[cache_key][local_bin_i]
        else:
            # Fallback: use yerr_all / std_raw ratio (less accurate — does
            # not correctly separate sky+noise and noise contributions)
            print("  WARNING: data_dict not provided; falling back to scalar rescaling.")
            print("           Pass data_dict and nside for exact physical corrections.")
            correction_vector = np.ones(n_block)
            for i in range(n_block):
                global_idx = idx_list[i]
                std_raw_i = np.std(block_sims[i, :], ddof=1)
                if std_raw_i > 0:
                    correction_vector[i] = yerr_all[global_idx] / std_raw_i
                else:
                    correction_vector[i] = 1.0

        # Apply correction factors
        block_sims_corr = block_sims * correction_vector[:, np.newaxis]
        block_noise_sims_corr = None
        if block_noise_sims is not None:
            block_noise_sims_corr = block_noise_sims * correction_vector[:, np.newaxis]

        # ----- Compute covariance in the corrected domain -----
        cov_block = np.cov(block_sims_corr, ddof=1)
        if block_noise_sims_corr is not None:
            print("  Adding noise covariance term...")
            cov_noise = np.cov(block_noise_sims_corr, ddof=1)
            cov_block = cov_block + cov_noise

        # ----- Diagnostic: compare diagonal to yerr_all² -----
        print("  Diagonal consistency check (sqrt(cov_ii) vs yerr_all):")
        for i in range(min(n_block, 6)):
            global_idx = idx_list[i]
            cov_diag_i = np.sqrt(cov_block[i, i]) if cov_block[i, i] > 0 else 0.0
            ratio = cov_diag_i / yerr_all[global_idx] if yerr_all[global_idx] > 0 else np.nan
            info_i, local_bin_i = idx_to_dataset_map[global_idx]
            print(f"    [{info_i['pair']}, bin {local_bin_i}]  "
                  f"sqrt(Cov_ii)={cov_diag_i:.4e}  yerr={yerr_all[global_idx]:.4e}  "
                  f"ratio={ratio:.4f}")
        if n_block > 6:
            print(f"    ... ({n_block - 6} more entries)")

        # Fill diagonal for any zero entries (missing sims)
        for i in range(n_block):
            if cov_block[i, i] == 0:
                global_idx_i = idx_list[i]
                cov_block[i, i] = yerr_all[global_idx_i]**2

        # ----- Hartlap correction for finite-simulation bias -----
        # The inverse of a sample covariance estimated from N_sim
        # simulations is biased.  The Hartlap factor corrects this:
        #     C^{-1}_unbiased = (N_sim - n_block - 2) / (N_sim - 1) * C^{-1}_sample
        hartlap = (n_sims_actual - n_block - 2) / (n_sims_actual - 1)
        if hartlap <= 0:
            print(f"  WARNING: Hartlap factor <= 0 ({hartlap:.3f}). "
                  f"n_sims={n_sims_actual} too small for n_block={n_block}! "
                  f"Keeping diagonal for this block.")
            return np.array(idx_list, dtype=int), None
        print(f"  Hartlap correction factor: {hartlap:.4f} "
              f"(n_sims={n_sims_actual}, n_block={n_block})")

        # ----- Invert block -----
        print(f"  Inverting {n_block}x{n_block} covariance block...")
        try:
            reg = 1e-12 * np.max(np.diag(cov_block))
            cov_block_reg = cov_block + reg * np.eye(n_block)
            cov_block_inv = np.linalg.inv(cov_block_reg) * hartlap

            # Insert into main inverse matrix
            for i, idx_i in enumerate(idx_list):
                for j, idx_j in enumerate(idx_list):
                    cov_inv[idx_i, idx_j] = cov_block_inv[i, j]

            print(f"  ✓ Block inverted and inserted.")
        except np.linalg.LinAlgError:
            print(f"  ✗ Failed to invert block, keeping diagonal.")
            return np.array(idx_list, dtype=int), None

        return np.array(idx_list, dtype=int), cov_block_inv

    # ------------------------------------------------------------------
    # Process both QUIJOTE blocks
    # ------------------------------------------------------------------
    idx_11_13, cov_inv_11_13 = _process_block("QUIJOTE 11-13", quijote_bands_11_13)
    idx_17_19, cov_inv_17_19 = _process_block("QUIJOTE 17-19", quijote_bands_17_19)

    print(f"\n  Block-diagonal inverse covariance matrix built")
    print(f"  Total size: {n_data} x {n_data}")
    print(f"  QUIJOTE 11-13 block: {len(idx_11_13)} x {len(idx_11_13)}")
    print(f"  QUIJOTE 17-19 block: {len(idx_17_19)} x {len(idx_17_19)}")
    print(f"  Diagonal elements: {n_data - len(idx_11_13) - len(idx_17_19)}")

    # Return block information for efficient chi-squared computation
    block_info = {
        'indices_11_13': idx_11_13,
        'indices_17_19': idx_17_19,
        'cov_inv_11_13': cov_inv_11_13,
        'cov_inv_17_19': cov_inv_17_19,
    }

    return block_info


def _with_suffix_before_fits(path, suffix):
    """Insert `suffix` before .fits or .fits.gz (case-insensitive).

    Examples
    --------
    a.fits    -> a{suffix}.fits
    a.fits.gz -> a{suffix}.fits.gz
    """
    p = str(path)
    pl = p.lower()
    if pl.endswith('.fits.gz'):
        return p[:-len('.fits.gz')] + f"{suffix}.fits.gz"
    if pl.endswith('.fits'):
        return p[:-len('.fits')] + f"{suffix}.fits"
    base, ext = os.path.splitext(p)
    return f"{base}{suffix}{ext}"


def save_sims_to_fits(
    avg_std_dict=None,
    band_list=None,
    out_file=None,
    use_white_noise=False,
    sims_npz=None,
    avg_std_out_file=None,
    dtype='float32',
    sims_layout='vector',
    hdu_grouping='by_cl',
):
    """
    Save per-simulation spectra into a new FITS file.

    This function accepts either:
      - an `avg_std_dict` that contains per-simulation 2D arrays under
        avg_std_dict['band_i_band_j'][cl_key][subkey] (subkey != 'MEAN'/'STD'),
      - or a compressed NPZ file (`sims_npz`) with keys like '11_11__EE'
        containing arrays shape (n_sim, n_bins).

        The output FITS can be written in two layouts:
            - hdu_grouping='by_cl' (default): one BinTableHDU per (band_pair, cl_key, subkey)
            - hdu_grouping='by_bandpair': one BinTableHDU per (band_pair, subkey) with
                columns TT/EE/BB/TE/TB/EB, each storing all simulations as a fixed-length
                vector per bin. This significantly reduces FITS header overhead.

    Parameters
    ----------
    avg_std_dict : dict or None
        Dictionary that may contain per-simulation arrays (optional if sims_npz provided).
    band_list : list of str
        Ordered list of bands.
    out_file : str
        Output FITS filename (same naming convention as save_avg_std_to_fits).
    use_white_noise : bool
        If True, append '_wn' before '.fits' (or '.fits.gz') in the filename.
            dtype : {'float32','float64'} or numpy dtype
                    Numeric dtype used for per-simulation spectra. Using float32 typically halves file size.
                sims_layout : {'vector','columns'}
                    - 'vector': store all simulations in a single fixed-length vector column named 'SIMS'
                      with shape (n_bins, n_sim). This is much more compact than one column per simulation.
                    - 'columns': legacy layout with columns SIM_1 ... SIM_N.
                hdu_grouping : {'by_cl','by_bandpair'}
                    Choose the HDU grouping strategy. 'by_bandpair' requires sims_layout='vector'.
    sims_npz : str or None
        Path to an external compressed NPZ with per-sim arrays. If provided, data
        will be read from this NPZ and saved into the FITS. If None, the function
        will inspect `avg_std_dict` for per-sim arrays.

    Returns
    -------
    out_file : str
        Path to the written FITS file.
    """
    if out_file is None:
        raise ValueError("out_file must be provided")
    if band_list is None:
        raise ValueError("band_list must be provided")

    sims_layout = str(sims_layout).lower().strip()
    if sims_layout not in ('vector', 'columns'):
        raise ValueError("sims_layout must be 'vector' or 'columns'")

    hdu_grouping = str(hdu_grouping).lower().strip()
    if hdu_grouping not in ('by_cl', 'by_bandpair'):
        raise ValueError("hdu_grouping must be 'by_cl' or 'by_bandpair'")
    if hdu_grouping == 'by_bandpair' and sims_layout != 'vector':
        raise ValueError("hdu_grouping='by_bandpair' requires sims_layout='vector'")

    dt = np.dtype(dtype)
    if dt not in (np.dtype('float32'), np.dtype('float64')):
        raise ValueError("dtype must be float32 or float64")
    fmt_char = 'E' if dt == np.dtype('float32') else 'D'

    # Adjust filename for white-noise convention
    if use_white_noise:
        out_file = _with_suffix_before_fits(out_file, '_wn')

    # Ensure directory exists
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    # Optionally also write the avg+std file if requested and available
    if avg_std_out_file is not None:
        if avg_std_dict is None:
            raise ValueError("avg_std_out_file was provided, but avg_std_dict is None")
        save_avg_std_to_fits(avg_std_dict, band_list, out_file=avg_std_out_file, use_white_noise=use_white_noise)

    # Prepare a mapping of per-sim arrays: key -> (cl_key, subkey, array)
    sims_map = {}

    # Prefer external NPZ if provided
    if sims_npz is not None and os.path.exists(sims_npz):
        npz = np.load(sims_npz)
        for k in npz.files:
            # Expect keys like '11_11__EE' or '11_11__EE__SUB'
            parts = k.split('__')
            if len(parts) >= 2:
                band_pair = parts[0]
                cl_key = parts[1]
                subkey = parts[2] if len(parts) > 2 else 'SIMS'
            else:
                continue
            arr = np.asarray(npz[k], dtype=dt)
            if arr.ndim != 2:
                continue
            sims_map.setdefault(band_pair, []).append((cl_key, subkey, arr))
    else:
        # Inspect avg_std_dict for 2D per-sim arrays under non-MEAN/STD subkeys
        if avg_std_dict is None:
            raise ValueError("Neither sims_npz found nor avg_std_dict provided with sims")
        for band_i in band_list:
            for band_j in band_list:
                band_pair = f"{band_i}_{band_j}"
                spec_dict = avg_std_dict.get(band_pair, {})
                for cl_key in ['ell1', 'ell2', 'ell_eff', 'TT', 'EE', 'BB', 'TE', 'TB', 'EB']:
                    cl_entry = spec_dict.get(cl_key, None)
                    if not isinstance(cl_entry, dict):
                        continue
                    for subk, subv in cl_entry.items():
                        if subk in ('MEAN', 'STD'):
                            continue
                        try:
                            arr = np.asarray(subv, dtype=dt)
                        except Exception:
                            continue
                        if arr.ndim != 2:
                            continue
                        sims_map.setdefault(band_pair, []).append((cl_key, subk, arr))

    # If nothing was discovered, fail fast to avoid writing empty FITS
    if len(sims_map) == 0:
        raise ValueError(
            "No per-simulation arrays found for save_sims_to_fits. "
            "Pass a valid sims_npz, or include 2D per-sim arrays under non-MEAN/STD keys in avg_std_dict."
        )

    # Write FITS
    hdu_list = fits.HDUList()
    hdu_list.append(fits.PrimaryHDU())

    def _get_ell_arrays(band_pair, n_bins):
        ell1 = ell2 = ell_eff = None
        if avg_std_dict is not None and band_pair in avg_std_dict:
            spec = avg_std_dict[band_pair]
            ell1 = spec.get('ell1', {}).get('MEAN', None)
            ell2 = spec.get('ell2', {}).get('MEAN', None)
            ell_eff = spec.get('ell_eff', {}).get('MEAN', None)
        if ell1 is None:
            ell1 = np.arange(n_bins)
        if ell2 is None:
            ell2 = np.arange(n_bins)
        if ell_eff is None:
            ell_eff = np.arange(n_bins)
        return (
            np.asarray(ell1, dtype=np.int32),
            np.asarray(ell2, dtype=np.int32),
            np.asarray(ell_eff, dtype=dt),
        )

    if hdu_grouping == 'by_cl':
        for band_pair, entries in sims_map.items():
            for cl_key, subkey, arr in entries:
                n_sim, n_bins = arr.shape
                ell1, ell2, ell_eff = _get_ell_arrays(band_pair, n_bins)
                arr = np.asarray(arr, dtype=dt)

                cols = [
                    fits.Column(name='ELL1', format='J', array=ell1),
                    fits.Column(name='ELL2', format='J', array=ell2),
                    fits.Column(name='ELL_EFF', format=fmt_char, array=ell_eff),
                ]

                if sims_layout == 'vector':
                    sims_by_bin = arr.T  # (n_bins, n_sim)
                    cols.append(fits.Column(name='SIMS', format=f"{n_sim}{fmt_char}", array=sims_by_bin))
                else:
                    for s in range(n_sim):
                        cols.append(fits.Column(name=f"SIM_{s+1}", format=fmt_char, array=arr[s]))

                hdu = fits.BinTableHDU.from_columns(cols)
                bi, bj = band_pair.split('_') if '_' in band_pair else (band_pair, '')
                hdu.header['BAND_I'] = bi
                hdu.header['BAND_J'] = bj
                hdu.header['CLKEY'] = cl_key
                hdu.header['SIMKEY'] = subkey
                hdu.header['SDTYPE'] = 'F32' if dt == np.dtype('float32') else 'F64'
                hdu.header['SLAYOUT'] = sims_layout.upper()
                hdu.name = f"{band_pair}__{cl_key}__{subkey}_SIMS"
                hdu_list.append(hdu)
    else:
        preferred_order = ['TT', 'EE', 'BB', 'TE', 'TB', 'EB']
        for band_pair, entries in sims_map.items():
            # group by subkey
            by_subkey = {}
            for cl_key, subkey, arr in entries:
                by_subkey.setdefault(subkey, []).append((cl_key, arr))

            for subkey, cl_entries in by_subkey.items():
                # reference shape
                ref_arr = None
                for _, a in cl_entries:
                    a = np.asarray(a)
                    if a.ndim == 2:
                        ref_arr = a
                        break
                if ref_arr is None:
                    continue
                n_sim, n_bins = ref_arr.shape
                ell1, ell2, ell_eff = _get_ell_arrays(band_pair, n_bins)

                cols = [
                    fits.Column(name='ELL1', format='J', array=ell1),
                    fits.Column(name='ELL2', format='J', array=ell2),
                    fits.Column(name='ELL_EFF', format=fmt_char, array=ell_eff),
                ]

                cl_map = {k: np.asarray(v, dtype=dt) for k, v in cl_entries if np.asarray(v).ndim == 2}
                ordered_keys = [k for k in preferred_order if k in cl_map] + [k for k in sorted(cl_map) if k not in preferred_order]
                for cl_key in ordered_keys:
                    a = cl_map[cl_key]
                    if a.shape != (n_sim, n_bins):
                        continue
                    cols.append(
                        fits.Column(name=str(cl_key), format=f"{n_sim}{fmt_char}", array=a.T)
                    )

                hdu = fits.BinTableHDU.from_columns(cols)
                bi, bj = band_pair.split('_') if '_' in band_pair else (band_pair, '')
                hdu.header['BAND_I'] = bi
                hdu.header['BAND_J'] = bj
                hdu.header['CLKEY'] = 'MULTI'
                hdu.header['SIMKEY'] = subkey
                hdu.header['SDTYPE'] = 'F32' if dt == np.dtype('float32') else 'F64'
                hdu.header['SLAYOUT'] = sims_layout.upper()
                hdu.header['SGROUP'] = 'BANDPAIR'
                hdu.name = f"{band_pair}__{subkey}_SIMS"
                hdu_list.append(hdu)

    # Write to disk
    hdu_list.writeto(out_file, overwrite=True)
    print(f"Saved per-simulation spectra to {out_file}")
    return out_file


'''
# ====================================
# 6
# ====================================
'''


def get_beam_for_band(band_name, data, ell_eff):
    """
    Return the interpolated beam transfer functions for a given frequency band.

    Parameters
    ----------
    band_name : str
        Name of the frequency band, e.g., '11', '30', '100'.
    data : dict
        Dictionary containing experiment and band information, including beam file paths.
    ell_eff : array_like
        Array of effective multipoles at which to interpolate the beam.

    Returns
    -------
    beam_interp : dict of numpy.ndarray
        Dictionary with keys 'T','E','B'. Each is the interpolated beam
        transfer function at the effective multipoles `ell_eff`.
    """

    # QUIJOTE
    if band_name in data.get('QUIJOTE', {}):
        with fits.open(data['QUIJOTE'][band_name]['beam']) as hdul:
            beam_hdu = hdul[1]
            col_map = {
                "11": "Bl_311",
                "13": "Bl_313",
                "17": "Bl_417",
                "19": "Bl_419",
            }
            colname = col_map.get(band_name)
            if colname is None:
                # fallback: take first column-like Bl_*
                for name in beam_hdu.columns.names:
                    if name.lower().startswith('bl_'):
                        colname = name
                        break
            beam_arr = beam_hdu.data[colname][0]
            beam_interp = np.interp(ell_eff, np.arange(len(beam_arr)), beam_arr)
            # Clip any negative beam values to zero (user request): make them explicit zeros
            neg_count = int(np.sum(beam_interp < 0))
            if neg_count > 0:
                print(f"[INFO] get_beam_for_band: band {band_name} had {neg_count} negative beam values after interpolation; setting them to 0")
                beam_interp[beam_interp < 0] = 0.0
        return {"T": beam_interp, "E": beam_interp, "B": beam_interp}

    # WMAP
    elif band_name in data.get('WMAP', {}):
        beam_arr = np.loadtxt(data['WMAP'][band_name]['beam']).T[1]
        beam_interp = np.interp(ell_eff, np.arange(len(beam_arr)), beam_arr)
        return {"T": beam_interp, "E": beam_interp, "B": beam_interp}

    # Planck
    elif band_name in data.get('Planck', {}):
        if int(band_name) <= 70:  # LFI
            hdul = fits.open(data['Planck'][band_name]['beam'])
            # try to find correct extension name
            extname = f'BEAMWF_0{band_name}X0{band_name}'
            if extname in hdul:
                beam_hdu = hdul[extname]
                Bl = beam_hdu.data['BL']
            else:
                # fallback: take first extension with 'BL' column
                beam_hdu = hdul[1]
                Bl = beam_hdu.data[beam_hdu.columns.names[0]]
            beam_interp = np.interp(ell_eff, np.arange(len(Bl)), Bl)
            hdul.close()
            return {"T": beam_interp, "E": beam_interp, "B": beam_interp}
        else:  # HFI
            hdul = fits.open(data['Planck'][band_name]['beam'])
            window_hdu = hdul['WINDOW FUNCTIONS']
            Bl_T = np.interp(ell_eff, np.arange(len(window_hdu.data['T'])), window_hdu.data['T'])
            Bl_E = np.interp(ell_eff, np.arange(len(window_hdu.data['E'])), window_hdu.data['E'])
            Bl_B = np.interp(ell_eff, np.arange(len(window_hdu.data['B'])), window_hdu.data['B'])
            hdul.close()
            return {"T": Bl_T, "E": Bl_E, "B": Bl_B}
    else:
        raise ValueError(f"Band '{band_name}' not found in data.")


def cmb_unit_conversion(nuGHz, option='KCMB2KRJ', help=False):
    """
    Compute unit conversion factors between CMB thermodynamic temperature (K_CMB),
    Rayleigh-Jeans temperature (K_RJ), and surface brightness (Jy/sr).

    Parameters
    ----------
    nuGHz : float or array-like
        Frequency in GHz.
    option : str, optional
        Type of conversion to perform. Available options:
            - 'KCMB2KRJ' : Convert from K_CMB to K_RJ
            - 'KRJ2KCMB' : Convert from K_RJ to K_CMB
            - 'KCMB2Jysr' : Convert from K_CMB to Jy/sr
            - 'Jysr2KCMB' : Convert from Jy/sr to K_CMB
            - 'KRJ2Jysr' : Convert from K_RJ to Jy/sr
            - 'Jysr2KRJ' : Convert from Jy/sr to K_RJ
    help : bool, optional
        If True, prints the available conversion options and syntax.

    Returns
    -------
    float or ndarray
        Conversion factor for the selected unit transformation.

    Notes
    -----
    - Based on standard CMB thermodynamic relations.
    - Tcmb = 2.72548 K (Planck 2018).
    - h, k, c are Planck's constant, Boltzmann constant, and speed of light.

    Examples
    --------
    >>> cmb_unit_conversion(30, 'KCMB2KRJ')
    1.0289...
    >>> cmb_unit_conversion(143, 'KCMB2Jysr')
    269.3...
    """
    # Physical constants
    h = 6.62607015e-34  # Planck constant [J s]
    k = 1.380649e-23    # Boltzmann constant [J/K]
    c = 2.99792458e8    # Speed of light [m/s]
    Tcmb = 2.72548       # CMB temperature [K]

    cases = ['KCMB2KRJ', 'KRJ2KCMB', 'KCMB2Jysr', 'Jysr2KCMB', 'KRJ2Jysr', 'Jysr2KRJ']
    if help:
        print("Syntax: cmb_unit_conversion(nuGHz, option=)")
        print("Available options:", cases)

    nu = np.asarray(nuGHz) * 1e9  # [Hz]
    x = h * nu / (k * Tcmb)
    thermo = x**2 * np.exp(x) / (np.exp(x) - 1.0)**2
    rj = (2.0 * k * nu**2 / c**2) * 1e26  # [Jy/sr per K_RJ]

    if option == 'KCMB2KRJ':
        fac = thermo
    elif option == 'KRJ2KCMB':
        fac = 1 / thermo
    elif option == 'KCMB2Jysr':
        fac = thermo * rj
    elif option == 'Jysr2KCMB':
        fac = 1 / (thermo * rj)
    elif option == 'KRJ2Jysr':
        fac = rj
    elif option == 'Jysr2KRJ':
        fac = 1 / rj
    else:
        print("Units not identified. Returning -1")
        fac = -1

    return fac


# Unit corrections for PLANCK HFI. Values extracted from PLA, explanatory supplement.
# https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/UC_CC_Tables
# RGS values sent via email, 26-27/Feb/2025
def planck_uc_hfi(use_bps=True):
    bands_hfi = np.array([100, 143, 217, 353, 545, 857], dtype=float) # GHz

    # Table 1. Coefficient MJy/sr/KCMB. Values correspond to "avg" entry.
    UC_HFI_KCMB2MJysr_PLA = np.array([244.0960, 371.7327, 483.6874, 287.4517, 58.0356, 2.2681])

    # Computed by RGS, including bandpass shift
    UC_HFI_KCMB2MJysr_rgs = np.array([242.09786, 370.53512, 481.93046, 287.22432, 56.659334, 2.1156277])

    # Table 2. KRJ/(MJy/sr).
    # Note: Coincides with Table 5 of Planck 2013, IX. HFI spectral response
    UC_HFI_MJysr2KRJ_PLA = np.array([0.0032548074, 0.0015916707, 0.00069120334, 0.00026120163, 0.00010958025, 4.4316316e-05 ])

    # Computed by RGS, evaluated at the center
    UC_HFI_MJysr2KRJ_rgs = 1./np.array([307.09143, 627.97125, 1446.0629, 3826.6356, 9121.3834, 22554.299])

    # Derived quantities (final outputs)
    UC_HFI_KCMB2KRJ = UC_HFI_KCMB2MJysr_rgs * UC_HFI_MJysr2KRJ_rgs #includes Bandpass shifts
    uc_hfi_no_bps   = UC_HFI_KCMB2MJysr_PLA * UC_HFI_MJysr2KRJ_PLA

    # Select output. Default is bandpass shift corrected value.
    output = UC_HFI_KCMB2KRJ
    if use_bps==False: output = uc_hfi_no_bps

    return output

# ---------------------------------------------------------------------
def load_cmb_spectrum_from_file(filepath, ell_values, planck_format=True):
    """
    Load CMB spectrum from a file and interpolate to requested multipoles.
    
    Parameters
    ----------
    filepath : str
        Path to file containing CMB spectrum.
        - Planck format: columns [L, TT, TE, EE, BB, PP] with D_l in μK²
        - CAMB format: columns [ell, TT, EE, BB, TE] with C_l in K²
    ell_values : array-like
        Multipole values at which to interpolate.
    planck_format : bool, optional
        If True (default), assumes Planck Legacy Archive format:
          - Columns: L, TT, TE, EE, BB, PP
          - Values are D_l = l(l+1)/(2π) x C_l in μK²
        If False, assumes CAMB format:
          - Columns: ell, TT, EE, BB, TE
          - Values are C_l in K²
        
    Returns
    -------
    dict
        Dictionary with keys 'TT', 'EE', 'BB', 'TE', 'TB', 'EB' 
        containing C_l values in K² at ell_values.
    """
    # Load spectrum file
    data = np.loadtxt(filepath)
    ell_file = data[:, 0].astype(int)
    
    ell_values = np.asarray(ell_values)
    spectra = {}
    
    if planck_format:
        # Planck format: L, TT, TE, EE, BB, PP
        # D_l in μK²
        col_map = {'TT': 1, 'TE': 2, 'EE': 3, 'BB': 4}
        
        for spec_type, col_idx in col_map.items():
            if data.shape[1] > col_idx:
                # Get D_l in μK²
                Dl_muK2 = data[:, col_idx]
                
                # Convert D_l to C_l: C_l = D_l / [l(l+1)/(2π)]
                # Avoid division by zero for l < 2
                Cl_muK2 = np.zeros_like(Dl_muK2)
                mask = ell_file >= 2
                Cl_muK2[mask] = Dl_muK2[mask] * (2 * np.pi) / (ell_file[mask] * (ell_file[mask] + 1))
                
                # Convert from μK² to K²
                Cl_K2 = Cl_muK2 * 1e-12
                
                # Interpolate to requested ell values
                spectra[spec_type] = np.interp(ell_values, ell_file, Cl_K2, left=0, right=0)
            else:
                spectra[spec_type] = np.zeros_like(ell_values)
        
        # TB and EB are zero in standard ΛCDM
        spectra['TB'] = np.zeros_like(ell_values)
        spectra['EB'] = np.zeros_like(ell_values)
        
    else:
        # CAMB format: ell, TT, EE, BB, TE
        # C_l already in K²
        col_map = {'TT': 1, 'EE': 2, 'BB': 3, 'TE': 4}
        
        for spec_type, col_idx in col_map.items():
            if data.shape[1] > col_idx:
                Cl_K2 = data[:, col_idx]
                spectra[spec_type] = np.interp(ell_values, ell_file, Cl_K2, left=0, right=0)
            else:
                spectra[spec_type] = np.zeros_like(ell_values)
        
        # TB and EB are zero
        spectra['TB'] = np.zeros_like(ell_values)
        spectra['EB'] = np.zeros_like(ell_values)
    
    return spectra


# ---------------------------------------------------------------------
def correct_power_spectra(path_spectra, path_avg_std_skyplusnoise, path_avg_std_noise,
                          band_list, data, nside,
                          correct_beam=True, correct_pixel=True,
                          save=False, path_out_file=None, use_white_noise=False,
                          use_noise=False,
                          path_hmdm_spectra=None, subtract_cmb=False, cmb_spectrum_path=None,
                          correct_unit=True):
    """
    Correct power spectra by removing noise bias and applying beam, pixel,
    and unit corrections. Optionally saves corrected spectra to a FITS file.

    Parameters
    ----------
    path_spectra : str
        Path to input FITS file containing raw (sky + noise) spectra.
    path_avg_std_skyplusnoise : str
        Path to FITS file with average and standard deviation for sky + noise simulations.
    path_avg_std_noise : str
        Path to FITS file with average and standard deviation for pure noise simulations.
    band_list : list of str
        List of band names or pairs (e.g., ['30_44', '44_70']).
    data : dict
        Metadata dictionary with beam file paths and band frequencies.
    nside : int
        HEALPix NSIDE resolution of the input maps.
    correct_beam : bool, optional
        If True, deconvolve beam window functions. Default: True.
    correct_pixel : bool, optional
        If True, deconvolve the HEALPix pixel window function. Default: True.
    save : bool, optional
        If True, save the corrected spectra to a FITS file. Default: False.
    path_out_file : str, optional
        Output FITS file path. Defaults to "corrected_cls.fits" if not provided.
    use_white_noise : bool, optional
        If True, subtract white noise simulation mean; if False, subtract HMDM spectra.
    use_noise : bool, optional
        If True, for QUIJOTE auto-spectra the noise bias is taken as the mean of the
        pure-noise simulations (avg_std_noise MEAN) instead of the HMDM.  For all
        other bands HMDM is still used.  Ignored when use_white_noise=True.
        Default: False.
    path_hmdm_spectra : str, optional
        Path to FITS file containing HMDM spectra. Required when use_white_noise=False.
    subtract_cmb : bool, optional
        If True, subtract the CMB spectrum from Planck best-fit cosmology. Default: False.
        When True, cmb_spectrum_path must be provided.
    cmb_spectrum_path : str, optional
        Path to file containing CMB spectrum. REQUIRED when subtract_cmb=True.
        Download from: http://pla.esac.esa.int/
        Recommended file: COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt
    correct_unit : bool, optional
        If True, convert from K_CMB to K_RJ units (Planck 2018 convention). Default: True.
        Set to False only if you want to keep spectra in K_CMB units.

    Returns
    -------
    tuple
        (corr_spectra, out_file)
        corr_spectra : dict
            Dictionary with corrected power spectra and errors.
            Units: K²_RJ if correct_unit=True, mK²_CMB if correct_unit=False.
        out_file : str or None
            Path to the output FITS file if saved, otherwise None.
    """
    if save and path_out_file is None:
        path_out_file = "corrected_cls.fits"

    # Load input spectra
    spectra = read_spectra_from_fits(path_spectra, band_list)
    avg_std_skyplusnoise = read_spectra_from_fits(
        path_avg_std_skyplusnoise, band_list, use_white_noise=use_white_noise
    )
    
    # Always load noise for error propagation
    avg_std_noise = read_spectra_from_fits(
        path_avg_std_noise, band_list, use_white_noise=use_white_noise
    )
    
    # Load HMDM spectra if needed
    if not use_white_noise:
        if path_hmdm_spectra is None:
            raise ValueError("path_hmdm_spectra must be provided when use_white_noise=False")
        hmdm_spectra = read_spectra_from_fits(path_hmdm_spectra, band_list)

    # Collect the set of QUIJOTE band names from the data dictionary
    quijote_bands = set(data.get('QUIJOTE', {}).keys())

    # Effective multipoles from first entry
    first_entry = next(iter(spectra.values()))
    ell_eff = np.array(first_entry['ell_eff'])

    # Pixel window function
    if correct_pixel:
        wpix = hp.pixwin(nside)
        wp_interp = np.interp(ell_eff, np.arange(len(wpix)), wpix)
    else:
        wp_interp = np.ones_like(ell_eff)

    # Precompute correction factors for each band
    all_bands = set()
    for key in spectra.keys():
        if "_" in key:
            band1, band2 = key.split('_', 1)
            all_bands.update([band1, band2])

    beam_dict, unit_dict, wp_dict = {}, {}, {}
    for band in all_bands:
        for exp in data:
            if band in data[exp]:
                # Beam correction
                if correct_beam:
                    beam_dict[band] = get_beam_for_band(band, data, ell_eff)
                else:
                    beam_dict[band] = {"T": np.ones_like(ell_eff),
                                       "E": np.ones_like(ell_eff),
                                       "B": np.ones_like(ell_eff)}

                # Clip small negative beam values
                for comp in ('T', 'E', 'B'):
                    arr = np.asarray(beam_dict[band][comp], dtype=float)
                    neg_count = int(np.sum(arr < 0))
                    if neg_count > 0:
                        small_neg_mask = (arr < 0) & (arr > -1e-3)
                        if np.sum(~small_neg_mask) > 0:
                            print(f"[WARN] beam {band} {comp} has {neg_count} negative values, including >1e-3; check input")
                        arr = np.clip(arr, 0.0, None)
                        beam_dict[band][comp] = arr

                # Frequency in GHz
                freq = data[exp][band].get('freq')
                try:
                    nuGHz = freq.to('GHz').value
                except Exception:
                    nuGHz = float(freq)

                # Unit conversion factor (K_CMB → K_RJ)
                if correct_unit:
                    # Use generic conversion for all bands except Planck HFI
                    is_planck = str(exp).lower() == 'planck'
                    hfi_band_set = {'100','143','217','353','545','857'}
                    if is_planck and str(band) in hfi_band_set:
                        # Select HFI KCMB->KRJ value with bandpass shift correction (use_bps=True)
                        try:
                            uc_hfi = planck_uc_hfi(use_bps=True)
                            hfi_order = [100, 143, 217, 353, 545, 857]
                            idx = hfi_order.index(int(float(band)))
                            unit_dict[band] = float(uc_hfi[idx])
                        except Exception:
                            # Fallback to generic conversion if mapping fails
                            print('HFI uc failed, using generic conversion instead')
                            unit_dict[band] = float(cmb_unit_conversion(nuGHz, 'KCMB2KRJ'))
                    else:
                        unit_dict[band] = float(cmb_unit_conversion(nuGHz, 'KCMB2KRJ'))
                else:
                    unit_dict[band] = 1.0
                    
                wp_dict[band] = wp_interp if correct_pixel else np.ones_like(ell_eff)
                break

    # Load CMB spectrum if subtraction is requested
    cmb_spectra = {}
    if subtract_cmb:
        if cmb_spectrum_path is None:
            raise ValueError(
                "cmb_spectrum_path must be provided when subtract_cmb=True. "
                "Download the Planck CMB spectrum from: http://pla.esac.esa.int/ "
                "(e.g., COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt)"
            )
        
        # Load from file
        print(f"[INFO] Loading CMB spectrum from {cmb_spectrum_path}")
        # Auto-detect Planck format (has .txt extension from PLA)
        is_planck_format = 'COM_PowerSpect' in cmb_spectrum_path or 'planck' in cmb_spectrum_path.lower()
        cmb_spectra = load_cmb_spectrum_from_file(cmb_spectrum_path, ell_eff, planck_format=is_planck_format)
        print(f"[INFO] Loaded CMB spectrum with format: {'Planck PLA' if is_planck_format else 'CAMB'}")

    # Define cross-band pairs with known correlated noise to subtract in cross-spectra
    correlated_cross_pairs = {('11', '13'), ('13', '11'), ('17', '19'), ('19', '17')}

    # Apply corrections and noise subtraction
    corr_spectra = {}
    for key, spec in spectra.items():
        if "_" not in key:
            continue
        band1, band2 = key.split('_', 1)
        corr_spectra[key] = {}

        for cl_key in ['TT', 'EE', 'BB', 'TE', 'TB', 'EB']:
            # Beam factor
            comp_map = {'TT':('T','T'),'EE':('E','E'),'BB':('B','B'),'TE':('T','E'),'TB':('T','B'),'EB':('E','B')}
            comp1, comp2 = comp_map.get(cl_key, ('T','T'))
            beam_factor = beam_dict[band1][comp1] * beam_dict[band2][comp2]

            # Physical deconvolution factor and unit factor
            phys_factor = beam_factor * wp_dict[band1] * wp_dict[band2]
            unit_factor = unit_dict[band1] * unit_dict[band2]

            # Retrieve raw Cl
            spec_val = spec.get(cl_key)
            if spec_val is None:
                continue
            if isinstance(spec_val, dict):
                if 'MEAN' in spec_val:
                    Cl_raw = np.array(spec_val['MEAN'])
                elif 'SPECTRUM' in spec_val:
                    Cl_raw = np.array(spec_val['SPECTRUM'])
                else:
                    raise ValueError(f"Unexpected dict format for spectrum {key} {cl_key}")
            else:
                Cl_raw = np.array(spec_val)

            # Check if this is a cross-spectrum (band1 != band2)
            is_cross_spectrum = (band1 != band2)
            
            # Step 1: Noise/HMDM subtraction
            if is_cross_spectrum:
                # For cross-spectra: subtract noise only for known correlated pairs (11-13 and 17-19)
                if (band1, band2) in correlated_cross_pairs:
                    if use_white_noise:
                        Nl = np.array(avg_std_noise[key][cl_key]['MEAN'])
                    else:
                        Nl = np.array(hmdm_spectra[key][cl_key])
                    Cl = Cl_raw - Nl  # mK²_CMB, convolved with beams/pixel
                else:
                    # Other cross-spectra: keep raw spectrum (no noise subtraction)
                    Cl = Cl_raw
            else:
                # For auto-spectra: subtract noise
                if use_white_noise:
                    Nl = np.array(avg_std_noise[key][cl_key]['MEAN'])
                elif use_noise and band1 in quijote_bands:
                    # QUIJOTE auto-spectra: use mean of pure-noise simulations
                    Nl = np.array(avg_std_noise[key][cl_key]['MEAN'])
                    # print(f"[INFO] QUIJOTE {key} {cl_key}: subtracting mean noise sim (N={Nl.shape})")
                else:
                    Nl = np.array(hmdm_spectra[key][cl_key])
                Cl = Cl_raw - Nl  # mK²_CMB, convolved with beams/pixel

            # Step 2: Subtract CMB contribution (before deconvolution)
            # CMB must be convolved with beams/pixel to match the observed spectrum
            if subtract_cmb and cl_key in cmb_spectra:
                # Get CMB spectrum in K_CMB²
                Cl_cmb_kcmb = cmb_spectra[cl_key]
                
                # Convert to mK_CMB² and convolve with beams/pixel to match observed spectrum
                Cl_cmb_mkcmb = Cl_cmb_kcmb * 1e6  # K² -> mK²
                Cl_cmb_conv = Cl_cmb_mkcmb * phys_factor  # Convolve with beams/pixel
                
                # Subtract CMB (both are now in mK_CMB² and convolved)
                Cl = Cl - Cl_cmb_conv
                
                
            # Step 3: Deconvolve beams and pixel window
            safe_phys = np.array(phys_factor, dtype=float)
            safe_phys[safe_phys == 0] = np.nan
            safe_phys[safe_phys < 0] = np.nan
            
            Cl_deconv = Cl / safe_phys  # mK²_CMB, deconvolved
            
            # Step 4: Apply unit conversion (K_CMB → K_RJ)
            spectrum_corr = Cl_deconv * unit_factor  # K²_RJ or mK²_CMB depending on correct_unit
            unit_correction = unit_factor

            # Propagate errors
            if is_cross_spectrum:
                if (band1, band2) in correlated_cross_pairs:
                    # When subtracting cross-noise, include both sky+noise and noise uncertainties
                    err_num = np.sqrt(
                        np.array(avg_std_skyplusnoise[key][cl_key]['STD'])**2 +
                        np.array(avg_std_noise[key][cl_key]['STD'])**2
                    )
                else:
                    # For other cross-spectra: error only from sky+noise std
                    err_num = np.array(avg_std_skyplusnoise[key][cl_key]['STD'])
            else:
                # For auto-spectra: quadratic sum as before
                err_num = np.sqrt(
                    np.array(avg_std_skyplusnoise[key][cl_key]['STD'])**2 +
                    np.array(avg_std_noise[key][cl_key]['STD'])**2
                )
            errbar = (err_num / safe_phys) * unit_correction
            errbar = np.abs(errbar)

            corr_spectra[key][cl_key] = {'SPECTRUM': spectrum_corr, 'ERROR': errbar}

        # Copy multipole info
        corr_spectra[key]['ell1'] = np.array(spec['ell1'])
        corr_spectra[key]['ell2'] = np.array(spec['ell2'])
        corr_spectra[key]['ell_eff'] = np.array(spec['ell_eff'])

    # Save to FITS file if requested
    out_file = None
    if save:
        out_file = path_out_file
        hdu_list = fits.HDUList([fits.PrimaryHDU()])

        for key, spec_dict in sorted(corr_spectra.items()):
            band_i, band_j = key.split('_', 1)
            cols = []
            for cl_key in ['ell1', 'ell2', 'ell_eff', 'TT', 'EE', 'BB', 'TE', 'TB', 'EB']:
                if cl_key in ['ell1', 'ell2', 'ell_eff']:
                    cols.append(fits.Column(name=cl_key, format='D', array=spec_dict[cl_key]))
                else:
                    if cl_key not in spec_dict:
                        continue
                    cols.append(fits.Column(name=f"{cl_key}_SPECTRUM", format='D', array=spec_dict[cl_key]['SPECTRUM']))
                    cols.append(fits.Column(name=f"{cl_key}_ERROR", format='D', array=spec_dict[cl_key]['ERROR']))

            hdu = fits.BinTableHDU.from_columns(cols)
            hdu.header['BAND_I'] = band_i
            hdu.header['BAND_J'] = band_j
            hdu.header['COMMENT'] = (
                "Corrected spectra: noise subtracted, beam/unit/pixel corrections applied"
            )
            hdu.name = key
            hdu_list.append(hdu)

        hdu_list.writeto(out_file, overwrite=True)
        print(f"[OK] Saved corrected spectra with errors to {out_file}")

    return corr_spectra, out_file


def correct_theoretical_spectra(path_theoretical_spectra, band_list, data, nside,
                               correct_beam=True, correct_pixel=True,
                               save=False, path_out_file=None, correct_unit=True):
    """
    Apply beam, pixel window, and unit corrections to theoretical spectra
    (no noise subtraction since these are pure theoretical spectra).

    Parameters
    ----------
    path_theoretical_spectra : str
        Path to input FITS file containing raw theoretical spectra.
    band_list : list of str
        List of band names or pairs (e.g., ['30_44', '44_70']).
    data : dict
        Metadata dictionary with beam file paths and band frequencies.
    nside : int
        HEALPix NSIDE resolution of the input maps.
    correct_beam : bool, optional
        If True, deconvolve beam window functions. Default: True.
    correct_pixel : bool, optional
        If True, deconvolve the HEALPix pixel window function. Default: True.
    save : bool, optional
        If True, save the corrected spectra to a FITS file. Default: False.
    path_out_file : str, optional
        Output FITS file path. Defaults to "corrected_theoretical_cls.fits" if not provided.
    correct_unit : bool, optional
        If True, convert from K_CMB to K_RJ units. Default: True.

    Returns
    -------
    dict
        Dictionary with corrected theoretical power spectra.
        Units: K²_RJ if correct_unit=True, mK²_CMB if correct_unit=False.
    """
    if save and path_out_file is None:
        path_out_file = "corrected_theoretical_cls.fits"

    # Load theoretical spectra
    spectra = read_spectra_from_fits(path_theoretical_spectra, band_list)

    # Effective multipoles from first entry
    first_entry = next(iter(spectra.values()))
    ell_eff = np.array(first_entry['ell_eff'])

    # Pixel window function
    if correct_pixel:
        wpix = hp.pixwin(nside)
        wp_interp = np.interp(ell_eff, np.arange(len(wpix)), wpix)
    else:
        wp_interp = np.ones_like(ell_eff)

    # Precompute correction factors for each band
    all_bands = set()
    for key in spectra.keys():
        if "_" in key:
            band1, band2 = key.split('_', 1)
            all_bands.update([band1, band2])

    beam_dict, unit_dict, wp_dict = {}, {}, {}
    for band in all_bands:
        for exp in data:
            if band in data[exp]:
                # Beam correction
                if correct_beam:
                    beam_dict[band] = get_beam_for_band(band, data, ell_eff)
                else:
                    beam_dict[band] = {"T": np.ones_like(ell_eff),
                                       "E": np.ones_like(ell_eff),
                                       "B": np.ones_like(ell_eff)}

                # Clip small negative beam values
                for comp in ('T', 'E', 'B'):
                    arr = np.asarray(beam_dict[band][comp], dtype=float)
                    neg_count = int(np.sum(arr < 0))
                    if neg_count > 0:
                        small_neg_mask = (arr < 0) & (arr > -1e-3)
                        if np.sum(~small_neg_mask) > 0:
                            print(f"[WARN] beam {band} {comp} has {neg_count} negative values, including >1e-3; check input")
                        arr = np.clip(arr, 0.0, None)
                        beam_dict[band][comp] = arr

                # Frequency in GHz
                freq = data[exp][band].get('freq')
                try:
                    nuGHz = freq.to('GHz').value
                except Exception:
                    nuGHz = float(freq)

                # Unit conversion factor (K_CMB → K_RJ)
                if correct_unit:
                    is_planck = str(exp).lower() == 'planck'
                    hfi_band_set = {'100','143','217','353','545','857'}
                    if is_planck and str(band) in hfi_band_set:
                        try:
                            uc_hfi = planck_uc_hfi(use_bps=True)
                            hfi_order = [100, 143, 217, 353, 545, 857]
                            idx = hfi_order.index(int(float(band)))
                            unit_dict[band] = float(uc_hfi[idx])
                        except Exception:
                            unit_dict[band] = float(cmb_unit_conversion(nuGHz, 'KCMB2KRJ'))
                    else:
                        unit_dict[band] = float(cmb_unit_conversion(nuGHz, 'KCMB2KRJ'))
                else:
                    unit_dict[band] = 1.0
                    
                wp_dict[band] = wp_interp if correct_pixel else np.ones_like(ell_eff)
                break

    # Apply corrections (no noise subtraction for theoretical spectra)
    corr_spectra = {}
    for key, spec in spectra.items():
        if "_" not in key:
            continue
        band1, band2 = key.split('_', 1)
        corr_spectra[key] = {}

        for cl_key in ['TT', 'EE', 'BB', 'TE', 'TB', 'EB']:
            if cl_key not in spec:
                continue
                
            # Beam factor
            comp_map = {'TT':('T','T'),'EE':('E','E'),'BB':('B','B'),'TE':('T','E'),'TB':('T','B'),'EB':('E','B')}
            comp1, comp2 = comp_map.get(cl_key, ('T','T'))
            beam_factor = beam_dict[band1][comp1] * beam_dict[band2][comp2]

            # Physical deconvolution factor and unit factor
            phys_factor = beam_factor * wp_dict[band1] * wp_dict[band2]
            unit_factor = unit_dict[band1] * unit_dict[band2]

            # Get raw theoretical spectrum (no noise to subtract)
            Cl = np.array(spec[cl_key])

            # Safe deconvolution
            safe_phys = np.array(phys_factor, dtype=float)
            safe_phys[safe_phys == 0] = np.nan
            safe_phys[safe_phys < 0] = np.nan

            # Deconvolve and apply unit conversion
            Cl_deconv = Cl / safe_phys
            spectrum_corr = Cl_deconv * unit_factor

            # Store corrected spectrum (no error bars for theoretical spectra)
            corr_spectra[key][cl_key] = spectrum_corr

        # Copy multipole info
        corr_spectra[key]['ell1'] = np.array(spec['ell1'])
        corr_spectra[key]['ell2'] = np.array(spec['ell2'])
        corr_spectra[key]['ell_eff'] = np.array(spec['ell_eff'])

    # Save to FITS file if requested
    if save:
        hdu_list = fits.HDUList([fits.PrimaryHDU()])

        for key, spec_dict in sorted(corr_spectra.items()):
            band_i, band_j = key.split('_', 1)
            cols = []
            
            # Add multipole columns
            for cl_key in ['ell1', 'ell2', 'ell_eff']:
                cols.append(fits.Column(name=cl_key, format='D', array=spec_dict[cl_key]))
            
            # Add spectrum columns
            for cl_key in ['TT', 'EE', 'BB', 'TE', 'TB', 'EB']:
                if cl_key in spec_dict:
                    cols.append(fits.Column(name=cl_key, format='D', array=spec_dict[cl_key]))

            hdu = fits.BinTableHDU.from_columns(cols)
            hdu.header['BAND_I'] = band_i
            hdu.header['BAND_J'] = band_j
            hdu.header['COMMENT'] = (
                "Corrected theoretical spectra: beam/unit/pixel corrections applied (no noise subtraction)"
            )
            hdu.name = key
            hdu_list.append(hdu)

        hdu_list.writeto(path_out_file, overwrite=True)
        print(f"[OK] Saved corrected theoretical spectra to {path_out_file}")

    return corr_spectra


'''
# ====================================
# 7
# ====================================
'''


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
        Scaling factor for dust emission in K_RJ units.
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
                      cc_dict=None,
                      freq_ref=23., ell_ref=80.0, freq_max_c=40.0):
    """
    Synchrotron angular power spectrum model.
    
    Parameters
    ----------
    theta : list or array
        Parameters [A_s, alpha_s, beta_s] (+ c_sync[band] if fit_c_terms,
        only for bands with freq <= freq_max_c).
    datasets : list of dict
        Prepared datasets with frequency pairs, spectra, and errors.
    ell : array
        Multipoles.
    fit_c_terms : bool
        Whether to fit constant terms for synchrotron auto-spectra
        at low frequencies (<= freq_max_c).
    freq_ref : float
        Reference frequency in GHz.
    ell_ref : float
        Reference multipole.
    freq_max_c : float
        Maximum frequency (GHz) for which constant terms are fitted.
        Default: 40.0 (i.e., only synchrotron-dominated bands).

    Returns
    -------
    model_vector : array
        Concatenated modeled spectra for all datasets.
    """
    A_s, alpha_s, beta_s = theta[:3]
    unique_freqs = sorted({f for d in datasets for f in d['freqs']})

    # Only fit c_terms for low-frequency (synchrotron-dominated) bands
    low_freqs = sorted({f for f in unique_freqs if f <= freq_max_c})
    low_freq_to_idx = {f: i for i, f in enumerate(low_freqs)}
    N_c = len(low_freqs)

    c_terms = np.zeros(N_c)
    if fit_c_terms:
        if len(theta) != 3 + N_c:
            raise ValueError(
                f"theta length mismatch for synch c_terms: expected {3 + N_c}, "
                f"got {len(theta)} (N_c={N_c} bands <= {freq_max_c} GHz)"
            )
        c_terms = np.asarray(theta[3:3+N_c])

    model_list = []

    for d in datasets:
        # Parse band identifiers (strings) from pair name like '11_353'
        band1_str, band2_str = d['pair'].split('_')
        f1, f2 = d['freqs']
        scale_f1 = (f1 / freq_ref) ** beta_s
        scale_f2 = (f2 / freq_ref) ** beta_s
        ell_scale = (ell / ell_ref) ** alpha_s
        # Apply per-band color corrections using alpha = 2 + beta_s (spectral index)
        if cc_dict is not None:
            alpha_cc = 2.0 + float(beta_s)
            poly1 = (cc_dict.get('synch', {}) or {}).get(str(band1_str))
            poly2 = (cc_dict.get('synch', {}) or {}).get(str(band2_str))
            cc_s1 = (poly1[0] + poly1[1]*alpha_cc + poly1[2]*(alpha_cc**2)) if poly1 is not None else 1.0
            cc_s2 = (poly2[0] + poly2[1]*alpha_cc + poly2[2]*(alpha_cc**2)) if poly2 is not None else 1.0
        else:
            cc_s1 = cc_s2 = 1.0
        # Physical model: power-law + constant term (for auto-spectra only)
        Cl = A_s * ell_scale * scale_f1 * scale_f2
        if fit_c_terms and (f1 == f2) and (f1 in low_freq_to_idx):
            Cl = Cl + c_terms[low_freq_to_idx[f1]]
        # Divide by color correction: model_obs = model_phys / (cc1 * cc2)
        # (data is raw/uncorrected, so the model must be divided by cc to match)
        Cl = Cl / (cc_s1 * cc_s2)
        model_list.append(Cl)
    return np.concatenate(model_list)


def model_synchrotron_joint(theta, datasets, ell, mode, fit_c_terms=False,
                           cc_dict=None,
                           freq_ref=23.0, ell_ref=80.0, freq_max_c=40.0):
    """
    Synchrotron angular power spectrum model for joint EE-BB analysis.
    
    Parameters
    ----------
    theta : list or array
        Parameters [A_s_EE, A_s_BB, alpha_s, beta_s] (+ c_sync[band] if fit_c_terms,
        only for bands with freq <= freq_max_c).
    datasets : list of dict
        Prepared datasets with frequency pairs, spectra, and errors.
    ell : array
        Multipoles.
    mode : str
        Either 'EE' or 'BB' - determines which amplitude to use.
    fit_c_terms : bool
        Whether to fit constant terms for synchrotron auto-spectra
        at low frequencies (<= freq_max_c).
    cc_dict : dict, optional
        Color correction polynomials.
    freq_ref : float
        Reference frequency in GHz.
    ell_ref : float
        Reference multipole.
    freq_max_c : float
        Maximum frequency (GHz) for which constant terms are fitted.
        Default: 40.0 (i.e., only synchrotron-dominated bands).

    Returns
    -------
    model_vector : array
        Concatenated modeled spectra for all datasets.
    """
    A_s_EE, A_s_BB, alpha_s, beta_s = theta[:4]
    
    # Select amplitude based on mode
    A_s = A_s_EE if mode == 'EE' else A_s_BB
    
    unique_freqs = sorted({f for d in datasets for f in d['freqs']})

    # Only fit c_terms for low-frequency (synchrotron-dominated) bands
    low_freqs = sorted({f for f in unique_freqs if f <= freq_max_c})
    low_freq_to_idx = {f: i for i, f in enumerate(low_freqs)}
    N_c = len(low_freqs)

    c_terms = np.zeros(N_c)
    if fit_c_terms:
        if len(theta) != 4 + N_c:
            raise ValueError(
                f"theta length mismatch for synch c_terms in joint analysis: "
                f"expected {4 + N_c}, got {len(theta)} (N_c={N_c} bands <= {freq_max_c} GHz)"
            )
        c_terms = np.asarray(theta[4:4+N_c])

    model_list = []

    for d in datasets:
        # Parse band identifiers from pair name like '11_353'
        band1_str, band2_str = d['pair'].split('_')
        f1, f2 = d['freqs']
        
        # Frequency scaling (shared beta_s for both modes)
        scale_f1 = (f1 / freq_ref) ** beta_s
        scale_f2 = (f2 / freq_ref) ** beta_s
        
        # Multipole scaling (shared alpha_s for both modes)
        ell_scale = (ell / ell_ref) ** alpha_s
        
        # Apply per-band color corrections using spectral index alpha = 2 + beta_s
        if cc_dict is not None:
            alpha_cc = 2.0 + float(beta_s)
            poly1 = (cc_dict.get('synch', {}) or {}).get(str(band1_str))
            poly2 = (cc_dict.get('synch', {}) or {}).get(str(band2_str))
            cc_s1 = (poly1[0] + poly1[1]*alpha_cc + poly1[2]*(alpha_cc**2)) if poly1 is not None else 1.0
            cc_s2 = (poly2[0] + poly2[1]*alpha_cc + poly2[2]*(alpha_cc**2)) if poly2 is not None else 1.0
        else:
            cc_s1 = cc_s2 = 1.0
        
        # Physical model: power-law + constant term (for auto-spectra only)
        Cl = A_s * ell_scale * scale_f1 * scale_f2
        if fit_c_terms and (f1 == f2) and (f1 in low_freq_to_idx):
            Cl = Cl + c_terms[low_freq_to_idx[f1]]
        # Divide by color correction: model_obs = model_phys / (cc1 * cc2)
        # (data is raw/uncorrected, so the model must be divided by cc to match)
        Cl = Cl / (cc_s1 * cc_s2)
        
        model_list.append(Cl)
    
    return np.concatenate(model_list)

def model_dust(theta, datasets, ell, fit_c_terms=False,
               cc_dict=None,
               freq_ref=353.0, T_d=19.6, ell_ref=80.0):
    """
    Dust angular power spectrum model with modified blackbody scaling in K_RJ units.

    Parameters
    ----------
    theta : list or array
        Parameters [A_d, alpha_d, beta_d]. No constant terms for dust.
    datasets : list of dict
        Prepared datasets.
    ell : array
        Multipoles.
    fit_c_terms : bool
        Ignored (kept for API compatibility). No c_terms for dust.
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

    # Precompute per-frequency dust scaling (K_RJ units)
    freqs_all = np.array(unique_freqs)
    S = mbb_scaling_KRJ(freqs_all, nu0_GHz=freq_ref, beta=beta_d, T_d=T_d)

    model_list = []

    for d in datasets:
        band1_str, band2_str = d['pair'].split('_')
        f1, f2 = d['freqs']
        s1 = S[freq_to_idx[f1]]
        s2 = S[freq_to_idx[f2]]
        ell_scale = (ell / ell_ref) ** alpha_d
        # Apply per-band color corrections: alpha_cc = 2 + beta_d (frequency spectral index)
        if cc_dict is not None:
            alpha_cc = 2.0 + float(beta_d)
            poly1 = (cc_dict.get('dust', {}) or {}).get(str(band1_str))
            poly2 = (cc_dict.get('dust', {}) or {}).get(str(band2_str))
            cc_d1 = (poly1[0] + poly1[1]*alpha_cc + poly1[2]*(alpha_cc**2)) if poly1 is not None else 1.0
            cc_d2 = (poly2[0] + poly2[1]*alpha_cc + poly2[2]*(alpha_cc**2)) if poly2 is not None else 1.0
        else:
            cc_d1 = cc_d2 = 1.0
        Cl = A_d * ell_scale * s1 * s2 / (cc_d1 * cc_d2)
        model_list.append(Cl)
    return np.concatenate(model_list)



def model_dust_joint(theta, datasets, ell, mode, fit_c_terms=False,
                    cc_dict=None,
                    freq_ref=353.0, T_d=19.6, ell_ref=80.0):
    """
    Dust angular power spectrum model for joint EE-BB analysis with modified 
    blackbody scaling in K_RJ units.

    Parameters
    ----------
    theta : list or array
        Parameters [A_d_EE, A_d_BB, alpha_d, beta_d]. No constant terms for dust.
    datasets : list of dict
        Prepared datasets.
    ell : array
        Multipoles.
    mode : str
        Either 'EE' or 'BB' - determines which amplitude to use.
    fit_c_terms : bool
        Ignored (kept for API compatibility). No c_terms for dust.
    cc_dict : dict, optional
        Color correction polynomials.
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
    A_d_EE, A_d_BB, alpha_d, beta_d = theta[:4]
    
    # Select amplitude based on mode
    A_d = A_d_EE if mode == 'EE' else A_d_BB
    
    unique_freqs = sorted({f for d in datasets for f in d['freqs']})
    freq_to_idx = {f: i for i, f in enumerate(unique_freqs)}

    # Precompute per-frequency dust scaling (K_RJ units)
    freqs_all = np.array(unique_freqs)
    S = mbb_scaling_KRJ(freqs_all, nu0_GHz=freq_ref, beta=beta_d, T_d=T_d)

    model_list = []

    for d in datasets:
        band1_str, band2_str = d['pair'].split('_')
        f1, f2 = d['freqs']
        
        # Frequency scaling using modified blackbody (shared beta_d for both modes)
        s1 = S[freq_to_idx[f1]]
        s2 = S[freq_to_idx[f2]]
        
        # Multipole scaling (shared alpha_d for both modes)
        ell_scale = (ell / ell_ref) ** alpha_d
        
        # Apply per-band color corrections: alpha_cc = 2 + beta_d (frequency spectral index)
        if cc_dict is not None:
            alpha_cc = 2.0 + float(beta_d)
            poly1 = (cc_dict.get('dust', {}) or {}).get(str(band1_str))
            poly2 = (cc_dict.get('dust', {}) or {}).get(str(band2_str))
            cc_d1 = (poly1[0] + poly1[1]*alpha_cc + poly1[2]*(alpha_cc**2)) if poly1 is not None else 1.0
            cc_d2 = (poly2[0] + poly2[1]*alpha_cc + poly2[2]*(alpha_cc**2)) if poly2 is not None else 1.0
        else:
            cc_d1 = cc_d2 = 1.0
        
        # Build model: amplitude (mode-specific) × ell_scale (shared) × freq_scale (shared) / cc
        Cl = A_d * ell_scale * s1 * s2 / (cc_d1 * cc_d2)
        
        model_list.append(Cl)
    
    return np.concatenate(model_list)


def model_cross(theta, datasets, ell,
                cc_dict=None,
                ref_sync=23., ref_dust=353.0, T_d=19.6, ell_ref=80.0):
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
    # Use K_RJ units for dust (Planck convention)
    Sd = mbb_scaling_KRJ(freqs_all, nu0_GHz=ref_dust, beta=beta_d, T_d=T_d)

    model_list = []

    for d in datasets:
        band1_str, band2_str = d['pair'].split('_')
        f1, f2 = d['freqs']
        s1 = (f1 / ref_sync) ** beta_s
        s2 = (f2 / ref_sync) ** beta_s
        ell_scale_s = (ell / ell_ref) ** alpha_s
        # synch-only term if needed
        C_s_ij = A_s * ell_scale_s * s1 * s2

        d1 = Sd[freq_to_idx[f1]]
        d2 = Sd[freq_to_idx[f2]]
        ell_scale_d = (ell / ell_ref) ** alpha_d
        C_d_ij = A_d * ell_scale_d * d1 * d2

        # cross term (rho * sqrt(C_s * C_d))
        ell_scale_cross = (ell / ell_ref) ** ((alpha_s + alpha_d) / 2)
        # Apply per-band color corrections to the mixing terms using alpha_s=2+beta_s and alpha_d=2+beta_d
        if cc_dict is not None:
            alpha_s_cc = 2.0 + float(beta_s)
            alpha_d_cc = 2.0 + float(beta_d)
            syn1 = (cc_dict.get('synch', {}) or {}).get(str(band1_str))
            syn2 = (cc_dict.get('synch', {}) or {}).get(str(band2_str))
            dus1 = (cc_dict.get('dust', {}) or {}).get(str(band1_str))
            dus2 = (cc_dict.get('dust', {}) or {}).get(str(band2_str))
            cc_s1 = (syn1[0] + syn1[1]*alpha_s_cc + syn1[2]*(alpha_s_cc**2)) if syn1 is not None else 1.0
            cc_s2 = (syn2[0] + syn2[1]*alpha_s_cc + syn2[2]*(alpha_s_cc**2)) if syn2 is not None else 1.0
            cc_d1 = (dus1[0] + dus1[1]*alpha_d_cc + dus1[2]*(alpha_d_cc**2)) if dus1 is not None else 1.0
            cc_d2 = (dus2[0] + dus2[1]*alpha_d_cc + dus2[2]*(alpha_d_cc**2)) if dus2 is not None else 1.0
        else:
            cc_s1 = cc_s2 = cc_d1 = cc_d2 = 1.0
        mix = ( (s1 / cc_s1) * (d2 / cc_d2) + (s2 / cc_s2) * (d1 / cc_d1) )
        As_Ad = A_s * A_d
        C_sd_ij = rho * np.sign(As_Ad) * np.sqrt(As_Ad) * mix * ell_scale_cross

        model_list.append(C_sd_ij)
    return np.concatenate(model_list)



def model_cross_joint(theta, datasets, ell, mode,
                     cc_dict=None,
                     ref_sync=23.0, ref_dust=353.0, T_d=19.6, ell_ref=80.0):
    """
    Cross-correlation between synchrotron and dust components for joint EE-BB analysis.

    Parameters
    ----------
    theta : list or array
        Parameters [rho, A_s_EE, A_s_BB, A_d_EE, A_d_BB, alpha_s, alpha_d, beta_s, beta_d].
    datasets : list of dict
        Prepared datasets.
    ell : array
        Multipoles.
    mode : str
        Either 'EE' or 'BB' - determines which amplitudes to use.
    cc_dict : dict, optional
        Color correction polynomials.
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
    rho, A_s_EE, A_s_BB, A_d_EE, A_d_BB, alpha_s, alpha_d, beta_s, beta_d = theta
    
    # Select amplitudes based on mode
    A_s = A_s_EE if mode == 'EE' else A_s_BB
    A_d = A_d_EE if mode == 'EE' else A_d_BB
    
    unique_freqs = sorted({f for d in datasets for f in d['freqs']})
    freq_to_idx = {f: i for i, f in enumerate(unique_freqs)}

    freqs_all = np.array(unique_freqs)
    # Use K_RJ units for dust (Planck convention)
    Sd = mbb_scaling_KRJ(freqs_all, nu0_GHz=ref_dust, beta=beta_d, T_d=T_d)

    model_list = []

    for d in datasets:
        band1_str, band2_str = d['pair'].split('_')
        f1, f2 = d['freqs']
        
        # Synchrotron frequency scaling (shared beta_s)
        s1 = (f1 / ref_sync) ** beta_s
        s2 = (f2 / ref_sync) ** beta_s
        
        # Dust frequency scaling (shared beta_d)
        d1 = Sd[freq_to_idx[f1]]
        d2 = Sd[freq_to_idx[f2]]
        
        # Multipole scaling: average of synch and dust slopes (shared)
        ell_scale_cross = (ell / ell_ref) ** ((alpha_s + alpha_d) / 2)
        
        # Apply per-band color corrections
        if cc_dict is not None:
            alpha_s_cc = 2.0 + float(beta_s)
            alpha_d_cc = 2.0 + float(beta_d)
            
            syn1 = (cc_dict.get('synch', {}) or {}).get(str(band1_str))
            syn2 = (cc_dict.get('synch', {}) or {}).get(str(band2_str))
            dus1 = (cc_dict.get('dust', {}) or {}).get(str(band1_str))
            dus2 = (cc_dict.get('dust', {}) or {}).get(str(band2_str))
            
            cc_s1 = (syn1[0] + syn1[1]*alpha_s_cc + syn1[2]*(alpha_s_cc**2)) if syn1 is not None else 1.0
            cc_s2 = (syn2[0] + syn2[1]*alpha_s_cc + syn2[2]*(alpha_s_cc**2)) if syn2 is not None else 1.0
            cc_d1 = (dus1[0] + dus1[1]*alpha_d_cc + dus1[2]*(alpha_d_cc**2)) if dus1 is not None else 1.0
            cc_d2 = (dus2[0] + dus2[1]*alpha_d_cc + dus2[2]*(alpha_d_cc**2)) if dus2 is not None else 1.0
        else:
            cc_s1 = cc_s2 = cc_d1 = cc_d2 = 1.0
        
        # Cross term: rho × sqrt(A_s × A_d) × (s1/cc_s1×d2/cc_d2 + s2/cc_s2×d1/cc_d1) × ell_scale
        # This represents the correlation between synchrotron and dust
        # Division by cc because data is raw/uncorrected
        mix = ((s1 / cc_s1) * (d2 / cc_d2) + (s2 / cc_s2) * (d1 / cc_d1))
        As_Ad = A_s * A_d
        C_sd_ij = rho * np.sign(As_Ad) * np.sqrt(As_Ad) * mix * ell_scale_cross

        model_list.append(C_sd_ij)
    
    return np.concatenate(model_list)


def log_multivariate_gamma(p, a):
    """
    log Γ_p(a) = (p(p-1)/4) log π + sum_{i=1}^p log Γ(a + (1-i)/2)
    """
    return (
        0.25 * p * (p - 1) * log(pi)
        + sum(gammaln(a + 0.5 * (1 - i)) for i in range(1, p + 1))
    )


def make_spd(M, eps=0.0):
    """
    Symmetrize matrix and add diagonal jitter if needed
    to ensure positive definiteness.
    """
    M = 0.5 * (M + M.T)

    sign, _ = np.linalg.slogdet(M)
    if sign > 0 and eps == 0.0:
        return M

    # adaptive jitter
    scale = np.max(np.abs(np.diag(M)))
    jitter = eps if eps > 0 else (1e-6 * scale if scale > 0 else 1e-12)

    return M + jitter * np.eye(M.shape[0])

def loglik_wishart(C_hat, C_model, nu, drop_const=True, jitter=0.0):
    """
    Log-likelihood for the (scaled) Wishart distribution.

    Parameters
    ----------
    C_hat : array
        Empirical covariance(s):
        - scalar case: (L,)
        - multivariate case: (L, p, p)
    C_model : array
        Model covariance(s), same shape as C_hat
    nu : array (L,)
        Degrees of freedom (typically 2ell + 1)
    """
    C_hat = np.asarray(C_hat)
    C_model = np.asarray(C_model)
    nu = np.asarray(nu)

    if C_hat.shape != C_model.shape:
        raise ValueError("C_hat and C_model must have the same shape.")

    L = nu.size
    logL = 0.0

    # --------------------------------------------------
    # Scalar case (p = 1)
    # --------------------------------------------------
    if C_hat.ndim == 1:
        p = 1
        eps = jitter if jitter > 0 else 1e-30

        for i in range(L):
            Sh = max(float(C_hat[i]), eps)
            Sm = max(float(C_model[i]), eps)

            logL += 0.5 * (
                (nu[i] - p - 1) * np.log(Sh)
                - nu[i] * (Sh / Sm)
                - nu[i] * np.log(Sm)
            )

    # --------------------------------------------------
    # Multivariate case
    # --------------------------------------------------
    else:
        if C_hat.ndim != 3 or C_hat.shape[1] != C_hat.shape[2]:
            raise ValueError("Multivariate case requires shape (L, p, p).")

        p = C_hat.shape[1]

        for i in range(L):
            Sh = make_spd(C_hat[i], jitter)
            Sm = make_spd(C_model[i], jitter)

            s_sign, s_logdet = np.linalg.slogdet(Sh)
            m_sign, m_logdet = np.linalg.slogdet(Sm)

            if s_sign <= 0 or m_sign <= 0:
                raise ValueError("Covariance matrices must be positive definite.")

            # Tr(C_model^{-1} C_hat)
            tr_term = np.trace(np.linalg.solve(Sm, Sh))

            logL += 0.5 * (
                (nu[i] - p - 1) * s_logdet
                - nu[i] * tr_term
                - nu[i] * m_logdet
            )

    # --------------------------------------------------
    # Normalization constant (optional)
    # --------------------------------------------------
    if not drop_const:
        const = 0.0
        for i in range(L):
            const += 0.5 * nu[i] * p * np.log(2.0)
            const += log_multivariate_gamma(p, nu[i] / 2.0)
        logL -= const

    return float(logL)


def lnlike(theta_full, datasets, ell, y_all, yerr_all,
           fit_c_terms=False, fit_components=('sync', 'dust', 'cross'),
           cc_dict=None, cov_matrix=None, freq_max_c=40.0):
    """
    Compute the log-likelihood (-0.5 chi^2) for the model given data.

    Parameters
    ----------
    theta_full : array
        Full parameter vector:
        [A_s, alpha_s, beta_s, A_d, alpha_d, beta_d, rho]
        (+ c_sync[low-freq bands] if fit_c_terms=True, synchrotron only, freq <= freq_max_c)
    datasets : list of dict
        Prepared datasets.
    ell : array
        Multipoles.
    y_all : array
        Observed spectra concatenated.
    yerr_all : array
        Errors associated with y_all (used if cov_matrix is None).
    fit_c_terms : bool
        Whether to fit constant c terms (synchrotron only, low-freq bands).
    fit_components : tuple
        Components to include ('sync','dust','cross').
    cov_matrix : dict, optional
        Dictionary with keys 'indices_11_13', 'indices_17_19', 'cov_inv'.
        Uses block-diagonal covariance ONLY for QUIJOTE blocks,
        diagonal (1/sigma^2) for all others (FAST!).
        If None, uses diagonal covariance: chi^2 = sum((data - model)^2 / sigma^2).
    freq_max_c : float
        Maximum frequency (GHz) for synchrotron constant terms. Default: 40.0.

    Returns
    -------
    lnL : float
        Log-likelihood value.
    """
    unique_freqs = sorted({f for d in datasets for f in d['freqs']})
    # Only low-frequency bands get c_terms (synchrotron only, no dust)
    low_freqs = sorted({f for f in unique_freqs if f <= freq_max_c})
    N_c = len(low_freqs)

    A_s, alpha_s, beta_s, A_d, alpha_d, beta_d, rho = theta_full[:7]
    c_sync = np.zeros(N_c)
    offset = 7
    if fit_c_terms:
        c_sync = np.asarray(theta_full[offset:offset+N_c]); offset += N_c

    y_model = np.zeros_like(y_all)

    if 'sync' in fit_components:
        y_model += model_synchrotron([A_s, alpha_s, beta_s, *c_sync] if fit_c_terms else
                                     [A_s, alpha_s, beta_s],
                                     datasets, ell, fit_c_terms=fit_c_terms, cc_dict=cc_dict,
                                     freq_max_c=freq_max_c)

    if 'dust' in fit_components:
        y_model += model_dust([A_d, alpha_d, beta_d],
                              datasets, ell, cc_dict=cc_dict)

    if 'cross' in fit_components:
        y_model += model_cross([rho, A_s, A_d, alpha_s, alpha_d, beta_s, beta_d],
                               datasets, ell, cc_dict=cc_dict)

    residual = y_all - y_model
    
    if cov_matrix is not None:
        # FAST block-diagonal approach: only use covariance for QUIJOTE blocks
        idx_11_13 = cov_matrix['indices_11_13']
        idx_17_19 = cov_matrix['indices_17_19']
        cov_inv_11_13 = cov_matrix['cov_inv_11_13']
        cov_inv_17_19 = cov_matrix['cov_inv_17_19']
        
        # Start with diagonal chi-squared for ALL points
        chi2 = np.sum((residual / yerr_all) ** 2)
        
        # Subtract diagonal contribution for QUIJOTE blocks and add covariance
        # (do indexing only once per block)
        if len(idx_11_13) > 0:
            res_block = residual[idx_11_13]
            yerr_block = yerr_all[idx_11_13]
            chi2 -= np.sum((res_block / yerr_block) ** 2)  # Subtract diagonal
            chi2 += res_block @ cov_inv_11_13 @ res_block  # Add covariance
        
        if len(idx_17_19) > 0:
            res_block = residual[idx_17_19]
            yerr_block = yerr_all[idx_17_19]
            chi2 -= np.sum((res_block / yerr_block) ** 2)  # Subtract diagonal
            chi2 += res_block @ cov_inv_17_19 @ res_block  # Add covariance
    else:
        # Use diagonal covariance: chi^2 = sum((residual / sigma)^2)
        chi2 = np.sum((residual / yerr_all) ** 2)
    
    return -0.5 * chi2



def lnlike_joint(theta_full, datasets_EE, datasets_BB, ell, 
                 y_EE, yerr_EE, y_BB, yerr_BB,
                 fit_c_terms=False, fit_components=('sync', 'dust', 'cross'),
                 cc_dict=None, cov_matrix=None, freq_max_c=40.0):
    """
    Compute the log-likelihood for joint EE-BB analysis.

    Parameters
    ----------
    theta_full : array
        Full parameter vector:
        [A_s_EE, A_s_BB, alpha_s, beta_s, A_d_EE, A_d_BB, alpha_d, beta_d, rho]
        (+ c_sync_EE[low-freq] + c_sync_BB[low-freq] if fit_c_terms=True,
         synchrotron only, for bands with freq <= freq_max_c)
    datasets_EE : list of dict
        Prepared datasets for EE mode.
    datasets_BB : list of dict
        Prepared datasets for BB mode.
    ell : array
        Multipoles.
    y_EE : array
        Observed EE spectra concatenated.
    yerr_EE : array
        Errors for EE spectra (used if cov_matrix is None).
    y_BB : array
        Observed BB spectra concatenated.
    yerr_BB : array
        Errors for BB spectra (used if cov_matrix is None).
    fit_c_terms : bool
        Whether to fit constant c terms (synchrotron only, low-freq bands).
    fit_components : tuple
        Components to include ('sync','dust','cross').
    cc_dict : dict, optional
        Color correction polynomials.
    cov_matrix : dict, optional
        Dictionary with keys 'indices_11_13', 'indices_17_19', 'cov_inv'.
        Uses block-diagonal covariance ONLY for QUIJOTE blocks,
        diagonal (1/sigma^2) for all others (FAST!).
        If None, uses diagonal covariance separately for EE and BB.
    freq_max_c : float
        Maximum frequency (GHz) for synchrotron constant terms. Default: 40.0.

    Returns
    -------
    lnL : float
        Log-likelihood value.
    """
    # Extract base parameters
    A_s_EE, A_s_BB = theta_full[0], theta_full[1]
    alpha_s, beta_s = theta_full[2], theta_full[3]
    A_d_EE, A_d_BB = theta_full[4], theta_full[5]
    alpha_d, beta_d = theta_full[6], theta_full[7]
    rho = theta_full[8]
    
    # Extract c_terms if present (synchrotron only, low-freq bands only)
    unique_freqs = sorted({f for d in datasets_EE + datasets_BB for f in d['freqs']})
    low_freqs = sorted({f for f in unique_freqs if f <= freq_max_c})
    N_c = len(low_freqs)
    
    c_sync_EE = np.zeros(N_c)
    c_sync_BB = np.zeros(N_c)
    
    offset = 9
    if fit_c_terms:
        c_sync_EE = np.asarray(theta_full[offset:offset+N_c]); offset += N_c
        c_sync_BB = np.asarray(theta_full[offset:offset+N_c]); offset += N_c
    
    # =========================================================================
    # Build model for EE mode
    # =========================================================================
    y_model_EE = np.zeros_like(y_EE)
    
    if 'sync' in fit_components:
        theta_sync = [A_s_EE, A_s_BB, alpha_s, beta_s]
        if fit_c_terms:
            theta_sync.extend(c_sync_EE)
        y_model_EE += model_synchrotron_joint(
            theta_sync, datasets_EE, ell, mode='EE',
            fit_c_terms=fit_c_terms, cc_dict=cc_dict,
            freq_max_c=freq_max_c
        )
    
    if 'dust' in fit_components:
        theta_dust = [A_d_EE, A_d_BB, alpha_d, beta_d]
        y_model_EE += model_dust_joint(
            theta_dust, datasets_EE, ell, mode='EE',
            cc_dict=cc_dict
        )
    
    if 'cross' in fit_components:
        theta_cross = [rho, A_s_EE, A_s_BB, A_d_EE, A_d_BB, 
                      alpha_s, alpha_d, beta_s, beta_d]
        y_model_EE += model_cross_joint(
            theta_cross, datasets_EE, ell, mode='EE',
            cc_dict=cc_dict
        )
    
    # =========================================================================
    # Build model for BB mode
    # =========================================================================
    y_model_BB = np.zeros_like(y_BB)
    
    if 'sync' in fit_components:
        theta_sync = [A_s_EE, A_s_BB, alpha_s, beta_s]
        if fit_c_terms:
            theta_sync.extend(c_sync_BB)
        y_model_BB += model_synchrotron_joint(
            theta_sync, datasets_BB, ell, mode='BB',
            fit_c_terms=fit_c_terms, cc_dict=cc_dict,
            freq_max_c=freq_max_c
        )
    
    if 'dust' in fit_components:
        theta_dust = [A_d_EE, A_d_BB, alpha_d, beta_d]
        y_model_BB += model_dust_joint(
            theta_dust, datasets_BB, ell, mode='BB',
            cc_dict=cc_dict
        )
    
    if 'cross' in fit_components:
        theta_cross = [rho, A_s_EE, A_s_BB, A_d_EE, A_d_BB,
                      alpha_s, alpha_d, beta_s, beta_d]
        y_model_BB += model_cross_joint(
            theta_cross, datasets_BB, ell, mode='BB',
            cc_dict=cc_dict
        )
    
    # =========================================================================
    # Compute combined chi-squared
    # =========================================================================
    if cov_matrix is not None:
        # FAST block-diagonal approach for joint EE+BB
        idx_11_13 = cov_matrix['indices_11_13']
        idx_17_19 = cov_matrix['indices_17_19']
        cov_inv_11_13 = cov_matrix['cov_inv_11_13']
        cov_inv_17_19 = cov_matrix['cov_inv_17_19']
        
        # Concatenate EE and BB residuals and errors
        residual_combined = np.concatenate([y_EE - y_model_EE, y_BB - y_model_BB])
        yerr_combined = np.concatenate([yerr_EE, yerr_BB])
        
        # Start with diagonal chi-squared for ALL points
        chi2_total = np.sum((residual_combined / yerr_combined) ** 2)
        
        # Subtract diagonal contribution for QUIJOTE blocks and add covariance
        # (do indexing only once per block)
        if len(idx_11_13) > 0:
            res_block = residual_combined[idx_11_13]
            yerr_block = yerr_combined[idx_11_13]
            chi2_total -= np.sum((res_block / yerr_block) ** 2)  # Subtract diagonal
            chi2_total += res_block @ cov_inv_11_13 @ res_block  # Add covariance
        
        if len(idx_17_19) > 0:
            res_block = residual_combined[idx_17_19]
            yerr_block = yerr_combined[idx_17_19]
            chi2_total -= np.sum((res_block / yerr_block) ** 2)  # Subtract diagonal
            chi2_total += res_block @ cov_inv_17_19 @ res_block  # Add covariance
    else:
        # Use diagonal covariance (separate for EE and BB)
        chi2_EE = np.sum(((y_EE - y_model_EE) / yerr_EE) ** 2)
        chi2_BB = np.sum(((y_BB - y_model_BB) / yerr_BB) ** 2)
        chi2_total = chi2_EE + chi2_BB
    
    return -0.5 * chi2_total



def compute_chi2_reduced(theta_full, datasets, ell, y_all, yerr_all, 
                         fit_c_terms=False, fit_components=('sync', 'dust', 'cross'),
                         cc_dict=None, freq_max_c=40.0, n_free_params=None):
    """
    Compute the reduced chi-squared for the model given data.

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
        Whether to fit constant c terms (synchrotron only, low-freq bands).
    fit_components : tuple
        Components to include ('sync','dust','cross').
    freq_max_c : float
        Maximum frequency (GHz) for synchrotron constant terms. Default: 40.0.
    n_free_params : int or None, optional
        Exact number of free parameters actually fitted (from the param_map).
        When provided this overrides the internal component-based counting,
        which is necessary when parameters are frozen via ``freeze_params``
        (e.g., ``beta_s`` frozen → n_free_params should be 2 not 3 for sync-only).

    Returns
    -------
    chi2_reduced : float
        Reduced chi-squared value (chi2 / degrees of freedom).
    """
    unique_freqs = sorted({f for d in datasets for f in d['freqs']})
    low_freqs = sorted({f for f in unique_freqs if f <= freq_max_c})
    N_c = len(low_freqs)

    A_s, alpha_s, beta_s, A_d, alpha_d, beta_d, rho = theta_full[:7]
    c_sync = np.zeros(N_c)
    offset = 7
    if fit_c_terms:
        c_sync = np.asarray(theta_full[offset:offset+N_c]); offset += N_c

    y_model = np.zeros_like(y_all)

    if 'sync' in fit_components:
        y_model += model_synchrotron([A_s, alpha_s, beta_s, *c_sync] if fit_c_terms else
                                     [A_s, alpha_s, beta_s],
                                     datasets, ell, fit_c_terms=fit_c_terms, cc_dict=cc_dict,
                                     freq_max_c=freq_max_c)

    if 'dust' in fit_components:
        y_model += model_dust([A_d, alpha_d, beta_d],
                              datasets, ell, cc_dict=cc_dict)

    if 'cross' in fit_components:
        y_model += model_cross([rho, A_s, A_d, alpha_s, alpha_d, beta_s, beta_d],
                               datasets, ell, cc_dict=cc_dict)

    chi2 = np.sum(((y_all - y_model) / yerr_all) ** 2)
    
    # Calculate degrees of freedom: number of data points - number of free parameters
    n_data = len(y_all)
    
    if n_free_params is not None:
        # Use the exact count passed by the caller (accounts for frozen params)
        n_params = int(n_free_params)
    else:
        # Fallback: infer from fit_components (may over-count if params were frozen)
        n_params = 0
        if 'sync' in fit_components:
            n_params += 3  # A_s, alpha_s, beta_s
            if fit_c_terms:
                n_params += N_c  # c_sync terms (low-freq only)
        if 'dust' in fit_components:
            n_params += 3  # A_d, alpha_d, beta_d
        if 'cross' in fit_components:
            n_params += 1  # rho
    
    dof = n_data - n_params
    
    if dof <= 0:
        return np.inf  # Invalid case: zero or negative degrees of freedom
    
    return chi2 / dof


def compute_chi2_reduced_joint(theta_full, datasets_EE, datasets_BB, ell,
                               y_EE, yerr_EE, y_BB, yerr_BB,
                               fit_c_terms=False, fit_components=('sync', 'dust', 'cross'),
                               cc_dict=None, freq_max_c=40.0):
    """
    Compute the reduced chi-squared for joint EE-BB analysis.

    Parameters
    ----------
    theta_full : array
        Full parameter vector.
    datasets_EE : list of dict
        Prepared datasets for EE mode.
    datasets_BB : list of dict
        Prepared datasets for BB mode.
    ell : array
        Multipoles.
    y_EE : array
        Observed EE spectra.
    yerr_EE : array
        Errors for EE spectra.
    y_BB : array
        Observed BB spectra.
    yerr_BB : array
        Errors for BB spectra.
    fit_c_terms : bool
        Whether constant c terms are fitted (synchrotron only, low-freq bands).
    fit_components : tuple
        Components included in fit.
    cc_dict : dict, optional
        Color correction polynomials.
    freq_max_c : float
        Maximum frequency (GHz) for synchrotron constant terms. Default: 40.0.

    Returns
    -------
    chi2_reduced : float
        Reduced chi-squared value.
    """
    # Extract parameters
    A_s_EE, A_s_BB = theta_full[0], theta_full[1]
    alpha_s, beta_s = theta_full[2], theta_full[3]
    A_d_EE, A_d_BB = theta_full[4], theta_full[5]
    alpha_d, beta_d = theta_full[6], theta_full[7]
    rho = theta_full[8]
    
    unique_freqs = sorted({f for d in datasets_EE + datasets_BB for f in d['freqs']})
    low_freqs = sorted({f for f in unique_freqs if f <= freq_max_c})
    N_c = len(low_freqs)
    
    c_sync_EE = np.zeros(N_c)
    c_sync_BB = np.zeros(N_c)
    
    offset = 9
    if fit_c_terms:
        c_sync_EE = np.asarray(theta_full[offset:offset+N_c]); offset += N_c
        c_sync_BB = np.asarray(theta_full[offset:offset+N_c]); offset += N_c
    
    # Build models
    y_model_EE = np.zeros_like(y_EE)
    y_model_BB = np.zeros_like(y_BB)
    
    if 'sync' in fit_components:
        theta_sync = [A_s_EE, A_s_BB, alpha_s, beta_s]
        if fit_c_terms:
            theta_sync.extend(c_sync_EE)
        y_model_EE += model_synchrotron_joint(
            theta_sync, datasets_EE, ell, mode='EE',
            fit_c_terms=fit_c_terms, cc_dict=cc_dict,
            freq_max_c=freq_max_c
        )
        
        theta_sync_BB = [A_s_EE, A_s_BB, alpha_s, beta_s]
        if fit_c_terms:
            theta_sync_BB.extend(c_sync_BB)
        y_model_BB += model_synchrotron_joint(
            theta_sync_BB, datasets_BB, ell, mode='BB',
            fit_c_terms=fit_c_terms, cc_dict=cc_dict,
            freq_max_c=freq_max_c
        )
    
    if 'dust' in fit_components:
        theta_dust = [A_d_EE, A_d_BB, alpha_d, beta_d]
        y_model_EE += model_dust_joint(
            theta_dust, datasets_EE, ell, mode='EE',
            cc_dict=cc_dict
        )
        
        theta_dust_BB = [A_d_EE, A_d_BB, alpha_d, beta_d]
        y_model_BB += model_dust_joint(
            theta_dust_BB, datasets_BB, ell, mode='BB',
            cc_dict=cc_dict
        )
    
    if 'cross' in fit_components:
        theta_cross = [rho, A_s_EE, A_s_BB, A_d_EE, A_d_BB,
                      alpha_s, alpha_d, beta_s, beta_d]
        y_model_EE += model_cross_joint(
            theta_cross, datasets_EE, ell, mode='EE',
            cc_dict=cc_dict
        )
        y_model_BB += model_cross_joint(
            theta_cross, datasets_BB, ell, mode='BB',
            cc_dict=cc_dict
        )
    
    # Compute chi-squared
    chi2_EE = np.sum(((y_EE - y_model_EE) / yerr_EE) ** 2)
    chi2_BB = np.sum(((y_BB - y_model_BB) / yerr_BB) ** 2)
    chi2_total = chi2_EE + chi2_BB
    
    # Calculate degrees of freedom
    n_data = len(y_EE) + len(y_BB)
    n_params = 9  # Base parameters: A_s_EE, A_s_BB, alpha_s, beta_s, A_d_EE, A_d_BB, alpha_d, beta_d, rho
    
    if fit_c_terms:
        # 2 sets of N_c c-terms (c_sync_EE, c_sync_BB) - synchrotron only, low-freq only
        n_params += 2 * N_c
    
    # Adjust for components not being fitted
    if 'sync' not in fit_components:
        n_params -= 4  # A_s_EE, A_s_BB, alpha_s, beta_s
        if fit_c_terms:
            n_params -= 2 * N_c  # c_sync_EE, c_sync_BB
    if 'dust' not in fit_components:
        n_params -= 4  # A_d_EE, A_d_BB, alpha_d, beta_d
    if 'cross' not in fit_components:
        n_params -= 1  # rho
    
    dof = n_data - n_params
    
    if dof <= 0:
        return np.inf
    
    return chi2_total / dof


def lnprior(theta_full, datasets, fit_c_terms=False, fit_components=('sync', 'dust', 'cross'),
            freq_max_c=40.0):
    """
    Apply priors on parameters.

    Returns -np.inf if any prior is violated.
    """
    unique_freqs = sorted({f for d in datasets for f in d['freqs']})
    # Only low-frequency bands get c_terms (synchrotron only)
    low_freqs = sorted({f for f in unique_freqs if f <= freq_max_c})
    N_c = len(low_freqs)

    A_s, alpha_s, beta_s, A_d, alpha_d, beta_d, rho = theta_full[:7]

    if 'sync' in fit_components:
        if A_s <= 0: return -np.inf                  # amplitude must be positive
        if not (-6 <= alpha_s <= 0): return -np.inf
        if not (-6 <= beta_s <= 0): return -np.inf

    if 'dust' in fit_components:
        if A_d <= 0: return -np.inf                  # amplitude must be positive
        if not (-6 <= alpha_d <= 0): return -np.inf
        if not (0 <= beta_d <= 6): return -np.inf

    if 'cross' in fit_components:
        if not (-1 <= rho <= 1): return -np.inf

    # Add optional Gaussian priors
    lp = 0.0

    if fit_c_terms:
        offset = 7
        if 'sync' in fit_components:
            c_sync = np.asarray(theta_full[offset:offset+N_c]); offset += N_c
            if not np.all(np.isfinite(c_sync)):
                return -np.inf
            # Gaussian prior on each c_sync centred at 0 with scale = typical |A_s|
            # This prevents walkers from drifting to ±∞ on the flat likelihood surface.
            sigma_c = abs(A_s) if A_s > 0 else 1.0
            lp += -0.5 * np.sum((c_sync / sigma_c) ** 2)
    try:
        if 'beta_s' in _GAUSSIAN_PRIORS:
            mu, sig = _GAUSSIAN_PRIORS['beta_s']
            if sig > 0:
                lp += -0.5 * ((beta_s - mu)/sig)**2
        if 'beta_d' in _GAUSSIAN_PRIORS:
            mu, sig = _GAUSSIAN_PRIORS['beta_d']
            if sig > 0:
                lp += -0.5 * ((beta_d - mu)/sig)**2
        if 'rho' in _GAUSSIAN_PRIORS:
            mu, sig = _GAUSSIAN_PRIORS['rho']
            if sig > 0:
                lp += -0.5 * ((rho - mu)/sig)**2
        if 'alpha_s' in _GAUSSIAN_PRIORS:
            mu, sig = _GAUSSIAN_PRIORS['alpha_s']
            if sig > 0:
                lp += -0.5 * ((alpha_s - mu)/sig)**2
        if 'alpha_d' in _GAUSSIAN_PRIORS:
            mu, sig = _GAUSSIAN_PRIORS['alpha_d']
            if sig > 0:
                lp += -0.5 * ((alpha_d - mu)/sig)**2
        if 'A_s' in _GAUSSIAN_PRIORS:
            mu, sig = _GAUSSIAN_PRIORS['A_s']
            if sig > 0:
                lp += -0.5 * ((A_s - mu)/sig)**2
        if 'A_d' in _GAUSSIAN_PRIORS:
            mu, sig = _GAUSSIAN_PRIORS['A_d']
            if sig > 0:
                lp += -0.5 * ((A_d - mu)/sig)**2
    except Exception:
        # Fail-safe: do not crash if priors are malformed
        pass

    return lp


def lnprior_joint(theta_full, fit_c_terms=False, fit_components=('sync', 'dust', 'cross'),
                  n_c_terms=0):
    """
    Apply priors on parameters for joint EE-BB analysis.

    Parameters
    ----------
    theta_full : array
        Full parameter vector.
    fit_c_terms : bool
        Whether c terms are included (synchrotron only, low-freq bands).
    fit_components : tuple
        Components included in fit.
    n_c_terms : int
        Number of low-frequency bands with c_terms (per mode).

    Returns
    -------
    lnp : float
        Log-prior value. Returns -np.inf if any prior is violated.
    """
    A_s_EE, A_s_BB = theta_full[0], theta_full[1]
    alpha_s, beta_s = theta_full[2], theta_full[3]
    A_d_EE, A_d_BB = theta_full[4], theta_full[5]
    alpha_d, beta_d = theta_full[6], theta_full[7]
    rho = theta_full[8]
    
    # Apply bounds for synchrotron parameters
    if 'sync' in fit_components:
        if A_s_EE <= 0: return -np.inf               # amplitude must be positive
        if A_s_BB <= 0: return -np.inf               # amplitude must be positive
        if not (-6 <= alpha_s <= 0): return -np.inf
        if not (-6 <= beta_s <= 0): return -np.inf
    
    # Apply bounds for dust parameters
    if 'dust' in fit_components:
        if A_d_EE <= 0: return -np.inf               # amplitude must be positive
        if A_d_BB <= 0: return -np.inf               # amplitude must be positive
        if not (-6 <= alpha_d <= 0): return -np.inf
        if not (0 <= beta_d <= 6): return -np.inf
    
    # Apply bounds for correlation parameter
    if 'cross' in fit_components:
        if not (-1 <= rho <= 1): return -np.inf
    
    # Check c_terms if present (synchrotron only, low-freq bands)
    if fit_c_terms and n_c_terms > 0:
        offset = 9
        
        if 'sync' in fit_components:
            c_sync_EE = theta_full[offset:offset+n_c_terms]
            offset += n_c_terms
            c_sync_BB = theta_full[offset:offset+n_c_terms]
            offset += n_c_terms
            if not np.all(np.isfinite(c_sync_EE)):
                return -np.inf
            if not np.all(np.isfinite(c_sync_BB)):
                return -np.inf
    
    # Add optional Gaussian priors
    lp = 0.0
    try:
        if 'beta_s' in _GAUSSIAN_PRIORS:
            mu, sig = _GAUSSIAN_PRIORS['beta_s']
            if sig > 0:
                lp += -0.5 * ((beta_s - mu)/sig)**2
        if 'beta_d' in _GAUSSIAN_PRIORS:
            mu, sig = _GAUSSIAN_PRIORS['beta_d']
            if sig > 0:
                lp += -0.5 * ((beta_d - mu)/sig)**2
        if 'rho' in _GAUSSIAN_PRIORS:
            mu, sig = _GAUSSIAN_PRIORS['rho']
            if sig > 0:
                lp += -0.5 * ((rho - mu)/sig)**2
        if 'alpha_s' in _GAUSSIAN_PRIORS:
            mu, sig = _GAUSSIAN_PRIORS['alpha_s']
            if sig > 0:
                lp += -0.5 * ((alpha_s - mu)/sig)**2
        if 'alpha_d' in _GAUSSIAN_PRIORS:
            mu, sig = _GAUSSIAN_PRIORS['alpha_d']
            if sig > 0:
                lp += -0.5 * ((alpha_d - mu)/sig)**2
        if 'A_s_EE' in _GAUSSIAN_PRIORS:
            mu, sig = _GAUSSIAN_PRIORS['A_s_EE']
            if sig > 0:
                lp += -0.5 * ((A_s_EE - mu)/sig)**2
        if 'A_s_BB' in _GAUSSIAN_PRIORS:
            mu, sig = _GAUSSIAN_PRIORS['A_s_BB']
            if sig > 0:
                lp += -0.5 * ((A_s_BB - mu)/sig)**2
        if 'A_d_EE' in _GAUSSIAN_PRIORS:
            mu, sig = _GAUSSIAN_PRIORS['A_d_EE']
            if sig > 0:
                lp += -0.5 * ((A_d_EE - mu)/sig)**2
        if 'A_d_BB' in _GAUSSIAN_PRIORS:
            mu, sig = _GAUSSIAN_PRIORS['A_d_BB']
            if sig > 0:
                lp += -0.5 * ((A_d_BB - mu)/sig)**2
    except Exception:
        pass
    
    return lp


def lnprob(theta_free, datasets, ell, y_all, yerr_all,
           fit_c_terms=False, fit_components=('sync', 'dust', 'cross'),
           param_map=None, fixed_values=None, cc_dict=None, cov_matrix=None):
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
    cov_matrix : ndarray, optional
        Full covariance matrix to use in likelihood calculation.

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
                fit_c_terms=fit_c_terms, fit_components=fit_components,
                cc_dict=cc_dict, cov_matrix=cov_matrix)
    return lp + ll


def lnprob_joint(theta_full, datasets_EE, datasets_BB, ell,
                y_EE, yerr_EE, y_BB, yerr_BB,
                fit_c_terms=False, fit_components=('sync', 'dust', 'cross'),
                cc_dict=None, cov_matrix=None, freq_max_c=40.0):
    """
    Compute log-posterior for joint EE-BB analysis (prior + likelihood).

    Parameters
    ----------
    theta_full : array
        Full parameter vector.
    datasets_EE : list of dict
        Prepared datasets for EE mode.
    datasets_BB : list of dict
        Prepared datasets for BB mode.
    ell : array
        Multipoles.
    y_EE : array
        Observed EE spectra.
    yerr_EE : array
        Errors for EE spectra (used if cov_matrix is None).
    y_BB : array
        Observed BB spectra.
    yerr_BB : array
        Errors for BB spectra (used if cov_matrix is None).
    fit_c_terms : bool
        Whether c terms are fitted (synchrotron only, low-freq bands).
    fit_components : tuple
        Components included.
    cc_dict : dict, optional
        Color correction polynomials.
    cov_matrix : ndarray, optional
        Full covariance matrix for joint EE+BB data.
    freq_max_c : float
        Maximum frequency (GHz) for synchrotron constant terms. Default: 40.0.

    Returns
    -------
    lnpost : float
        Log-posterior probability.
    """
    # Compute N_c for prior
    unique_freqs = sorted({f for d in datasets_EE + datasets_BB for f in d['freqs']})
    N_c = len([f for f in unique_freqs if f <= freq_max_c])
    
    lp = lnprior_joint(theta_full, fit_c_terms=fit_c_terms, fit_components=fit_components,
                       n_c_terms=N_c)
    if not np.isfinite(lp):
        return -np.inf
    
    ll = lnlike_joint(theta_full, datasets_EE, datasets_BB, ell,
                     y_EE, yerr_EE, y_BB, yerr_BB,
                     fit_c_terms=fit_c_terms, fit_components=fit_components,
                     cc_dict=cc_dict, cov_matrix=cov_matrix, freq_max_c=freq_max_c)
    
    return lp + ll


def run_mcmc(fit_data, fit_components=('sync', 'dust', 'cross'),
            fit_c_terms=False, nwalkers=100, ninter=5000,
            discard_fraction=0.5, verbose=True,
            fit_mode='power-law', color_correction=False,
            joint_analysis=False, cov_matrix=None, n_processes=None,
            freeze_params=None, print_residuals=False):
    """
    Run MCMC fit with optional joint EE-BB analysis.

    Parameters
    ----------
    fit_data : dict
        Output from prepare_mcmc_data.
    fit_components : tuple
        Components to include ('sync', 'dust', 'cross').
    fit_c_terms : bool
        Whether to fit constant terms.
    nwalkers : int
        Number of emcee walkers.
    ninter : int
        Number of iterations per walker.
    discard_fraction : float
        Fraction to discard as burn-in.
    verbose : bool
        Whether to print progress.
    fit_mode : str
        'power-law' or 'bin-to-bin'.
    color_correction : bool
        Whether to apply color corrections.
    joint_analysis : bool
        If True, perform joint EE-BB analysis with shared spectral indices.
    cov_matrix : ndarray or dict, optional
        Covariance matrix for likelihood calculation.
    n_processes : int or None, optional
        Number of parallel processes to use for MCMC.
        If None, uses min(available_cores, nwalkers//2).
        Set to specific value to limit CPU usage (e.g., n_processes=20 on 50-core machine).
    freeze_params : dict or None, optional
        Parameters to freeze at fixed values, overriding the free_mask derived from
        fit_components. Example: ``freeze_params={'beta_s': 0.0}`` freezes the
        synchrotron frequency spectral index to 0 (useful for single-band auto-spectrum
        fits where beta_s is degenerate with A_s). Only supported with fit_mode='power-law'.

    Returns
    -------
    sampler : emcee.EnsembleSampler or list
        Sampler(s) after running.
    samples_full : ndarray or list
        Full parameter chains.
    samples_free : ndarray or list
        Free parameter chains.
    param_map : list or param_names : list
        Parameter information.
    chi2_reduced : float or list
        Reduced chi-squared value(s).
    """
    if joint_analysis:
        # Check that both EE and BB modes are present
        modes_present = set()
        for d in fit_data['datasets']:
            modes_present.add(d['mode'])
        
        if modes_present != {'EE', 'BB'}:
            raise ValueError(
                "joint_analysis=True requires both 'EE' and 'BB' modes in fit_data. "
                "Use prepare_mcmc_data with modes=['EE', 'BB']"
            )
        
        if fit_mode != 'power-law':
            raise ValueError(
                "joint_analysis=True is only supported with fit_mode='power-law'. "
                f"Got fit_mode='{fit_mode}'"
            )
        
        return _run_mcmc_joint(
            fit_data, fit_components, fit_c_terms,
            nwalkers, ninter, discard_fraction, verbose,
            color_correction, cov_matrix, n_processes
        )
    
    # Standard analysis (single mode or separate EE/BB)
    if fit_mode == 'power-law':
        return _run_mcmc_powerlaw(
            fit_data, fit_components, fit_c_terms,
            nwalkers, ninter, discard_fraction, verbose,
            color_correction, cov_matrix, n_processes,
            freeze_params=freeze_params, print_residuals=print_residuals
        )
    elif fit_mode == 'bin-to-bin':
        return _run_mcmc_bin_to_bin(
            fit_data, fit_components, nwalkers, ninter,
            discard_fraction, verbose, color_correction, n_processes
        )
    else:
        raise ValueError(f"fit_mode must be 'power-law' or 'bin-to-bin', got '{fit_mode}'")


def _run_mcmc_joint(fit_data, fit_components, fit_c_terms,
                   nwalkers, ninter, discard_fraction, verbose,
                   color_correction, cov_matrix=None, n_processes=None):
    """
    Run MCMC for joint EE-BB analysis.

    Parameters
    ----------
    fit_data : dict
        Output from prepare_mcmc_data with modes=['EE', 'BB'].
    fit_components : tuple
        Components to include ('sync', 'dust', 'cross').
    fit_c_terms : bool
        Whether to fit constant terms.
    nwalkers : int
        Number of emcee walkers.
    ninter : int
        Number of iterations per walker.
    discard_fraction : float
        Fraction of samples to discard as burn-in.
    verbose : bool
        Whether to print progress.
    color_correction : bool
        Whether to apply color corrections.

    Returns
    -------
    sampler : emcee.EnsembleSampler
        The sampler after running.
    samples_full : ndarray
        Flattened chain after burn-in.
    samples_free : ndarray
        Same as samples_full (no fixed parameters in joint analysis).
    param_names : list
        List of parameter names.
    chi2_reduced : float
        Reduced chi-squared at best fit.
    """
    # Load color-correction polynomials if requested
    cc_dict = None
    if color_correction:
        try:
            cc_dict = load_color_correction_polynomials()
            if verbose:
                print("[run_mcmc_joint] Color corrections enabled.")
        except Exception as e:
            print(f"[run_mcmc_joint] WARNING: Failed to load color corrections: {e}")
            cc_dict = None
    
    # Separate datasets by mode
    datasets_EE = [d for d in fit_data['datasets'] if d['mode'] == 'EE']
    datasets_BB = [d for d in fit_data['datasets'] if d['mode'] == 'BB']
    
    if len(datasets_EE) == 0 or len(datasets_BB) == 0:
        raise ValueError(
            "Joint analysis requires both EE and BB modes in fit_data. "
            "Use prepare_mcmc_data with modes=['EE', 'BB']"
        )
    
    # Reconstruct y and yerr for each mode
    y_EE, yerr_EE = [], []
    y_BB, yerr_BB = [], []
    
    for d in datasets_EE:
        y_EE.append(d['spectrum'])
        yerr_EE.append(d['error'])
    for d in datasets_BB:
        y_BB.append(d['spectrum'])
        yerr_BB.append(d['error'])
    
    y_EE = np.concatenate(y_EE)
    yerr_EE = np.concatenate(yerr_EE)
    y_BB = np.concatenate(y_BB)
    yerr_BB = np.concatenate(yerr_BB)
    
    ell = fit_data['ell_eff']
    
    # Build parameter names
    param_names = [
        'A_s_EE', 'A_s_BB',      # Synchrotron amplitudes
        'alpha_s', 'beta_s',     # Synchrotron spectral indices (shared)
        'A_d_EE', 'A_d_BB',      # Dust amplitudes
        'alpha_d', 'beta_d',     # Dust spectral indices (shared)
        'rho'                    # Cross-correlation (shared)
    ]
    
    # Add c_terms if requested (synchrotron only, low-freq bands only)
    unique_freqs = sorted({f for d in fit_data['datasets'] for f in d['freqs']})
    FREQ_MAX_C = 40.0
    low_freqs = sorted({f for f in unique_freqs if f <= FREQ_MAX_C})
    N_c = len(low_freqs)
    
    if fit_c_terms:
        for f in low_freqs:
            param_names.append(f'c_sync_EE[{int(f)}]')
        for f in low_freqs:
            param_names.append(f'c_sync_BB[{int(f)}]')
    
    ndim = len(param_names)
    
    if verbose:
        print(f"[run_mcmc_joint] Joint EE-BB analysis")
        print(f"[run_mcmc_joint] Parameters: {param_names[:9]}")  # Show base params
        print(f"[run_mcmc_joint] Total parameters: {ndim}")
        print(f"[run_mcmc_joint] EE data points: {len(y_EE)}")
        print(f"[run_mcmc_joint] BB data points: {len(y_BB)}")
    
    # Initialize walkers
    rng = np.random.default_rng()
    p0_center = np.array([
        1.0,    # A_s_EE
        0.25,   # A_s_BB
        -3.0,   # alpha_s
        -3.0,   # beta_s
        1.0,    # A_d_EE
        0.5,    # A_d_BB (typically smaller than EE)
        -2.3,   # alpha_d
        1.59,   # beta_d
        0.05    # rho
    ], dtype=float)
    
    # Add zeros for c_terms (synchrotron only, low-freq only: N_c per mode × 2 modes)
    if fit_c_terms:
        p0_center = np.concatenate([p0_center, np.zeros(2 * N_c)])
    
    p0_walkers = p0_center + 1e-2 * rng.standard_normal((nwalkers, ndim))
    
    # Determine number of processes
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    
    if n_processes is None:
        n_procs = max(1, min(available, max(1, nwalkers // 2)))
    else:
        n_procs = max(1, min(n_processes, available))
    
    if verbose:
        print(f"[run_mcmc_joint] Using {n_procs} processes with {nwalkers} walkers")
        print(f"[run_mcmc_joint] Running {ninter} iterations...")
    
    # Run MCMC
    with mp.get_context("fork").Pool(processes=n_procs, maxtasksperchild=200) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, lnprob_joint,
            args=(datasets_EE, datasets_BB, ell, y_EE, yerr_EE, y_BB, yerr_BB,
                  fit_c_terms, fit_components, cc_dict, cov_matrix, FREQ_MAX_C),
            pool=pool
        )
        sampler.run_mcmc(p0_walkers, ninter, progress=verbose)
    
    # Post-processing
    discard = int(ninter * discard_fraction)
    samples = sampler.get_chain(discard=discard, flat=True)
    
    if verbose:
        print(f"[run_mcmc_joint] MCMC completed. {samples.shape[0]} samples after burn-in.")
    
    # Compute reduced chi-squared at best fit
    best_idx = np.argmax(sampler.get_log_prob(discard=discard, flat=True))
    best_params = samples[best_idx]
    
    chi2_reduced = compute_chi2_reduced_joint(
        best_params, datasets_EE, datasets_BB, ell,
        y_EE, yerr_EE, y_BB, yerr_BB,
        fit_c_terms=fit_c_terms, fit_components=fit_components,
        cc_dict=cc_dict, freq_max_c=FREQ_MAX_C
    )
    
    if verbose:
        print(f"[run_mcmc_joint] Reduced chi-squared at best fit: {chi2_reduced:.4f}")
    
    return sampler, samples, samples, param_names, chi2_reduced



def _run_mcmc_powerlaw(fit_data, fit_components, fit_c_terms, nwalkers, ninter, discard_fraction, verbose,
                       color_correction, cov_matrix=None, n_processes=None, freeze_params=None,
                       print_residuals=False):
    # Load color-correction polynomials dict if requested
    cc_dict = None
    if color_correction:
        try:
            cc_dict = load_color_correction_polynomials()
            if verbose:
                print("[run_mcmc] Color corrections enabled (alpha-polynomials per band).")
        except Exception as e:
            print(f"[run_mcmc] WARNING: Failed to load color-correction polynomials: {e}. Proceeding without.")
            cc_dict = None
    """Power-law mode: fit global spectral indices across all ells."""
    datasets = fit_data['datasets']
    ell = fit_data['ell_eff']
    y_all = fit_data['y_all']
    yerr_all = fit_data['yerr_all']

    # -------------------------------
    # Parameter mapping
    # -------------------------------
    unique_freqs = sorted({f for d in datasets for f in d['freqs']})
    FREQ_MAX_C = 40.0
    low_freqs = sorted({f for f in unique_freqs if f <= FREQ_MAX_C})
    N_c = len(low_freqs)

    param_names = ['A_s', 'alpha_s', 'beta_s', 'A_d', 'alpha_d', 'beta_d', 'rho']
    if fit_c_terms:
        param_names += [f'c_sync[{int(f)}]' for f in low_freqs]

    free_mask = {
        'A_s':    ('sync' in fit_components),
        'alpha_s':('sync' in fit_components),
        'beta_s': ('sync' in fit_components),
        'A_d':    ('dust' in fit_components),
        'alpha_d':('dust' in fit_components),
        'beta_d': ('dust' in fit_components),
        'rho':    ('cross' in fit_components)
    }

    # Apply caller-supplied parameter freezes (e.g., freeze beta_s=0 for single-band fits)
    _freeze = freeze_params or {}
    _fixed_overrides = {}
    for pname, pval in _freeze.items():
        if pname in free_mask:
            free_mask[pname] = False
            _fixed_overrides[pname] = float(pval)

    param_map = []
    for name in ['A_s','alpha_s','beta_s','A_d','alpha_d','beta_d','rho']:
        param_map.append((name, free_mask[name]))
    if fit_c_terms:
        for f in low_freqs:
            param_map.append((f'c_sync[{int(f)}]', True if 'sync' in fit_components else False))

    fixed_values = {name: 0.0 for name, is_free in param_map if not is_free}
    # Apply caller-supplied frozen values (overrides default 0.0)
    fixed_values.update(_fixed_overrides)
    ndim = sum(1 for _, is_free in param_map if is_free)

    # -------------------------------
    # Initialize walkers
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
        else:
            # c_sync term: initialise at a small fraction of the data scale
            # (1e-2 is many orders of magnitude off for mK² spectra)
            p0_center.append(float(np.median(np.abs(y_all))) * 0.1)

    p0_center = np.array(p0_center, dtype=float)
    # Per-parameter spread: small absolute value for shape/index parameters
    # so walkers stay close and linearly independent; relative for amplitudes.
    p0_scale = []
    free_names = [name for name, is_free in param_map if is_free]
    for idx, name in enumerate(free_names):
        if name in ('A_s', 'A_d'):
            p0_scale.append(0.1 * abs(p0_center[idx]))
        elif name.startswith('c_sync'):
            p0_scale.append(0.3 * abs(p0_center[idx]) if p0_center[idx] != 0 else float(np.median(np.abs(y_all))) * 0.05)
        elif name == 'rho':
            # rho must stay in (-1, 1): use a tight spread so no walker starts outside
            p0_scale.append(0.02)
        else:
            # alpha, beta: fixed small absolute spread
            p0_scale.append(0.05)
    p0_scale = np.array(p0_scale, dtype=float)
    p0_walkers = p0_center + p0_scale * rng.standard_normal((nwalkers, ndim))

    # Clip walkers to valid prior bounds so none start at -inf log-prob
    # (prevents large condition-number errors in emcee)
    for idx, name in enumerate(free_names):
        if name == 'rho':
            p0_walkers[:, idx] = np.clip(p0_walkers[:, idx], -0.99, 0.99)
        elif name in ('A_s', 'A_d'):
            p0_walkers[:, idx] = np.clip(p0_walkers[:, idx], 1e-6, None)
        elif name == 'beta_d':
            p0_walkers[:, idx] = np.clip(p0_walkers[:, idx], 0.5, 3.0)
        elif name == 'beta_s':
            p0_walkers[:, idx] = np.clip(p0_walkers[:, idx], -5.0, -0.5)

    # -------------------------------
    # Run the sampler
    # -------------------------------
    try:
        available_cores = len(os.sched_getaffinity(0))
    except AttributeError:
        available_cores = os.cpu_count() or 1

    if n_processes is None:
        n_procs = available_cores
    else:
        n_procs = max(1, min(n_processes, available_cores))

    if verbose:
        print(f"[run_mcmc] Available cores: {available_cores}")
        print(f"[run_mcmc] Using {n_procs} processes for walker parallelization")

    with mp.Pool(processes=n_procs) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, lnprob,
            args=(datasets, ell, y_all, yerr_all, fit_c_terms, fit_components, param_map, fixed_values, cc_dict, cov_matrix),
            pool=pool
        )
        if verbose:
            print("[run_mcmc] Starting burn-in...")
        burn = max(100, int(0.2 * ninter))
        p0_walkers, _, _ = sampler.run_mcmc(p0_walkers, burn, progress=verbose)
        sampler.reset()
        if verbose:
            print("[run_mcmc] Starting production run...")
        sampler.run_mcmc(p0_walkers, ninter, progress=verbose)
        

    # -------------------------------
    # Post-processing
    # -------------------------------
    discard = int(ninter * discard_fraction)
    samples_free = sampler.get_chain(discard=discard, flat=True)

    n_full = len(param_map)
    samples_full = np.zeros((samples_free.shape[0], n_full))
    free_cols = [i for i, (_, is_free) in enumerate(param_map) if is_free]
    fixed_cols = [i for i, (_, is_free) in enumerate(param_map) if not is_free]
    samples_full[:, free_cols] = samples_free
    for i in fixed_cols:
        samples_full[:, i] = fixed_values[param_map[i][0]]

    # -------------------------------
    # Compute reduced chi-squared at best fit
    # -------------------------------
    # Find best-fit parameters (maximum likelihood)
    best_idx = np.argmax(sampler.get_log_prob(discard=discard, flat=True))
    best_params_free = samples_free[best_idx]
    
    # Reconstruct full parameter vector
    best_params_full = np.zeros(n_full)
    best_params_full[free_cols] = best_params_free
    for i in fixed_cols:
        best_params_full[i] = fixed_values[param_map[i][0]]
    
    # Calculate reduced chi-squared
    chi2_reduced = compute_chi2_reduced(
        best_params_full, datasets, ell, y_all, yerr_all, 
        fit_c_terms=fit_c_terms, fit_components=fit_components, cc_dict=cc_dict,
        n_free_params=ndim  # ndim = actual number of free parameters in the sampler
    )

    if verbose:
        print(f"[run_mcmc] MCMC completed. {samples_free.shape[0]} usable samples after burn-in.")
        # Additional diagnostics: report data points, free params, dof and total chi2
        try:
            n_data = len(y_all)
            n_free = int(ndim)
            dof = n_data - n_free
            if np.isfinite(chi2_reduced):
                chi2_total = float(chi2_reduced) * float(dof) if dof > 0 else float('nan')
            else:
                chi2_total = float('inf')
            print(f"[run_mcmc] Reduced chi-squared at best fit: {chi2_reduced:.4f}")
            print(f"[run_mcmc] chi2 diagnostics: n_data={n_data}, n_free_params={n_free}, dof={dof}, chi2_total={chi2_total}")
        except Exception:
            # Fallback to minimal reporting
            print(f"[run_mcmc] Reduced chi-squared at best fit: {chi2_reduced}")

        # If requested, print per-bin residuals for inspection (best-fit model)
        if print_residuals:
            try:
                # Rebuild the model vector at the best-fit parameters
                y_model = np.zeros_like(y_all)
                if 'sync' in fit_components:
                    # Extract sync-related params from best_params_full
                    # Build theta for model_synchrotron
                    unique_freqs = sorted({f for d in datasets for f in d['freqs']})
                    low_freqs = sorted({f for f in unique_freqs if f <= 40.0})
                    N_c = len(low_freqs)
                    if fit_c_terms:
                        theta_sync = list(best_params_full[:3]) + list(best_params_full[7:7+N_c])
                    else:
                        theta_sync = list(best_params_full[:3])
                    y_model += model_synchrotron(theta_sync, datasets, ell, fit_c_terms=fit_c_terms, cc_dict=cc_dict)
                if 'dust' in fit_components:
                    y_model += model_dust(list(best_params_full[3:6]), datasets, ell, cc_dict=cc_dict)
                if 'cross' in fit_components:
                    # build theta_cross = [rho, A_s, A_d, alpha_s, alpha_d, beta_s, beta_d]
                    theta_cross = [
                        float(best_params_full[6]),  # rho
                        float(best_params_full[0]),  # A_s
                        float(best_params_full[3]),  # A_d
                        float(best_params_full[1]),  # alpha_s
                        float(best_params_full[4]),  # alpha_d
                        float(best_params_full[2]),  # beta_s
                        float(best_params_full[5])   # beta_d
                    ]
                    y_model += model_cross(theta_cross, datasets, ell, cc_dict=cc_dict)

                # Print per-pair/per-bin table of residuals
                print('\n[run_mcmc] Per-bin residuals (best-fit):')
                cursor = 0
                for d in datasets:
                    start, stop = d['slice']
                    ell_here = ell[start:stop]
                    y_here = d['spectrum']
                    err_here = d['error']
                    m_here = y_model[start:stop]
                    res_here = y_here - m_here
                    chi2_here = np.sum((res_here / err_here) ** 2)
                    print(f"\n Pair: {d['pair']}  mode: {d['mode']}  chi2_bin={chi2_here:.4e}  n_points={len(ell_here)}")
                    print(" ell   data    model   err   (res/err)^2")
                    for e, yy, mm, ee, rr in zip(ell_here, y_here, m_here, err_here, (res_here / err_here) ** 2):
                        print(f"{int(e):4d}  {yy: .4e}  {mm: .4e}  {ee: .4e}  {rr: .4e}")
            except Exception as exc:
                print(f"[run_mcmc] Failed to print per-bin residuals: {exc}")

    return sampler, samples_full, samples_free, param_map, chi2_reduced


def _run_mcmc_bin_to_bin(fit_data, fit_components, nwalkers, ninter, discard_fraction, verbose,
                         color_correction, n_processes=None):
    """
    Run MCMC in "bin-to-bin" mode, fitting each ell bin independently.

    For every ell bin, an independent MCMC is performed on a compact parameter set:
    - If 'sync' in fit_components:   [A_s, beta_s]
    - If 'dust' in fit_components:   [A_d, beta_d]
    - If 'cross' in fit_components:  [rho]

    Parameters
    ----------
    fit_data : dict
        Output of `prepare_mcmc_data`, with at least the keys:
        - 'datasets': list of dataset dicts (pair, mode, freqs, slice)
        - 'ell_eff' : array of effective ells (one per bin)
        - 'y_all'   : concatenated spectra (not directly used here)
        - 'yerr_all': concatenated errors   (not directly used here)
    fit_components : tuple of str
        Which components to include: any subset of ('sync', 'dust', 'cross').
    nwalkers : int
        Number of emcee walkers per bin.
    ninter : int
        Total number of iterations per walker (per bin).
    discard_fraction : float
        Fraction of initial samples to discard as burn-in when summarizing results.
    verbose : bool
        If True, prints progress and resource usage.
    color_correction : bool
        If True, applies per-band color-correction polynomials for each component in every bin.
        The correction is evaluated at alpha_cc = 2 + beta (spectral index in frequency) and
        divides the per-band scale factors in the synch, dust, and cross terms. Polynomials are
        loaded once via `load_color_correction_polynomials()` and reused across bins.

    Returns
    -------
    samplers_list : list[emcee.EnsembleSampler]
        The sampler for each ell bin.
    samples_full_list : list[np.ndarray]
        Flattened chains (after burn-in) for each bin and its free parameters.
    samples_free_list : list[np.ndarray]
        Alias of samples_full_list (no fixed params in this mode).
    param_names_base : list[str]
        Names of parameters fitted per bin (depends on `fit_components`).
    chi2_reduced_list : list[float]
        Reduced chi-squared per bin at the best-fit parameters.

    Notes
    -----
    - Bins are processed sequentially; within each bin, emcee uses a multiprocessing Pool.
    - Color-corrections are applied in the same way as in power-law mode, but with the
      bin-specific beta_s and beta_d in each likelihood evaluation.
    - Each band uses its own polynomial; for dust, Planck HFI bands use the HFI set when present.
    """
    datasets = fit_data['datasets']
    ell = fit_data['ell_eff']
    
    n_ell = len(ell)
    
    # For bin-to-bin, we fit: A_s, beta_s, A_d, beta_d, rho (no alpha_s, alpha_d)
    param_names_base = []
    if 'sync' in fit_components:
        param_names_base.extend(['A_s', 'beta_s'])
    if 'dust' in fit_components:
        param_names_base.extend(['A_d', 'beta_d'])
    if 'cross' in fit_components:
        param_names_base.append('rho')
    
    if verbose:
        print(f"[run_mcmc bin-to-bin] Fitting {n_ell} ell bins independently")
        print(f"[run_mcmc bin-to-bin] Parameters per bin: {param_names_base}")
    
    # Determine number of cores for parallelization
    try:
        available_cores = len(os.sched_getaffinity(0))
    except AttributeError:
        available_cores = os.cpu_count() or 1
    
    if n_processes is None:
        n_procs = available_cores
    else:
        n_procs = max(1, min(n_processes, available_cores))
    
    if verbose:
        print(f"[run_mcmc bin-to-bin] Available cores: {available_cores}")
        print(f"[run_mcmc bin-to-bin] Using {n_procs} processes for walker parallelization")
    
    # Load color-correction polynomials dict if requested
    cc_dict = None
    if color_correction:
        try:
            cc_dict = load_color_correction_polynomials()
            if verbose:
                print("[run_mcmc bin-to-bin] Color corrections enabled (alpha=2+beta per band).")
        except Exception as e:
            print(f"[run_mcmc bin-to-bin] WARNING: Failed to load color-correction polynomials: {e}. Proceeding without.")
            cc_dict = None

    # Storage for results
    samplers_list = []
    samples_full_list = []
    samples_free_list = []
    chi2_reduced_list = []
    
    # Loop over ell bins sequentially
    for i_ell in range(n_ell):
        if verbose:
            print(f"\n[run_mcmc bin-to-bin] Processing bin {i_ell+1}/{n_ell} (ell={ell[i_ell]:.1f})")
        
        # Extract data for this ell bin only
        y_bin = []
        yerr_bin = []
        datasets_bin = []
        
        for d in datasets:
            start, stop = d['slice']
            if i_ell < (stop - start):
                y_bin.append(d['spectrum'][i_ell])
                yerr_bin.append(d['error'][i_ell])
                datasets_bin.append({
                    'pair': d['pair'],
                    'mode': d['mode'],
                    'freqs': d['freqs'],
                    'slice': (len(y_bin)-1, len(y_bin))
                })
        
        y_bin = np.array(y_bin)
        yerr_bin = np.array(yerr_bin)
        ell_bin = np.array([ell[i_ell]])
        
        ndim = len(param_names_base)
        
        # Initialize walkers
        # Use a data-driven initialization for amplitude parameters (A_s, A_d)
        # to avoid starting all walkers at extremely small values which can
        # hinder mixing in low-S/N bins. Use median of y_bin as a rough estimate.
        rng = np.random.default_rng()
        p0_center = []
        # rough_scale is an estimate of the data amplitude in this bin
        try:
            rough_scale = float(np.median(np.abs(y_bin))) if y_bin.size > 0 else 1e-6
        except Exception:
            rough_scale = 1e-6

        for name in param_names_base:
            if name == 'A_s':
                # Start near the observed median but ensure a positive floor
                p0_center.append(max(rough_scale, 1e-12))
            elif name == 'beta_s':
                p0_center.append(-3.0)
            elif name == 'A_d':
                p0_center.append(max(rough_scale, 1e-12))
            elif name == 'beta_d':
                p0_center.append(1.59)
            elif name == 'rho':
                p0_center.append(0.05)
            else:
                p0_center.append(0.0)

        p0_center = np.array(p0_center, dtype=float)
        # Use a relative spread but also a minimum absolute spread to ensure walkers
        # are not identical when p0_center is near zero.
        abs_spread = np.maximum(np.abs(p0_center) * 0.2, 1e-12)
        p0_walkers = p0_center + abs_spread * rng.standard_normal((nwalkers, ndim))
        
        # Run sampler for this bin with multiprocessing Pool
        with mp.Pool(processes=n_procs) as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, _lnprob_bin_to_bin,
                args=(datasets_bin, ell_bin, y_bin, yerr_bin, fit_components, param_names_base, cc_dict),
                pool=pool
            )
            sampler.run_mcmc(p0_walkers, ninter, progress=verbose)
        
        # Post-processing for this bin
        discard = int(ninter * discard_fraction)
        samples_free = sampler.get_chain(discard=discard, flat=True)
        samples_full = samples_free.copy()
        
        # Compute chi2 for best fit
        best_idx = np.argmax(sampler.get_log_prob(discard=discard, flat=True))
        best_params = samples_free[best_idx]
        chi2_reduced = _compute_chi2_reduced_bin_to_bin(
            best_params, datasets_bin, ell_bin, y_bin, yerr_bin, fit_components, param_names_base, cc_dict
        )
        
        samplers_list.append(sampler)
        samples_full_list.append(samples_full)
        samples_free_list.append(samples_free)
        chi2_reduced_list.append(chi2_reduced)
        
        if verbose:
            print(f"[run_mcmc bin-to-bin] Bin {i_ell+1} completed. χ²_red = {chi2_reduced:.4f}")
    
    if verbose:
        print(f"\n[run_mcmc bin-to-bin] All {n_ell} bins completed successfully")
        print(f"[run_mcmc bin-to-bin] Mean χ²_red = {np.mean(chi2_reduced_list):.4f} ± {np.std(chi2_reduced_list):.4f}")
    
    return samplers_list, samples_full_list, samples_free_list, param_names_base, chi2_reduced_list


def _lnprob_bin_to_bin(theta, datasets, ell, y_all, yerr_all, fit_components, param_names, cc_dict=None):
    """Log-probability for bin-to-bin fitting (single ell bin)."""
    # Prior
    lp = _lnprior_bin_to_bin(theta, param_names)
    if not np.isfinite(lp):
        return -np.inf
    
    # Likelihood
    ll = _lnlike_bin_to_bin(theta, datasets, ell, y_all, yerr_all, fit_components, param_names, cc_dict)
    if not np.isfinite(ll):
        return -np.inf
    
    return lp + ll


def _lnprior_bin_to_bin(theta, param_names):
    """Prior for bin-to-bin parameters."""
    param_dict = {name: theta[i] for i, name in enumerate(param_names)}
    
    # Check bounds
    # A_s and A_d must be positive (physical amplitudes)
    if 'A_s' in param_dict and param_dict['A_s'] <= 0:
        return -np.inf
    if 'A_d' in param_dict and param_dict['A_d'] <= 0:
        return -np.inf
    if 'beta_s' in param_dict and not (-10 < param_dict['beta_s'] < 0):
        return -np.inf
    if 'beta_d' in param_dict and not (-2 < param_dict['beta_d'] < 5):
        return -np.inf
    if 'rho' in param_dict and not (-1 < param_dict['rho'] < 1):
        return -np.inf
    
    # Optional Gaussian priors
    lp = 0.0
    try:
        if 'beta_s' in param_dict and 'beta_s' in _GAUSSIAN_PRIORS:
            mu, sig = _GAUSSIAN_PRIORS['beta_s']
            if sig > 0:
                lp += -0.5 * ((param_dict['beta_s'] - mu)/sig)**2
        if 'beta_d' in param_dict and 'beta_d' in _GAUSSIAN_PRIORS:
            mu, sig = _GAUSSIAN_PRIORS['beta_d']
            if sig > 0:
                lp += -0.5 * ((param_dict['beta_d'] - mu)/sig)**2
        if 'rho' in param_dict and 'rho' in _GAUSSIAN_PRIORS:
            mu, sig = _GAUSSIAN_PRIORS['rho']
            if sig > 0:
                lp += -0.5 * ((param_dict['rho'] - mu)/sig)**2
    except Exception:
        pass
    
    return lp


def _lnlike_bin_to_bin(theta, datasets, ell, y_all, yerr_all, fit_components, param_names, cc_dict=None):
    """Likelihood for bin-to-bin fitting (single ell bin)."""
    param_dict = {name: theta[i] for i, name in enumerate(param_names)}
    
    # Extract parameters (set to 0 if not fitted)
    A_s = param_dict.get('A_s', 0.0)
    beta_s = param_dict.get('beta_s', -3.0)
    A_d = param_dict.get('A_d', 0.0)
    beta_d = param_dict.get('beta_d', 1.59)
    rho = param_dict.get('rho', 0.0)
    
    # Build model (no ell scaling since we fit each bin independently)
    y_model = np.zeros_like(y_all)
    
    for idx, d in enumerate(datasets):
        f1, f2 = d['freqs']
        
        model_val = 0.0
        
        if 'sync' in fit_components:
            # Synchrotron: A_s * (f1/f_ref)^beta_s * (f2/f_ref)^beta_s
            # Use 23 GHz as the reference frequency for synchrotron to be
            # consistent across the codebase.
            freq_ref_sync = 23.0
            scale_f1 = (f1 / freq_ref_sync) ** beta_s
            scale_f2 = (f2 / freq_ref_sync) ** beta_s
            if cc_dict is not None:
                alpha_s_cc = 2.0 + float(beta_s)
                poly1 = (cc_dict.get('synch', {}) or {}).get(str(d['pair'].split('_')[0]))
                poly2 = (cc_dict.get('synch', {}) or {}).get(str(d['pair'].split('_')[1]))
                cc_s1 = (poly1[0] + poly1[1]*alpha_s_cc + poly1[2]*(alpha_s_cc**2)) if poly1 is not None else 1.0
                cc_s2 = (poly2[0] + poly2[1]*alpha_s_cc + poly2[2]*(alpha_s_cc**2)) if poly2 is not None else 1.0
            else:
                cc_s1 = cc_s2 = 1.0
            model_val += A_s * (scale_f1 / cc_s1) * (scale_f2 / cc_s2)
        
        if 'dust' in fit_components:
            # Dust: A_d * mbb_scaling
            freq_ref_dust = 353.0
            T_d = 19.6
            scale_f1 = mbb_scaling_KRJ(np.array([f1]), nu0_GHz=freq_ref_dust, beta=beta_d, T_d=T_d)[0]
            scale_f2 = mbb_scaling_KRJ(np.array([f2]), nu0_GHz=freq_ref_dust, beta=beta_d, T_d=T_d)[0]
            if cc_dict is not None:
                alpha_d_cc = 2.0 + float(beta_d)
                poly1 = (cc_dict.get('dust', {}) or {}).get(str(d['pair'].split('_')[0]))
                poly2 = (cc_dict.get('dust', {}) or {}).get(str(d['pair'].split('_')[1]))
                cc_d1 = (poly1[0] + poly1[1]*alpha_d_cc + poly1[2]*(alpha_d_cc**2)) if poly1 is not None else 1.0
                cc_d2 = (poly2[0] + poly2[1]*alpha_d_cc + poly2[2]*(alpha_d_cc**2)) if poly2 is not None else 1.0
            else:
                cc_d1 = cc_d2 = 1.0
            model_val += A_d * (scale_f1 / cc_d1) * (scale_f2 / cc_d2)
        
        if 'cross' in fit_components:
            # Cross term: rho * sqrt(A_s * A_d) * (sync_scale1 * dust_scale2 + sync_scale2 * dust_scale1)
            freq_ref_sync = 23.
            freq_ref_dust = 353.0
            T_d = 19.6
            
            s1 = (f1 / freq_ref_sync) ** beta_s
            s2 = (f2 / freq_ref_sync) ** beta_s
            d1 = mbb_scaling_KRJ(np.array([f1]), nu0_GHz=freq_ref_dust, beta=beta_d, T_d=T_d)[0]
            d2 = mbb_scaling_KRJ(np.array([f2]), nu0_GHz=freq_ref_dust, beta=beta_d, T_d=T_d)[0]
            if cc_dict is not None:
                alpha_s_cc = 2.0 + float(beta_s)
                alpha_d_cc = 2.0 + float(beta_d)
                b1, b2 = d['pair'].split('_')
                syn1 = (cc_dict.get('synch', {}) or {}).get(str(b1))
                syn2 = (cc_dict.get('synch', {}) or {}).get(str(b2))
                dus1 = (cc_dict.get('dust', {}) or {}).get(str(b1))
                dus2 = (cc_dict.get('dust', {}) or {}).get(str(b2))
                cc_s1 = (syn1[0] + syn1[1]*alpha_s_cc + syn1[2]*(alpha_s_cc**2)) if syn1 is not None else 1.0
                cc_s2 = (syn2[0] + syn2[1]*alpha_s_cc + syn2[2]*(alpha_s_cc**2)) if syn2 is not None else 1.0
                cc_d1 = (dus1[0] + dus1[1]*alpha_d_cc + dus1[2]*(alpha_d_cc**2)) if dus1 is not None else 1.0
                cc_d2 = (dus2[0] + dus2[1]*alpha_d_cc + dus2[2]*(alpha_d_cc**2)) if dus2 is not None else 1.0
            else:
                cc_s1 = cc_s2 = cc_d1 = cc_d2 = 1.0

            model_val += rho * np.sqrt(A_s * A_d) * ((s1 / cc_s1) * (d2 / cc_d2) + (s2 / cc_s2) * (d1 / cc_d1))
        
        y_model[idx] = model_val
    
    # Chi-squared
    chi2 = np.sum(((y_all - y_model) / yerr_all) ** 2)
    return -0.5 * chi2


def _compute_chi2_reduced_bin_to_bin(theta, datasets, ell, y_all, yerr_all, fit_components, param_names, cc_dict=None):
    """Compute reduced chi-squared for bin-to-bin fit."""
    param_dict = {name: theta[i] for i, name in enumerate(param_names)}
    
    # Extract parameters
    A_s = param_dict.get('A_s', 0.0)
    beta_s = param_dict.get('beta_s', -3.0)
    A_d = param_dict.get('A_d', 0.0)
    beta_d = param_dict.get('beta_d', 1.59)
    rho = param_dict.get('rho', 0.0)
    
    # Build model
    y_model = np.zeros_like(y_all)
    
    for idx, d in enumerate(datasets):
        f1, f2 = d['freqs']
        
        model_val = 0.0
        
        if 'sync' in fit_components:
            freq_ref_sync = 23.
            scale_f1 = (f1 / freq_ref_sync) ** beta_s
            scale_f2 = (f2 / freq_ref_sync) ** beta_s
            if cc_dict is not None:
                alpha_s_cc = 2.0 + float(beta_s)
                b1, b2 = d['pair'].split('_')
                poly1 = (cc_dict.get('synch', {}) or {}).get(str(b1))
                poly2 = (cc_dict.get('synch', {}) or {}).get(str(b2))
                cc_s1 = (poly1[0] + poly1[1]*alpha_s_cc + poly1[2]*(alpha_s_cc**2)) if poly1 is not None else 1.0
                cc_s2 = (poly2[0] + poly2[1]*alpha_s_cc + poly2[2]*(alpha_s_cc**2)) if poly2 is not None else 1.0
            else:
                cc_s1 = cc_s2 = 1.0
            model_val += A_s * (scale_f1 / cc_s1) * (scale_f2 / cc_s2)
        
        if 'dust' in fit_components:
            freq_ref_dust = 353.0
            T_d = 19.6
            scale_f1 = mbb_scaling_KRJ(np.array([f1]), nu0_GHz=freq_ref_dust, beta=beta_d, T_d=T_d)[0]
            scale_f2 = mbb_scaling_KRJ(np.array([f2]), nu0_GHz=freq_ref_dust, beta=beta_d, T_d=T_d)[0]
            if cc_dict is not None:
                alpha_d_cc = 2.0 + float(beta_d)
                b1, b2 = d['pair'].split('_')
                poly1 = (cc_dict.get('dust', {}) or {}).get(str(b1))
                poly2 = (cc_dict.get('dust', {}) or {}).get(str(b2))
                cc_d1 = (poly1[0] + poly1[1]*alpha_d_cc + poly1[2]*(alpha_d_cc**2)) if poly1 is not None else 1.0
                cc_d2 = (poly2[0] + poly2[1]*alpha_d_cc + poly2[2]*(alpha_d_cc**2)) if poly2 is not None else 1.0
            else:
                cc_d1 = cc_d2 = 1.0
            model_val += A_d * (scale_f1 / cc_d1) * (scale_f2 / cc_d2)
        
        if 'cross' in fit_components:
            freq_ref_sync = 23.
            freq_ref_dust = 353.0
            T_d = 19.6
            
            s1 = (f1 / freq_ref_sync) ** beta_s
            s2 = (f2 / freq_ref_sync) ** beta_s
            d1 = mbb_scaling_KRJ(np.array([f1]), nu0_GHz=freq_ref_dust, beta=beta_d, T_d=T_d)[0]
            d2 = mbb_scaling_KRJ(np.array([f2]), nu0_GHz=freq_ref_dust, beta=beta_d, T_d=T_d)[0]
            if cc_dict is not None:
                alpha_s_cc = 2.0 + float(beta_s)
                alpha_d_cc = 2.0 + float(beta_d)
                b1, b2 = d['pair'].split('_')
                syn1 = (cc_dict.get('synch', {}) or {}).get(str(b1))
                syn2 = (cc_dict.get('synch', {}) or {}).get(str(b2))
                dus1 = (cc_dict.get('dust', {}) or {}).get(str(b1))
                dus2 = (cc_dict.get('dust', {}) or {}).get(str(b2))
                cc_s1 = (syn1[0] + syn1[1]*alpha_s_cc + syn1[2]*(alpha_s_cc**2)) if syn1 is not None else 1.0
                cc_s2 = (syn2[0] + syn2[1]*alpha_s_cc + syn2[2]*(alpha_s_cc**2)) if syn2 is not None else 1.0
                cc_d1 = (dus1[0] + dus1[1]*alpha_d_cc + dus1[2]*(alpha_d_cc**2)) if dus1 is not None else 1.0
                cc_d2 = (dus2[0] + dus2[1]*alpha_d_cc + dus2[2]*(alpha_d_cc**2)) if dus2 is not None else 1.0
            else:
                cc_s1 = cc_s2 = cc_d1 = cc_d2 = 1.0

            model_val += rho * np.sqrt(A_s * A_d) * ((s1 / cc_s1) * (d2 / cc_d2) + (s2 / cc_s2) * (d1 / cc_d1))
        
        y_model[idx] = model_val
    
    # Chi-squared
    chi2 = np.sum(((y_all - y_model) / yerr_all) ** 2)
    n_data = len(y_all)
    n_params = len(param_names)
    dof = max(1, n_data - n_params)
    
    return chi2 / dof



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


def plot_corner(samples_free, param_map, save_path=None, title=None):
    """
    Generate a publication-quality corner plot for MCMC samples.
    
    Supports both standard single-mode analysis and joint EE-BB analysis.
    Automatically detects parameter names with _EE and _BB suffixes.
    
    Parameters
    ----------
    samples_free : ndarray
        MCMC samples for the free parameters (after burn-in).
    param_map : list
        List of (parameter_name, is_free) tuples for standard analysis,
        or list of parameter names (strings) for joint analysis.
    save_path : str or None
        If provided, save the figure to this path (e.g., 'corner_plot.png').
    title : str or None
        Optional title to display at the top of the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The corner plot figure.
    """

    # -------------------------------
    # Handle both param_map formats
    # -------------------------------
    # Check if param_map is list of tuples (standard) or list of strings (joint)
    if len(param_map) > 0 and isinstance(param_map[0], tuple):
        # Standard format: [(name, is_free), ...]
        labels_free = [name for name, is_free in param_map if is_free]
    else:
        # Joint format: [name, name, ...]
        labels_free = list(param_map)

    # -------------------------------
    # Prepare labels and scaling
    # -------------------------------
    scale_map = {
        # Standard single-mode parameters
        'A_s': (1e9, r'$10^{-3}\,\mu\mathrm{K}^2$'),
        'A_d': (1e9, r'$10^{-3}\,\mu\mathrm{K}^2$'),
        
        # Joint EE-BB parameters (mode-specific amplitudes)
        'A_s_EE': (1e9, r'$10^{-3}\,\mu\mathrm{K}^2$'),
        'A_s_BB': (1e9, r'$10^{-3}\,\mu\mathrm{K}^2$'),
        'A_d_EE': (1e9, r'$10^{-3}\,\mu\mathrm{K}^2$'),
        'A_d_BB': (1e9, r'$10^{-3}\,\mu\mathrm{K}^2$'),
    }

    # Add scaling for c_terms (both standard and joint formats)
    for name in labels_free:
        if name.startswith('c_sync') or name.startswith('c_dust'):
            scale_map[name] = (1e9, r'$10^{-3}\,\mu\mathrm{K}^2$')

    samples_plot = apply_corner_scales(samples_free, labels_free, scale_map)

    # -------------------------------
    # LaTeX labels
    # -------------------------------
    latex_labels = {
        # Standard single-mode parameters
        'A_s': r'$A_{\mathrm{s}}\,[10^{-3}\,\mu\mathrm{K}^2]$',
        'alpha_s': r'$\alpha_{\mathrm{s}}$',
        'beta_s': r'$\beta_{\mathrm{s}}$',
        'A_d': r'$A_{\mathrm{d}}\,[10^{-3}\,\mu\mathrm{K}^2]$',
        'alpha_d': r'$\alpha_{\mathrm{d}}$',
        'beta_d': r'$\beta_{\mathrm{d}}$',
        'rho': r'$\rho$',
        
        # Joint EE-BB parameters (mode-specific amplitudes)
        'A_s_EE': r'$A_{\mathrm{s}}^{\mathrm{EE}}\,[10^{-3}\,\mu\mathrm{K}^2]$',
        'A_s_BB': r'$A_{\mathrm{s}}^{\mathrm{BB}}\,[10^{-3}\,\mu\mathrm{K}^2]$',
        'A_d_EE': r'$A_{\mathrm{d}}^{\mathrm{EE}}\,[10^{-3}\,\mu\mathrm{K}^2]$',
        'A_d_BB': r'$A_{\mathrm{d}}^{\mathrm{BB}}\,[10^{-3}\,\mu\mathrm{K}^2]$',
        # Note: alpha_s, alpha_d, beta_s, beta_d, rho are shared (no suffix)
    }

    # Handle c_terms with automatic frequency extraction
    for name in labels_free:
        if name.startswith('c_sync_EE'):
            freq = name.split('[')[-1].strip(']')
            latex_labels[name] = rf'$c_{{\mathrm{{sync}}}}^{{\mathrm{{EE}}}}_{{,{freq}}}\,[10^{{-3}}\,\mu\mathrm{{K}}^2]$'
        elif name.startswith('c_sync_BB'):
            freq = name.split('[')[-1].strip(']')
            latex_labels[name] = rf'$c_{{\mathrm{{sync}}}}^{{\mathrm{{BB}}}}_{{,{freq}}}\,[10^{{-3}}\,\mu\mathrm{{K}}^2]$'
        elif name.startswith('c_dust_EE'):
            freq = name.split('[')[-1].strip(']')
            latex_labels[name] = rf'$c_{{\mathrm{{dust}}}}^{{\mathrm{{EE}}}}_{{,{freq}}}\,[10^{{-3}}\,\mu\mathrm{{K}}^2]$'
        elif name.startswith('c_dust_BB'):
            freq = name.split('[')[-1].strip(']')
            latex_labels[name] = rf'$c_{{\mathrm{{dust}}}}^{{\mathrm{{BB}}}}_{{,{freq}}}\,[10^{{-3}}\,\mu\mathrm{{K}}^2]$'
        elif name.startswith('c_sync'):
            freq = name.split('[')[-1].strip(']')
            latex_labels[name] = rf'$c_{{\mathrm{{sync}},\,{freq}}}\,[10^{{-3}}\,\mu\mathrm{{K}}^2]$'
        elif name.startswith('c_dust'):
            freq = name.split('[')[-1].strip(']')
            latex_labels[name] = rf'$c_{{\mathrm{{dust}},\,{freq}}}\,[10^{{-3}}\,\mu\mathrm{{K}}^2]$'

    labels_plot = [latex_labels.get(name, name) for name in labels_free]

    # -------------------------------
    # Compute automatic ranges centered on posterior distributions
    # -------------------------------
    # Use 0.5th to 98th percentile to center the plot on actual data
    ranges = []
    for i in range(samples_plot.shape[1]):
        q_low = np.percentile(samples_plot[:, i], 0.5)
        q_high = np.percentile(samples_plot[:, i], 98)
        # Add small padding
        margin = (q_high - q_low) * 0.7
        ranges.append((q_low - margin, q_high + margin))

    # -------------------------------
    # Corner plot settings
    # -------------------------------
    corner_kwargs = dict(
        labels=labels_plot,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".3f",
        label_kwargs={"fontsize": 13},
        title_kwargs={"fontsize": 12, "color": "k"},
        smooth=1.9,
        smooth1d=1.0,
        plot_datapoints=False,
        fill_contours=True,
        plot_density=True,
        levels=(0.16, 0.5, 0.84, 0.99),
        color="steelblue",
        hist_kwargs={"color": "steelblue", "alpha": 0.35, "linewidth": 0},
        contour_kwargs={"linewidths": 1.5},
        max_n_ticks=3,
        range=ranges,  # Center plots on actual posterior distributions
    )

    fig = corner.corner(samples_plot, **corner_kwargs)
    fig.set_facecolor("white")

    # -------------------------------
    # Adjust tick size and spacing
    # -------------------------------
    for ax in fig.get_axes():
        ax.tick_params(axis='both', labelsize=12)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(12)

    plt.subplots_adjust(
        left=0.12, right=0.95,
        bottom=0.12, top=0.93,
    )

    # Add title if requested
    if title is not None:
        fig.suptitle(title, fontsize=16, y=0.995)

    # -------------------------------
    # Fill histograms and mark medians
    # -------------------------------
    axes = np.array(fig.axes).reshape(len(labels_plot), len(labels_plot))
    for i in range(len(labels_plot)):
        ax = axes[i, i]
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
        if not filled:
            data = samples_plot[:, i]
            try:
                kde = gaussian_kde(data)
                x_vals = np.linspace(np.min(data), np.max(data), 400)
                y_vals = kde(x_vals)
                ax.cla()
                ax.fill_between(x_vals, 0, y_vals, color="steelblue", alpha=0.5, linewidth=0)
            except Exception:
                ax.cla()
                ax.hist(samples_plot[:, i], bins=50, color="steelblue", alpha=0.5)

        # Median line
        median_val = np.median(samples_plot[:, i])
        ax.axvline(median_val, color="k", lw=1.2, ls="--")
        if ax.get_title():
            ax.set_title(ax.get_title(), fontsize=12, color='k')

    # -------------------------------
    # Save or show
    # -------------------------------
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[plot_corner] Saved corner plot to: {save_path}")
    else:
        plt.show()

    return fig


def create_bin_to_bin_table(
    fit_data_EE, 
    fit_data_BB, 
    samples_free_list_EE, 
    samples_free_list_BB, 
    param_names, 
    ell1, 
    ell2,
    save_path=None,
    format='latex',
    chi2_reduced_EE=None,
    chi2_reduced_BB=None,
):
    """
    Create a publication-quality table showing bin-to-bin fit results for both EE and BB modes.
    Table shows EE results first, then BB results with amplitude ratios (BB/EE).
    
    Parameters
    ----------
    fit_data_EE : dict
        Output from prepare_mcmc_data for EE mode.
    fit_data_BB : dict
        Output from prepare_mcmc_data for BB mode.
    samples_free_list_EE : list of ndarray
        MCMC samples for each ell bin (EE mode).
    samples_free_list_BB : list of ndarray
        MCMC samples for each ell bin (BB mode).
    param_names : list of str
        Names of fitted parameters (e.g., ['A_s', 'beta_s', 'A_d', 'beta_d', 'rho']).
    ell1 : array-like
        Lower edges of ell bins.
    ell2 : array-like
        Upper edges of ell bins.
    save_path : str or None
        If provided, save the table to this path (e.g., 'table.tex' or 'table.txt').
    format : str, default 'latex'
        Output format: 'latex' for LaTeX table, 'ascii' for plain text.
    chi2_reduced_EE : list of float or None
        Per-bin reduced chi-squared values for EE mode (from run_mcmc bin-to-bin).
        If None, the chi2 column is omitted.
    chi2_reduced_BB : list of float or None
        Per-bin reduced chi-squared values for BB mode (from run_mcmc bin-to-bin).
        If None, the chi2 column is omitted.
    
    Returns
    -------
    table_str : str
        The formatted table as a string.
    """
    import pandas as pd
    
    n_bins = len(fit_data_EE['ell_eff'])
    ell_eff_EE = fit_data_EE['ell_eff']
    ell_eff_BB = fit_data_BB['ell_eff']
    
    # Check consistency
    if len(samples_free_list_EE) != n_bins or len(samples_free_list_BB) != n_bins:
        raise ValueError("Number of samples lists does not match number of ell bins")
    
    # Prepare data for EE and BB tables separately
    rows_EE = []
    rows_BB = []
    
    for i in range(n_bins):
        row_EE = {
            'ell_range': f"{int(ell1[i])}--{int(ell2[i])}",
            'ell_eff': f"{ell_eff_EE[i]:.1f}"
        }
        row_BB = {
            'ell_range': f"{int(ell1[i])}--{int(ell2[i])}",
            'ell_eff': f"{ell_eff_BB[i]:.1f}"
        }
        
        # Extract statistics for EE mode
        samples_EE = samples_free_list_EE[i]
        samples_BB = samples_free_list_BB[i]
        
        # Store median values for ratio calculation
        medians_EE = {}
        medians_BB = {}
        
        for j, param_name in enumerate(param_names):
            # --- EE ---
            values_EE = samples_EE[:, j]
            median_EE  = np.median(values_EE)
            lower_EE   = np.percentile(values_EE, 16)
            upper_EE   = np.percentile(values_EE, 84)
            medians_EE[param_name] = median_EE

            # --- BB ---
            values_BB = samples_BB[:, j]
            median_BB  = np.median(values_BB)
            lower_BB   = np.percentile(values_BB, 16)
            upper_BB   = np.percentile(values_BB, 84)
            medians_BB[param_name] = median_BB

            # Apply unit scale (amplitudes: K² → 10⁻³ µK², i.e. ×1e9)
            if param_name in ('A_s', 'A_d'):
                scale = 1e9
            else:
                scale = 1.0

            def _cell(med, lo, hi, s=scale):
                """Format as $med^{+upper}_{-lower}$ after applying scale s."""
                m = med * s
                u = (hi - med) * s
                l = (med - lo) * s
                prec = 3 if param_name == 'rho' else 3 if param_name in ('A_s', 'A_d') else 3
                return f"${m:.{prec}f}^{{+{u:.{prec}f}}}_{{-{l:.{prec}f}}}$"

            row_EE[param_name] = _cell(median_EE, lower_EE, upper_EE)
            row_BB[param_name] = _cell(median_BB, lower_BB, upper_BB)
        
        # Add per-bin chi2 reduced if provided
        if chi2_reduced_EE is not None:
            v = chi2_reduced_EE[i]
            row_EE['chi2'] = f"{v:.3f}" if np.isfinite(v) else "---"
        if chi2_reduced_BB is not None:
            v = chi2_reduced_BB[i]
            row_BB['chi2'] = f"{v:.3f}" if np.isfinite(v) else "---"

        rows_EE.append(row_EE)
        rows_BB.append(row_BB)
    
    # Whether to include a chi2 column
    show_chi2 = (chi2_reduced_EE is not None) or (chi2_reduced_BB is not None)

    if format == 'latex':
        # Create LaTeX table — same style as power-law table (booktabs, footnotesize, scalebox)
        table_lines = []
        table_lines.append(r"\begin{table*}[h]")
        table_lines.append(r"\centering")
        table_lines.append(r"\setlength{\tabcolsep}{4pt}")
        table_lines.append(r"\caption{Bin-to-bin fit results. EE mode (top) and BB mode (bottom).}")
        table_lines.append(r"\label{tab:bin_to_bin_results}")

        # Build column specification: 2 fixed cols + 1 per parameter + optional chi2 col
        n_params = len(param_names)
        n_extra = 1 if show_chi2 else 0
        col_spec = "cc" + "c" * (n_params + n_extra)
        table_lines.append(r"\begin{tabular}{" + col_spec + "}")
        table_lines.append(r"\toprule")

        # Shared helper: build a header row for a given mode suffix (EE or BB)
        def _btb_header(suffix):
            cols = [r"$\ell$ range", r"$\ell_{\rm eff}$"]
            latex_map = {
                'A_s':    rf"$A^{{\rm {suffix}}}_{{\rm s}}\,[10^{{-3}}\,\mu\mathrm{{K}}^2]$",
                'beta_s': rf"$\beta^{{\rm {suffix}}}_{{\rm s}}$",
                'A_d':    rf"$A^{{\rm {suffix}}}_{{\rm d}}\,[10^{{-3}}\,\mu\mathrm{{K}}^2]$",
                'beta_d': rf"$\beta^{{\rm {suffix}}}_{{\rm d}}$",
                'rho':    rf"$\rho^{{\rm {suffix}}}$",
            }
            for p in param_names:
                cols.append(latex_map.get(p, p + rf"$^{{\rm {suffix}}}$"))
            if show_chi2:
                cols.append(r'$\chi^2_\mathrm{red}$')
            return cols

        # ---- EE header + rows ----
        table_lines.append(" & ".join(_btb_header("EE")) + r" \\")
        table_lines.append(r"\midrule")
        for i in range(n_bins):
            row_vals = [rows_EE[i]['ell_range'], rows_EE[i]['ell_eff']]
            for param_name in param_names:
                row_vals.append(rows_EE[i][param_name])
            if show_chi2:
                row_vals.append(rows_EE[i].get('chi2', '---'))
            table_lines.append(" & ".join(row_vals) + r" \\")

        table_lines.append(r"\midrule")

        # ---- BB header + rows ----
        table_lines.append(" & ".join(_btb_header("BB")) + r" \\")
        table_lines.append(r"\midrule")
        for i in range(n_bins):
            row_vals = [rows_BB[i]['ell_range'], rows_BB[i]['ell_eff']]
            for param_name in param_names:
                row_vals.append(rows_BB[i][param_name])
            if show_chi2:
                row_vals.append(rows_BB[i].get('chi2', '---'))
            table_lines.append(" & ".join(row_vals) + r" \\")

        table_lines.append(r"\bottomrule")
        table_lines.append(r"\end{tabular}")
        table_lines.append(r"\end{table*}")

        table_str = "\n".join(table_lines)

    elif format == 'ascii':
        # Create ASCII table with EE results on top, BB (with ratios) below
        table_lines = []
        table_lines.append("=" * 150)
        table_lines.append("Bin-to-bin fit results")
        table_lines.append("EE mode (top) and BB mode (bottom)")
        table_lines.append("For BB mode, amplitudes A_s and A_d are shown in the same units as EE")
        table_lines.append("=" * 150)
        
        # Header for EE (with EE superscript)
        header = f"{'ell range':^12} | {'ell_eff':^8} |"
        for param_name in param_names:
            header += f" {param_name + '_EE':^30} |"
        if show_chi2:
            header += f" {'chi2_red_EE':^12} |"
        table_lines.append(header)
        table_lines.append("-" * 150)
        
        # EE mode data rows
        for i in range(n_bins):
            row_str = f"{rows_EE[i]['ell_range']:^12} | {rows_EE[i]['ell_eff']:^8} |"
            for param_name in param_names:
                row_str += f" {rows_EE[i][param_name]:^30} |"
            if show_chi2:
                row_str += f" {rows_EE[i].get('chi2', '---'):^12} |"
            table_lines.append(row_str)

        table_lines.append("=" * 150)

        # Header for BB
        header_BB = f"{'ell range':^12} | {'ell_eff':^8} |"
        for param_name in param_names:
            header_BB += f" {param_name + '_BB':^30} |"
        if show_chi2:
            header_BB += f" {'chi2_red_BB':^12} |"
        table_lines.append(header_BB)
        table_lines.append("-" * 150)

        # BB mode data rows
        for i in range(n_bins):
            row_str = f"{rows_BB[i]['ell_range']:^12} | {rows_BB[i]['ell_eff']:^8} |"
            for param_name in param_names:
                row_str += f" {rows_BB[i][param_name]:^30} |"
            if show_chi2:
                row_str += f" {rows_BB[i].get('chi2', '---'):^12} |"
            table_lines.append(row_str)
        
        table_lines.append("=" * 150)
        table_str = "\n".join(table_lines)
    
    else:
        raise ValueError(f"format must be 'latex' or 'ascii', got '{format}'")
    
    # Save if requested
    if save_path:
        with open(save_path, 'w') as f:
            f.write(table_str)
        print(f"[create_bin_to_bin_table] Table saved to: {save_path}")
    
    return table_str


def plot_bin_to_bin_results(
    fit_data_EE,
    fit_data_BB,
    samples_free_list_EE,
    samples_free_list_BB,
    param_names,
    chi2_reduced_EE=None,
    chi2_reduced_BB=None,
    save_path=None,
    figsize=(14, 10)
):
    """
    Plot the evolution of fitted parameters with ell for bin-to-bin fitting.
    
    Parameters
    ----------
    fit_data_EE : dict
        Output from prepare_mcmc_data for EE mode.
    fit_data_BB : dict
        Output from prepare_mcmc_data for BB mode.
    samples_free_list_EE : list of ndarray
        MCMC samples for each ell bin (EE mode).
    samples_free_list_BB : list of ndarray
        MCMC samples for each ell bin (BB mode).
    param_names : list of str
        Names of fitted parameters.
    chi2_reduced_EE : list of float, optional
        Reduced chi-squared values for each ell bin (EE mode).
    chi2_reduced_BB : list of float, optional
        Reduced chi-squared values for each ell bin (BB mode).
    save_path : str or None
        If provided, save the figure to this path.
    figsize : tuple
        Figure size (width, height) in inches.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    """
    
    n_params = len(param_names)
    n_bins = len(samples_free_list_EE)
    ell_eff_EE = fit_data_EE['ell_eff']
    ell_eff_BB = fit_data_BB['ell_eff']
    
    # Create subplots (always use 3x2 layout to have space for chi2)
    n_cols = 2
    n_rows = 3
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    # Parameter labels for plotting
    param_labels = {
        'A_s': r'$A_{\rm sync}$ [$\mu$K$^2$]',
        'beta_s': r'$\beta_{\rm sync}$',
        'A_d': r'$A_{\rm dust}$ [$10^{-3}$ $\mu$K$^2$]',
        'beta_d': r'$\beta_{\rm dust}$',
        'rho': r'$\rho$'
    }
    
    # Unit conversion factors (from mK² to desired units)
    unit_conversions = {
        'A_s': 1e6,    
        'A_d': 1e9,    
        'beta_s': 1.0,
        'beta_d': 1.0,
        'rho': 1.0
    }
    
    for i, param_name in enumerate(param_names):
        ax = axes[i]
        
        # Get the conversion factor for this parameter
        conversion_factor = unit_conversions.get(param_name, 1.0)
        
        # Extract values and errors for EE mode
        medians_EE = []
        lower_EE = []
        upper_EE = []
        
        for j in range(n_bins):
            samples = samples_free_list_EE[j][:, i]
            median = np.median(samples) * conversion_factor
            lower = np.percentile(samples, 16) * conversion_factor
            upper = np.percentile(samples, 84) * conversion_factor
            
            medians_EE.append(median)
            lower_EE.append(median - lower)
            upper_EE.append(upper - median)
        
        # Extract values and errors for BB mode
        medians_BB = []
        lower_BB = []
        upper_BB = []
        
        for j in range(n_bins):
            samples = samples_free_list_BB[j][:, i]
            median = np.median(samples) * conversion_factor
            lower = np.percentile(samples, 16) * conversion_factor
            upper = np.percentile(samples, 84) * conversion_factor
            
            medians_BB.append(median)
            lower_BB.append(median - lower)
            upper_BB.append(upper - median)
        
        # Plot EE mode
        ax.errorbar(ell_eff_EE, medians_EE, 
                   yerr=[lower_EE, upper_EE],
                   fmt='o-', color='k', markersize=6, 
                   capsize=3, label='EE', alpha=0.8)
        
        # Plot BB mode
        ax.errorbar(ell_eff_BB, medians_BB,
                   yerr=[lower_BB, upper_BB],
                   fmt='s-', color='goldenrod', markersize=6,
                   capsize=3, label='BB', alpha=0.8)
        
        # Formatting
        ax.set_xlabel(r'$\ell_{\rm eff}$', fontsize=12)
        ylabel = param_labels.get(param_name, param_name)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(loc='best', fontsize=10, frameon=False)
        
        # Use log scale for amplitudes if appropriate
        if 'A_' in param_name:
            try:
                if all(v > 0 for v in medians_EE + medians_BB):
                    ax.set_yscale('log')
            except:
                pass
    
    # Plot chi-squared evolution in the last panel (index n_params)
    if chi2_reduced_EE is not None and chi2_reduced_BB is not None:
        ax_chi2 = axes[n_params]
        
        # Plot EE mode chi2
        ax_chi2.plot(ell_eff_EE, chi2_reduced_EE, 
                    'o-', color='k', markersize=6, 
                    label='EE', alpha=0.8)
        
        # Plot BB mode chi2
        ax_chi2.plot(ell_eff_BB, chi2_reduced_BB,
                    's-', color='goldenrod', markersize=6,
                    label='BB', alpha=0.8)
        
        # Add horizontal line at chi2 = 1
        # ax_chi2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label=r'$\chi^2_{\rm red} = 1$')
        
        # Formatting
        ax_chi2.set_xlabel(r'$\ell_{\rm eff}$', fontsize=12)
        ax_chi2.set_ylabel(r'$\chi^2_{\rm red}$', fontsize=12)
        ax_chi2.legend(loc='best', fontsize=10, frameon=False)
    
    # Remove remaining unused subplots
    for i in range(n_params + 1, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"[plot_bin_to_bin_results] Figure saved to: {save_path}")
    
    return fig


def plot_bin_to_bin_convergence(
    samplers_EE,
    samplers_BB,
    ell_1,
    ell_2,
    ninter,
    discard_fraction=0.5,
    save_path=None,
    figsize=(14, 10)
):
    """
    Plot convergence diagnostics for bin-to-bin MCMC fitting.
    
    Parameters
    ----------
    samplers_EE : list of emcee.EnsembleSampler
        List of samplers for each ell bin (EE mode).
    samplers_BB : list of emcee.EnsembleSampler
        List of samplers for each ell bin (BB mode).
    ell_1 : list of int
        Lower edges of ell bins.
    ell_2 : list of int
        Upper edges of ell bins.
    ninter : int
        Number of MCMC iterations used.
    discard_fraction : float, optional
        Fraction of samples to discard as burn-in (default: 0.5).
    save_path : str or None, optional
        If provided, save the figure to this path.
    figsize : tuple, optional
        Figure size (width, height) in inches.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Extract diagnostics from EE
    tau_max_EE = []
    ESS_min_EE = []
    accept_EE = []
    ell_centers = []
    
    for bin_idx, sampler in enumerate(samplers_EE):
        try:
            tau = sampler.get_autocorr_time(tol=0)
            chain = sampler.get_chain()
            nsteps = chain.shape[0]
            burn = int(discard_fraction * nsteps)
            post = chain[burn:].reshape(-1, chain.shape[2])
            ESS = post.shape[0] / tau
            
            tau_max_EE.append(np.max(tau))
            ESS_min_EE.append(np.min(ESS))
            accept_EE.append(np.mean(sampler.acceptance_fraction))
            ell_centers.append((ell_1[bin_idx] + ell_2[bin_idx]) / 2)
        except:
            tau_max_EE.append(np.nan)
            ESS_min_EE.append(np.nan)
            accept_EE.append(np.nan)
    
    # Extract diagnostics from BB
    tau_max_BB = []
    ESS_min_BB = []
    accept_BB = []
    
    for sampler in samplers_BB:
        try:
            tau = sampler.get_autocorr_time(tol=0)
            chain = sampler.get_chain()
            nsteps = chain.shape[0]
            burn = int(discard_fraction * nsteps)
            post = chain[burn:].reshape(-1, chain.shape[2])
            ESS = post.shape[0] / tau
            
            tau_max_BB.append(np.max(tau))
            ESS_min_BB.append(np.min(ESS))
            accept_BB.append(np.mean(sampler.acceptance_fraction))
        except:
            tau_max_BB.append(np.nan)
            ESS_min_BB.append(np.nan)
            accept_BB.append(np.nan)
    
    # Plot 1: Max autocorrelation time vs ell
    axes[0, 0].plot(ell_centers, tau_max_EE, 'o-', label='EE', color='C0')
    axes[0, 0].plot(ell_centers, tau_max_BB, 's-', label='BB', color='C1')
    axes[0, 0].axhline(0.05 * ninter, ls='--', color='red', alpha=0.5, label='5% threshold')
    axes[0, 0].set_xlabel(r'$\ell$', fontsize=12)
    axes[0, 0].set_ylabel(r'Max $\tau$', fontsize=12)
    axes[0, 0].set_title('Autocorrelation Time', fontsize=13)
    axes[0, 0].legend()
    
    # Plot 2: Min ESS vs ell
    axes[0, 1].plot(ell_centers, ESS_min_EE, 'o-', label='EE', color='C0')
    axes[0, 1].plot(ell_centers, ESS_min_BB, 's-', label='BB', color='C1')
    axes[0, 1].axhline(1000, ls='--', color='orange', alpha=0.5, label='Min threshold (1000)')
    axes[0, 1].axhline(3000, ls='--', color='green', alpha=0.5, label='Good threshold (3000)')
    axes[0, 1].set_xlabel(r'$\ell$', fontsize=12)
    axes[0, 1].set_ylabel('Min ESS', fontsize=12)
    axes[0, 1].set_title('Effective Sample Size', fontsize=13)
    axes[0, 1].legend()
    
    # Plot 3: Acceptance fraction vs ell
    axes[1, 0].plot(ell_centers, accept_EE, 'o-', label='EE', color='C0')
    axes[1, 0].plot(ell_centers, accept_BB, 's-', label='BB', color='C1')
    axes[1, 0].axhline(0.2, ls='--', color='red', alpha=0.3)
    axes[1, 0].axhline(0.6, ls='--', color='red', alpha=0.3)
    axes[1, 0].set_xlabel(r'$\ell$', fontsize=12)
    axes[1, 0].set_ylabel('Acceptance Fraction', fontsize=12)
    axes[1, 0].set_title('MCMC Acceptance', fontsize=13)
    axes[1, 0].legend()
    
    # Plot 4: Summary histogram
    axes[1, 1].hist([ESS_min_EE, ESS_min_BB], bins=15, alpha=0.6, label=['EE', 'BB'])
    axes[1, 1].axvline(1000, ls='--', color='orange', label='Min (1000)')
    axes[1, 1].axvline(3000, ls='--', color='green', label='Good (3000)')
    axes[1, 1].set_xlabel('Min ESS per bin', fontsize=12)
    axes[1, 1].set_ylabel('Count', fontsize=12)
    axes[1, 1].set_title('ESS Distribution', fontsize=13)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[plot_bin_to_bin_convergence] Figure saved to: {save_path}")
    
    return fig


def read_corrected_cls(path_file, band_list):
    """
    Read a FITS file generated by correct_power_spectra and rebuild the dictionary
    with the same structure, so you can access values like:
        spectra['band1_band2']['EE']['SPECTRUM']

    Parameters
    ----------
    path_file : str
        Path to the FITS file created by correct_power_spectra.
    band_list : list of str
        List of frequency bands to read (e.g., ['11','30']).

    Returns
    -------
    spectra : dict
        Dictionary with the same structure as corr_spectra in correct_power_spectra:
            spectra['band1_band2']['EE']['SPECTRUM']
            spectra['band1_band2']['EE']['ERROR']
            spectra['band1_band2']['ell_eff']
    """
    spectra = {}

    with fits.open(path_file) as hdul:
        for band_i in band_list:
            for band_j in band_list:
                key = f"{band_i}_{band_j}"
                if key not in hdul:
                    continue
                data = hdul[key].data

                # Rebuild dictionary for this band pair
                spec_dict = {}

                # Multipole info
                spec_dict['ell1'] = data['ell1']
                spec_dict['ell2'] = data['ell2']
                spec_dict['ell_eff'] = data['ell_eff']

                # Power spectra + errors
                for cl_key in ['TT','EE','BB','TE','TB','EB']:
                    spec_dict[cl_key] = {
                        'SPECTRUM': data[f'{cl_key}_SPECTRUM'],
                        'ERROR':    data[f'{cl_key}_ERROR']
                    }

                spectra[key] = spec_dict

    return spectra

# ============================================================================================
# COLOR CORRECTIONS: simple loader returning dict of polynomials per band
# ============================================================================================

def load_color_correction_polynomials():
    """
    Load color-correction polynomials from the FITS file into a simple dictionary.

    Returns
    -------
    cc_dict : dict
        {
          'synch': { '11': (a0,a1,a2), '23': (...), ... },
          'dust' : { '11': (a0,a1,a2), '353': (...), ... }
        }
    The polynomials are intended to be evaluated at the component's alpha parameter.
    If both 'dust_HFI' and 'dust_nonHFI' are present for a band, 'dust_HFI' is preferred.
    """
    cc_path = color_corrections.get('cc_polynoms')
    if cc_path is None or not os.path.exists(cc_path):
        raise FileNotFoundError(f"Color correction FITS not found at '{cc_path}'")
    with fits.open(cc_path) as hdul:
        tbl = hdul['CC_POLYNOMS'].data
        # Build maps
        synch_map = {}
        dust_map_hfi = {}
        dust_map_non = {}
        for row in tbl:
            comp = str(row['COMP'])
            band = str(row['BAND'])
            a0 = float(row['A0']); a1 = float(row['A1']); a2 = float(row['A2'])
            if comp == 'synch':
                synch_map[band] = (a0, a1, a2)
            elif comp == 'dust_HFI':
                # Convert beta-polynomial to alpha-polynomial using alpha = beta + 2
                # If P(beta) = a0 + a1*beta + a2*beta^2, then Q(alpha) = P(alpha-2)
                # => Q(alpha) = (a0 - 2 a1 + 4 a2) + (a1 - 4 a2) * alpha + (a2) * alpha^2
                A0p = a0 - 2.0*a1 + 4.0*a2
                A1p = a1 - 4.0*a2
                A2p = a2
                dust_map_hfi[band] = (A0p, A1p, A2p)
            elif comp == 'dust_nonHFI':
                dust_map_non[band] = (a0, a1, a2)
        # Merge dust maps, preferring HFI where available
        dust_map = {**dust_map_non, **dust_map_hfi}
        return {'synch': synch_map, 'dust': dust_map}



# ============================================================================================

def fastcc(freq, alpha=False, td=False, bd=False, detector=False, debug=False, option=3, returnfreq=False):
	# Define dictionaries containing the coefficients for different detectors and frequencies
	# WMAP9 + Planck 2013 + QUIJOTE MFI original
	frequencies_v1 = {
		'Q11': [0.99421638, 0.00712698, -0.00234507, 11.1],
		'Q13': [0.98683305, 0.01021644, -0.00182772, 12.9],
		'Q17': [1.00166741, 4.87517031e-04, -6.63221982e-04, 16.7],
		'Q19': [0.997955472, 1.66235456e-03, -4.36898290e-04, 18.7],
		'P30': [0.98520, 0.0131778, -0.00302, 28.4],
		'P44': [0.99059, 0.0079600, -0.00169, 44.1],
		'P70': [0.98149, 0.0152737, -0.00325, 70.4],
		'P100': [0.9957806,  -0.0079764,  -0.00431805, 100.0],
		'P143': [1.0115104,   0.00670483, -0.00465519, 143.0],
		'P217': [0.9843917,  -0.01793987, -0.00345606, 217.0],
		'P353': [0.9834889,  -0.01869916, -0.0031923, 353.0],
		'P545': [0.9852726,  -0.01683206, -0.00393032, 545.0],
		'P857': [1.0015814,  -0.00154559, -0.00421434, 857.0],
		'WK' : {'nu': [20.6, 22.8, 24.9], 'w': [0.332906, 0.374325, 0.292768], 'dT': 1.013438},
		'WKa': {'nu': [30.4, 33.0, 35.6], 'w': [0.322425, 0.387532, 0.290043], 'dT': 1.028413},
		'WQ' : {'nu': [37.8, 40.7, 43.8], 'w': [0.353635, 0.342752, 0.303613], 'dT': 1.043500},
		'WV' : {'nu': [55.7, 60.7, 66.2], 'w': [0.337805, 0.370797, 0.291399], 'dT': 1.098986},
		'WW' : {'nu': [87.0, 93.5, 100.8], 'w': [0.337633, 0.367513, 0.294854], 'dT': 1.247521},
		'DB10': [1.0056317,   -0.0052173,   -0.0119257, 1249],
		'DB9':  [1.0347912,    0.0245728,   -0.0095350, 2141],
		'DB8':  [0.9593942,   -0.0469581,   -0.0075185, 2997],
		'DB7':  [0.9079217,   -0.0942761,   -0.0068850, 4995],
		'DB6':  [0.8160551,   -0.1717965,    0.0103011, 11988],
		'DB5':  [0.9816717,   -0.0327394,   -0.0178211, 24975],
		'DB4':  [0.9947178,   -0.0060948,   -0.0008378, 61163],
		'DB3':  [1.0030533,   -0.0001524,   -0.0032236, 85629],
		'DB2':  [1.0064020,    0.0050962,   -0.0012763, 136227],
		'DB1':  [1.0073260,    0.0044317,   -0.0028791, 239760],
		'I100': [0.9899184805240909, -0.01970072476335274, -0.007706943391147654, 2997],
		'I60': [0.9510100573154361, -0.06111819772388552, -0.016457316462171055, 4995],
		'I25': [0.9123700815757735, -0.08972760549528734, -0.0056369170040417305, 11988],
		'I12': [0.9083422603187424, -0.09300485963152248, -0.008759896726856715, 24975]
	}
	detectors_v1 = {
		'Q111': [0.99278484, 0.00709364, -0.00182065, 11.2],
		'Q113': [0.99583466, 0.00512841, -0.00151745, 12.8],
		'Q217': [1.00166741, 4.87517031e-04, -6.63221982e-04, 16.7],
		'Q219': [0.997955472, 1.66235456e-03, -4.36898290e-04, 18.7],
		'Q311': [0.99421638, 0.00712698, -0.00234507, 11.1],
		'Q313': [0.98683305, 0.01021644, -0.00182772, 12.9],
		'Q417': [0.996066342, 3.10394890e-03, -5.71903939e-04, 17.0],
		'Q419': [0.997016199, 2.40564506e-03, -4.58929918e-04, 19.0],
		'P18': [0.98836, 0.0123556, -0.00394, 70.4],
		'P19': [0.93933, 0.0375844, -0.00225, 70.4],
		'P20': [0.95663, 0.0285644, -0.00273, 70.4],
		'P21': [0.97140, 0.0209690, -0.00318, 70.4],
		'P22': [1.02220,-0.0077263, -0.00327, 70.4],
		'P23': [1.00098, 0.0029940, -0.00240, 70.4],
		'P24': [0.99571, 0.0053247, -0.00175, 44.1],
		'P25': [0.98988, 0.0082248, -0.00161, 44.1],
		'P26': [0.98557, 0.0107023, -0.00175, 44.1],
		'P27': [0.98513, 0.0129780, -0.00288, 28.4],
		'P28': [0.98516, 0.0134605, -0.00318, 28.4]
	}
	mbb_v1 =  {
		'P100': [9.78302300e-01, -8.62236600e-04,  1.36959025e-05, -2.59891562e-02, -3.56237032e-03, 100.0],
		'P143': [1.02247059e+00, -9.17630852e-04,  1.45960412e-05, -1.14919655e-02, -4.26908862e-03, 143.0],
		'P217': [9.74371731e-01, -2.16696435e-03,  3.47218520e-05, -3.21109220e-02, -2.52205669e-03, 217.0],
		'P353': [1.00078213e+00, -3.71209905e-03,  5.97756043e-05, -3.04965433e-02, -2.43337080e-03, 353.0],
		'P545': [1.05366600e+00, -6.55682059e-03,  1.05510808e-04, -2.98897382e-02, -2.86604161e-03, 545.0],
		'P857': [1.09370363e+00, -6.18117163e-03,  9.40234822e-05, -9.27975681e-03, -3.77657381e-03, 857.0]
	}

	# WMAP9 modified by Paddy Leahy, Planck 2015, MFI 2019 pre-release, CBASS-N pre-release
	frequencies_v2 = {
		'CBASSNI': [1.000103213196618, -0.0007238230723748647, -0.0013363809246677324, 4.76],
		'CBASSNP': [0.998561438218094, -0.0059052869678747595, -0.001680320871507042, 4.76],
		'Q11': [0.9820571573496972, 0.011947716397192746, -0.0015316600265756597, 11.1],
		'Q13': [1.002376171424373, 0.00087053237460555, -0.0010541678118863593, 12.9],
		'Q11p': [0.9843034690253142, 0.011094990021336104, -0.0016791051042741416, 11.1],
		'Q13p': [0.9958959030027292, 0.0051175051270606255, -0.0014905894344412235, 12.9],
		'Q17': [1.0045219086496582, -0.0008239816165386079, -0.000711213836891805, 16.8],
		'Q19': [1.0081938704016171, -0.0024991942821279652, -0.0007922783464949779, 18.8],
		'Q17p': [1.0199160282346944, -0.007745808622016744, -0.001118027841018008, 16.8],
		'Q19p': [1.0049510074155987, 0.00014833510393174123, -0.0012521222004909067, 18.8],
		'P30': [1.00513, 0.00301399, -0.00300699, 28.4],
		'P44': [0.994769, 0.00596703, -0.00173626, 44.1],
		'P70': [0.989711, 0.0106943, -0.00328671, 70.4],
		'P100': [0.99806035, -0.00576334, -0.00433317, 100.0],
		'P143': [1.0113103,   0.00652353, -0.00465055, 143.0],
		'P217': [0.98461574, -0.01773364, -0.00346536, 217.0],
		'P353': [0.98313624, -0.01902254, -0.00317299, 353.0],
		'P545': [0.98094696, -0.02082102, -0.00375779, 545.0],
		'P857': [0.994686,   -0.00797382, -0.00402693, 857.0],
		'WK' : [0.972902, 0.0190469, -0.00276464, 22.8],
		'WKa': [0.983787, 0.0117567, -0.00183716, 33.0],
		'WQ' : [0.996854, 0.00496893, -0.00181359, 40.6],
		'WV' : [0.980322, 0.0143631, -0.00223596, 60.8],
		'WW' : [0.984848, 0.0112743, -0.00164595, 93.5],
		'DB10': [1.0056317,   -0.0052173,   -0.0119257, 1249],
		'DB9':  [1.0347912,    0.0245728,   -0.0095350, 2141],
		'DB8':  [0.9593942,   -0.0469581,   -0.0075185, 2997],
		'DB7':  [0.9079217,   -0.0942761,   -0.0068850, 4995],
		'DB6':  [0.8160551,   -0.1717965,    0.0103011, 11988],
		'DB5':  [0.9816717,   -0.0327394,   -0.0178211, 24975],
		'DB4':  [0.9947178,   -0.0060948,   -0.0008378, 61163],
		'DB3':  [1.0030533,   -0.0001524,   -0.0032236, 85629],
		'DB2':  [1.0064020,    0.0050962,   -0.0012763, 136227],
		'DB1':  [1.0073260,    0.0044317,   -0.0028791, 239760],
		'I100': [0.9899184805240909, -0.01970072476335274, -0.007706943391147654, 2997],
		'I60': [0.9510100573154361, -0.06111819772388552, -0.016457316462171055, 4995],
		'I25': [0.9123700815757735, -0.08972760549528734, -0.0056369170040417305, 11988],
		'I12': [0.9083422603187424, -0.09300485963152248, -0.008759896726856715, 24975]
	}
	detectors_v2 = {
		'P18': [0.977484, 0.0185055, -0.00391209, 70.4],
		'P19': [0.965314, 0.0234026, -0.00256943, 70.4],
		'P20': [0.968436, 0.0220869, -0.00285115, 70.4],
		'P21': [0.982854, 0.0142877, -0.00317682, 70.4],
		'P22': [1.049, -0.0237173, -0.00288312, 70.4],
		'P23': [0.990172, 0.0091968, -0.00238961, 70.4],
		'P1823': [0.983195, 0.0141778, -0.00317682, 70.4],
		'P1922': [1.00978, -0.000698302, -0.00328272, 70.4],
		'P2021': [0.97712, 0.0175904, -0.00308092, 70.4],
		'P24': [0.999958, 0.00309391, -0.00177223, 44.1],
		'P25': [0.994381, 0.00591109, -0.00162038, 44.1],
		'P26': [0.990046, 0.00854446, -0.00177223, 44.1],
		'P2526': [0.992115, 0.00717982, -0.00167233, 44.1],
		'P27': [1.00503, 0.00276424, -0.00282717, 28.4],
		'P28': [1.00491, 0.00334266, -0.00313287, 28.4],
		'WK11': [0.939366, 0.0346715, -0.00214346, 22.8],
		'WK12': [1.00894, 0.000982418, -0.00276923, 22.8],
		'WK1': [0.972902, 0.0190469, -0.00276464, 22.8],
		'WKa11': [0.974784, 0.0159578, -0.00161958, 33.0],
		'WKa12': [0.992978, 0.00737502, -0.00200839, 33.0],
		'WKa1': [0.983787, 0.0117567, -0.00183716, 33.0],
		'WQ11': [0.990948, 0.00846474, -0.00204555, 40.6],
		'WQ12': [0.998159, 0.00404356, -0.00167233, 40.6],
		'WQ1': [0.994548, 0.00627672, -0.00186693, 40.6],
		'WQ21': [0.981607, 0.0126181, -0.00166893, 40.6],
		'WQ22': [1.01705, -0.00573297, -0.0016989, 40.6],
		'WQ2': [0.998986, 0.00378172, -0.00176723, 40.6],
		'WV11': [0.939474, 0.0354285, -0.00155105, 60.8],
		'WV12': [0.994737, 0.006396, -0.00217822, 60.8],
		'WV1': [0.966309, 0.0217416, -0.00209331, 60.8],
		'WV21': [1.00662, -0.000113686, -0.00217942, 60.8],
		'WV22': [0.977227, 0.0160255, -0.00220999, 60.8],
		'WV2': [0.991701, 0.0082012, -0.00226214, 60.8],
		'WW11': [0.988343, 0.00956424, -0.00211948, 93.5],
		'WW12': [0.9838, 0.0120015, -0.00173207, 93.5],
		'WW1': [0.986087, 0.0107974, -0.00193167, 93.5],
		'WW21': [0.978714, 0.0149705, -0.00148032, 93.5],
		'WW22': [0.992004, 0.00655744, -0.00146334, 93.5],
		'WW2': [0.985324, 0.0108262, -0.00149331, 93.5],
		'WW31': [0.977457, 0.0155997, -0.00131688, 93.5],
		'WW32': [0.993636, 0.0054001, -0.00134126, 93.5],
		'WW3': [0.985473, 0.0105855, -0.00135485, 93.5],
		'WW41': [0.991452, 0.0072962, -0.00181239, 93.5],
		'WW42': [0.973071, 0.0185705, -0.00153746, 93.5],
		'WW4': [0.982185, 0.0130277, -0.00170889, 93.5],
		'Q111': [0.9925526276311687, 0.007161407883127753, -0.0017891332783659003, 11.2],
		'Q113': [0.9949649181723395, 0.005394042720463821, -0.0014382949498306515, 12.8],
		'Q111p': [0.9925526276311687, 0.007161407883127753, -0.0017891332783659003, 11.2],
		'Q113p': [0.9949649181723395, 0.005394042720463821, -0.0014382949498306515, 12.8],
		'Q217': [1.0129709167366905, -0.005101169068593444, -0.000683203301567257, 16.7],
		'Q219': [1.0175292512259428, -0.007544762181222079, -0.0006248503973819523, 18.7],
		'Q217p': [1.0165157773037472, -0.006419637626097819, -0.000927550610591879, 16.7],
		'Q219p': [1.0136255722145442, -0.004544629474433171, -0.0010878048910708117, 18.7],
		'Q311': [0.9820571573496972, 0.011947716397192746, -0.0015316600265756597, 11.1],
		'Q313': [1.002376171424373, 0.00087053237460555, -0.0010541678118863593, 12.9],
		'Q311p': [0.9843034690253142, 0.011094990021336104, -0.0016791051042741416, 11.1],
		'Q313p': [0.9958959030027292, 0.0051175051270606255, -0.0014905894344412235, 12.9],
		'Q417': [0.9888926591458963, 0.006986281593109233, -0.0007107060258669477, 17.0],
		'Q419': [0.9886433438856964, 0.007541246196285672, -0.0009156379893644272, 19.0],
		'Q417p': [1.0029823815141663, 0.0008640490621996672, -0.0012140291472187072, 17.0],
		'Q419p': [0.9843617866675408, 0.010330165097893152, -0.0012211561793953436, 19.0]
	}
	mbb_v2 =  {
		'P100': [9.8406219e-01, -8.1796874e-04,  1.2984993e-05, -2.3725530e-02, -3.6248658e-03, 100.0],
		'P143': [1.0219676e+00, -9.1921160e-04,  1.4604784e-05, -1.1657137e-02, -4.2607225e-03, 143.0],
		'P217': [9.7481853e-01, -2.1586758e-03,  3.4575922e-05, -3.1927716e-02, -2.5357329e-03, 217.0],
		'P353': [1.0002152e+00, -3.7297627e-03,  6.0069218e-05, -3.0771509e-02, -2.4096791e-03, 353.0],
		'P545': [1.0512086e+00, -7.0331488e-03,  1.1341097e-04, -3.3633444e-02, -2.6284012e-03, 545.0],
		'P857': [1.1041638e+00, -7.7668922e-03,  1.2043497e-04, -1.5862772e-02, -3.4758949e-03, 857.0]
	}

	# WMAP9 modified by Paddy Leahy, Planck LFI 2015, MFI 2019 pre-release, CBASS-N pre-release, HFI 2018
	frequencies_v3 = {
		'CBASSNI': [1.000103213196618, -0.0007238230723748647, -0.0013363809246677324, 4.76],
		'CBASSNP': [0.998561438218094, -0.0059052869678747595, -0.001680320871507042, 4.76],
		'Q11': [0.9821236098857145, 0.011920227628586971, -0.0015347819996889738, 11.1],
		'Q13': [1.0009126144672769, 0.001850712851951476, -0.0011606369297726346, 12.9],
		'Q11p': [0.9829079423583542, 0.011650135267413536, -0.0015947702677977047, 11.1],
		'Q13p': [0.999421869156669, 0.0027138111927979794, -0.0012174359452614659, 12.9],
		'Q17': [1.0067957381497863, -0.0019621413830759947, -0.000715949539348768, 16.8],
		'Q19': [1.0081384744982096, -0.002537677057807221, -0.0007667684588267012, 18.8],
		'Q17p': [1.014273653770507, -0.00557224894906659, -0.0007686809342150011, 16.8],
		'Q19p': [1.0098821312461561, -0.0032961843042380937, -0.0008186810627282263, 18.8],
		'P30': [1.00513, 0.00301399, -0.00300699, 28.4],
		'P44': [0.994769, 0.00596703, -0.00173626, 44.1],
		'P70': [0.989711, 0.0106943, -0.00328671, 70.4],
		'P100': [0.99868757, -0.00512203, -0.00428818, 100.0],
		'P143': [1.0125835,   0.00767883, -0.00468418, 143.0],
		'P217': [0.98670965, -0.01582359, -0.00362294, 217.0],
		'P353': [0.98479307, -0.0174746,  -0.00318638, 353.0],
		'P545': [0.98083174, -0.0209257,  -0.00374975, 545.0],
		'P857': [0.9957921,  -0.00693394, -0.00402974, 857.0],
		'WK' : [0.972902, 0.0190469, -0.00276464, 22.8],
		'WKa': [0.983787, 0.0117567, -0.00183716, 33.0],
		'WQ' : [0.996854, 0.00496893, -0.00181359, 40.6],
		'WV' : [0.980322, 0.0143631, -0.00223596, 60.8],
		'WW' : [0.984848, 0.0112743, -0.00164595, 93.5],
		'DB10': [1.0056317e+00, -6.4834543e-03, -1.1925717e-02,  2.3065088e-04, 1249],
		'DB9':  [1.0347912e+00,  2.5405537e-02, -9.5349792e-03, -1.5169126e-04, 2141],
		'DB8':  [9.5939422e-01, -4.8364848e-02, -7.5184624e-03,  2.5625768e-04, 2997],
		'DB7':  [9.0792167e-01, -9.9896029e-02, -6.8849851e-03,  1.0237540e-03, 4995],
		'DB6':  [8.1605506e-01, -1.7407244e-01,  1.0301138e-02,  4.1460118e-04, 11988],
		'DB5':  [9.8167169e-01, -3.6703132e-02, -1.7821072e-02,  7.2205620e-04, 24975],
		'DB4':  [9.9471784e-01, -6.1251973e-03, -8.3776598e-04,  5.5378068e-06, 61163],
		'DB3':  [1.0030533e+00, -1.7146974e-04, -3.2236280e-03,  3.4781212e-06, 85629],
		'DB2':  [1.0064020e+00,  5.1325657e-03, -1.2763232e-03, -6.6162706e-06, 136227],
		'DB1':  [1.0073260e+00,  4.4526956e-03, -2.8791293e-03, -3.8249291e-06, 239760],
		'I100': [9.8682648e-01, -2.0946426e-02, -7.6484028e-03,  1.4026849e-04, 2997],
		'I60': [9.5354623e-01, -6.4501286e-02, -1.7309224e-02,  8.4418210e-04, 4995],
		'I25': [9.1217232e-01, -9.4449066e-02, -5.8258590e-03,  8.7096798e-04, 11988],
		'I12': [9.0728289e-01, -1.0321426e-01, -9.2115393e-03,  1.4914416e-03, 24975]
  	}
	detectors_v3 = {
		'P18': [0.977484, 0.0185055, -0.00391209, 70.4],
		'P19': [0.965314, 0.0234026, -0.00256943, 70.4],
		'P20': [0.968436, 0.0220869, -0.00285115, 70.4],
		'P21': [0.982854, 0.0142877, -0.00317682, 70.4],
		'P22': [1.049, -0.0237173, -0.00288312, 70.4],
		'P23': [0.990172, 0.0091968, -0.00238961, 70.4],
		'P1823': [0.983195, 0.0141778, -0.00317682, 70.4],
		'P1922': [1.00978, -0.000698302, -0.00328272, 70.4],
		'P2021': [0.97712, 0.0175904, -0.00308092, 70.4],
		'P24': [0.999958, 0.00309391, -0.00177223, 44.1],
		'P25': [0.994381, 0.00591109, -0.00162038, 44.1],
		'P26': [0.990046, 0.00854446, -0.00177223, 44.1],
		'P2526': [0.992115, 0.00717982, -0.00167233, 44.1],
		'P27': [1.00503, 0.00276424, -0.00282717, 28.4],
		'P28': [1.00491, 0.00334266, -0.00313287, 28.4],
		'WK11': [0.939366, 0.0346715, -0.00214346, 22.8],
		'WK12': [1.00894, 0.000982418, -0.00276923, 22.8],
		'WK1': [0.972902, 0.0190469, -0.00276464, 22.8],
		'WKa11': [0.974784, 0.0159578, -0.00161958, 33.0],
		'WKa12': [0.992978, 0.00737502, -0.00200839, 33.0],
		'WKa1': [0.983787, 0.0117567, -0.00183716, 33.0],
		'WQ11': [0.990948, 0.00846474, -0.00204555, 40.6],
		'WQ12': [0.998159, 0.00404356, -0.00167233, 40.6],
		'WQ1': [0.994548, 0.00627672, -0.00186693, 40.6],
		'WQ21': [0.981607, 0.0126181, -0.00166893, 40.6],
		'WQ22': [1.01705, -0.00573297, -0.0016989, 40.6],
		'WQ2': [0.998986, 0.00378172, -0.00176723, 40.6],
		'WV11': [0.939474, 0.0354285, -0.00155105, 60.8],
		'WV12': [0.994737, 0.006396, -0.00217822, 60.8],
		'WV1': [0.966309, 0.0217416, -0.00209331, 60.8],
		'WV21': [1.00662, -0.000113686, -0.00217942, 60.8],
		'WV22': [0.977227, 0.0160255, -0.00220999, 60.8],
		'WV2': [0.991701, 0.0082012, -0.00226214, 60.8],
		'WW11': [0.988343, 0.00956424, -0.00211948, 93.5],
		'WW12': [0.9838, 0.0120015, -0.00173207, 93.5],
		'WW1': [0.986087, 0.0107974, -0.00193167, 93.5],
		'WW21': [0.978714, 0.0149705, -0.00148032, 93.5],
		'WW22': [0.992004, 0.00655744, -0.00146334, 93.5],
		'WW2': [0.985324, 0.0108262, -0.00149331, 93.5],
		'WW31': [0.977457, 0.0155997, -0.00131688, 93.5],
		'WW32': [0.993636, 0.0054001, -0.00134126, 93.5],
		'WW3': [0.985473, 0.0105855, -0.00135485, 93.5],
		'WW41': [0.991452, 0.0072962, -0.00181239, 93.5],
		'WW42': [0.973071, 0.0185705, -0.00153746, 93.5],
		'WW4': [0.982185, 0.0130277, -0.00170889, 93.5],
		'Q111': [0.9925526276311687, 0.007161407883127753, -0.0017891332783659003, 11.2],
		'Q113': [0.9949649181723395, 0.005394042720463821, -0.0014382949498306515, 12.8],
		'Q111p': [0.9925526276311687, 0.007161407883127753, -0.0017891332783659003, 11.2],
		'Q113p': [0.9949649181723395, 0.005394042720463821, -0.0014382949498306515, 12.8],
		'Q217': [1.013613544170594, -0.005386090287803821, -0.0007046039427570969, 16.7],
		'Q219': [1.0156406755924114, -0.006295206555422138, -0.0007590413283371535, 18.7],
		'Q217p': [1.0138278113763701, -0.005438125088975354, -0.0007305386490667834, 16.7],
		'Q219p': [1.0169781391858363, -0.006911736638829896, -0.0007823458097656444, 18.7],
		'Q311': [0.9821236098857145, 0.011920227628586971, -0.0015347819996889738, 11.1],
		'Q313': [1.0009126144672769, 0.001850712851951476, -0.0011606369297726346, 12.9],
		'Q311p': [0.9829079423583542, 0.011650135267413536, -0.0015947702677977047, 11.1],
		'Q313p': [0.999421869156669, 0.0027138111927979794, -0.0012174359452614659, 12.9],
		'Q417': [0.9940133255090668, 0.004348655146028017, -0.0006806931151648366, 17.0],
		'Q419': [0.9900977831410904, 0.006399634237968239, -0.0007378319379189141, 19.0],
		'Q417p': [0.996802897834079, 0.003140068389040273, -0.0007717250294159369, 17.0],
		'Q419p': [0.9901652495861841, 0.006494018875299342, -0.0008034365663807574, 19.0],
	}
	mbb_v3 = {
		'P100': [9.8575562e-01, -7.9658168e-04,  1.2639553e-05, -2.2861226e-02, -3.6062312e-03, 100.0],
		'P143': [1.0249784e+00, -8.9829491e-04,  1.4292214e-05, -1.0535011e-02, -4.3411409e-03, 143.0],
		'P217': [9.7928268e-01, -2.1219356e-03,  3.3978060e-05, -3.0585570e-02, -2.6902291e-03, 217.0],
		'P353': [1.0023545e+00, -3.5956064e-03,  5.7863985e-05, -2.9128991e-02, -2.4766673e-03, 353.0],
		'P545': [1.0511112e+00, -7.0444597e-03,  1.1361618e-04, -3.3710260e-02, -2.6224528e-03, 545.0],
		'P857': [1.1017295e+00, -7.4657206e-03,  1.1545943e-04, -1.4661285e-02, -3.5227763e-03, 857.0]
	}
	retfreq = 0.0
	# Pull out the desired coefficients from the above arrays
	if (detector != False):
		if (debug == True):
			print('Using detector ' + str(detector))
		if td:
			print('Not coded for black-body spectra and detectors, returning zero.')
			if returnfreq:
				return [0,0]
			else:
				return 0
		else:
			if (option == 3):
				cc = detectors_v3.get(detector, 0)
			elif (option == 2):
				cc = detectors_v2.get(detector, 0)
			else:
				cc = detectors_v1.get(detector, 0)

		if (cc == 0):
			print('Invalid detector specified for fastcc ('+str(detector)+'), returning zero.')
			if returnfreq:
				return [0,0]
			else:
				return 0

	else:
		if (debug == True):
			print('Using frequency ' + str(freq))

		if td:
			if (option == 3):
				cc = mbb_v3.get(freq, 0)
			elif (option == 2):
				cc = mbb_v2.get(freq, 0)
			else:
				cc = mbb_v1.get(freq, 0)
		else:
			if (option == 3):
				cc = frequencies_v3.get(freq, 0)
			elif (option == 2):
				cc = frequencies_v2.get(freq, 0)
			else:
				cc = frequencies_v1.get(freq, 0)

		if (cc == 0):
			print('Invalid frequency specified for fastcc ('+str(freq)+'), returning zero.')
			if returnfreq:
				return [0,0]
			else:
				return 0

	if (type(cc) is dict):
		# We have WMAP values
		beta=-2.0+alpha
		T0 = 1.0 * (cc['nu'][0]/cc['nu'][1]) ** beta
		T1 = 1.0
		T2 = 1.0 * (cc['nu'][2]/cc['nu'][1]) ** beta
		dT = 1.0 # Because conversion from T_CMB to T_RJ is done elsewhere.
		fastCC = 1.0 / (cc['dT'] * (cc['w'][0]*T0 + cc['w'][1]*T1 + cc['w'][2]*T2))
		retfreq = cc['nu'][1]
	else:
		if len(cc) == 6:
			fastCC = cc[0] + cc[1]*td + cc[2]*(td**2) + cc[3] * bd + cc[4]*(bd**2)
			retfreq = cc[5]
		elif len(cc) == 5:
			fastCC = cc[0] + cc[1]*alpha + cc[2]*(alpha**2) + cc[3]*(alpha**2)
			retfreq = cc[4]
		else:
			fastCC = cc[0] + cc[1]*alpha + cc[2]*(alpha**2)
			retfreq = cc[3]

	if returnfreq:
		return [retfreq,fastCC]
	else:
		return fastCC
    

def interpcc_setup(infile,band,td_limit=40,method=2):
	# Read in the fits file with precomputed values
	dat = fits.open(infile)
	# print(dat.info())
	# print(dat[1].header)
	bands = dat[1].data[0][0]
	td = dat[1].data[0][1]
	beta = dat[1].data[0][2]
	#
	doing_planck = False
	if band.startswith('DB'):
		band = band.split('DB')[1]
	elif band.startswith('P'):
		band = band.split('P')[1]
		doing_planck = True
	elif band.startswith('I'):
		iras_bands = {'I100': '4', 'I60': '3', 'I25': '2', 'I12': '1'}
		band = iras_bands.get(band, 0)
	idx_band = [ii for ii,bb in enumerate(bands) if band ==bb]

	if len(idx_band) == 0:
		raise ValueError(f"Invalid band name '{band}'")

	if doing_planck:
		map_cc = dat[1].data[0][3][1][idx_band[0]]
		# map_cc = map_cc[::2,::2]
		# td = td[::2]
		# beta = beta[::2]
	else:
		map_cc = dat[1].data[0][3][idx_band[0]]

	# Limit the dust tempertaure
	sel_td = (td <= td_limit)
	X, Y = np.meshgrid(beta,td[sel_td])
	Z = map_cc[:,sel_td].T

	# Interpolation
	if method == 1:
		# Method 1: using interp2d
		return interpolate.interp2d(X, Y, Z, kind ='cubic')
	elif method == 2:
		# Method 2: using Rbf
		return interpolate.Rbf(X,Y,Z,function='cubic')
	else:
		# Method 3, using RegularGridInterpolator
		return interpolate.RegularGridInterpolator((td[sel_td],beta),Z,method='linear')

def interpcc(interp,td,bd):
	try:
		return np.around(interp(bd,td)[()],4)
	except:
		return np.around(interp([td,bd])[()],4)[0]

'''
# ====================================
# Test functions
# ====================================
'''


def compute_and_plot_spectra(experiment, band, mask_path, lmax=1535, target_nside=512, save=False, save_path=None):
    """
    Compute and plot the EE and BB spectra of a CMB map along with:
    - white noise map
    - observed data map
    - simulated map
    - HMDM map
    - noise simulation map

    Notes
    -----
    - All maps (data, simulation, noise, HMDM, and mask) are downgraded to the same NSIDE = target_nside
      before applying the mask and computing spectra. This ensures consistency across inputs.
    - HMDM (Half-Mission Difference Map or similar) is now explicitly downgraded to target_nside.
    - A zoomed-in range in multipole space is also plotted to better visualize ratios.
    
    Parameters
    ----------
    experiment : str
        Name of the experiment ('QUIJOTE', 'WMAP', 'Planck')
    band : str
        Band identifier (e.g., '11', '13', '23', '30', etc.)
    mask_path : str
        Path to the mask FITS file
    lmax : int
        Maximum multipole for spectrum computation
    target_nside : int
        NSIDE to downgrade all maps to (default 512)
    save : bool
        If True, saves the figure to save_path
    save_path : str
        Directory where to save figures (created if it doesn't exist)
    """
    
    # Import data dictionary
    from data import data
    
    # Get map_info from experiment and band
    try:
        map_info = data[experiment][band]
    except KeyError:
        raise ValueError(f"Invalid experiment '{experiment}' or band '{band}'. "
                        f"Available experiments: {list(data.keys())}")
    
    # Helper function to downgrade maps to target NSIDE
    def downgrade_map(m, target_nside):
        """Downgrade a map (I, Q/U, or IQU) to target_nside."""
        if hp.get_nside(m) == target_nside:
            return m
        if m.ndim == 1:  # intensity map
            return hp.ud_grade(m, target_nside)
        else:  # polarization or IQU map
            return np.array([hp.ud_grade(m_ch, target_nside) for m_ch in m])

    # Load maps
    map_data = hp.read_map(map_info['path'], field=(0,1,2), verbose=False)
    map_sim = hp.read_map(map_info['path_simulated'], field=(0,1,2), verbose=False)
    map_hmdm = hp.read_map(map_info['hmdm'], field=(0,1,2), verbose=False)
    
    # Load white noise simulation
    white_noise_file = map_info['white_noise_simulation_1']
    white_noise_path = map_info['path_white_noise_simulations'] + white_noise_file
    map_white_noise = hp.read_map(white_noise_path, field=(0,1,2), verbose=False)
    
    # Load noise simulation
    noise_file = map_info['noise_simulation_1']
    noise_path = map_info['path_noise_simulations'] + noise_file
    map_noise_sim = hp.read_map(noise_path, field=(0,1,2), verbose=False)
    
    # Load mask
    mask = hp.read_map(mask_path, verbose=False)
    
    # Downgrade all maps and mask to the same NSIDE
    map_data = downgrade_map(map_data, target_nside)
    map_sim = downgrade_map(map_sim, target_nside)
    map_white_noise = downgrade_map(map_white_noise, target_nside)
    map_noise_sim = downgrade_map(map_noise_sim, target_nside)
    map_hmdm = downgrade_map(map_hmdm, target_nside)
    mask = downgrade_map(mask, target_nside)
    
    # Convert hp.UNSEEN pixels to 0 before applying mask
    def unseen_to_zero(m):
        """Convert hp.UNSEEN pixels to 0."""
        if m.ndim == 1:  # single component map
            m[m == hp.UNSEEN] = 0.0
        else:  # multi-component map (I, Q, U)
            for i in range(m.shape[0]):
                m[i][m[i] == hp.UNSEEN] = 0.0
        return m
    
    map_data = unseen_to_zero(map_data)
    map_sim = unseen_to_zero(map_sim)
    map_white_noise = unseen_to_zero(map_white_noise)
    map_noise_sim = unseen_to_zero(map_noise_sim)
    map_hmdm = unseen_to_zero(map_hmdm)
    
    # Apply the mask
    map_data *= mask
    map_sim *= mask
    map_white_noise *= mask
    map_noise_sim *= mask
    map_hmdm *= mask
    
    # Compute spectra
    cl_data = hp.anafast(map_data, lmax=lmax)
    cl_sim = hp.anafast(map_sim, lmax=lmax)
    cl_white_noise = hp.anafast(map_white_noise, lmax=lmax)
    cl_noise_sim = hp.anafast(map_noise_sim, lmax=lmax)
    cl_hmdm = hp.anafast(map_hmdm, lmax=lmax)
    
    # Create figure - only EE and BB modes (indices 1 and 2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    spectra_labels = ['EE', 'BB']
    ell1 = 600
    ell2 = 800
    
    fig.suptitle(f'{experiment} {band} GHz', fontsize=14, y=0.9)
    
    for i, ax_pair in enumerate(axes):
        ax_main, ax_zoom = ax_pair
        # Use i+1 to skip TT spectrum (index 0) and plot EE (index 1) and BB (index 2)
        spectrum_idx = i + 1
        
        # Main plot
        ax_main.plot(cl_white_noise[spectrum_idx], label='White Noise', color='C0')
        ax_main.plot(cl_data[spectrum_idx], label='Real Map', color='C1')
        ax_main.plot(cl_sim[spectrum_idx], label='Simulated Map', color='C2')
        ax_main.plot(cl_hmdm[spectrum_idx], label='HMDM', color='C3')
        ax_main.plot(cl_noise_sim[spectrum_idx], label='Noise Simulation 1', color='C4')
        ax_main.set_yscale('log')
        ax_main.set_xlabel(r'Multipole $\ell$')
        ax_main.set_ylabel(rf'$C_\ell^{{{spectra_labels[i]}}}$')
        ax_main.legend(frameon=False)
        ax_main.set_xlim(0,300)
        
        # Set tighter y-axis range for main plot
        main_vals = np.concatenate([cl_white_noise[spectrum_idx][:300], cl_data[spectrum_idx][:300], 
                                   cl_sim[spectrum_idx][:300], cl_hmdm[spectrum_idx][:300], 
                                   cl_noise_sim[spectrum_idx][:300]])
        main_positive_vals = main_vals[main_vals > 0]
        if len(main_positive_vals) > 0:
            y_min_main = np.min(main_positive_vals) * 0.8
            y_max_main = np.max(main_vals) * 1.2
            ax_main.set_ylim(y_min_main, y_max_main)
        
        ax_main.set_title(f'{spectra_labels[i]}')
        
        # Zoom plot
        ell_zoom = np.arange(ell1, ell2)
        white_noise_zoom = cl_white_noise[spectrum_idx][ell1:ell2]
        data_zoom = cl_data[spectrum_idx][ell1:ell2]
        sim_zoom = cl_sim[spectrum_idx][ell1:ell2]
        hmdm_zoom = cl_hmdm[spectrum_idx][ell1:ell2]
        noise_sim_zoom = cl_noise_sim[spectrum_idx][ell1:ell2]
        
        ax_zoom.plot(ell_zoom, white_noise_zoom, label='White Noise', color='C0')
        ax_zoom.plot(ell_zoom, data_zoom, label='Real Map', color='C1')
        # ax_zoom.plot(ell_zoom, sim_zoom, label='Simulated Map', color='C2')
        ax_zoom.plot(ell_zoom, hmdm_zoom, label='HMDM', color='C3')
        ax_zoom.plot(ell_zoom, noise_sim_zoom, label='Noise Simulation 1', color='C4')
        ax_zoom.set_yscale('log')
        ax_zoom.set_xlabel(r'Multipole $\ell$')
        ax_zoom.set_ylabel(rf'$C_\ell^{{{spectra_labels[i]}}}$')
        ax_zoom.legend(frameon=False)
        ax_zoom.set_title(f'{spectra_labels[i]}')
        
        # Y-axis range - more closed/tight range
        all_vals = np.concatenate([white_noise_zoom, data_zoom, hmdm_zoom, noise_sim_zoom])
        positive_vals = all_vals[all_vals > 0]
        if len(positive_vals) > 0:
            y_min = np.min(positive_vals) * 0.5
            y_max = np.max(all_vals) * 2.
            ax_zoom.set_ylim(y_min, y_max)
        
        # Compute mean and ratio
        mean_data = np.mean(data_zoom)
        mean_sim = np.mean(sim_zoom)
        ratio = mean_data / mean_sim if mean_sim != 0 else np.nan
        print(f"{spectra_labels[i]} zoom range (ell={ell1}-{ell2}): "
              f"mean(Real Map)={mean_data:.3e}, mean(Simulated Map)={mean_sim:.3e}, "
              f"ratio={ratio:.3f}")
        
        # Add ratio text
        # ax_zoom.text(0.05, 0.95, f'ratio(Real Map / Simulated Map) = {ratio:.3f}',
                    #  transform=ax_zoom.transAxes,
                    #  fontsize=12, verticalalignment='top',
                    #  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.6))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])  # Leave space for main title
    
    if save and save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        map_name = f"{experiment}_{band}"
        filename = os.path.join(save_path, f"spectra_{map_name}.png")
        plt.savefig(filename)
        print(f"Figure saved to {filename}")
    
    plt.show()




def plot_maps_mollview(map_info, component='I', use_white_noise=True, target_nside=512, 
                       min_val=None, max_val=None, save=False, save_path=None):
    """
    Plot Mollweide projection (hp.mollview) of the real map, simulated map, and noise map
    for a chosen component (I, Q, or U), WITHOUT applying a mask.

    Parameters
    ----------
    map_info : dict
        Dictionary with map information (e.g., data['WMAP']['23'] or data['Planck']['30'])
    component : str
        Which component to plot: 'I', 'Q', or 'U' (default: 'I')
    use_white_noise : bool
        If True, use white noise simulation; if False, use regular noise simulation
    target_nside : int
        NSIDE to downgrade all maps to (default 512)
    min_val : float or None
        Minimum value for color scale
    max_val : float or None
        Maximum value for color scale
    save : bool
        If True, save the figure to save_path
    save_path : str
        Directory where to save the figure (created if it doesn't exist)
    """
    # Map component index
    comp_dict = {'I': 0, 'Q': 1, 'U': 2}
    if component not in comp_dict:
        raise ValueError("Component must be one of 'I', 'Q', or 'U'")
    comp_idx = comp_dict[component]

    # Load maps
    map_data = hp.read_map(map_info['path'], field=(0,1,2), verbose=False)
    map_sim = hp.read_map(map_info['path_simulated'], field=(0,1,2), verbose=False)

    noise_file = map_info['white_noise_simulation_1'] if use_white_noise else map_info['noise_simulation_1']
    noise_path = (map_info['path_white_noise_simulations'] + noise_file 
                  if use_white_noise else map_info['path_noise_simulations'] + noise_file)
    map_noise = hp.read_map(noise_path, field=(0,1,2), verbose=False)

    # Ensure NSIDE consistency
    def downgrade_map(m):
        if hp.get_nside(m) != target_nside:
            return np.array([hp.ud_grade(m_ch, target_nside) for m_ch in m])
        return m

    map_data = downgrade_map(map_data)
    map_sim = downgrade_map(map_sim)
    map_noise = downgrade_map(map_noise)

    # Extract chosen component
    map_data_comp = map_data[comp_idx]
    map_sim_comp = map_sim[comp_idx]
    map_noise_comp = map_noise[comp_idx]

    # Plot in a single figure with 3 subplots
    fig = plt.figure(figsize=(6, 10))
    hp.mollview(map_data_comp, title=f"Data ({component})", norm='hist', min=min_val, max=max_val, sub=(3,1,1), fig=fig)
    hp.mollview(map_sim_comp, title=f"Simulated ({component})", norm='hist', min=min_val, max=max_val, sub=(3,1,2), fig=fig)
    hp.mollview(map_noise_comp, title=f"Noise ({component})", norm='hist', min=min_val, max=max_val, sub=(3,1,3), fig=fig)

    plt.tight_layout()

    # Save if requested
    if save and save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        map_name = map_info.get('name', 'map')  # use map name if provided
        filename = os.path.join(save_path, f"mollview_{map_name}_{component}.png")
        plt.savefig(filename)
        print(f"Figure saved to {filename}")

    plt.show()

	
def plot_cls_auto_cross(spectra_dict, band1, band2, save=False, save_path=None):
    """
    Plot EE, BB, and EB power spectra with error bars for autos (band1_band1, band2_band2)
    and cross (band1_band2) at the same time.

    Parameters
    ----------
    spectra_dict : dict
        Dictionary from read_corrected_cls.
    band1, band2 : str
        Frequency bands to plot (e.g., '11', '30').
    save : bool, default False
        Whether to save the plot to file.
    save_path : str, optional
        Directory where to save the plot. The filename is automatically generated.
    """

    keys = [f"{band1}_{band1}", f"{band2}_{band2}", f"{band1}_{band2}"]
    colors = {
        f"{band1}_{band1}": "steelblue",
        f"{band1}_{band2}": "k",
        f"{band2}_{band2}": "goldenrod"
    }
    modes = ['EE', 'BB', 'EB']

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    for i, mode in enumerate(modes):
        for key in keys:
            if key not in spectra_dict:
                continue
            spec_dict = spectra_dict[key]
            ell = spec_dict['ell_eff']
            spectrum = spec_dict[mode]['SPECTRUM']
            error = spec_dict[mode]['ERROR']

            band_label = key.replace("_", r"$\times$") + " GHz"

            axes[i].errorbar(
                ell, spectrum, yerr=error, fmt='o', markersize=2.5, capsize=1.5, 
                label=band_label, color=colors[key]
            )

        axes[i].set_ylabel(rf"$C_\ell^{{{mode}}} \; [\mathrm{{mK}}^2]$")
        axes[i].legend(frameon=False)
        axes[i].set_yscale('log')
        axes[i].set_xlim(0, 300)
        
		
        ymin = np.min(spectrum[np.isfinite(spectrum)])
        ymax = np.max(spectrum[np.isfinite(spectrum)])
        axes[i].set_ylim(ymin*0.01, ymax*10) 

    axes[-1].set_xlabel(r"$\ell$")
    plt.suptitle(f"Power Spectra: {band1} GHz & {band2} GHz", fontsize=14)

    plt.tight_layout()

    if save:
        if save_path is None:
            raise ValueError("save_path must be provided if save=True")
        os.makedirs(save_path, exist_ok=True)
        filename = os.path.join(save_path, f"cls_{band1}_{band2}.png")
        plt.savefig(filename, dpi=150)
        print(f"Figure saved to {filename}")
    
    plt.show()



def plot_cls_auto_bands(spectra_dict, bands, save=False, save_path=None):
    """
    Plot EE and BB power spectra with error bars for autos of any number of bands.

    Parameters
    ----------
    spectra_dict : dict
        Dictionary from read_corrected_cls containing the spectra.
    bands : list of str
        List of frequency bands (e.g., ['11', '30', '44', '70', '100']).
    save : bool, default False
        Whether to save the plot to a file.
    save_path : str, optional
        Directory where the plot will be saved. The filename is automatically generated.
    """
    if len(bands) == 0:
        raise ValueError("At least one band must be provided")
    
    modes = ['EE', 'BB']
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # Use viridis colormap for better visual distinction
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / max(1, len(bands)-1)) for i in range(len(bands))]

    for i, mode in enumerate(modes):
        for j, band in enumerate(bands):
            key = f"{band}_{band}"
            if key not in spectra_dict:
                print(f"Warning: {key} not found in spectra_dict, skipping")
                continue
            spec_dict = spectra_dict[key]
            ell = spec_dict['ell_eff']
            spectrum = spec_dict[mode]['SPECTRUM']
            error = spec_dict[mode]['ERROR']

            # Pretty label for the legend
            band_label = rf"${band}\mathrm{{GHz}}$"

            axes[i].errorbar(
                ell, spectrum, yerr=error, fmt='o', markersize=3, capsize=2,
                label=band_label, color=colors[j]
            )

        axes[i].set_ylabel(rf"$C_\ell^{{{mode}}} \; [\mathrm{{mK}}^2]$")
        axes[i].legend(frameon=False)
        axes[i].set_yscale('log')
        axes[i].set_xlim(0, 300)

        # Dynamic y-axis adjustment
        finite_vals = np.concatenate([
            spectra_dict[f"{b}_{b}"][mode]['SPECTRUM'] 
            for b in bands if f"{b}_{b}" in spectra_dict
        ])
        finite_vals = finite_vals[np.isfinite(finite_vals)]
        if len(finite_vals) > 0:
            ymin = np.min(finite_vals)
            ymax = np.max(finite_vals)
            axes[i].set_ylim(ymin*0.01, ymax*10)

    axes[-1].set_xlabel(r"$\ell$")
    plt.suptitle("Auto Power Spectra: EE and BB", fontsize=14)
    plt.tight_layout()

    if save:
        if save_path is None:
            raise ValueError("save_path must be provided if save=True")
        os.makedirs(save_path, exist_ok=True)
        filename = os.path.join(save_path, "cls_autos.png")
        plt.savefig(filename, dpi=150)
        print(f"Figure saved to {filename}")
    
    plt.show()


def plot_corrected_vs_theoretical(corrected_spectra_dict, theoretical_spectra_dict, band_pairs, mask_name, save=False, save_path=None, plot_dl=False, filename=None):
    """
    Plot multiple corrected spectra with error bars alongside the theoretical spectra 
    from pre-computed theoretical spectra files. All spectra are plotted on the same figure.

    Parameters
    ----------
    corrected_spectra_dict : dict
        Dictionary from read_corrected_cls containing the corrected spectra.
    theoretical_spectra_dict : dict
        Dictionary from read_spectra_from_fits containing the theoretical spectra.
    band_pairs : list of str or str
        List of band pairs to plot (e.g., ['11_11', '11_30']) or single band pair.
    mask_name : str
        Name of the mask used (for the title).
    save : bool, default False
        Whether to save the plot to a file.
    save_path : str, optional
        Directory where the plot will be saved.
    plot_dl : bool, default False
        If True, plot D_l = l(l+1)C_l/(2π) instead of C_l.
    filename : str, optional
        Custom filename for the saved figure. If None, generates automatic name.
    """
    # Handle single band_pair input for backward compatibility
    if isinstance(band_pairs, str):
        band_pairs = [band_pairs]
    
    # Define color scheme starting with requested colors
    colors = ['steelblue', 'k', 'goldenrod', 'crimson', 'forestgreen', 
              'darkorange', 'mediumorchid', 'brown', 'pink', 'gray']
    
    # Validate all band pairs exist
    for band_pair in band_pairs:
        if band_pair not in corrected_spectra_dict:
            raise ValueError(f"Band pair '{band_pair}' not found in corrected_spectra_dict")
        if band_pair not in theoretical_spectra_dict:
            raise ValueError(f"Band pair '{band_pair}' not found in theoretical_spectra_dict")
    
    # Create plot - EE and BB modes only, vertical layout
    modes_to_plot = ['EE', 'BB']
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Keep track of all values for y-limits
    all_values_per_mode = {mode: [] for mode in modes_to_plot}
    
    # Plot each band pair with its own color
    for band_idx, band_pair in enumerate(band_pairs):
        color = colors[band_idx % len(colors)]  # Cycle through colors if more bands than colors
        
        # Get corrected and theoretical spectrum data
        corr_spec = corrected_spectra_dict[band_pair]
        theo_spec = theoretical_spectra_dict[band_pair]
        ell_eff = corr_spec['ell_eff']
        
        for i, mode in enumerate(modes_to_plot):
            if mode in corr_spec and mode in theo_spec:
                # Filter data to plot only ell range [0, 200]
                ell_mask = (ell_eff >= 0) & (ell_eff <= 200)
                ell_plot = ell_eff[ell_mask]
                
                # Get raw corrected spectrum data and apply mask
                spectrum_raw = corr_spec[mode]['SPECTRUM'][ell_mask]
                error_raw = corr_spec[mode]['ERROR'][ell_mask]

                # Obtain theoretical spectrum values and corresponding ell array.
                # Theoretical data may come in two flavours (simple arrays or dicts
                # with 'MEAN'/'STD' when average+std were stored). Normalize both
                # to plain arrays, then interpolate the theoretical spectrum to
                # the corrected `ell_eff` bins (ell_plot) so boolean indexing
                # lengths match and comparison is meaningful.
                # Get theoretical ell array
                theo_ell = None
                if isinstance(theo_spec.get('ell_eff', None), dict):
                    theo_ell = theo_spec['ell_eff'].get('MEAN')
                else:
                    theo_ell = theo_spec.get('ell_eff')

                # Get theoretical spectrum values for this mode
                if isinstance(theo_spec[mode], dict):
                    # average+std case
                    theo_vals = theo_spec[mode].get('MEAN')
                else:
                    theo_vals = theo_spec[mode]

                # Defensive checks
                if theo_ell is None or theo_vals is None:
                    raise ValueError(f"Theoretical spectrum for '{band_pair}' mode '{mode}' is missing ell or values")

                # Interpolate theoretical spectrum onto the corrected ell bins
                # Use ell_plot (which is subset of corr_spec['ell_eff']) as target
                theo_spectrum_raw = np.interp(ell_plot, theo_ell, theo_vals)
                
                # Apply D_l transformation if requested
                if plot_dl:
                    # D_l = l(l+1)/(2π) * C_l
                    dl_factor = ell_plot * (ell_plot + 1) / (2 * np.pi)
                    spectrum = spectrum_raw * dl_factor
                    error = error_raw * dl_factor
                    theo_spectrum = theo_spectrum_raw * dl_factor
                    legend_symbol_corr = rf'$D_{{\ell}}^{{{mode}}}$'
                    legend_symbol_theo = rf'$D_{{\ell,\mathrm{{theo}}}}^{{{mode}}}$'
                else:
                    spectrum = spectrum_raw
                    error = error_raw
                    theo_spectrum = theo_spectrum_raw
                    legend_symbol_corr = rf'$C_{{\ell}}^{{{mode}}}$'
                    legend_symbol_theo = rf'$C_{{\ell,\mathrm{{theo}}}}^{{{mode}}}$'
                
                # Corrected spectrum with error bars (points) - only plot filtered data
                axes[i].errorbar(
                    ell_plot, spectrum, yerr=error, fmt='o', color=color,
                    markersize=4, capsize=2, alpha=0.8,
                    label=rf'{legend_symbol_corr} {band_pair.replace("_", "x")}'
                )
                
                # Theoretical spectrum (line) - only plot filtered data
                axes[i].plot(
                    ell_plot, theo_spectrum, color=color, linewidth=2, alpha=0.9,
                    label=rf'{legend_symbol_theo} {band_pair.replace("_", "x")}'
                )
                
                # Collect values for y-limits (only from plotted data)
                all_values_per_mode[mode].extend(spectrum[np.isfinite(spectrum) & (spectrum > 0)])
                all_values_per_mode[mode].extend(theo_spectrum[np.isfinite(theo_spectrum) & (theo_spectrum > 0)])
    
    # Set labels, limits and legend for each subplot
    for i, mode in enumerate(modes_to_plot):
        # Set appropriate y-label based on plot_dl option
        if plot_dl:
            axes[i].set_ylabel(rf"$D_\ell^{{{mode}}} \; [\mathrm{{mK}}^2]$")
        else:
            axes[i].set_ylabel(rf"$C_\ell^{{{mode}}} \; [\mathrm{{mK}}^2]$")
        axes[i].legend(frameon=False, fontsize=9)
        axes[i].set_yscale('log')
        # axes[i].set_xlim(30, 185)
        axes[i].set_xlim(0, 200)
        
        # Set reasonable y-limits based on all plotted data
        if all_values_per_mode[mode]:
            ymin = np.min(all_values_per_mode[mode])
            ymax = np.max(all_values_per_mode[mode])
            axes[i].set_ylim(ymin * 0.5, ymax * 2)
    
    axes[-1].set_xlabel(r"$\ell$")
    plt.tight_layout()
    
    if save:
        if save_path is None:
            raise ValueError("save_path must be provided if save=True")
        os.makedirs(save_path, exist_ok=True)
        # Create filename based on band pairs and plot type
        if filename is None:
            bands_str = "_".join(band_pairs)
            plot_type = "Dl" if plot_dl else "Cl"
            filename = f"corrected_vs_theoretical_{plot_type}_{bands_str}_{mask_name}.pdf"
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path)
        print(f"Figure saved to {full_path}")
    
    plt.show()


def plot_auto_cross_spectra(bands, spectra_dict, save=False, save_path=None, figsize=(14, 10), 
                           show_errors=True):
    """
    Create a 2x2 grid of plots showing auto-spectra and cross-spectra for EE and BB modes.
    
    Parameters
    ----------
    bands : list of str
        List of band identifiers to plot (e.g., ['11', '23', '30']).
    spectra_dict : dict
        Dictionary containing spectra data with structure:
        spectra_dict[band_pair]['ell_eff']: effective multipoles
        spectra_dict[band_pair]['EE']['SPECTRUM']: EE mode spectrum
        spectra_dict[band_pair]['EE']['ERROR']: EE mode error
        spectra_dict[band_pair]['BB']['SPECTRUM']: BB mode spectrum
        spectra_dict[band_pair]['BB']['ERROR']: BB mode error
    save : bool, optional
        If True, save the figure to save_path. Default is False.
    save_path : str, optional
        Path where to save the figure. Required if save=True.
    figsize : tuple, optional
        Figure size as (width, height). Default is (14, 10).
    show_errors : bool, optional
        If True, display error bars. Default is True.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object.
    
    Notes
    -----
    The function creates a 2x2 grid with:
    - Top-left: Auto-spectra for EE mode
    - Top-right: Cross-spectra for EE mode
    - Bottom-left: Auto-spectra for BB mode
    - Bottom-right: Cross-spectra for BB mode
    
    Auto-spectra show band_i x band_i correlations (circles for EE, squares for BB).
    Cross-spectra show band_i x band_j correlations with i != j (absolute values plotted).
    Error bars are shown as vertical bars.
    """
    
    # Create 2x2 figure for auto and cross spectra (EE on top, BB on bottom)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Define colors and labels
    default_colors = ['steelblue', 'black', 'goldenrod', 'crimson', 'forestgreen', 'darkorange']
    colors = {band: default_colors[i % len(default_colors)] for i, band in enumerate(bands)}
    labels = {band: f'{band} GHz' for band in bands}
    
    # Generate cross-pairs
    cross_pairs = [(bands[i], bands[j]) for i in range(len(bands)) for j in range(i+1, len(bands))]
    cross_colors = {f'{b1}_{b2}': default_colors[i % len(default_colors)] for i, (b1, b2) in enumerate(cross_pairs)}
    cross_labels = {f'{b1}_{b2}': f'{b1}x{b2} GHz' for b1, b2 in cross_pairs}
    
    # ========== TOP ROW: EE MODE ==========
    # Plot auto-spectra EE (top-left)
    ax_auto_ee = axes[0, 0]
    for band in bands:
        pair = f'{band}_{band}'
        if pair in spectra_dict:
            ell = spectra_dict[pair]['ell_eff']
            cl_ee = spectra_dict[pair]['EE']['SPECTRUM']
            err_ee = spectra_dict[pair]['EE']['ERROR'] if show_errors else None
            
            if show_errors and err_ee is not None:
                ax_auto_ee.errorbar(ell, cl_ee, yerr=err_ee, fmt='o', 
                                   color=colors[band], label=f'{labels[band]}', 
                                   alpha=0.7, capsize=3, markersize=5)
            else:
                ax_auto_ee.plot(ell, cl_ee, 'o', color=colors[band], 
                              label=f'{labels[band]}', alpha=0.7, markersize=5)
    
    ax_auto_ee.set_xlabel(r'$\ell$', fontsize=12)
    ax_auto_ee.set_ylabel(r'$C_\ell^{EE}$ [$\mu$K$^2$]', fontsize=12)
    ax_auto_ee.set_title('Auto-Spectra (EE)', fontsize=14)
    ax_auto_ee.set_yscale('log')
    ax_auto_ee.legend(fontsize=9, frameon=False)
    
    # Plot cross-spectra EE (top-right)
    ax_cross_ee = axes[0, 1]
    for b1, b2 in cross_pairs:
        pair = f'{b1}_{b2}'
        if pair in spectra_dict:
            ell = spectra_dict[pair]['ell_eff']
            cl_ee = spectra_dict[pair]['EE']['SPECTRUM']
            err_ee = spectra_dict[pair]['EE']['ERROR'] if show_errors else None
            
            if show_errors and err_ee is not None:
                ax_cross_ee.errorbar(ell, cl_ee, yerr=err_ee, fmt='o', 
                                    color=cross_colors[pair], label=f'{cross_labels[pair]}', 
                                    alpha=0.7, capsize=3, markersize=5)
            else:
                ax_cross_ee.plot(ell, cl_ee, 'o', color=cross_colors[pair], 
                               label=f'{cross_labels[pair]}', alpha=0.7, markersize=5)
    
    ax_cross_ee.set_xlabel(r'$\ell$', fontsize=12)
    ax_cross_ee.set_ylabel(r'$C_\ell^{EE}$ [$\mu$K$^2$]', fontsize=12)
    ax_cross_ee.set_title('Cross-Spectra (EE)', fontsize=14)
    ax_cross_ee.set_yscale('log')
    ax_cross_ee.legend(fontsize=9, frameon=False)
    
    # ========== BOTTOM ROW: BB MODE ==========
    # Plot auto-spectra BB (bottom-left)
    ax_auto_bb = axes[1, 0]
    for band in bands:
        pair = f'{band}_{band}'
        if pair in spectra_dict:
            ell = spectra_dict[pair]['ell_eff']
            cl_bb = spectra_dict[pair]['BB']['SPECTRUM']
            err_bb = spectra_dict[pair]['BB']['ERROR'] if show_errors else None
            
            if show_errors and err_bb is not None:
                ax_auto_bb.errorbar(ell, cl_bb, yerr=err_bb, fmt='s', 
                                   color=colors[band], label=f'{labels[band]}', 
                                   alpha=0.7, capsize=3, markersize=5)
            else:
                ax_auto_bb.plot(ell, cl_bb, 's', color=colors[band], 
                              label=f'{labels[band]}', alpha=0.7, markersize=5)
    
    ax_auto_bb.set_xlabel(r'$\ell$', fontsize=12)
    ax_auto_bb.set_ylabel(r'$C_\ell^{BB}$ [$\mu$K$^2$]', fontsize=12)
    ax_auto_bb.set_title('Auto-Spectra (BB)', fontsize=14)
    ax_auto_bb.set_yscale('log')
    ax_auto_bb.legend(fontsize=9, frameon=False)
    
    # Plot cross-spectra BB (bottom-right)
    ax_cross_bb = axes[1, 1]
    for b1, b2 in cross_pairs:
        pair = f'{b1}_{b2}'
        if pair in spectra_dict:
            ell = spectra_dict[pair]['ell_eff']
            cl_bb = spectra_dict[pair]['BB']['SPECTRUM']
            err_bb = spectra_dict[pair]['BB']['ERROR'] if show_errors else None
            
            if show_errors and err_bb is not None:
                ax_cross_bb.errorbar(ell, cl_bb, yerr=err_bb, fmt='s', 
                                    color=cross_colors[pair], label=f'{cross_labels[pair]}', 
                                    alpha=0.7, capsize=3, markersize=5)
            else:
                ax_cross_bb.plot(ell, cl_bb, 's', color=cross_colors[pair], 
                               label=f'{cross_labels[pair]}', alpha=0.7, markersize=5)
    
    ax_cross_bb.set_xlabel(r'$\ell$', fontsize=12)
    ax_cross_bb.set_ylabel(r'$C_\ell^{BB}$ [$\mu$K$^2$]', fontsize=12)
    ax_cross_bb.set_title('Cross-Spectra (BB)', fontsize=14)
    ax_cross_bb.set_yscale('log')
    ax_cross_bb.legend(fontsize=9, frameon=False)
    
    plt.tight_layout()
    
    if save:
        if save_path is None:
            raise ValueError("save_path must be provided if save=True")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    return fig


def plot_spectra_with_bestfit(fit_data, results_entry, fit_components, fit_c_terms,
                              bands_to_plot=('11', '23', '30'),
                              color_correction=True,
                              save_path=None, title=None):
    """
    Plot auto-spectra data points with error bars for selected bands,
    overlaid with the best-fit synchrotron model curves, for both EE and BB
    in a single figure (two panels side-by-side).

    Parameters
    ----------
    fit_data : dict
        Dictionary with two keys, ``'EE'`` and ``'BB'``, each being the
        output of :func:`prepare_mcmc_data` for the respective mode.
    results_entry : dict
        Dictionary with two keys, ``'EE'`` and ``'BB'``, each being the
        result dict stored in *results_list* (must contain
        ``'samples_free'``, ``'param_map'``, ``'chi2_reduced'``).
    fit_components : tuple
        Components that were fit, e.g. ``('sync', 'dust', 'cross')``.
    fit_c_terms : bool
        Whether constant terms were fitted.
    bands_to_plot : tuple of str
        Band identifiers whose **auto-spectra** will be plotted.
    color_correction : bool
        Whether to apply color-correction polynomials to the model curves.
    save_path : str, optional
        If given, figure is saved to this path.
    title : str, optional
        Super-title for the figure.

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt

    # Load color-correction polynomials if needed
    cc_dict = None
    if color_correction:
        try:
            cc_dict = load_color_correction_polynomials()
        except Exception:
            cc_dict = None

    FREQ_MAX_C = 40.0

    # ---- colour / marker per band
    band_styles = {
        '11': dict(color='steelblue', marker='o', label='11 GHz'),
        '13': dict(color='#ff7f0e', marker='o', label='13 GHz'),
        '17': dict(color='#2ca02c', marker='o', label='17 GHz'),
        '19': dict(color='#d62728', marker='o', label='19 GHz'),
        '23': dict(color='k', marker='o', label='23 GHz'),
        '30': dict(color='goldenrod', marker='o', label='30 GHz'),
        '33': dict(color='#e377c2', marker='o', label='33 GHz'),
    }
    default_style = dict(color='grey', marker='*', label='?')

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    mode_list = ['EE', 'BB']

    for ax, mode in zip(axes, mode_list):
        fd = fit_data[mode]
        res = results_entry[mode]
        datasets = fd['datasets']
        ell = fd['ell_eff']
        y_all = fd['y_all']
        yerr_all = fd['yerr_all']
        samples_free = res['samples_free']
        param_map = res['param_map']

        # ---- Reconstruct best-fit full parameter vector
        if len(param_map) > 0 and isinstance(param_map[0], tuple):
            free_names = [name for name, is_free in param_map if is_free]
            all_names = [name for name, _ in param_map]
            free_cols = [i for i, (_, f) in enumerate(param_map) if f]
            fixed_cols = [i for i, (_, f) in enumerate(param_map) if not f]
        else:
            free_names = list(param_map)
            all_names = free_names
            free_cols = list(range(len(free_names)))
            fixed_cols = []

        # Use median of posterior as best-fit
        best_free = np.median(samples_free, axis=0)
        best_full = np.zeros(len(all_names))
        for j, col in enumerate(free_cols):
            best_full[col] = best_free[j]

        # Extract synchrotron parameters
        idx_map_names = {name: i for i, name in enumerate(all_names)}
        A_s = best_full[idx_map_names['A_s']] if 'A_s' in idx_map_names else 0.0
        alpha_s = best_full[idx_map_names['alpha_s']] if 'alpha_s' in idx_map_names else 0.0
        beta_s = best_full[idx_map_names['beta_s']] if 'beta_s' in idx_map_names else 0.0

        unique_freqs = sorted({f for d in datasets for f in d['freqs']})
        low_freqs = sorted({f for f in unique_freqs if f <= FREQ_MAX_C})
        c_sync = {}
        for lf in low_freqs:
            cname = f'c_sync[{int(lf)}]'
            if cname in idx_map_names:
                c_sync[lf] = best_full[idx_map_names[cname]]
            else:
                c_sync[lf] = 0.0

        freq_ref = 23.
        ell_ref = 80.0

        # ---- Plot data + model for each requested auto band
        ell_fine = np.linspace(ell.min() - 5, ell.max() + 5, 200)

        for band in bands_to_plot:
            pair_name = f'{band}_{band}'
            style = band_styles.get(band, default_style)

            # Find the dataset for this auto-spectrum
            ds_match = [d for d in datasets if d['pair'] == pair_name and d['mode'] == mode]
            if not ds_match:
                continue
            d = ds_match[0]
            s0, s1 = d['slice']
            spec = y_all[s0:s1] * 1e9   # K² → μK²
            err = yerr_all[s0:s1] * 1e9
            f1, f2 = d['freqs']

            # Model curve on fine ell grid
            scale_f = (f1 / freq_ref) ** beta_s * (f2 / freq_ref) ** beta_s
            ell_scale = (ell_fine / ell_ref) ** alpha_s
            model_curve = A_s * ell_scale * scale_f * 1e9  # → μK²

            # Add c_term if applicable
            if fit_c_terms and f1 in c_sync:
                model_curve = model_curve + c_sync[f1] * 1e9

            # Apply color correction to model
            if cc_dict is not None:
                alpha_cc = 2.0 + float(beta_s)
                poly1 = (cc_dict.get('synch', {}) or {}).get(str(band))
                poly2 = (cc_dict.get('synch', {}) or {}).get(str(band))
                cc_s1 = (poly1[0] + poly1[1]*alpha_cc + poly1[2]*(alpha_cc**2)) if poly1 is not None else 1.0
                cc_s2 = (poly2[0] + poly2[1]*alpha_cc + poly2[2]*(alpha_cc**2)) if poly2 is not None else 1.0
                model_curve = model_curve / (cc_s1 * cc_s2)

            # Data points
            ax.errorbar(ell, spec, yerr=err, fmt=style['marker'],
                        color=style['color'], markersize=6, capsize=3,
                        label=f"{style['label']} data", zorder=3)
            # Model curve
            ax.plot(ell_fine, model_curve, '-', color=style['color'],
                    alpha=0.8, linewidth=2.0,
                    label=f"{style['label']} model", zorder=2)

        ax.set_yscale('log')
        ax.set_xlim(20, 210)
        ax.set_ylim(1e-13, 1e-3)
        ax.set_xlabel(r'$\ell$', fontsize=20)
        ax.set_ylabel(r'$C_\ell\;[\mu\mathrm{K}^2_\mathrm{RJ}]$', fontsize=20)
        ax.set_title(f'{mode}', fontsize=20)
        ax.tick_params(axis='both', labelsize=18)
        ax.legend(fontsize=12, frameon=False, ncol=2)
        ax.axhline(0, color='grey', ls='--', lw=0.5)

    if title:
        fig.suptitle(title, fontsize=18, y=1.02)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"[plot_spectra_with_bestfit] Saved to {save_path}")

    plt.show()
    return fig


def _expand_free(theta_free, param_map):
    """Expand free-parameter vector into full vector using param_map."""
    if len(param_map) > 0 and isinstance(param_map[0], tuple):
        all_names = [name for name, _ in param_map]
        free_cols = [i for i, (_, f) in enumerate(param_map) if f]
    else:
        return theta_free
    full = np.zeros(len(all_names))
    for j, col in enumerate(free_cols):
        full[col] = theta_free[j]
    return full


def _model_full(theta_full, datasets, ell, fit_c_terms, fit_components, cc_dict, freq_max_c=40.0):
    """Evaluate the full model (sync + dust + cross) from a full theta vector."""
    unique_freqs = sorted({f for d in datasets for f in d['freqs']})
    low_freqs = sorted({f for f in unique_freqs if f <= freq_max_c})
    N_c = len(low_freqs)

    A_s, alpha_s, beta_s, A_d, alpha_d, beta_d, rho = theta_full[:7]
    offset = 7
    c_sync = np.zeros(N_c)
    if fit_c_terms:
        c_sync = np.asarray(theta_full[offset:offset+N_c])
        offset += N_c

    y_model = np.zeros(sum(d['slice'][1] - d['slice'][0] for d in datasets))

    if 'sync' in fit_components:
        y_model += model_synchrotron(
            [A_s, alpha_s, beta_s, *c_sync] if fit_c_terms else [A_s, alpha_s, beta_s],
            datasets, ell, fit_c_terms=fit_c_terms, cc_dict=cc_dict, freq_max_c=freq_max_c)
    if 'dust' in fit_components:
        y_model += model_dust([A_d, alpha_d, beta_d], datasets, ell, cc_dict=cc_dict)
    if 'cross' in fit_components:
        y_model += model_cross([rho, A_s, A_d, alpha_s, alpha_d, beta_s, beta_d],
                               datasets, ell, cc_dict=cc_dict)
    return y_model


def create_fitting_results_table(results_list, save_path=None, 
                                 caption=None, label='tab:fit_results',
                                 ell_range='20-120', mask_name='$10^{\\circ}$ Galactic cut',
                                 include_c_terms=False):
    """
    Generate a LaTeX table with posterior constraints from MCMC fitting results.
    
    When ``include_c_terms=True`` the constant offset terms
    (``c_sync[freq]``) are placed in a **separate** companion table whose
    values are expressed in units of :math:`10^{-3}\\,\\mu\\mathrm{K}^2`.
    
    Parameters
    ----------
    results_list : list of dict
        Each dict should contain:
        - 'data_label': str (e.g., 'WMAP+Planck', 'QUIJOTE+WMAP+Planck')
        - 'mode': str ('EE' or 'BB')
        - 'samples_free': ndarray (MCMC samples after burn-in)
        - 'param_map': list (parameter map from run_mcmc)
        - 'chi2_reduced': float (optional, reduced chi-squared)
    save_path : str, optional
        Path to save the LaTeX table file.  When *include_c_terms* is True a
        second file with ``_c_terms`` appended to the stem is saved alongside.
    caption : str, optional
        Custom caption for the main table. If None, a default is generated.
    label : str, optional
        LaTeX label for the table (default: 'tab:fit_results').
    ell_range : str, optional
        Multipole range string for caption (default: '20--200').
    mask_name : str, optional
        Mask description for caption (default: '$10^{\\circ}$ Galactic cut').
    include_c_terms : bool, optional
        If True, produce a second table with the c_sync constant terms
        in units of :math:`10^{-3}\\,\\mu\\mathrm{K}^2`.  Default: False.
    
    Returns
    -------
    str
        LaTeX table code (main table).  When *include_c_terms* is True
        the returned string contains **both** tables separated by a blank
        line.
    """
    
    # Generate default caption if not provided
    if caption is None:
        caption = (
            f"Posterior constraints (median and $68\\%$ credible intervals) on the "
            f"synchrotron and dust amplitudes, spectral indices, and on the "
            f"synchrotron--dust correlation coefficient $\\rho$, from fits to the "
            f"$EE$ and $BB$ spectra using the $b$ $>$ $|$10º$|$ mask."
        )
    
    # -----------------------------------------------------------------
    # Identify parameters
    # -----------------------------------------------------------------
    # Base (physical) parameters – always in the main table
    base_param_order = ['A_s', 'alpha_s', 'beta_s', 'A_d', 'alpha_d', 'beta_d', 'rho']
    
    # Auto-detect c_sync frequencies from all results
    c_sync_names = []
    if include_c_terms:
        c_freq_set = set()
        for result in results_list:
            pm = result['param_map']
            if len(pm) > 0 and isinstance(pm[0], tuple):
                names = [name for name, is_free in pm if is_free]
            else:
                names = list(pm)
            for n in names:
                if n.startswith('c_sync['):
                    c_freq_set.add(n)
        c_sync_names = sorted(c_freq_set, key=lambda s: int(s.split('[')[1].rstrip(']')))
    
    # -----------------------------------------------------------------
    # LaTeX column names
    # -----------------------------------------------------------------
    latex_headers = {
        'A_s': r'$A_s\,[10^{-3}\,\mu\mathrm{K}^2]$',
        'alpha_s': r'$\alpha_s$',
        'beta_s': r'$\beta_s$',
        'A_d': r'$A_d\,[10^{-3}\,\mu\mathrm{K}^2]$',
        'alpha_d': r'$\alpha_d$',
        'beta_d': r'$\beta_d$',
        'rho': r'$\rho$'
    }
    for cname in c_sync_names:
        freq_str = cname.split('[')[1].rstrip(']')
        latex_headers[cname] = (
            f'$c_{{\\mathrm{{sync}}}}^{{{freq_str}}}\\,'
            r'[10^{-3}\,\mu\mathrm{K}^2]$'
        )
    
    # -----------------------------------------------------------------
    # Scaling factors
    # -----------------------------------------------------------------
    scale_factors = {
        'A_s': 1e9,      # K² → μK²
        'A_d': 1e9,      # K² → 10^-3 μK²
        'alpha_s': 1.0,
        'beta_s': 1.0,
        'alpha_d': 1.0,
        'beta_d': 1.0,
        'rho': 1.0
    }
    # c_sync terms: display in 10^-3 μK²
    for cname in c_sync_names:
        scale_factors[cname] = 1e9
    
    # -----------------------------------------------------------------
    # Compute median ± 68 % CI for every parameter in every result
    # -----------------------------------------------------------------
    all_param_values = []          # one dict per result
    for result in results_list:
        samples_free = result['samples_free']
        param_map = result['param_map']
        
        if len(param_map) > 0 and isinstance(param_map[0], tuple):
            param_names = [name for name, is_free in param_map if is_free]
        else:
            param_names = list(param_map)
        
        param_values = {}
        for i, pname in enumerate(param_names):
            if i >= samples_free.shape[1]:
                continue
            samples = samples_free[:, i]
            median = np.median(samples)
            lower = np.percentile(samples, 16)
            upper = np.percentile(samples, 84)
            scale = scale_factors.get(pname, 1.0)
            median *= scale
            lower *= scale
            upper *= scale
            param_values[pname] = {
                'median': median,
                'lower': median - lower,
                'upper': upper - median
            }
        all_param_values.append(param_values)
    
    # -----------------------------------------------------------------
    # Helper: format a single value cell
    # -----------------------------------------------------------------
    def _fmt(param_values, pname):
        if pname in param_values:
            p = param_values[pname]
            return f"${p['median']:.3f}^{{+{p['upper']:.3f}}}_{{-{p['lower']:.3f}}}$"
        return '---'
    
    # -----------------------------------------------------------------
    # Helper: build a complete table* environment
    # -----------------------------------------------------------------
    def _build_table(param_list, cap, lab, chi2_col=False):
        lines = []
        lines.append(r'\begin{table*}[h]')
        lines.append(r'\centering')
        lines.append(r'\footnotesize')
        lines.append(r'\setlength{\tabcolsep}{4pt}')
        lines.append(f'\\caption{{{cap}}}')
        lines.append(f'\\label{{{lab}}}')

        extra = 1 if chi2_col else 0
        colspec = 'll' + 'c' * (len(param_list) + extra)
        lines.append(r'\scalebox{0.95}{')
        lines.append(f'\\begin{{tabular}}{{{colspec}}}')
        lines.append(r'\toprule')

        header = ['Data', 'Mode'] + [latex_headers[p] for p in param_list]
        if chi2_col:
            header.append(r'$\chi^2_\mathrm{red}$')
        lines.append(' & '.join(header) + r' \\')
        lines.append(r'\midrule')

        prev_label = None
        for idx, result in enumerate(results_list):
            # Normalize data label to short form required by tables
            raw_label = result.get('data_label', '')
            if 'QUIJOTE' in raw_label.upper() or raw_label.upper().startswith('Q'):
                cur_label = 'QJ+WP+Pl'
            elif ('WMAP' in raw_label.upper() and 'PLANCK' in raw_label.upper()) or raw_label.upper().startswith('W'):
                cur_label = 'WP+Pl'
            else:
                cur_label = raw_label

            # Insert \midrule between distinct data_label groups
            if prev_label is not None and cur_label != prev_label:
                lines.append(r'\midrule')
            prev_label = cur_label

            pv = all_param_values[idx]
            row = [cur_label, result['mode']]
            for pname in param_list:
                row.append(_fmt(pv, pname))
            if chi2_col:
                chi2 = result.get('chi2_reduced', None)
                row.append(f'${chi2:.2f}$' if chi2 is not None else '---')
            lines.append(' & '.join(row) + r' \\')

        lines.append(r'\bottomrule')
        lines.append(r'\end{tabular}}')
        lines.append(r'\end{table*}')
        return '\n'.join(lines)
    
    # -----------------------------------------------------------------
    # Build main table (physical parameters + chi2)
    # -----------------------------------------------------------------
    main_table = _build_table(base_param_order, caption, label, chi2_col=True)
    
    # -----------------------------------------------------------------
    # Build c_terms table (if requested)
    # -----------------------------------------------------------------
    c_table = ''
    if include_c_terms and c_sync_names:
        c_caption = (
            f"Constant offset terms $c_{{\\mathrm{{sync}}}}$ "
            f"(in $10^{{-3}}\\,\\mu\\mathrm{{K}}^2$) for the auto-spectra, "
            f"from the same fits as Table~\\ref{{{label}}} "
            f"(multipole range $\\ell={ell_range}$)."
        )
        c_label = label + '_c_terms'
        c_table = _build_table(c_sync_names, c_caption, c_label, chi2_col=False)
    
    # -----------------------------------------------------------------
    # Combine and save
    # -----------------------------------------------------------------
    if c_table:
        full_output = main_table + '\n\n' + c_table
    else:
        full_output = main_table
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(full_output)
        print(f"LaTeX table saved to {save_path}")
    
    return full_output
