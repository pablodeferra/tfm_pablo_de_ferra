#%%
import numpy as np
import pysm3
import healpy as hp
from astropy import units as u
import os
from data import data, path_map
from tqdm import tqdm 
import pymaster as nmt
from astropy.io import fits
import re
from scipy.constants import c,h,k
import matplotlib.pyplot as plt
import sys
import pysm3.units as u_pysm
import emcee
import corner
import multiprocessing as mp
from scipy.stats import gaussian_kde

from data import data

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


def generate_band_beams(BANDS, beam_path, save_path):
    '''
    Generate averaged beams for each frequency band from DA beams.
    
    Parameters
    ----------
    BANDS : dict
        Dictionary of bands and their DAs.
    beam_path : str
        Path to the directory containing original DA beam txt files.
    save_path : str
        Path where the new band beams will be saved.

    Returns
    -------
    None
        The function writes a text file for each band to disk and prints a confirmation message.
    '''
    
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
            all_data.append(data)
        if len(all_data) == 0:
            print(f'No beams found for band {band}')
            continue
        # Average B_l (column 2) and fractional error (column 3) across DAs
        avg_data = np.copy(all_data[0])
        if len(all_data) > 1:
            for i in range(1, len(all_data)):
                avg_data[:,1] += all_data[i][:,1]
                avg_data[:,2] += all_data[i][:,2]
            avg_data[:,1] /= len(all_data)
            avg_data[:,2] /= len(all_data)
        save_band_beam(band, header_template, avg_data, save_path)


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
                                 workspaces=None, lmax=None):
    """
    Compute the average and standard deviation of auto- and cross-spectra
    over multiple noise realizations using precomputed NaMaster workspaces.

    This function returns a unified dictionary structure:
        spectra['band_i_band_j']['TT']['MEAN']
        spectra['band_i_band_j']['TT']['STD']

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
        base, ext = os.path.splitext(out_file)
        if ext.lower() != ".fits":
            ext = ".fits"
        out_file = f"{base}_wn{ext}"

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
        base, ext = os.path.splitext(path_fits)
        if ext.lower() != ".fits":
            ext = ".fits"
        path_fits = f"{base}_wn{ext}"

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
                unit_dict[band] = cmb_unit_conversion(nuGHz, 'KCMB2KRJ') if correct_unit else 1.0
                    
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
            
            # Step 1: Noise/HMDM subtraction (only for auto-spectra)
            if is_cross_spectrum:
                # For cross-spectra: no noise subtraction, use raw spectrum
                Cl = Cl_raw
            else:
                # For auto-spectra: subtract noise
                if use_white_noise:
                    # Subtract white noise simulation mean
                    Nl = np.array(avg_std_noise[key][cl_key]['MEAN'])
                else:
                    # Subtract HMDM spectra (same format as regular spectra from save_spectra_to_fits)
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
                # For cross-spectra: error only from sky+noise std
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
                unit_dict[band] = cmb_unit_conversion(nuGHz, 'KCMB2KRJ') if correct_unit else 1.0
                    
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
    Dust angular power spectrum model with modified blackbody scaling in K_RJ units.

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

    # Precompute per-frequency dust scaling (K_RJ units)
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
    # Use K_RJ units for dust (Planck convention)
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


def compute_chi2_reduced(theta_full, datasets, ell, y_all, yerr_all, 
                         fit_c_terms=False, fit_components=('sync', 'dust', 'cross')):
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
        Whether to fit constant c terms.
    fit_components : tuple
        Components to include ('sync','dust','cross').

    Returns
    -------
    chi2_reduced : float
        Reduced chi-squared value (chi2 / degrees of freedom).
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
    
    # Calculate degrees of freedom: number of data points - number of free parameters
    n_data = len(y_all)
    n_params = 0
    
    # Count free parameters
    if 'sync' in fit_components:
        n_params += 3  # A_s, alpha_s, beta_s
        if fit_c_terms:
            n_params += N  # c_sync terms
    
    if 'dust' in fit_components:
        n_params += 3  # A_d, alpha_d, beta_d
        if fit_c_terms:
            n_params += N  # c_dust terms
    
    if 'cross' in fit_components:
        n_params += 1  # rho
    
    dof = n_data - n_params
    
    if dof <= 0:
        return np.inf  # Invalid case
    
    return chi2 / dof

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


def run_mcmc(
    fit_data,
    fit_components=('sync', 'dust', 'cross'),
    fit_c_terms=False,
    nwalkers=100,
    ninter=5000,
    discard_fraction=0.5,
    verbose=True,
    fit_mode='power-law',
):
    """
    Run an MCMC fit using data prepared by `prepare_mcmc_data`.

    Parameters
    ----------
    fit_data : dict
        Output from `prepare_mcmc_data`.
    fit_components : tuple
        Components to include in the fit ('sync', 'dust', 'cross').
    fit_c_terms : bool
        Whether to include constant terms for auto-spectra.
    nwalkers : int
        Number of walkers for emcee.
    ninter : int
        Total number of iterations per walker.
    discard_fraction : float
        Fraction of initial samples to discard as burn-in.
    verbose : bool
        If True, print progress information.
    fit_mode : str, default 'power-law'
        Fitting mode: 'power-law' or 'bin-to-bin'.
        - 'power-law': fit global spectral indices (alpha_s, alpha_d) across all ells.
        - 'bin-to-bin': fit amplitudes and spectral indices independently for each ell bin.

    Returns
    -------
    sampler : emcee.EnsembleSampler or list
        The sampler object(s) after running MCMC. For 'bin-to-bin', returns list of samplers.
    samples_full : ndarray or list
        Full chain including both free and fixed parameters. For 'bin-to-bin', returns list of arrays.
    samples_free : ndarray or list
        Chain containing only free parameters. For 'bin-to-bin', returns list of arrays.
    param_map : list
        List of (name, is_free) tuples describing each parameter.
    chi2_reduced : float or list
        Reduced chi-squared value(s) at the best-fit parameters.
    """

    if fit_mode == 'power-law':
        return _run_mcmc_powerlaw(
            fit_data, fit_components, fit_c_terms, nwalkers, ninter, discard_fraction, verbose
        )
    elif fit_mode == 'bin-to-bin':
        return _run_mcmc_bin_to_bin(
            fit_data, fit_components, nwalkers, ninter, discard_fraction, verbose
        )
    else:
        raise ValueError(f"fit_mode must be 'power-law' or 'bin-to-bin', got '{fit_mode}'")


def _run_mcmc_powerlaw(fit_data, fit_components, fit_c_terms, nwalkers, ninter, discard_fraction, verbose):
    """Power-law mode: fit global spectral indices across all ells."""
    datasets = fit_data['datasets']
    ell = fit_data['ell_eff']
    y_all = fit_data['y_all']
    yerr_all = fit_data['yerr_all']

    # -------------------------------
    # Parameter mapping
    # -------------------------------
    unique_freqs = sorted({f for d in datasets for f in d['freqs']})
    N = len(unique_freqs)

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

    fixed_values = {name: 0.0 for name, is_free in param_map if not is_free}
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
            p0_center.append(0.0)

    p0_center = np.array(p0_center, dtype=float)
    p0_walkers = p0_center + 1e-2 * rng.standard_normal((nwalkers, ndim))

    # -------------------------------
    # Run the sampler
    # -------------------------------
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    n_procs = max(1, min(available, max(1, nwalkers // 2)))

    if verbose:
        print(f"[run_mcmc] Using {n_procs} processes (of {available}) with {nwalkers} walkers")

    with mp.get_context("fork").Pool(processes=n_procs, maxtasksperchild=200) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, lnprob,
            args=(datasets, ell, y_all, yerr_all, fit_c_terms, fit_components, param_map, fixed_values),
            pool=pool
        )
        if verbose:
            print("[run_mcmc] Starting MCMC...")
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
        fit_c_terms=fit_c_terms, fit_components=fit_components
    )

    if verbose:
        print(f"[run_mcmc] MCMC completed. {samples_free.shape[0]} usable samples after burn-in.")
        print(f"[run_mcmc] Reduced chi-squared at best fit: {chi2_reduced:.4f}")

    return sampler, samples_full, samples_free, param_map, chi2_reduced


def _run_mcmc_bin_to_bin(fit_data, fit_components, nwalkers, ninter, discard_fraction, verbose):
    """
    Bin-to-bin mode: fit amplitudes and spectral indices independently for each ell bin.
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
    
    if verbose:
        print(f"[run_mcmc bin-to-bin] Available cores: {available_cores}")
        print(f"[run_mcmc bin-to-bin] Using {available_cores} cores for walker parallelization")
    
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
        with mp.Pool(processes=available_cores) as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, _lnprob_bin_to_bin,
                args=(datasets_bin, ell_bin, y_bin, yerr_bin, fit_components, param_names_base),
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
            best_params, datasets_bin, ell_bin, y_bin, yerr_bin, fit_components, param_names_base
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


def _lnprob_bin_to_bin(theta, datasets, ell, y_all, yerr_all, fit_components, param_names):
    """Log-probability for bin-to-bin fitting (single ell bin)."""
    # Prior
    lp = _lnprior_bin_to_bin(theta, param_names)
    if not np.isfinite(lp):
        return -np.inf
    
    # Likelihood
    ll = _lnlike_bin_to_bin(theta, datasets, ell, y_all, yerr_all, fit_components, param_names)
    if not np.isfinite(ll):
        return -np.inf
    
    return lp + ll


def _lnprior_bin_to_bin(theta, param_names):
    """Prior for bin-to-bin parameters."""
    param_dict = {name: theta[i] for i, name in enumerate(param_names)}
    
    # Check bounds
    if 'A_s' in param_dict and not (0 < param_dict['A_s'] < 1e3):
        return -np.inf
    if 'beta_s' in param_dict and not (-10 < param_dict['beta_s'] < 0):
        return -np.inf
    if 'A_d' in param_dict and not (0 < param_dict['A_d'] < 1e3):
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


def _lnlike_bin_to_bin(theta, datasets, ell, y_all, yerr_all, fit_components, param_names):
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
            freq_ref_sync = 11.1
            scale_f1 = (f1 / freq_ref_sync) ** beta_s
            scale_f2 = (f2 / freq_ref_sync) ** beta_s
            model_val += A_s * scale_f1 * scale_f2
        
        if 'dust' in fit_components:
            # Dust: A_d * mbb_scaling
            freq_ref_dust = 353.0
            T_d = 19.6
            scale_f1 = mbb_scaling_KRJ(np.array([f1]), nu0_GHz=freq_ref_dust, beta=beta_d, T_d=T_d)[0]
            scale_f2 = mbb_scaling_KRJ(np.array([f2]), nu0_GHz=freq_ref_dust, beta=beta_d, T_d=T_d)[0]
            model_val += A_d * scale_f1 * scale_f2
        
        if 'cross' in fit_components:
            # Cross term: rho * sqrt(A_s * A_d) * (sync_scale1 * dust_scale2 + sync_scale2 * dust_scale1)
            freq_ref_sync = 11.1
            freq_ref_dust = 353.0
            T_d = 19.6
            
            s1 = (f1 / freq_ref_sync) ** beta_s
            s2 = (f2 / freq_ref_sync) ** beta_s
            d1 = mbb_scaling_KRJ(np.array([f1]), nu0_GHz=freq_ref_dust, beta=beta_d, T_d=T_d)[0]
            d2 = mbb_scaling_KRJ(np.array([f2]), nu0_GHz=freq_ref_dust, beta=beta_d, T_d=T_d)[0]
            
            model_val += rho * np.sqrt(A_s * A_d) * (s1 * d2 + s2 * d1)
        
        y_model[idx] = model_val
    
    # Chi-squared
    chi2 = np.sum(((y_all - y_model) / yerr_all) ** 2)
    return -0.5 * chi2


def _compute_chi2_reduced_bin_to_bin(theta, datasets, ell, y_all, yerr_all, fit_components, param_names):
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
            freq_ref_sync = 11.1
            scale_f1 = (f1 / freq_ref_sync) ** beta_s
            scale_f2 = (f2 / freq_ref_sync) ** beta_s
            model_val += A_s * scale_f1 * scale_f2
        
        if 'dust' in fit_components:
            freq_ref_dust = 353.0
            T_d = 19.6
            scale_f1 = mbb_scaling_KRJ(np.array([f1]), nu0_GHz=freq_ref_dust, beta=beta_d, T_d=T_d)[0]
            scale_f2 = mbb_scaling_KRJ(np.array([f2]), nu0_GHz=freq_ref_dust, beta=beta_d, T_d=T_d)[0]
            model_val += A_d * scale_f1 * scale_f2
        
        if 'cross' in fit_components:
            freq_ref_sync = 11.1
            freq_ref_dust = 353.0
            T_d = 19.6
            
            s1 = (f1 / freq_ref_sync) ** beta_s
            s2 = (f2 / freq_ref_sync) ** beta_s
            d1 = mbb_scaling_KRJ(np.array([f1]), nu0_GHz=freq_ref_dust, beta=beta_d, T_d=T_d)[0]
            d2 = mbb_scaling_KRJ(np.array([f2]), nu0_GHz=freq_ref_dust, beta=beta_d, T_d=T_d)[0]
            
            model_val += rho * np.sqrt(A_s * A_d) * (s1 * d2 + s2 * d1)
        
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
    
    Parameters
    ----------
    samples_free : ndarray
        MCMC samples for the free parameters (after burn-in).
    param_map : list
        List of (parameter_name, is_free) tuples.
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
    # Prepare labels and scaling
    # -------------------------------
    labels_free = [name for name, is_free in param_map if is_free]

    scale_map = {
        'A_s': (1e6, r'$\mu\mathrm{K}^2$'),
        'A_d': (1e9, r'$10^{-3}\,\mu\mathrm{K}^2$'),
    }

    for name in labels_free:
        if name.startswith('c_sync') or name.startswith('c_dust'):
            scale_map[name] = (1e9, r'$10^{-3}\,\mu\mathrm{K}^2$')

    samples_plot = apply_corner_scales(samples_free, labels_free, scale_map)

    # -------------------------------
    # LaTeX labels
    # -------------------------------
    latex_labels = {
        'A_s': r'$A_{\mathrm{s}}\,[\mu\mathrm{K}^2]$',
        'alpha_s': r'$\alpha_{\mathrm{s}}$',
        'beta_s': r'$\beta_{\mathrm{s}}$',
        'A_d': r'$A_{\mathrm{d}}\,[10^{-3}\,\mu\mathrm{K}^2]$',
        'alpha_d': r'$\alpha_{\mathrm{d}}$',
        'beta_d': r'$\beta_{\mathrm{d}}$',
        'rho': r'$\rho$',
    }

    for name in labels_free:
        if name.startswith('c_sync'):
            freq = name.split('[')[-1].strip(']')
            latex_labels[name] = rf'$c_{{\mathrm{{sync}},\,{freq}}}\,[10^{-3}\,\mu\mathrm{{K}}^2]$'
        elif name.startswith('c_dust'):
            freq = name.split('[')[-1].strip(']')
            latex_labels[name] = rf'$c_{{\mathrm{{dust}},\,{freq}}}\,[10^{-3}\,\mu\mathrm{{K}}^2]$'

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
    format='latex'
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
            values_EE = samples_EE[:, j]
            median_EE = np.median(values_EE)
            std_EE = np.std(values_EE)
            medians_EE[param_name] = median_EE

            values_BB = samples_BB[:, j]
            median_BB = np.median(values_BB)
            std_BB = np.std(values_BB)
            medians_BB[param_name] = median_BB

            if param_name == 'A_s':
                # Convert from mK^2 to μK^2 (multiply by 1e6)
                median_EE_conv = median_EE * 1e6
                std_EE_conv = std_EE * 1e6
                row_EE[param_name] = f"{median_EE_conv:.2f}"
                row_EE[f'{param_name}_err'] = f"{std_EE_conv:.2f}"
            elif param_name == 'A_d':
                # Convert from mK^2 to 10^-3 μK^2 (multiply by 1e9)
                median_EE_conv = median_EE * 1e9
                std_EE_conv = std_EE * 1e9
                row_EE[param_name] = f"{median_EE_conv:.2f}"
                row_EE[f'{param_name}_err'] = f"{std_EE_conv:.2f}"
            elif param_name == 'rho':
                # Correlation coefficient
                row_EE[param_name] = f"{median_EE:.3f}"
                row_EE[f'{param_name}_err'] = f"{std_EE:.3f}"
            else:
                # Spectral indices
                row_EE[param_name] = f"{median_EE:.2f}"
                row_EE[f'{param_name}_err'] = f"{std_EE:.2f}"

            if param_name == 'A_s' or param_name == 'A_d':
                # Apply same conversion as for EE mode
                if param_name == 'A_s':
                    median_BB_conv = median_BB * 1e6
                    std_BB_conv = std_BB * 1e6
                else:  # A_d
                    median_BB_conv = median_BB * 1e9
                    std_BB_conv = std_BB * 1e9
                row_BB[param_name] = f"{median_BB_conv:.2f}"
                row_BB[f'{param_name}_err'] = f"{std_BB_conv:.2f}"
            elif 'A_' in param_name:
                # Fallback for any other amplitude-like parameter: compute BB/EE ratio (dimensionless)
                if median_EE != 0:
                    ratio = median_BB / median_EE
                    rel_err_EE = std_EE / median_EE if median_EE != 0 else 0
                    rel_err_BB = std_BB / median_BB if median_BB != 0 else 0
                    ratio_err = ratio * np.sqrt(rel_err_EE**2 + rel_err_BB**2)
                else:
                    ratio = 0
                    ratio_err = 0
                row_BB[param_name] = f"{ratio:.3f}"
                row_BB[f'{param_name}_err'] = f"{ratio_err:.3f}"
            elif param_name == 'rho':
                row_BB[param_name] = f"{median_BB:.3f}"
                row_BB[f'{param_name}_err'] = f"{std_BB:.3f}"
            else:
                # Spectral indices
                row_BB[param_name] = f"{median_BB:.2f}"
                row_BB[f'{param_name}_err'] = f"{std_BB:.2f}"
        
        rows_EE.append(row_EE)
        rows_BB.append(row_BB)
    
    if format == 'latex':
        # Create LaTeX table with EE results on top, BB (absolute amplitudes) below
        table_lines = []
        table_lines.append(r"\begin{table*}[htbp]")
        table_lines.append(r"\centering")
        table_lines.append(r"\caption{Bin-to-bin fit results. EE mode (top) and BB mode (bottom).}")
        table_lines.append(r"\label{tab:bin_to_bin_results}")
        
        # Build column specification: 2 fixed cols + 1 per parameter (value ± error in same column)
        n_params = len(param_names)
        col_spec = "c c " + " ".join(["c"] * n_params)
        table_lines.append(r"\begin{tabular}{" + col_spec + "}")
        table_lines.append(r"\hline\hline")
        
        # Header row for EE mode
        header_cols = [r"$\ell$ range", r"$\ell_{\rm eff}$"]
        for param_name in param_names:
            # Convert parameter names to LaTeX with EE superscript and updated units
            if param_name == 'A_s':
                latex_name = r"$A^{\rm EE}_{\rm sync}$ [$\mu$K$^2$]"
            elif param_name == 'beta_s':
                latex_name = r"$\beta^{\rm EE}_{\rm sync}$"
            elif param_name == 'A_d':
                latex_name = r"$A^{\rm EE}_{\rm dust}$ [$10^{-3}$ $\mu$K$^2$]"
            elif param_name == 'beta_d':
                latex_name = r"$\beta^{\rm EE}_{\rm dust}$"
            elif param_name == 'rho':
                latex_name = r"$\rho^{\rm EE}$"
            else:
                latex_name = param_name + r"$^{\rm EE}$"
            
            header_cols.append(latex_name)
        
        table_lines.append(" & ".join(header_cols) + r" \\")
        table_lines.append(r"\hline")
        
        # EE mode data rows
        for i in range(n_bins):
            row_vals = [rows_EE[i]['ell_range'], rows_EE[i]['ell_eff']]
            for param_name in param_names:
                # Combine value and error in same column
                val_err = f"{rows_EE[i][param_name]} $\\pm$ {rows_EE[i][f'{param_name}_err']}"
                row_vals.append(val_err)
            table_lines.append(" & ".join(row_vals) + r" \\")
        
        table_lines.append(r"\hline")
        
        # Header row for BB mode (with ratio notation for amplitudes)
        header_cols_BB = [r"$\ell$ range", r"$\ell_{\rm eff}$"]
        for param_name in param_names:
            if param_name == 'A_s':
                latex_name = r"$A^{\rm BB}_{\rm sync}$ [$\mu$K$^2$]"
            elif param_name == 'beta_s':
                latex_name = r"$\beta^{\rm BB}_{\rm sync}$"
            elif param_name == 'A_d':
                latex_name = r"$A^{\rm BB}_{\rm dust}$ [$10^{-3}$ $\mu$K$^2$]"
            elif param_name == 'beta_d':
                latex_name = r"$\beta^{\rm BB}_{\rm dust}$"
            elif param_name == 'rho':
                latex_name = r"$\rho^{\rm BB}$"
            else:
                latex_name = param_name + r"$^{\rm BB}$"
            
            header_cols_BB.append(latex_name)
        
        table_lines.append(" & ".join(header_cols_BB) + r" \\")
        table_lines.append(r"\hline")
        
        # BB mode data rows
        for i in range(n_bins):
            row_vals = [rows_BB[i]['ell_range'], rows_BB[i]['ell_eff']]
            for param_name in param_names:
                # Combine value and error in same column
                val_err = f"{rows_BB[i][param_name]} $\\pm$ {rows_BB[i][f'{param_name}_err']}"
                row_vals.append(val_err)
            table_lines.append(" & ".join(row_vals) + r" \\")
        
        table_lines.append(r"\hline\hline")
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
            # Add EE suffix to parameter names
            label = f"{param_name}_EE"
            header += f" {label:^20} |"
        table_lines.append(header)
        table_lines.append("-" * 150)
        
        # EE mode data rows
        for i in range(n_bins):
            row_str = f"{rows_EE[i]['ell_range']:^12} | {rows_EE[i]['ell_eff']:^8} |"
            for param_name in param_names:
                val = rows_EE[i][param_name]
                err = rows_EE[i][f'{param_name}_err']
                row_str += f" {val:>9} ± {err:<9} |"
            table_lines.append(row_str)
        
        table_lines.append("=" * 150)
        
        # Header for BB (with ratio notation)
        header_BB = f"{'ell range':^12} | {'ell_eff':^8} |"
        for param_name in param_names:
            if param_name in ('A_s', 'A_d'):
                label = f"{param_name}_BB"
            elif 'A_' in param_name:
                label = f"{param_name}_BB/EE"
            else:
                label = f"{param_name}_BB"
            header_BB += f" {label:^20} |"
        table_lines.append(header_BB)
        table_lines.append("-" * 150)
        
        # BB mode data rows
        for i in range(n_bins):
            row_str = f"{rows_BB[i]['ell_range']:^12} | {rows_BB[i]['ell_eff']:^8} |"
            for param_name in param_names:
                val = rows_BB[i][param_name]
                err = rows_BB[i][f'{param_name}_err']
                row_str += f" {val:>9} ± {err:<9} |"
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
                
                # Get raw spectrum data and apply mask
                spectrum_raw = corr_spec[mode]['SPECTRUM'][ell_mask]
                error_raw = corr_spec[mode]['ERROR'][ell_mask]
                theo_spectrum_raw = theo_spec[mode][ell_mask]
                
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
