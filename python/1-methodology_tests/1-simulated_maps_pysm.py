#%%
import os
import sys
import numpy as np
import healpy as hp
from astropy import units as u
from astropy.io import fits

import pysm3
import pysm3.units as u_pysm

from data import data

# Default configuration
nside = 512
path_save = '/home/pablo/Desktop/Paper/maps/PYSM/'


# ---------------------------------------------------------------------
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
    almT, almE, almB = hp.map2alm(iqu64, lmax=lmax, pol=True, iter=0)
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
    - Simulates synchrotron ('s1') and dust ('d1') components.
    - Saves FITS files with metadata in headers.
    """
    os.makedirs(path_save, exist_ok=True)
    sky = pysm3.Sky(nside=nside, preset_strings=['s1', 'd1'])  # synchrotron + dust

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
                nu = data[exp][band]['freq']  # e.g., 28.4 * u.GHz
                nu_GHz = nu.to_value(u.GHz)

                iqu = sky.get_emission(nu).to(
                    u_pysm.uK_CMB, equivalencies=u_pysm.cmb_equivalencies(nu)
                ).value

                iqu_conv = _convolve_IQU_with_beam(iqu, exp, band, data, lmax=3 * nside - 1)

                out_dir = os.path.join(path_save, exp, str(band))
                os.makedirs(out_dir, exist_ok=True)
                out_f = os.path.join(out_dir, f"map_{exp}_{band}_nside{nside}_beamconv.fits")

                header = [
                    ("FREQ_GHZ", float(nu_GHz), "Band center frequency [GHz]"),
                    ("NSIDE", int(nside), "HEALPix NSIDE"),
                    ("UNIT", "uK_CMB", "Units"),
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


# ---------------------------------------------------------------------
quijote_bands = ['11']
wmap_bands = ['23', '33', '41', '61', '94']
planck_bands = ['30', '44', '70', '100', '143', '217', '353']

# Generate WMAP maps
generate_sky_maps(nside, path_save, experiment_select='WMAP', band_select=wmap_bands)