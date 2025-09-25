#%%
import os
import healpy as hp
import numpy as np
from data import data 

base_dir = data['WMAP']['23']['hmdm']
save_path = data['WMAP']['23']['hmdm']

# Differential Assemblies (DAs) per frequency band
BANDS = {
    "K": ["K1"],
    "Ka": ["Ka1"],
    "Q": ["Q1", "Q2"],
    "V": ["V1", "V2"],
    "W": ["W1", "W2", "W3", "W4"],
}

def coadd_da(files):
    """
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
    """
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
    """
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
    """
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
    """
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
    """
    # If bands is 'all', process all available bands
    if bands == 'all':
        bands = list(BANDS.keys())

    # Step 1: Generate band maps per year
    band_year_maps = {band: {} for band in bands}
    for band in bands:
        for year in range(1, 10):
            das = BANDS[band]
            files = [os.path.join(base_dir, f"wmap_iqumap_r9_yr{year}_{da}_v5.fits") for da in das]
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
            filename = f"wmap_iqumap_r9_{year_1}to{year_2}_{band}_v5.fits"
            full_path = os.path.join(save_path, filename)
            hp.write_map(full_path, combined_map, overwrite=True)
            print(f"Saved {full_path}")

    return combined_maps


combined_1to4 = coadd_year_range(base_dir=base_dir, year_1=1, year_2=4, save=True, save_path=save_path)
combined_5to9 = coadd_year_range(base_dir=base_dir, year_1=5, year_2=9, save=True, save_path=save_path)
