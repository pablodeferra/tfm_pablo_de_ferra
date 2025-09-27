#%%
import os
import healpy as hp
import numpy as np
from data import data 


'''
# ==================================================
# Build half maps for WMAP
# ==================================================
'''

base_dir = os.path.dirname(data['WMAP']['23']['hmdm'])
save_path = os.path.dirname(data['WMAP']['23']['hmdm'])


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


combined_1to4 = coadd_year_range(base_dir=base_dir, year_1=1, year_2=4, save=True, save_path=save_path)
combined_5to9 = coadd_year_range(base_dir=base_dir, year_1=5, year_2=9, save=True, save_path=save_path)

'''
# ==================================================
# Build HMDM for each experiment and frequency band
# ==================================================
'''


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

                sigma1 = np.sqrt(hp.read_map(path_half1, field=[4, 7, 9]))
                sigma2 = np.sqrt(hp.read_map(path_half2, field=[4, 7, 9]))

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


# Bands to use
quijote_bands = ['11']
wmap_bands = ['23', '33', '41', '61', '94']
planck_bands = ['30', '44', '70', '100', '143', '217', '353']

bands = quijote_bands + wmap_bands + planck_bands

make_hmdm(data, bands, save=True)
