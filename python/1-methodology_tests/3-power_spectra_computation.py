#%%
import numpy as np
import healpy as hp
import os
from tqdm import tqdm
import pymaster as nmt
from data import data, path_map, masks, path_masks
from astropy.io import fits
import re

nside = 512
lmax = 2 * nside - 1
dl = 10
n_sim = 100 

# --- Choose experiment or band to run ---
experiment_select = None  # 'QUIJOTE', 'WMAP' or 'Planck'
band_select = None
mask_select = masks['quijote_galcut']['galcut10']
mask_name = mask_select['name']
use_simulated_maps = True
use_white_noise = True
out_path = '/home/pablo/Desktop/master/tfm/spectra/'
path_spectra = os.path.join(out_path, f'power_spectra_{mask_name}.fits')
avg_std_skyplusnoise_name = os.path.join(out_path, f'spectra_avg_std_{mask_name}_avg_std10_skyplusnoise.fits')
avg_std_noise_name = os.path.join(out_path, f'spectra_avg_std_{mask_name}_avg_std10_noise.fits')

# Bands to use
quijote_band_select = ['11']
wmap_band_select = ['23_1', '33_1', '41_1', '61_1', '94_1']
planck_band_select = 'all'

mask = hp.read_map(mask_select['path'])

#Create binning scheme
ell_1 = [30, 50, 70, 90, 110, 130]
ell_2 = [49, 69, 89, 109, 129, 149]

binning_params = {
    'type': 'linear',  # or 'edges'
    'lmax': lmax,
    'dl': dl,
    # For edges
    'ell1': ell_1,
    'ell2': ell_2
}

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

b = create_binning(binning_params)

# ======================================================
# Build list of bands to compute spectra
# ======================================================


def build_band_list(
    data, 
    quijote_band_select="all", 
    wmap_band_select="all", 
    planck_band_select="all"
):
    """
    Build a list of (experiment, band) tuples based on selection options.

    Parameters
    ----------
    data : dict
        Dictionary containing experiment and band info.
    quijote_band_select : "all", str, list of str, or None
        - "all": use all QUIJOTE bands
        - str: use only that band
        - list of str: use only those bands
        - None: use none
    wmap_band_select : "all", str, list of str, or None
        Same as above but for WMAP
    planck_band_select : "all", str, list of str, or None
        Same as above but for Planck

    Returns
    -------
    band_list : list of tuples
        List of (experiment, band) tuples in order.
    """
    band_list = []

    # Normaliza selects
    def normalize(select, keys, sort_key):
        if select == "all":
            return sorted(keys, key=sort_key)
        elif isinstance(select, str):
            return [select]
        elif isinstance(select, list):
            return select
        elif select is None:
            return []
        else:
            raise ValueError(f"Invalid band_select value: {select}")

    # QUIJOTE
    for band in normalize(quijote_band_select, data["QUIJOTE"].keys(), lambda x: float(x)):
        band_list.append((band))

    # WMAP
    for band in normalize(wmap_band_select, data["WMAP"].keys(), lambda x: float(x.split('_')[0])):
        band_list.append((band))

    # Planck
    for band in normalize(planck_band_select, data["Planck"].keys(), lambda x: float(x)):
        band_list.append((band))

    return band_list


# ======================================================
# Compute auto- and cross-spectra functions
# ======================================================

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



def cross_spectrum(mask, map_1, map_2, b, purify_e=False, purify_b=False, beam=None):
    """Compute the cross power spectrum between two maps.

    Parameters
    ----------
    mask : array
        Healpy map mask (0 or 1) for the observed region.
    map_1 : array
        First input map of shape (3, npix) with I, Q, U components.
    map_2 : array
        Second input map of shape (3, npix) with I, Q, U components.
    purify_e : bool, optional
        Whether to purify E modes in spin-2 field (default False).
    purify_b : bool, optional
        Whether to purify B modes in spin-2 field (default False).
    beam : array, optional
        Beam window function to apply (default None).

    Returns
    -------
    power_spectrum_modes : dict
        Dictionary with first two bin edges ell1, ell2, effective ell and spectra:
        ['ell1', 'ell2', 'ell_eff', 'TT', 'EE', 'BB', 'TE', 'TB', 'EB']
    """
    
    ell_min_list = [b.get_ell_min(i) for i in range(b.get_n_bands())]
    ell_max_list = [b.get_ell_max(i) for i in range(b.get_n_bands())]
    ell_eff = b.get_effective_ells()

    # Spin-0 and spin-2 fields
    f0_1 = nmt.NmtField(mask, [map_1[0,:]], beam=beam)
    f2_1 = nmt.NmtField(mask, [map_1[1,:], map_1[2,:]],
                        purify_e=purify_e, purify_b=purify_b, beam=beam)

    f0_2 = nmt.NmtField(mask, [map_2[0,:]], beam=beam)
    f2_2 = nmt.NmtField(mask, [map_2[1,:], map_2[2,:]],
                        purify_e=purify_e, purify_b=purify_b, beam=beam)

    # Workspaces
    w00 = nmt.NmtWorkspace(); w00.compute_coupling_matrix(f0_1, f0_2, b)
    w02 = nmt.NmtWorkspace(); w02.compute_coupling_matrix(f0_1, f2_2, b)
    w22 = nmt.NmtWorkspace(); w22.compute_coupling_matrix(f2_1, f2_2, b)

    # Compute spectra
    cl_tt = compute_master(f0_1, f0_2, w00)[0]
    cl_te, cl_tb = compute_master(f0_1, f2_2, w02)
    cl_ee, cl_eb, _, cl_bb = compute_master(f2_1, f2_2, w22)

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

# ======================================================
# Function to compute over all selected bands
# ======================================================

def compute_all_power_spectra(data, band_list, mask, b, 
                              use_simulated_maps, use_white_noise=False, 
                              noise_realization=1, only_noise=False):
    """
    Compute auto and cross spectra for a list of bands.

    Parameters
    ----------
    data : dict
        Dictionary with experiment and band info.
    band_list : list of str
        Ordered list of bands to use.
    mask : array
        Healpy mask to apply (0=masked, 1=unmasked).
    b : nmt.NmtBin
        Binning scheme object from NaMaster.
    use_simulated_maps : bool
        If True, load the simulated maps + noise.
        If False, load the real maps from 'path'.
    use_white_noise : bool, optional
        If True, use 'path_white_noise_simulations'.
        If False (default), use 'path_noise_simulations'.
    noise_realization : int, optional
        Noise realization number to use (default 1).
    only_noise : bool, optional
        If True, ignore the simulated sky and use only noise maps (default False).

    Returns
    -------
    spectra_matrix : ndarray
        Matrix of shape (N_band, N_band) with dictionaries containing the auto
        and cross spectra for each band pair.
    """
    N_band = len(band_list)
    spectra_matrix = np.empty((N_band, N_band), dtype=object)

    for i, band_i in enumerate(band_list):
        exp_i = next(exp for exp, bands in data.items() if band_i in bands)

        # --- load sky map ---
        if use_simulated_maps:
            sky_map_i = 0
            if not only_noise:
                sky_i_path = data[exp_i][band_i]['path_simulated']
                sky_map_i = hp.read_map(sky_i_path, field=[0,1,2])
                sky_map_i = np.where(sky_map_i == hp.UNSEEN, 0, sky_map_i)
                
                # --- Planck data in mK_CMB ---
                if exp_i == 'Planck':
                    sky_map_i *= 1e3

            # --- choose noise path ---
            if use_white_noise:
                noise_dir = data[exp_i][band_i]['path_white_noise_simulations']
                base_name = data[exp_i][band_i]['white_noise_simulation_1']
            else:
                noise_dir = data[exp_i][band_i]['path_noise_simulations']
                base_name = data[exp_i][band_i]['noise_simulation_1']

            noise_fname = get_noise_filename(base_name, noise_realization)
            noise_path = os.path.join(noise_dir, noise_fname)
            noise_map = hp.read_map(noise_path, field=[0,1,2])
            noise_map = np.where(noise_map == hp.UNSEEN, 0, noise_map)

            # --- Planck data in mK_CMB ---
            if exp_i == 'Planck':
                noise_map *= 1e3

            if isinstance(sky_map_i, int):  # if only_noise=True
                sky_map_i = np.zeros_like(noise_map)
            sky_map_i = sky_map_i + noise_map

        else:
            sky_i_path = data[exp_i][band_i]['path']
            sky_map_i = hp.read_map(sky_i_path, field=[0,1,2])
            sky_map_i = np.where(sky_map_i == hp.UNSEEN, 0, sky_map_i)

        for j, band_j in enumerate(band_list):
            if j < i:
                spectra_matrix[i, j] = spectra_matrix[j, i]
                continue

            exp_j = next(exp for exp, bands in data.items() if band_j in bands)

            if use_simulated_maps:
                sky_map_j = 0
                if not only_noise:
                    sky_j_path = data[exp_j][band_j]['path_simulated']
                    sky_map_j = hp.read_map(sky_j_path, field=[0,1,2])
                    sky_map_j = np.where(sky_map_j == hp.UNSEEN, 0, sky_map_j)
                    
                    # --- Planck data in mK_CMB ---
                    if exp_j == 'Planck':
                        sky_map_j *= 1e3

                if use_white_noise:
                    noise_dir = data[exp_j][band_j]['path_white_noise_simulations']
                    base_name = data[exp_j][band_j]['white_noise_simulation_1']
                else:
                    noise_dir = data[exp_j][band_j]['path_noise_simulations']
                    base_name = data[exp_j][band_j]['noise_simulation_1']

                noise_fname = get_noise_filename(base_name, noise_realization)
                noise_path = os.path.join(noise_dir, noise_fname)
                noise_map = hp.read_map(noise_path, field=[0,1,2])
                noise_map = np.where(noise_map == hp.UNSEEN, 0, noise_map)

                # --- Planck data in mK_CMB ---
                if exp_j == 'Planck':
                    noise_map *= 1e3

                if isinstance(sky_map_j, int):  # if only_noise=True
                    sky_map_j = np.zeros_like(noise_map)
                sky_map_j = sky_map_j + noise_map
            else:
                sky_j_path = data[exp_j][band_j]['path']
                sky_map_j = hp.read_map(sky_j_path, field=[0,1,2])
                sky_map_j = np.where(sky_map_j == hp.UNSEEN, 0, sky_map_j)

            cl = cross_spectrum(mask, sky_map_i, sky_map_j, b=b, purify_e=True, purify_b=True)
            spectra_matrix[i, j] = cl

    return spectra_matrix


def save_spectra_to_fits(spectra_matrix, band_list, mask_name, out_path):
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
    mask_name : str
        Name of the mask, used to label the output FITS file.
    out_path : str
        Directory path where the FITS file will be saved.

    Returns
    -------
    None
        The function writes the spectra into a FITS file at the specified path.
    """
    N_band = len(band_list)
    hdu_list = fits.HDUList()
    hdu_list.append(fits.PrimaryHDU())  

    for i in range(N_band):
        for j in range(N_band):
            cl_dict = spectra_matrix[i,j]
            cols = [fits.Column(name=key, format='D', array=cl_dict[key]) 
                    for key in ['ell1','ell2','ell_eff','TT','EE','BB','TE','TB','EB']]
            hdu = fits.BinTableHDU.from_columns(cols)
            hdu.header['BAND_I'] = band_list[i]
            hdu.header['BAND_J'] = band_list[j]
            hdu.name = f'{band_list[i]}_{band_list[j]}'
            hdu_list.append(hdu)

    out_file = os.path.join(out_path, f'power_spectra_{mask_name}.fits')
    hdu_list.writeto(out_file, overwrite=True)
    print(f"Saved power spectra matrix for mask \"{mask_name}\" to {out_file}")


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
        raise ValueError(f"No se encontró un número adecuado en {base_name}")
    
    match = matches[-1]  # Last numbers
    num_str = match.group(0)
    width = len(num_str)

    # Reference number
    base_num = int(num_str)
    new_num = base_num + (sim_number - 1)
    
    new_num_str = str(new_num).zfill(width)
    return base_name[:match.start()] + new_num_str + base_name[match.end():]


def average_and_std_spectra(data, spectra_matrix, band_list, mask, b,
                            use_white_noise=False, n_sim=100, only_noise=False):
    """
    Compute average and standard deviation of auto- and cross-spectra
    over many noise realizations.

    Parameters
    ----------
    data : dict
        Experiment and band info.
    spectra_matrix : ndarray
        Initial spectra_matrix already computed for noise_realization=1.
    band_list : list of str
        List of bands to compute.
    mask : array
        Healpy mask.
    b : nmt.NmtBin
        NaMaster binning scheme.
    use_white_noise : bool, optional
        If True, use white noise simulations. Default False (full noise sims).
    n_sim : int, optional
        Number of noise realizations. Default 100.
    only_noise : bool, optional
        If True, compute spectra using only noise maps.

    Returns
    -------
    avg_matrix : ndarray
        Matrix (N_band, N_band) with dictionaries of averaged spectra.
    std_matrix : ndarray
        Matrix (N_band, N_band) with dictionaries of std deviations.
    """
    N_band = len(band_list)

    # Start acumulators
    sum_matrix = np.empty((N_band, N_band), dtype=object)
    sumsq_matrix = np.empty((N_band, N_band), dtype=object)
    for i in range(N_band):
        for j in range(N_band):
            sum_matrix[i, j] = {}
            sumsq_matrix[i, j] = {}
            for key, val in spectra_matrix[i, j].items():
                sum_matrix[i, j][key] = np.zeros_like(val, dtype=float)
                sumsq_matrix[i, j][key] = np.zeros_like(val, dtype=float)

    # Loop over simulations
    for sim in tqdm(range(1, n_sim + 1), desc="Simulations"):
        spectra_matrix_sim = compute_all_power_spectra(
            data, band_list, mask, b,
            use_simulated_maps=True,
            use_white_noise=use_white_noise,
            noise_realization=sim,
            only_noise=only_noise
        )
        for i in range(N_band):
            for j in range(N_band):
                for key in spectra_matrix_sim[i, j]:
                    arr = np.array(spectra_matrix_sim[i, j][key], dtype=float)
                    sum_matrix[i, j][key] += arr
                    sumsq_matrix[i, j][key] += arr**2

    # Compute mean and std
    avg_matrix = np.empty((N_band, N_band), dtype=object)
    std_matrix = np.empty((N_band, N_band), dtype=object)
    for i in range(N_band):
        for j in range(N_band):
            avg_matrix[i, j] = {}
            std_matrix[i, j] = {}
            for key in sum_matrix[i, j]:
                mean = sum_matrix[i, j][key] / n_sim
                var = (sumsq_matrix[i, j][key] / n_sim) - mean**2
                var = np.where(var < 0, 0, var)
                std = np.sqrt(var)
                avg_matrix[i, j][key] = mean
                std_matrix[i, j][key] = std

    return avg_matrix, std_matrix



def save_avg_std_to_fits(avg_matrix, std_matrix, band_list, file_name, out_path):
    """
    Save average and std spectra into a FITS file.

    Parameters
    ----------
    avg_matrix : ndarray
        Matrix (N_band, N_band) with dictionaries of mean spectra.
    std_matrix : ndarray
        Matrix (N_band, N_band) with dictionaries of std deviations.
    band_list : list of str
        Ordered list of bands corresponding to the rows/cols of matrices.
    file_name : str
        Name of the mask, used for labeling the output file.
    out_path : str
        Directory where FITS will be saved.
    n_sim : int
        Number of simulations averaged.
    """
    N_band = len(band_list)
    hdu_list = fits.HDUList()
    hdu_list.append(fits.PrimaryHDU())  

    for i in range(N_band):
        for j in range(N_band):
            avg_dict = avg_matrix[i, j]
            std_dict = std_matrix[i, j]

            cols = []
            for key in ['ell1','ell2','ell_eff','TT','EE','BB','TE','TB','EB']:
                cols.append(fits.Column(name=f'{key}_MEAN', format='D', array=avg_dict[key]))
                cols.append(fits.Column(name=f'{key}_STD', format='D', array=std_dict[key]))

            hdu = fits.BinTableHDU.from_columns(cols)
            hdu.header['BAND_I'] = band_list[i]
            hdu.header['BAND_J'] = band_list[j]
            hdu.name = f'{band_list[i]}_{band_list[j]}'
            hdu_list.append(hdu)

    out_file = file_name
    hdu_list.writeto(out_file, overwrite=True)
    print(f"Saved avg+std spectra for mask \"{file_name}\" to {out_file}")


def read_spectra_from_fits(path_spectra, band_list):
    """
    Read a matrix of power spectra from a FITS file and return it as a NumPy 
    ndarray of dictionaries in the same format as produced by compute_all_power_spectra.

    Parameters
    ----------
    path_spectra : str
        Path to the FITS file containing the spectra.
    band_list : list of str
        Ordered list of bands corresponding to the spectra matrix.

    Returns
    -------
    spectra_matrix : ndarray
        Square matrix (N_band x N_band) where each element is a dictionary 
        with keys corresponding to the spectrum columns ('ell1','ell2','ell_eff',
        'TT','EE','BB','TE','TB','EB').
    """
    N_band = len(band_list)
    spectra_matrix = np.empty((N_band, N_band), dtype=object)

    with fits.open(path_spectra) as hdul:
        # Loop over each band pair
        for i, band_i in enumerate(band_list):
            for j, band_j in enumerate(band_list):
                # Build the HDU name for the current band pair
                hdu_name = f'{band_i}_{band_j}'
                hdu = None

                # Search for the HDU with the matching name
                for h in hdul[1:]:
                    if h.name == hdu_name:
                        hdu = h
                        break

                if hdu is None:
                    raise ValueError(f"HDU {hdu_name} not found in FITS file")

                # Read the columns from the HDU into a dictionary
                cl_dict = {name: hdu.data[name] for name in hdu.data.names}
                spectra_matrix[i, j] = cl_dict

    return spectra_matrix


def read_spectra_or_avg_std_fits(path_spectra, band_list):
    """
    Read a matrix of spectra from a FITS file. Supports both:
    - simple spectra files (keys: 'ell1','ell2','ell_eff','TT',...)
    - averaged spectra files with mean/std (keys like 'TT_MEAN', 'TT_STD', ...)

    Parameters
    ----------
    path_spectra : str
        Path to the FITS file containing the spectra.
    band_list : list of str
        Ordered list of bands corresponding to the spectra matrix.

    Returns
    -------
    spectra_matrix : ndarray
        N_band x N_band matrix of dictionaries.
        For simple spectra files: dict with keys ['ell1','ell2','ell_eff','TT',...].
        For avg/std files: dict with keys like 'ell1', 'ell2', 'ell_eff', 
        and each spectrum key contains another dict with 'MEAN' and 'STD'.
    """
    import numpy as np
    from astropy.io import fits

    N_band = len(band_list)
    spectra_matrix = np.empty((N_band, N_band), dtype=object)

    with fits.open(path_spectra) as hdul:
        for i, band_i in enumerate(band_list):
            for j, band_j in enumerate(band_list):
                # Buscar el HDU correspondiente
                hdu_name = f'{band_i}_{band_j}'
                hdu = next((h for h in hdul[1:] if h.name == hdu_name), None)
                if hdu is None:
                    raise ValueError(f"HDU {hdu_name} not found in FITS file")

                # Detectar si el archivo contiene _MEAN/_STD
                columns = hdu.data.names
                if any(col.endswith('_MEAN') for col in columns):
                    cl_dict = {}
                    for key in ['ell1','ell2','ell_eff','TT','EE','BB','TE','TB','EB']:
                        if key in ['ell1','ell2','ell_eff']:
                            cl_dict[key] = hdu.data[key]
                        else:
                            cl_dict[key] = {
                                'MEAN': hdu.data[f'{key}_MEAN'],
                                'STD' : hdu.data[f'{key}_STD']
                            }
                else:
                    # Formato simple
                    cl_dict = {name: hdu.data[name] for name in columns}

                spectra_matrix[i, j] = cl_dict

    return spectra_matrix


def read_avg_std_spectra_from_fits(path_fits, band_list):
    """
    Read a FITS file containing average and std spectra 
    (saved with save_avg_std_to_fits) into a spectra_matrix.

    Parameters
    ----------
    path_fits : str
        Path to the FITS file with avg+std spectra.
    band_list : list of str
        Ordered list of bands corresponding to the spectra.

    Returns
    -------
    spectra_matrix : ndarray
        Matrix (N_band, N_band) with dicts of the form:
        {
          'ell1': array,
          'ell2': array,
          'ell_eff': array,
          'TT': {'MEAN': array, 'STD': array},
          'EE': {...}, 'BB': {...},
          'TE': {...}, 'TB': {...}, 'EB': {...}
        }
    """
    N_band = len(band_list)
    spectra_matrix = np.empty((N_band, N_band), dtype=object)

    with fits.open(path_fits) as hdul:
        for i, band_i in enumerate(band_list):
            for j, band_j in enumerate(band_list):
                hdu_name = f"{band_i}_{band_j}"
                hdu = next((h for h in hdul[1:] if h.name == hdu_name), None)
                if hdu is None:
                    raise ValueError(f"HDU {hdu_name} not found in {path_fits}")

                cl_dict = {}
                for key in ['ell1','ell2','ell_eff','TT','EE','BB','TE','TB','EB']:
                    if key in ['ell1','ell2','ell_eff']:
                        cl_dict[key] = hdu.data[f"{key}_MEAN"]  # idénticas en MEAN y STD
                    else:
                        cl_dict[key] = {
                            'MEAN': hdu.data[f"{key}_MEAN"],
                            'STD' : hdu.data[f"{key}_STD"]
                        }

                spectra_matrix[i, j] = cl_dict

    return spectra_matrix

# 1. Build band list
band_list = build_band_list(data, quijote_band_select=quijote_band_select, 
                            wmap_band_select=wmap_band_select, 
                            planck_band_select=planck_band_select)

# 2. Compute spectra
# spectra_matrix = compute_all_power_spectra(
#     data, band_list, mask, b, use_simulated_maps=use_simulated_maps, 
#     use_white_noise=use_white_noise, noise_realization=1
# )

# 3. Save spectra matrix into a FITS
# save_spectra_to_fits(spectra_matrix, band_list, mask_name=mask_name, out_path=out_path)

# 4. Read spectra matrix (if already computed)
spectra_matrix_read = read_spectra_from_fits(path_spectra, band_list)


# 5. Compute mean and std from 100 noise simulations
avg_matrix, std_matrix = average_and_std_spectra(
    data, spectra_matrix_read, band_list, mask, b,
    use_white_noise=True,
    n_sim=10,
    only_noise=False
)

save_avg_std_to_fits(avg_matrix, std_matrix, band_list,
                     file_name=avg_std_skyplusnoise_name,
                     out_path=out_path)


avg_matrix_noise, std_matrix_noise = average_and_std_spectra(
    data, spectra_matrix_read, band_list, mask, b,
    use_white_noise=True,
    n_sim=10,
    only_noise=True
)

save_avg_std_to_fits(avg_matrix_noise, std_matrix_noise, band_list,
                     file_name=avg_std_noise_name,
                     out_path=out_path)


# %%
