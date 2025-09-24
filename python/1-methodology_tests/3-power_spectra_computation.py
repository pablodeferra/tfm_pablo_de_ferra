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
quijote_bands = ['11']
wmap_bands = ['23', '33', '41', '61', '94']
planck_bands = ['30', '44', '70', '100', '143', '217', '353']

band_list = quijote_bands + wmap_bands + planck_bands

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

def prepare_workspaces(mask, b, nside):
    """
    Precompute NaMaster workspaces for spin-0 and spin-2 fields.

    Parameters
    ----------
    mask : array
        Healpy mask (0=masked, 1=unmasked)
    b : nmt.NmtBin
        Binning scheme object
    nside : int
        Healpix nside

    Returns
    -------
    dict
        Dictionary with precomputed workspaces: {'w00': ..., 'w02': ..., 'w22': ...}
    """
    npix = hp.nside2npix(nside)
    f0_dummy = nmt.NmtField(mask, [np.zeros(npix)])
    f2_dummy = nmt.NmtField(mask, [np.zeros(npix), np.zeros(npix)])
    
    w00 = nmt.NmtWorkspace(); w00.compute_coupling_matrix(f0_dummy, f0_dummy, b)
    w02 = nmt.NmtWorkspace(); w02.compute_coupling_matrix(f0_dummy, f2_dummy, b)
    w22 = nmt.NmtWorkspace(); w22.compute_coupling_matrix(f2_dummy, f2_dummy, b)
    
    return {'w00': w00, 'w02': w02, 'w22': w22}

def cross_spectrum(mask, map_1, map_2, b, workspaces, purify_e=True, purify_b=True, beam=None):
    """
    Compute cross-spectrum using precomputed workspaces.

    Parameters
    ----------
    mask : array
        Healpy mask.
    map_1, map_2 : array
        Maps of shape (3, npix) with I,Q,U.
    b : NmtBin
        Binning object.
    workspaces : dict
        Precomputed workspaces: {'00': w00, '02': w02, '22': w22}.
    purify_e, purify_b : bool
        Whether to purify E/B modes.
    beam : array or None
        Optional beam window function.

    Returns
    -------
    dict with keys ['ell1','ell2','ell_eff','TT','EE','BB','TE','TB','EB']
    """
    ell_min_list = [b.get_ell_min(i) for i in range(b.get_n_bands())]
    ell_max_list = [b.get_ell_max(i) for i in range(b.get_n_bands())]
    ell_eff = b.get_effective_ells()

    f0_1 = nmt.NmtField(mask, [map_1[0,:]], beam=beam)
    f2_1 = nmt.NmtField(mask, [map_1[1,:], map_1[2,:]], purify_e=purify_e, purify_b=purify_b, beam=beam)
    f0_2 = nmt.NmtField(mask, [map_2[0,:]], beam=beam)
    f2_2 = nmt.NmtField(mask, [map_2[1,:], map_2[2,:]], purify_e=purify_e, purify_b=purify_b, beam=beam)

    w00, w02, w22 = workspaces['w00'], workspaces['w02'], workspaces['w22']

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


def compute_all_power_spectra(
    data, band_list, mask, b,
    use_simulated_maps=True,
    use_white_noise=False,
    noise_realization=1,
    only_noise=False,
    workspaces=None
):
    """
    Compute auto- and cross-power spectra for a list of bands using precomputed NaMaster workspaces.

    Avoids recomputing the NaMaster workspaces for each map pair.

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
    use_simulated_maps : bool, optional
        If True, use simulated sky + noise maps; if False, use real maps.
    use_white_noise : bool, optional
        If True, use white noise simulations; otherwise use full noise simulations.
    noise_realization : int, optional
        Noise realization number (1-based).
    only_noise : bool, optional
        If True, ignore sky and use only noise maps.
    workspaces : dict, optional
        Precomputed NaMaster workspaces with keys: 'w00', 'w02', 'w22'.

    Returns
    -------
    spectra_matrix : ndarray
        N_band x N_band matrix of dictionaries containing the spectra for each band pair.
        Each dictionary has keys: ['ell1','ell2','ell_eff','TT','EE','BB','TE','TB','EB'].
    """
    N_band = len(band_list)
    spectra_matrix = np.empty((N_band, N_band), dtype=object)

    # Loop over band pairs
    for i, band_i in enumerate(band_list):
        # Identify experiment
        exp_i = next(exp for exp, bands in data.items() if band_i in bands)

        # Load map i
        if use_simulated_maps:
            sky_map_i = 0
            if not only_noise:
                sky_i_path = data[exp_i][band_i]['path_simulated']
                sky_map_i = hp.read_map(sky_i_path, field=[0,1,2])
                sky_map_i = np.where(sky_map_i == hp.UNSEEN, 0, sky_map_i)
                if exp_i == 'Planck':
                    sky_map_i *= 1e3

            # Load noise
            if use_white_noise:
                noise_dir = data[exp_i][band_i]['path_white_noise_simulations']
                base_name = data[exp_i][band_i]['white_noise_simulation_1']
            else:
                noise_dir = data[exp_i][band_i]['path_noise_simulations']
                base_name = data[exp_i][band_i]['noise_simulation_1']

            noise_fname = get_noise_filename(base_name, noise_realization)
            noise_map = hp.read_map(os.path.join(noise_dir, noise_fname), field=[0,1,2])
            noise_map = np.where(noise_map == hp.UNSEEN, 0, noise_map)
            if exp_i == 'Planck':
                noise_map *= 1e3

            if isinstance(sky_map_i, int):
                sky_map_i = np.zeros_like(noise_map)
            sky_map_i = sky_map_i + noise_map
        else:
            sky_i_path = data[exp_i][band_i]['path']
            sky_map_i = hp.read_map(sky_i_path, field=[0,1,2])
            sky_map_i = np.where(sky_map_i == hp.UNSEEN, 0, sky_map_i)

        for j, band_j in enumerate(band_list):
            if j < i:
                spectra_matrix[i,j] = spectra_matrix[j,i]
                continue

            exp_j = next(exp for exp, bands in data.items() if band_j in bands)

            # Load map j
            if use_simulated_maps:
                sky_map_j = 0
                if not only_noise:
                    sky_j_path = data[exp_j][band_j]['path_simulated']
                    sky_map_j = hp.read_map(sky_j_path, field=[0,1,2])
                    sky_map_j = np.where(sky_map_j == hp.UNSEEN, 0, sky_map_j)
                    if exp_j == 'Planck':
                        sky_map_j *= 1e3

                if use_white_noise:
                    noise_dir = data[exp_j][band_j]['path_white_noise_simulations']
                    base_name = data[exp_j][band_j]['white_noise_simulation_1']
                else:
                    noise_dir = data[exp_j][band_j]['path_noise_simulations']
                    base_name = data[exp_j][band_j]['noise_simulation_1']

                noise_fname = get_noise_filename(base_name, noise_realization)
                noise_map = hp.read_map(os.path.join(noise_dir, noise_fname), field=[0,1,2])
                noise_map = np.where(noise_map == hp.UNSEEN, 0, noise_map)
                if exp_j == 'Planck':
                    noise_map *= 1e3

                if isinstance(sky_map_j, int):
                    sky_map_j = np.zeros_like(noise_map)
                sky_map_j = sky_map_j + noise_map
            else:
                sky_j_path = data[exp_j][band_j]['path']
                sky_map_j = hp.read_map(sky_j_path, field=[0,1,2])
                sky_map_j = np.where(sky_map_j == hp.UNSEEN, 0, sky_map_j)

            # Compute cross-spectrum using precomputed workspaces
            cl = cross_spectrum(mask, sky_map_i, sky_map_j, b, workspaces)
            spectra_matrix[i,j] = cl

    return spectra_matrix



def load_map(data, exp, band, use_simulated_maps, use_white_noise, noise_realization, only_noise):
    """
    Load a sky+noise map for a given experiment and band.

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
    array
        Combined sky+noise map (3, npix)
    """
    if use_simulated_maps:
        sky_map = 0
        if not only_noise:
            sky_path = data[exp][band]['path_simulated']
            sky_map = hp.read_map(sky_path, field=[0,1,2])
            sky_map = np.where(sky_map == hp.UNSEEN, 0, sky_map)
            if exp == 'Planck':
                sky_map *= 1e3
        
        if use_white_noise:
            noise_dir = data[exp][band]['path_white_noise_simulations']
            base_name = data[exp][band]['white_noise_simulation_1']
        else:
            noise_dir = data[exp][band]['path_noise_simulations']
            base_name = data[exp][band]['noise_simulation_1']
        
        noise_fname = get_noise_filename(base_name, noise_realization)
        noise_path = os.path.join(noise_dir, noise_fname)
        noise_map = hp.read_map(noise_path, field=[0,1,2])
        noise_map = np.where(noise_map == hp.UNSEEN, 0, noise_map)
        if exp == 'Planck':
            noise_map *= 1e3
        
        if isinstance(sky_map, int):
            sky_map = np.zeros_like(noise_map)
        return sky_map + noise_map
    else:
        path = data[exp][band]['path']
        sky_map = hp.read_map(path, field=[0,1,2])
        sky_map = np.where(sky_map == hp.UNSEEN, 0, sky_map)
        return sky_map

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


def average_and_std_spectra(data, spectra_dict, band_list, mask, b,
                                 use_white_noise=False, n_sim=100, only_noise=False,
                                 workspaces=None):
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
            workspaces=workspaces
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


def save_avg_std_to_fits(avg_std_dict, band_list, file_name, out_path):
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
    file_name : str
        Output FITS file name.
    out_path : str
        Directory where FITS will be saved.
    """
    hdu_list = fits.HDUList()
    hdu_list.append(fits.PrimaryHDU())  

    # Loop over all band pairs in band_list
    for band_i in band_list:
        for band_j in band_list:
            key = f"{band_i}_{band_j}"
            spec_dict = avg_std_dict[key]

            cols = []
            for cl_key in ['ell1','ell2','ell_eff','TT','EE','BB','TE','TB','EB']:
                cols.append(fits.Column(name=f"{cl_key}_MEAN", format="D", array=spec_dict[cl_key]['MEAN']))
                cols.append(fits.Column(name=f"{cl_key}_STD", format="D", array=spec_dict[cl_key]['STD']))

            # Create HDU for this band pair
            hdu = fits.BinTableHDU.from_columns(cols)
            hdu.header['BAND_I'] = band_i
            hdu.header['BAND_J'] = band_j
            hdu.name = key
            hdu_list.append(hdu)

    out_file = os.path.join(out_path, file_name)
    hdu_list.writeto(out_file, overwrite=True)
    print(f"Saved avg+std spectra to {out_file}")


def read_spectra_from_fits(path_fits, band_list):
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
        Path to the FITS file containing the spectra.
    band_list : list of str
        Ordered list of frequency bands.

    Returns
    -------
    spectra_dict : dict
        Dictionary with spectra for all band pairs.
    """
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

                # Case 2: avg+std
                if any(name.endswith("_MEAN") for name in colnames):
                    for cl_key in ['ell1','ell2','ell_eff','TT','EE','BB','TE','TB','EB']:
                        spec_dict[cl_key] = {
                            "MEAN": hdu.data[f"{cl_key}_MEAN"],
                            "STD":  hdu.data[f"{cl_key}_STD"],
                        }

                # Case 1: simple spectra
                else:
                    for cl_key in ['ell1','ell2','ell_eff','TT','EE','BB','TE','TB','EB']:
                        spec_dict[cl_key] = hdu.data[cl_key]

                spectra_dict[key] = spec_dict

    return spectra_dict


# 2. Precompute workspaces
workspaces = prepare_workspaces(mask, b, nside)

# 3. Compute all spectra
# spectra_matrix = compute_all_power_spectra(
#     data, band_list, mask, b,
#     use_simulated_maps=True,
#     use_white_noise=use_white_noise,
#     noise_realization=1,
#     only_noise=False,
#     workspaces=workspaces
# )

# 4. Save spectra matrix into a FITS file
# save_spectra_to_fits(spectra_matrix, band_list, mask_name=mask_name, out_path=out_path)

# 5. Read spectra matrix from FITS
spectra_dict = read_spectra_from_fits(path_spectra, band_list)

# 6. Compute mean and std over noise realizations (sky + noise)
avg_matrix, std_matrix = average_and_std_spectra(
    data, spectra_dict, band_list, mask, b,
    use_white_noise=True,
    n_sim=5, 
    only_noise=False,
    workspaces=workspaces
)

save_avg_std_to_fits(avg_matrix, std_matrix, band_list,
                     file_name=avg_std_skyplusnoise_name,
                     out_path=out_path)

# 7. Compute mean and std for noise-only maps
avg_matrix_noise, std_matrix_noise = average_and_std_spectra(
    data, spectra_dict, band_list, mask, b,
    use_white_noise=True,
    n_sim=5,
    only_noise=True,
    workspaces=workspaces
)

save_avg_std_to_fits(avg_matrix_noise, std_matrix_noise, band_list,
                     file_name=avg_std_noise_name,
                     out_path=out_path)

