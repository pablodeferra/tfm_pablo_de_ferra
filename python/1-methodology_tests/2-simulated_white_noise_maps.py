#%%
import numpy as np
import healpy as hp
from tqdm import tqdm
from astropy import units as u
import os
from data import data, path_map, masks, path_masks


nside = 512

def white_noise_maps(data, nside, experiment_select="all", band_select="all", n_sim=100):
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
                else:  # case string
                    if band != band_select:
                        continue

            try:
                band_path = band_info['path']
                print(f" [{experiment}] Band {band} GHz -> {band_path}")

                # Load N_obs depending on experiment
                if experiment == "QUIJOTE":
                    nobs = hp.read_map(band_path, field=[3, 4])
                    nobs_i, nobs_qu = nobs

                    nobs_i = np.where(nobs_i == hp.UNSEEN, 0, nobs_i)
                    nobs_qu = np.where(nobs_qu == hp.UNSEEN, 0, nobs_qu)

                    sigma_i = np.zeros_like(nobs_i, dtype=float)
                    sigma_qu = np.zeros_like(nobs_qu, dtype=float)

                    valid_i = nobs_i > 0
                    valid_qu = nobs_qu > 0

                    sigma_i[valid_i] = band_info['noise_I'].value / np.sqrt(nobs_i[valid_i])
                    sigma_qu[valid_qu] = band_info['noise_QU'].value / np.sqrt(nobs_qu[valid_qu])

                elif experiment == "WMAP":
                    nobs = hp.read_map(band_path, hdu=2, field=[0,1,2,3])
                    nobs_i, nobs_q, nobs_qu, nobs_u = nobs
                    nobs_q_eff = nobs_q - nobs_qu**2/nobs_u
                    nobs_u_eff = nobs_u - nobs_qu**2/nobs_q
                    sigma_i = band_info['noise_I'].value / np.sqrt(nobs_i)
                    sigma_q = band_info['noise_QU'].value / np.sqrt(nobs_q_eff)
                    sigma_u = band_info['noise_QU'].value / np.sqrt(nobs_u_eff)

                elif experiment == "Planck":
                    nobs = hp.read_map(band_path, field=[3])
                    nside_in = hp.get_nside(nobs)
                    if nside_in != nside:
                        nobs = hp.ud_grade(nobs, nside_out=nside) * (nside_in / nside)**2 # Sum of the exposure time
                    nobs_i = nobs_qu = nobs
                    sigma_i = band_info['noise_I'].value / np.sqrt(nobs_i)
                    sigma_qu = band_info['noise_QU'].value / np.sqrt(nobs_qu)

                # Generate n_sim white noise realizations
                noise_map = np.zeros([3, npix])
                for ii in tqdm(range(n_sim), desc='generating maps'):
                    if experiment == "WMAP":
                        noise_map[0] = np.random.normal(0, sigma_i, npix)
                        noise_map[1] = np.random.normal(0, sigma_q, npix)
                        noise_map[2] = np.random.normal(0, sigma_u, npix)
                    else:
                        noise_map[0] = np.random.normal(0, sigma_i, npix)
                        noise_map[1] = np.random.normal(0, sigma_qu, npix)
                        noise_map[2] = np.random.normal(0, sigma_qu, npix)

                    # Write map
                    out_dir = os.path.join(path_map, experiment, "noise_simulations", band)
                    os.makedirs(out_dir, exist_ok=True)
                    out_file = os.path.join(out_dir, f"white_noise_{band}ghz_{str(ii+1).zfill(4)}.fits")
                    hp.write_map(out_file, noise_map, dtype=np.float64, overwrite=True)

            except KeyError:
                print(f" ! Band {band} not found in {experiment}")


# Example run
white_noise_maps(data, nside, experiment_select='QUIJOTE', band_select='11', n_sim=1)
