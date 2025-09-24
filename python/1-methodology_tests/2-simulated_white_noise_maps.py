#%%
import numpy as np
import healpy as hp
from tqdm import tqdm
from astropy import units as u
import os
from data import data, path_map, masks, path_masks


nside = 512


import os
import numpy as np
import healpy as hp
from tqdm import tqdm

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


white_noise_maps(data, nside, experiment_select='Planck', band_select='all', n_sim=100, path_map=path_map)
