#%%
import functions
import os
from data import data, path_map, masks, path_masks
import healpy as hp
import numpy as np

# Configuration
nside = 512
lmax = 2 * nside - 1
dl = 10

# Band selection
quijote_bands = ['11']
wmap_bands = ['23', '33', '41', '61', '94']  
planck_bands = ['30', '44', '70', '100', '143', '217', '353']
band_list = quijote_bands + wmap_bands + planck_bands

# Mask selection
mask_select = masks['QUIJOTE_galcut']['galcut10_10mk']
mask_name = mask_select['name']
mask = hp.read_map(mask_select['path'])

# Output paths
out_path = '/home/pablo/Desktop/master/tfm/spectra/'
path_theoretical_spectra = os.path.join(out_path, f'theoretical_power_spectra_{mask_name}_dl20.fits')
path_corrected_theoretical_spectra = os.path.join(out_path, f'corrected_theoretical_power_spectra_{mask_name}_dl20.fits')

# Create binning scheme (same as in main analysis)
ell_1 = [30, 50, 70, 90,  110, 130, 150, 170]
ell_2 = [49, 69, 89, 109, 129, 149, 169, 189]


binning_params = {
    'type': 'edges',  # 'linear' or 'edges'
    'lmax': lmax,
    'dl': dl,
    # For edges (if needed)
    'ell1': ell_1,
    'ell2': ell_2
}

# 1. Prepare binning scheme
b = functions.create_binning(binning_params)

# 2. Precompute workspaces
workspaces = functions.prepare_workspaces(mask, b, nside, lmax=lmax, purify_e=True, purify_b=True)


# 3. Compute theoretical spectra from pure simulated maps (no noise added)
print("Computing theoretical power spectra from pure simulated maps (no noise)...")
theoretical_spectra_matrix = functions.compute_pure_theoretical_spectra(
    data, band_list, mask, b,
    workspaces=workspaces,
    lmax=lmax
)

# 4. Save theoretical spectra matrix into a FITS file
functions.save_spectra_to_fits(theoretical_spectra_matrix, band_list, out_file=path_theoretical_spectra)


# 5. Apply corrections to theoretical spectra (beam, pixel window, unit conversion)
corrected_theoretical_spectra = functions.correct_theoretical_spectra(
    path_theoretical_spectra=path_theoretical_spectra,
    band_list=band_list,
    data=data,
    nside=nside,
    correct_beam=True,
    correct_unit=True, 
    correct_pixel=True,
    save=True,
    path_out_file=path_corrected_theoretical_spectra
)
