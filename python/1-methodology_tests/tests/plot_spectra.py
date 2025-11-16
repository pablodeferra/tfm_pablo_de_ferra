#%%
import sys
sys.path.append('../') 
from data import data, path_map, masks, path_masks
import functions 
import os

# Configuration
nside = 512
n_sim = 100
path_save = path_map + 'PYSM/'

quijote_bands = ['11']
wmap_bands = ['23', '33', '41', '61', '94']
planck_bands = ['30', '44', '70', '100', '143', '217', '353']
band_list = quijote_bands + wmap_bands + planck_bands

mask_select = masks['QUIJOTE_galcut']['galcut10_10mk']
mask_name = mask_select['name']
out_path = '/home/pablo/Desktop/master/tfm/spectra/'
name_suffix = '_full'

# File paths
path_corrected_spectra = os.path.join(out_path, f'corrected_power_spectra_{mask_name}{name_suffix}.fits')
path_theoretical_spectra = os.path.join(out_path, f'corrected_theoretical_power_spectra_{mask_name}.fits')

corrected_spectra_dict = functions.read_corrected_cls(path_corrected_spectra, band_list)
theoretical_spectra_dict = functions.read_spectra_from_fits(path_theoretical_spectra, band_list)


band_pairs_to_plot = [
    '11_11',
    '23_23',
    '30_30',
]

band_pairs_to_plot = [
    '11_23',
    '11_30',
    '23_30',
]

# Output paths for figures
mask_path = masks['QUIJOTE_galcut']['galcut10']['path']
save_dir = '/home/pablo/Desktop/master/tfm/figures/corrected_vs_theoretical'
bands_str = "_".join(band_pairs_to_plot)
figure_name = f'corrected_vs_theoretical_Cl_{bands_str}_{mask_name}{name_suffix}.pdf' 


# Plot corrected vs theoretical spectra for each band pair
functions.plot_corrected_vs_theoretical(
    corrected_spectra_dict=corrected_spectra_dict,
    theoretical_spectra_dict=theoretical_spectra_dict,
    band_pairs=band_pairs_to_plot,
    mask_name=mask_name,
    save=False,
    save_path=save_dir,
    filename=figure_name,
    plot_dl = False
)
    