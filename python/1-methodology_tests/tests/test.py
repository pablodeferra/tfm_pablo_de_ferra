#%%
import numpy as np
import healpy as hp
import os
from tqdm import tqdm
import pymaster as nmt
import sys
sys.path.append('../') 
from data import data, path_map, masks, path_masks
import functions 
import matplotlib.pyplot as plt

#%%
mask_path = masks['quijote_galcut']['galcut10']['path']
save_base = '/home/pablo/Desktop/master/tfm/figures/noise_spectra'

experiments = {
    # 'QUIJOTE': ['11'],
    # 'WMAP': ['23'],
    'Planck': ['100', '143', '217', '353']
}

map_info = data['Planck']['100']

functions.compute_and_plot_spectra(map_info, mask_path, use_white_noise=True, save=False, save_path=save_base)

# for experiment, map_names in experiments.items():
#     save_path = os.path.join(save_base, experiment)
#     for map_name in map_names:
#         map_info = data[experiment][map_name]
#         map_info['name'] = map_name  # add 'name' key for saving
#         print(f"Processing {experiment} {map_name}...")
#         functions.compute_and_plot_spectra(map_info, mask_path, use_white_noise=True, save=True, save_path=save_path)
        
# functions.plot_maps_mollview(map_info, component='I', use_white_noise=True, save=False)

#%%
import numpy as np
import healpy as hp
import os
from data import data, path_map, masks, path_masks
import functions 
import itertools

# Bands to use
quijote_bands = ['11']
wmap_bands = ['23', '33', '41', '61', '94']
planck_bands = ['30', '44', '70', '30', '143', '217', '217']

band_list = quijote_bands + wmap_bands + planck_bands

mask_select = masks['quijote_galcut']['galcut10']
mask_name = mask_select['name']
out_path = '/home/pablo/Desktop/master/tfm/spectra/'
path_corrected_spectra = os.path.join(out_path, f'corrected_power_spectra_{mask_name}.fits')
save_path = '/home/pablo/Desktop/master/tfm/figures/spectra_test/'

spectra_dict = functions.read_corrected_cls(path_corrected_spectra, band_list)

# functions.plot_cls_auto_cross(spectra_dict, '11', '30', save=False, save_path=save_path)


# Plot all unique cross combinations (no repeats, no autos)
# for band1, band2 in itertools.combinations(band_list, 2):
#     print(f"Plotting {band1} x {band2}")
#     functions.plot_cls_auto_cross(spectra_dict, band1, band2, save=True, save_path=save_path)

bands_to_plot = ['11', '23', '30']

# functions.plot_cls_auto_bands(spectra_dict, bands_to_plot, save=False, save_path=save_path)

#%%
