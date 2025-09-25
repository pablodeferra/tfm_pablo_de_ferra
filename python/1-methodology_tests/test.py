#%%
import numpy as np
import healpy as hp
import os
from tqdm import tqdm
import pymaster as nmt
from data import data, path_map, masks, path_masks
import functions 

mask_path = masks['quijote_galcut']['galcut10']['path']
save_base = '/home/pablo/Desktop/master/tfm/figures/noise_spectra'

experiments = {
    'QUIJOTE': ['11'],
    'WMAP': ['23', '33', '41', '61', '94'],
    'Planck': ['30', '44', '70', '100', '143', '217', '353']
}

for experiment, map_names in experiments.items():
    save_path = os.path.join(save_base, experiment)
    for map_name in map_names:
        map_info = data[experiment][map_name]
        map_info['name'] = map_name  # add 'name' key for saving
        print(f"Processing {experiment} {map_name}...")
        functions.compute_and_plot_spectra(map_info, mask_path, use_white_noise=True, save=False, save_path=save_path)
        
# functions.plot_maps_mollview(map_info, component='I', use_white_noise=True, save=False)

#%%
import numpy as np
import healpy as hp
import os
from data import data, path_map, masks, path_masks
import functions 

n_sim = 5

mask_select = masks['quijote_galcut']['galcut10']
mask_name = mask_select['name']
use_simulated_maps = True
use_white_noise = True
out_path = '/home/pablo/Desktop/master/tfm/spectra/'
path_spectra = os.path.join(out_path, f'power_spectra_{mask_name}.fits')
path_avg_std_skyplusnoise = os.path.join(out_path, f'spectra_avg_std_{mask_name}_avg_std{n_sim}_skyplusnoise.fits')
path_avg_std_noise = os.path.join(out_path, f'spectra_avg_std_{mask_name}_avg_std{n_sim}_noise.fits')

save_path = '/home/pablo/Desktop/master/tfm/figures/spectra_auto_cross_test/'

# Bands to use
quijote_bands = ['11']
wmap_bands = ['23', '33', '41', '61', '94']
planck_bands = ['30', '44', '70', '100', '143', '217', '353']

band_list = quijote_bands + wmap_bands + planck_bands

# mask = hp.read_map(mask_select['path'])

# spectra_matrix = functions.read_spectra_from_fits(path_spectra, band_list)
# avg_std_spn_matrix = functions.read_spectra_from_fits(path_avg_std_skyplusnoise, band_list)
# avg_std_n_matrix = functions.read_spectra_from_fits(path_avg_std_noise, band_list)

# functions.plot_band_spectra(path_spectra, band_list, "11", "30", save=False, save_path=save_path)



#%%

map_path = data['WMAP']['23']['path']

map_23, h = hp.read_map(map_path, field=[0], hdu=2, h=True)

