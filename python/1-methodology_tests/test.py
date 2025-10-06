#%%
import numpy as np
import healpy as hp
import os
from tqdm import tqdm
import pymaster as nmt
from data import data, path_map, masks, path_masks
import functions 
import matplotlib.pyplot as plt

mask_path = masks['quijote_galcut']['galcut10']['path']
save_base = '/home/pablo/Desktop/master/tfm/figures/noise_spectra'

experiments = {
    # 'QUIJOTE': ['11'],
    # 'WMAP': ['23'],
    'Planck': ['353']
}

# functions.compute_and_plot_spectra(map_info, mask_path, use_white_noise=True, save=False, save_path=save_path)

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
import itertools

# Bands to use
quijote_bands = ['11']
wmap_bands = ['23', '33', '41', '61', '94']
planck_bands = ['30', '44', '70', '100', '143', '217', '353']

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

bands_to_plot = ['100', '217', '353']

functions.plot_cls_auto_bands(spectra_dict, bands_to_plot, save=False, save_path=save_path)

#%%

hmdm_100_path = data['Planck']['100']['hmdm']
map_100_path = data['Planck']['100']['path']

def downgrade_map(m, target_nside):
        """Downgrade a map (I, Q/U, or IQU) to target_nside."""
        if hp.get_nside(m) == target_nside:
            return m
        if m.ndim == 1:  # intensity map
            return hp.ud_grade(m, target_nside)
        else:  # polarization or IQU map
            return np.array([hp.ud_grade(m_ch, target_nside) for m_ch in m])

hmdm_100 = downgrade_map(hp.read_map(hmdm_100_path, field=[0,1,2]), 512)
map_100 = downgrade_map(hp.read_map(map_100_path, field=[0,1,2]), 512)

cl_hmdm_100 = hp.anafast(hmdm_100)
cl_map_100 = hp.anafast(map_100)
#%%

plt.plot(cl_hmdm_100[1])
plt.plot(cl_map_100[1])
plt.yscale('log')

