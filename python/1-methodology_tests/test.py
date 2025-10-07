#%%
import numpy as np
import healpy as hp
import os
from tqdm import tqdm
import pymaster as nmt
from data import data, path_map, masks, path_masks
import functions 
import matplotlib.pyplot as plt

#%%
mask_path = masks['quijote_galcut']['galcut10']['path']
save_base = '/home/pablo/Desktop/master/tfm/figures/noise_spectra'

experiments = {
    # 'QUIJOTE': ['11'],
    # 'WMAP': ['23'],
    'Planck': ['217']
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

bands_to_plot = ['30', '217', '217']

functions.plot_cls_auto_bands(spectra_dict, bands_to_plot, save=False, save_path=save_path)

#%%

hmdm_30_path = data['Planck']['30']['hmdm']
map_30_path = data['Planck']['30']['path']

def downgrade_map(m, target_nside):
        """Downgrade a map (I, Q/U, or IQU) to target_nside."""
        if hp.get_nside(m) == target_nside:
            return m
        if m.ndim == 1:  # intensity map
            return hp.ud_grade(m, target_nside)
        else:  # polarization or IQU map
            return np.array([hp.ud_grade(m_ch, target_nside) for m_ch in m])

hmdm_30 = downgrade_map(hp.read_map(hmdm_30_path, field=[0,1,2]), 512)
map_30 = downgrade_map(hp.read_map(map_30_path, field=[0,1,2]), 512)

cl_hmdm_30 = hp.anafast(hmdm_30)
cl_map_30 = hp.anafast(map_30)
#%%

plt.plot(cl_hmdm_30[1])
plt.plot(cl_map_30[1])
plt.yscale('log')


#%%

out_path = '/home/pablo/Desktop/master/tfm/spectra/'

mask_select = masks['quijote_galcut']['galcut10']
mask_name = mask_select['name']
n_sim = 100
path_avg_std_noise = os.path.join(out_path, f'spectra_avg_std_{mask_name}_avg_std{n_sim}_noise.fits')

quijote_bands = ['11']
wmap_bands = ['23', '33', '41', '61', '94']
planck_bands = ['30', '44', '70', '30', '143', '217', '353']

band_list = quijote_bands + wmap_bands + planck_bands
use_white_noise = True

path_spectra = os.path.join(out_path, f'power_spectra_{mask_name}.fits')

path_corrected_spectra = os.path.join(out_path, f'corrected_power_spectra_{mask_name}.fits')

avg_std_noise = functions.read_spectra_from_fits(path_avg_std_noise, band_list, use_white_noise=use_white_noise)
spectra = functions.read_spectra_from_fits(path_spectra, band_list)
corr_spectra = functions.read_corrected_cls(path_corrected_spectra, band_list)


ell = avg_std_noise['30_30']['ell_eff']['MEAN']
Nl_217 = avg_std_noise['30_30']
Nl_217_EE = np.abs(Nl_217['EE']['MEAN'])

Cl_217_EE = np.abs(spectra['30_30']['EE'])
Cl_corr_217_EE = np.abs(corr_spectra['30_30']['EE']['SPECTRUM'])

Bl_217 = functions.get_beam_for_band('30', data, ell)
uc_217 = functions.cmb_unit_conversion(data['Planck']['30']['freq'].value)

corr = np.abs(Cl_217_EE - Nl_217_EE) / (Bl_217['E']**2 * uc_217**0)

plt.plot(ell, Nl_217_EE)
plt.plot(ell, Cl_217_EE)
plt.plot(ell, Cl_corr_217_EE)
plt.plot(ell, corr)
plt.yscale('log')
plt.xlim(0,500)
