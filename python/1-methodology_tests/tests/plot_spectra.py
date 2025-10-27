#%%
import sys
sys.path.append('../') 
from data import data, path_map, masks, path_masks
import functions 

mask_path = masks['QUIJOTE_galcut']['galcut10']['path']
save_base = '/home/pablo/Desktop/master/tfm/figures/noise_spectra'

experiment = 'QUIJOTE'
band = '11'

functions.compute_and_plot_spectra(experiment=experiment, band=band, mask_path=mask_path, save=False, save_path=save_base)
