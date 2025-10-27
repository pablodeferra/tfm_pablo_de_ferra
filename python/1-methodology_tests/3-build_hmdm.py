#%%
import os
import healpy as hp
import numpy as np
from data import data 
import functions

#%%
'''
# ==================================================
# Build half maps for WMAP
# ==================================================
'''

base_dir = os.path.dirname(data['WMAP']['23']['hmdm'])
save_path = os.path.dirname(data['WMAP']['23']['hmdm'])

combined_1to4 = functions.coadd_year_range(base_dir=base_dir, year_1=1, year_2=4, save=True, save_path=save_path)
combined_5to9 = functions.coadd_year_range(base_dir=base_dir, year_1=5, year_2=9, save=True, save_path=save_path)

#%%

'''
# ==================================================
# Build HMDM for each experiment and frequency band
# ==================================================
'''

# Bands to use
quijote_bands = ['11']
wmap_bands = ['23', '33', '41', '61', '94']
planck_bands = ['100', '143', '217', '353']

bands = quijote_bands + wmap_bands + planck_bands

functions.make_hmdm(data, bands, save=True)