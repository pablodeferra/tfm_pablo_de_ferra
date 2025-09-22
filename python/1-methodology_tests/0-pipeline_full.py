#%%
import numpy as np
import pysm3
import healpy as hp
from astropy import units as u
import os
from tqdm import tqdm
from data import data, path_map
import functions

nside = 512

# --- Choose experiment or band to run ---
experiment_select = None  # 'QUIJOTE', 'WMAP' or 'Planck'
band_select = None 

'''
# ============================================
# Step 1: Simulate sky signal maps using PySM
# ============================================
'''

path_save = '/home/pablo/Desktop/Paper/maps/PYSM/'

functions.generate_sky_maps(nside, path_save, experiment_select, band_select)

'''
# ============================================
# Step 2: Generate white noise simulations 
# ============================================
'''

functions.run_experiments(data, nside, experiment_select, band_select)
