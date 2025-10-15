#%%
import os
import sys
import numpy as np
import healpy as hp
from astropy import units as u
from astropy.io import fits

import pysm3
import pysm3.units as u_pysm

from data import data
import functions

# Default configuration
nside = 512
path_save = '/home/pablo/Desktop/Paper/maps/PYSM/'

quijote_bands = ['11']
wmap_bands = ['23', '33', '41', '61', '94']
planck_bands = ['30', '44', '70', '100', '143', '217', '353']

# Generate WMAP maps
functions.generate_sky_maps(nside, path_save, experiment_select='WMAP', band_select=wmap_bands)