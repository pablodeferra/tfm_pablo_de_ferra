#%%
import numpy as np
import healpy as hp
from tqdm import tqdm
from astropy import units as u
import os
from data import data, path_map, masks, path_masks
import functions

nside = 512


functions.white_noise_maps(data, nside, experiment_select='QUIJOTE', band_select='11', n_sim=1, path_map=path_map)
