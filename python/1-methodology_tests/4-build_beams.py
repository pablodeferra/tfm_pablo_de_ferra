#%%
import os
import numpy as np
from data import data
import functions

beam_path = os.path.dirname(data['WMAP']['23']['beam'])
save_path = os.path.dirname(data['WMAP']['23']['beam'])

# Differential Assemblies per band
BANDS = {
    'K': ['K1'],
    'Ka': ['Ka1'],
    'Q': ['Q1', 'Q2'],
    'V': ['V1', 'V2'],
    'W': ['W1', 'W2', 'W3', 'W4'],
}

functions.generate_band_beams(BANDS, beam_path=beam_path, save_path=save_path)
