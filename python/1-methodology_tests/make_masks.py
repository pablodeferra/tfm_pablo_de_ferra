# %%
import healpy as hp
import numpy as np
import pymaster as nmt
from data import data, masks, path_masks

mask_lowdec_satband_path = masks['QUIJOTE']['lowdec_satband']['path']
mask_lowdec_satband = hp.read_map(mask_lowdec_satband_path)

map_quijote_11_path = data['QUIJOTE']['11']['path']
map_quijote_11 = hp.read_map(map_quijote_11_path, field=[0])

nside=512
npix = hp.nside2npix(nside)
aposcale=5

gb = hp.pix2ang(nside, np.arange(npix), lonlat=True)[1] 

mask_galcut10 = np.where(np.abs(gb) < 10., 0., 1.) 

mask_quijote_11_10mk = np.where(map_quijote_11 > 10., 0, 1)

mask_lowdec_satband_galcut10_10mk = mask_lowdec_satband * mask_galcut10 * mask_quijote_11_10mk

mask_lowdec_satband_galcut10_10mk_apodC2_5 = nmt.mask_apodization(mask_lowdec_satband_galcut10_10mk, aposcale, apotype="C2")
mask_name = 'mask_lowdec_satband_galcut10_10mk_apodC2_5.fits'
# =============================
# Saving masks 
# =============================

hp.write_map(path_masks + 'quijote_galcut/' + mask_name, mask_lowdec_satband_galcut10_10mk_apodC2_5, dtype='float64', overwrite=True)
