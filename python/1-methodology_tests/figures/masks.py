#%%
import numpy as np
import healpy as hp
import sys
sys.path.append('../')
import matplotlib.pyplot as plt
from data import data, masks

save_path = '/home/pablo/Desktop/master/tfm/figures/masks/'

mask_dict = masks['QUIJOTE_galcut']['galcut15']
mask_name = mask_dict['name']

mask = hp.read_map(mask_dict['path'])

hp.mollview(mask, title='')
# plt.savefig(f'{save_path}mask_{mask_name}.pdf')

wmap_k_path = data['WMAP']['23']['path']
wmap_k = hp.read_map(wmap_k_path, field=[0,1,2])
wmap_s = hp.smoothing(np.sqrt(wmap_k[1]**2 + wmap_k[2]**2), fwhm=np.deg2rad(1))

masks_path = masks['specific_regions']['nps']['path']
masks = hp.read_map(masks_path)

use_planck_cmap = True

cmap = None
if use_planck_cmap:
    ############### CMB colormap
    from matplotlib.colors import ListedColormap
    colombi1_cmap = ListedColormap(np.loadtxt("/home/pablo/Desktop/master/tfm/Planck_Parchment_RGB.txt")/255.)
    colombi1_cmap.set_bad("gray") # color of missing pixels
    colombi1_cmap.set_under("white") # color of background, necessary if you want to use
    # this colormap directly with hp.mollview(m, cmap=colombi1_cmap)
    cmap = colombi1_cmap


masks_inv = np.where(masks == 0, 0.8, 0)
fig0 = plt.figure(figsize=(10, 6))
hp.mollview(wmap_s, fig = fig0, cmap=cmap, norm='hist', cbar=False, title=None, alpha = np.ones(3145728), bgcolor='None' )
hp.mollview(masks, fig = fig0, cmap='gray', cbar=False, title=None, min = -0.25, max = 0.8, alpha = masks_inv, bgcolor='None')
hp.graticule(dmer=40)
# plt.suptitle('Fan', fontsize=16, color='w')
# hp.projtext(150, -5, 'Fan', lonlat=True, fontsize=16, color='w')

fig0.savefig('/home/pablo/Desktop/master/tfm/figures/masks/mask_nps_wmap_s.png', dpi=800, transparent=True)
