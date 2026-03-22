#%%
import numpy as np
import healpy as hp
import sys
sys.path.append('../')
import matplotlib.pyplot as plt
# Increase title fontsize for map plots (adjust as needed)
plt.rcParams['axes.titlesize'] = 20
from data import data, masks

save_path = '/home/pablo/Desktop/master/tfm/figures/masks/'

#%%
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

# fig0.savefig('/home/pablo/Desktop/master/tfm/figures/masks/mask_nps_wmap_s.png', dpi=800, transparent=True)


#%%

from data import data, masks

save_path = '/home/pablo/Desktop/master/tfm/figures/masks/'

mask_10_path = masks['QUIJOTE_galcut']['galcut10_noapod']['path']
mask_15_path = masks['QUIJOTE_galcut']['galcut15_noapod']['path']
mask_20_path = masks['QUIJOTE_galcut']['galcut20_noapod']['path']


mask_10 = hp.read_map(mask_10_path)
mask_15 = hp.read_map(mask_15_path)
mask_20 = hp.read_map(mask_20_path)


mask_sum = mask_10 + mask_15 + mask_20

# Choose exact colors for the 4 discrete mask values (0..3).
# Edit MASK_COLORS to pick your desired colors (hex codes or matplotlib names).
# The list must contain 4 entries corresponding to values 0,1,2,3 in `mask_sum`.
MASK_COLORS = ['#440154', '#31688e', '#35b765', '#fde725']
from matplotlib.colors import ListedColormap
mask_cmap = ListedColormap(MASK_COLORS)

hp.mollview(mask_sum, cmap=mask_cmap, title='', min=0, max=3, cbar=False)
plt.savefig(save_path + 'masks_sum_galcut_10-15-20.pdf')

#%%

mask_10_dict = masks['QUIJOTE_galcut']['galcut10']
mask_15_dict = masks['QUIJOTE_galcut']['galcut15']
mask_20_dict = masks['QUIJOTE_galcut']['galcut20']

mask_10_path = mask_10_dict['path']
mask_15_path = mask_15_dict['path']
mask_20_path = mask_20_dict['path']

mask_10_name = mask_10_dict['name']
mask_15_name = mask_15_dict['name']
mask_20_name = mask_20_dict['name']

mask_10 = hp.read_map(mask_10_path)
mask_15 = hp.read_map(mask_15_path)
mask_20 = hp.read_map(mask_20_path)


hp.mollview(mask_10, title=r'$b > |10^{\circ}|$', cbar=True)
try:
    cax = plt.gcf().axes[-1]
    pos = cax.get_position()
    extra = pos.width * 1
    new_pos = [pos.x0 - extra * 0.5, pos.y0, pos.width + extra, pos.height]
    cax.set_position(new_pos)
    cax.tick_params(labelsize=12)
    try:
        cax.set_yticklabels(['0', '1'])
    except Exception:
        pass
except Exception:
    pass
plt.savefig(save_path + f'mask_{mask_10_name}_cbar.pdf')

hp.mollview(mask_15, title=r'$b > |15^{\circ}|$', cbar=True)
try:
    cax = plt.gcf().axes[-1]
    pos = cax.get_position()
    extra = pos.width * 1
    new_pos = [pos.x0 - extra * 0.5, pos.y0, pos.width + extra, pos.height]
    cax.set_position(new_pos)
    cax.tick_params(labelsize=12)
    try:
        cax.set_yticklabels(['0', '1'])
    except Exception:
        pass
except Exception:
    pass
plt.savefig(save_path + f'mask_{mask_15_name}_cbar.pdf')

hp.mollview(mask_20, title=r'$b > |20^{\circ}|$', cbar=True)
try:
    cax = plt.gcf().axes[-1]
    pos = cax.get_position()
    # expand width slightly to make the colorbar larger (tweak as needed)
    extra = pos.width * 1
    new_pos = [pos.x0 - extra * 0.5, pos.y0, pos.width + extra, pos.height]
    cax.set_position(new_pos)
    cax.tick_params(labelsize=20)
    try:
        cax.set_yticklabels(['0', '1'])
    except Exception:
        pass
except Exception:
    pass
plt.savefig(save_path + f'mask_{mask_20_name}_cbar.pdf')


#%%

plt.rcParams.update({
    'axes.labelcolor': 'white',     # Axis labels
    'xtick.color': 'white',         # X-axis tick labels
    'ytick.color': 'white',         # Y-axis tick labels
    'axes.titlecolor': 'white',     # Title color
    'legend.facecolor': 'black',    # Legend background
    'legend.edgecolor': 'white',    # Legend border
    'legend.fontsize': 'medium',    # Legend font size
    'text.color': 'white',          # General text color
    'figure.facecolor': 'black',    # Figure background
    'figure.edgecolor': 'white',    # Figure edge color
    'axes.facecolor': 'black',      # Axes background
    'axes.edgecolor': 'white'       # Axes edge (border) color
})


from data import data, masks

save_path = '/home/pablo/Desktop/master/tfm/figures/masks/'

mask_10_path = masks['QUIJOTE_galcut']['galcut10']['path']
mask_15_path = masks['QUIJOTE_galcut']['galcut15']['path']
mask_20_path = masks['QUIJOTE_galcut']['galcut20']['path']

mask_20_name = masks['QUIJOTE_galcut']['galcut20']['name']

mask_10 = hp.read_map(mask_10_path)
mask_15 = hp.read_map(mask_15_path)
mask_20 = hp.read_map(mask_20_path)

save_path_ppt = '/home/pablo/Desktop/master/tfm/figures_ppt/masks/'


hp.mollview(mask_10, title=r'', cbar=False, bgcolor='None')

# plt.savefig(save_path_ppt + f'mask_{mask_10_name}_cbar.png', dpi=300, transparent=True)

hp.mollview(mask_15, title=r'', cbar=False, bgcolor='None')

# plt.savefig(save_path_ppt + f'mask_{mask_15_name}_cbar.png', dpi=300, transparent=True)

hp.mollview(mask_20, title=r'', cbar=False, bgcolor='None')
# try:
#     cax = plt.gcf().axes[-1]
#     pos = cax.get_position()
#     # expand width slightly to make the colorbar larger (tweak as needed)
#     extra = pos.width * 1
#     new_pos = [pos.x0 - extra * 0.5, pos.y0, pos.width + extra, pos.height]
#     cax.set_position(new_pos)
#     cax.tick_params(labelsize=20)
#     try:
#         cax.set_yticklabels(['0', '1'])
#     except Exception:
#         pass
# except Exception:
#     pass
plt.savefig(save_path_ppt + f'mask_{mask_20_name}.png', dpi=300, transparent=True)

#%%

wmap_k_path = data['WMAP']['23']['path']
wmap_k = hp.read_map(wmap_k_path, field=[0,1,2])
wmap_s = hp.smoothing(np.sqrt(wmap_k[1]**2 + wmap_k[2]**2), fwhm=np.deg2rad(1.2))

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

masks_10_inv = np.where(mask_10 == 0, 0.8, 0)
masks_15_inv = np.where(mask_15 == 0, 0.8, 0)
masks_20_inv = np.where(mask_20 == 0, 0.8, 0)

#%%
fig_10 = plt.figure(figsize=(10, 6))
hp.mollview(wmap_s, fig = fig_10, cmap=cmap, norm='hist', cbar=False, title=None, alpha = np.ones(3145728), bgcolor='None',
		remove_mono=True,
		remove_dip=False,)
hp.mollview(mask_10, fig = fig_10, cmap='gray', cbar=False, title=None, min = -0.25, max = 0.8, alpha = masks_10_inv, bgcolor='None')
hp.graticule(dmer=40)
plt.savefig(save_path_ppt + 'mask_galcut10_wmap_k.png', dpi=800, transparent=True)

fig_15 = plt.figure(figsize=(10, 6))
hp.mollview(wmap_s, fig = fig_15, cmap=cmap, norm='hist', cbar=False, title=None, alpha = np.ones(3145728), bgcolor='None' )
hp.mollview(mask_15, fig = fig_15, cmap='gray', cbar=False, title=None, min = -0.25, max = 0.8, alpha = masks_15_inv, bgcolor='None')
hp.graticule(dmer=40)
plt.savefig(save_path_ppt + 'mask_galcut15_wmap_k.png', dpi=800, transparent=True)

fig_20 = plt.figure(figsize=(10, 6))
hp.mollview(wmap_s, fig = fig_20, cmap=cmap, norm='hist', cbar=False, title=None, alpha = np.ones(3145728), bgcolor='None' )
hp.mollview(mask_20, fig = fig_20, cmap='gray', cbar=False, title=None, min = -0.25, max = 0.8, alpha = masks_20_inv, bgcolor='None')
hp.graticule(dmer=40)
plt.savefig(save_path_ppt + 'mask_galcut20_wmap_k.png', dpi=800, transparent=True)

#%%

hp.mollview(wmap_s, cmap=cmap, norm='hist', min = 0.027, max =2.070)
try:
    cax = plt.gcf().axes[-1]
    pos = cax.get_position()
    # expand width slightly to make the colorbar larger (tweak as needed)
    extra = pos.width * 1
    new_pos = [pos.x0 - extra * 0.5, pos.y0, pos.width + extra, pos.height]
    cax.set_position(new_pos)
    cax.tick_params(labelsize=20)
    try:
        cax.set_yticklabels(['0', '1'])
    except Exception:
        pass
except Exception:
    pass
plt.savefig(save_path_ppt + 'wmap_k.png', dpi=800, transparent=True)
