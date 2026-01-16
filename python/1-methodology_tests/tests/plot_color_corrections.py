#%%
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os
import sys
sys.path.append('../') 
import functions as functions

# Paths
cc_path = '/home/pablo/Desktop/master/tfm/cc/'
cc_td_hfi_bps = os.path.join(cc_path, 'c_td_10-40_beta_hfi_bps_pr3.fits')
cc_td_hfi = os.path.join(cc_path, 'c_td_10-40_beta_hfi_pr3.fits')

# Bands to plot (Planck HFI)
bands = ['P100', 'P143', 'P217', 'P353']
Tdust = 19.6  # K

# Load beta grid from the FITS CC table so we evaluate on the native grid
with fits.open(cc_td_hfi_bps) as hdul:
    beta_grid_bps = np.asarray(hdul[1].data[0]['BETA'], dtype=float)

with fits.open(cc_td_hfi) as hdul:
    beta_grid = np.asarray(hdul[1].data[0]['BETA'], dtype=float)

beta_grid_bps = np.ravel(beta_grid_bps)
beta_grid = np.ravel(beta_grid)

# Create a single figure with 2x2 subplots (no shared axes)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes_flat = axes.ravel()

# Consistent styling for curves
fastcc_style = dict(color='tab:blue', lw=2)
pr3_style = dict(color='tab:orange', lw=2, ls='--')
pr3_bps_style = dict(color='tab:green', lw=2, ls=':')

# Compute beta range across both tables for consistent x-limits
beta_min = float(min(beta_grid.min(), beta_grid_bps.min()))
beta_max = float(max(beta_grid.max(), beta_grid_bps.max()))


for idx, band in enumerate(bands):
    ax = axes_flat[idx]

    # 1) Curve from functions.fastcc (alpha = beta + 2)
    alpha_grid = beta_grid + 2.0
    cc_fastcc = np.array([functions.fastcc(band, alpha=a) for a in alpha_grid], dtype=float)

    # 2) Curves from fastcc tables at Td = 19.6 K (non-bps and bps)
    interp = functions.interpcc_setup(cc_td_hfi, band, td_limit=40, method=3)
    interp_bps = functions.interpcc_setup(cc_td_hfi_bps, band, td_limit=40, method=3)
    cc = np.array([functions.interpcc(interp, Tdust, b) for b in beta_grid], dtype=float)
    cc_bps = np.array([functions.interpcc(interp_bps, Tdust, b) for b in beta_grid_bps], dtype=float)

    # Plot on the subplot; ensure cc_bps uses its own beta grid
    ax.plot(beta_grid, cc_fastcc, label="fastcc (α=β+2)", **fastcc_style)
    ax.plot(beta_grid, cc, label=f"PR3 table (Td={Tdust}K)", **pr3_style)
    ax.plot(beta_grid_bps, cc_bps, label=f"PR3 bps table (Td={Tdust}K)", **pr3_bps_style)
    ax.set_title(f"{band}")
    # X limits
    ax.set_xlim(beta_min, beta_max)
    ax.set_xlabel(r'Spectral index $\beta_d$')
    ax.set_ylabel('Color correction factor')


# Put a single shared legend and place it a bit higher
handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.05))
# Make room for the legend above the subplots
plt.tight_layout(rect=[0, 0, 1, 1])

plt.show()
