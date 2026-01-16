#%%
import numpy as np
from astropy.io import fits
import sys
sys.path.append('../') 
import functions as functions
import matplotlib.pyplot as plt

cc_path = '/home/pablo/Desktop/master/tfm/cc/'

cc_td_hfi_bps = cc_path + 'c_td_10-40_beta_hfi_bps_pr3.fits'
cc_td_hfi = cc_path + 'c_td_10-40_beta_hfi_pr3.fits'

cc_dust_p100 = functions.fastcc('P100', alpha=3.50)
cc_dust_p143 = functions.fastcc('P143', alpha=3.50)
cc_dust_p217 = functions.fastcc('P217', alpha=3.50)
cc_dust_p353 = functions.fastcc('P353', alpha=3.50)

alpha = -1.1

cc_dust_q11 = functions.fastcc('Q11', alpha=alpha)
cc_p_q11 = 0.9821236098857145 + 0.011920227628586971*alpha -0.0015347819996889738 * alpha**2

print(cc_dust_q11)
print(cc_p_q11)
# 'Q11': [0.9821236098857145, 0.011920227628586971, -0.0015347819996889738, 11.1]



cc_dict = functions.load_color_correction_polynomials()

f = '100'
cc_dict_11_s = cc_dict['synch'][f]
cc_dict_11_d = cc_dict['dust'][f]

alpha = np.linspace(-2, 5, 100)

cc_poly_s = cc_dict_11_s[0] + cc_dict_11_s[1]*alpha + cc_dict_11_s[2]*alpha**2
cc_poly_d = cc_dict_11_d[0] + cc_dict_11_d[1]*alpha + cc_dict_11_d[2]*alpha**2

plt.plot(alpha, cc_poly_s)
plt.plot(alpha, cc_poly_d, ls='--')


# Add PR3 (non-bps and bps) color-corrections from FITS tables, evaluated on native beta grids
Tdust = 19.6
band_label = f'P{f}'

# Load beta grids
with fits.open(cc_td_hfi) as hdul:
	beta_grid = np.asarray(hdul[1].data[0]['BETA'], dtype=float).ravel()
with fits.open(cc_td_hfi_bps) as hdul:
	beta_grid_bps = np.asarray(hdul[1].data[0]['BETA'], dtype=float).ravel()

# Build interpolators and evaluate cc on the beta grids
interp = functions.interpcc_setup(cc_td_hfi, band_label, td_limit=40, method=3)
interp_bps = functions.interpcc_setup(cc_td_hfi_bps, band_label, td_limit=40, method=3)
cc_pr3 = np.array([functions.interpcc(interp, Tdust, b) for b in beta_grid], dtype=float)
cc_pr3_bps = np.array([functions.interpcc(interp_bps, Tdust, b) for b in beta_grid_bps], dtype=float)

# Convert beta grids to alpha grids (alpha = beta + 2) to overlay with alpha-polynomial curves
alpha_grid = beta_grid + 2.0
alpha_grid_bps = beta_grid_bps + 2.0

plt.plot(alpha_grid, cc_pr3, label=f"PR3 (Td={Tdust}K)")
plt.plot(alpha_grid_bps, cc_pr3_bps, ls=':', label=f"PR3 bps (Td={Tdust}K)")
plt.legend()
plt.xlabel(r'alpha = 2 + beta')
plt.ylabel('Color correction factor')
plt.title(f'Band {band_label}')


alpha = -1.1
cc_d_s = cc_dict_11_s[0] + cc_dict_11_s[1]*alpha + cc_dict_11_s[2]*alpha**2
cc_f_s = functions.fastcc(f'P{f}', alpha=alpha)

print('=================')
print(cc_d_s)
print(cc_f_s)
print('=================')


