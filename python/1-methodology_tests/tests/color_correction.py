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

# Summary prints: per-band color-correction factors
# - Synchrotron at alpha_s = -1.0
# - Dust at alpha_d = 3.5

#%%

def _eval_poly(poly, alpha_val):
	return float(poly[0] + poly[1]*alpha_val + poly[2]*(alpha_val**2))

alpha_s_print = -1.0
alpha_d_print = 3.5

print("Synchrotron color corrections at alpha_s = -1.0:")
for band in sorted(cc_dict.get('synch', {}).keys(), key=lambda x: float(x)):
	poly = cc_dict['synch'][band]
	val = _eval_poly(poly, alpha_s_print)
	print(f"  {band} GHz: {val:.6f}")

print("Dust color corrections at alpha_d = 3.5:")
for band in sorted(cc_dict.get('dust', {}).keys(), key=lambda x: float(x)):
	poly = cc_dict['dust'][band]
	val = _eval_poly(poly, alpha_d_print)
	print(f"  {band} GHz: {val:.6f}")


#%%

p_100 = [0.99868757, -0.00512203, -0.00428818, 100.0]
p_143 = [1.0125835,   0.00767883, -0.00468418, 143.0]
p_217 = [0.98670965, -0.01582359, -0.00362294, 217.0]
p_353 = [0.98479307, -0.0174746,  -0.00318638, 353.0]


# Compare HFI dust color-correction values and polynomials
bands_hfi = ['100', '143', '217', '353']
fastcc_polys = {
	'100': p_100[:3],
	'143': p_143[:3],
	'217': p_217[:3],
	'353': p_353[:3],
}

print("\nHFI dust color-correction polynomials (fastcc vs cc_dict):")
for band in bands_hfi:
	poly_fast = fastcc_polys[band]
	poly_dict = list(cc_dict['dust'][band])
	print(f"{band} fastcc: {poly_fast}")
	print(f"{band} cc_dict: {poly_dict}")

print("\nHFI dust color-correction values at alpha_d = 3.5 (fastcc vs cc_dict):")
for band in bands_hfi:
	val_fast = _eval_poly(fastcc_polys[band], alpha_d_print)
	val_dict = _eval_poly(cc_dict['dust'][band], alpha_d_print)
	# Print just the values (fastcc_value cc_dict_value)
	print(f"{band}: {val_fast:.6f}  {val_dict:.6f}")

#%%
# Interpolate HFI PR3-bps table to Tdust and fit polynomial; compare value at alpha_dust=3.5
TDUST = 19.6
alpha_d_fit = 3.5
beta_d_fit = alpha_d_fit - 2.0

def beta_to_alpha_poly(a0b, a1b, a2b):
	"""Convert polynomial in beta to polynomial in alpha, with alpha=beta+2.
	Q(alpha) = a0 + a1*(alpha-2) + a2*(alpha-2)^2
			 = (a0 - 2*a1 + 4*a2) + (a1 - 4*a2)*alpha + a2*alpha^2
	Returns [A0, A1, A2].
	"""
	A0 = a0b - 2.0*a1b + 4.0*a2b
	A1 = a1b - 4.0*a2b
	A2 = a2b
	return [A0, A1, A2]

print("\nInterpolation-based HFI dust polynomials (alpha-domain) and values vs table:")
print(f"Tdust = {TDUST}, alpha_dust = {alpha_d_fit}, beta_dust = {beta_d_fit}")

for band in bands_hfi:
	# Load beta grid and verify band exists in table
	with fits.open(cc_td_hfi_bps) as hdul:
		bands_arr = hdul[1].data[0][0]
		beta_grid = np.asarray(hdul[1].data[0][2], dtype=float).ravel()
	idx_band = [ii for ii, bb in enumerate(bands_arr) if str(band) == bb]
	if not idx_band:
		print(f"[WARN] Band {band} not found in HFI CC table")
		continue

	# Build cc(β, TDUST) using the table interpolator (method=3) for all β
	interp_bps_all = functions.interpcc_setup(cc_td_hfi_bps, f"P{band}", td_limit=40, method=3)
	cc_vals_beta = np.array([functions.interpcc(interp_bps_all, TDUST, b) for b in beta_grid], dtype=float)
	# Fit quadratic in beta
	p_beta = np.polyfit(beta_grid, cc_vals_beta, deg=2)
	a2b, a1b, a0b = map(float, p_beta)
	# Convert to alpha-domain coefficients
	A0, A1, A2 = beta_to_alpha_poly(a0b, a1b, a2b)

	# Value at alpha_dust from fitted poly (alpha-domain)
	val_poly_alpha = _eval_poly([A0, A1, A2], alpha_d_fit)

	# Value from direct table interpolation at (Tdust, beta_d_fit)
	val_table = float(functions.interpcc(interp_bps_all, TDUST, beta_d_fit))

	# print(f"{band} alpha-poly: [{A0:.8f}, {A1:.8f}, {A2:.8f}]")
	# print(f"{band} values: poly={val_poly_alpha:.6f}  table={val_table:.6f}")

	fastcc_cc = functions.fastcc(f'P{band}', alpha = alpha_d_fit)
	print(f'{band} GHz cc values:')
	print(f'fastcc = {fastcc_cc:.6f}')
	print(f'interpcc = {val_table:.6f}')
	print(f'interpolated polynom = {val_poly_alpha:.6f}')