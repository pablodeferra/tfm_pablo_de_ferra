#%%
import os
import numpy as np
from astropy.io import fits

# Local project modules
import sys
sys.path.append('./')
from data import path_cc

# Configuration
TDUST = 19.6

# Bands mapping: internal codes used by fastcc and friendly experiment/band labels
SYNCH_CODES = {
    "QUIJOTE": {"11": "Q11", "13": "Q13", "17": "Q17", "19": "Q19"},
    "WMAP": {"23": "WK", "33": "WKa", "41": "WQ", "61": "WV", "94": "WW"},
    "Planck": {"30": "P30", "44": "P44", "70": "P70", "100": "P100", "143": "P143", "217": "P217", "353": "P353"},
}
HFI_DUST_BANDS = ["100", "143", "217", "353"]

# Helper: fit quadratic to y(x)
def fit_quadratic(x, y):
    """Return coefficients [a0, a1, a2] for y ≈ a0 + a1*x + a2*x^2."""
    p = np.polyfit(x, y, deg=2)
    a2, a1, a0 = map(float, p)
    return [a0, a1, a2]

# Build synch polynomials from fastcc (exact coefficients)
synch_rows = []
"""Exact synch coefficients from frequencies_v3 (functions.fastcc), copied here
to avoid importing heavy dependencies. Values are [a0, a1, a2, nu_center]."""
FREQ_V3 = {
    # QUIJOTE
    'Q11': [0.9821236098857145, 0.011920227628586971, -0.0015347819996889738, 11.1],
    'Q13': [1.0009126144672769, 0.001850712851951476, -0.0011606369297726346, 12.9],
    'Q17': [1.0067957381497863, -0.0019621413830759947, -0.000715949539348768, 16.8],
    'Q19': [1.0081384744982096, -0.002537677057807221, -0.0007667684588267012, 18.8],
    # WMAP (band averages)
    'WK':  [0.972902, 0.0190469, -0.00276464, 22.8],
    'WKa': [0.983787, 0.0117567, -0.00183716, 33.0],
    'WQ':  [0.996854, 0.00496893, -0.00181359, 40.6],
    'WV':  [0.980322, 0.0143631, -0.00223596, 60.8],
    'WW':  [0.984848, 0.0112743, -0.00164595, 93.5],
    # Planck
    'P30':  [1.00513, 0.00301399, -0.00300699, 28.4],
    'P44':  [0.994769, 0.00596703, -0.00173626, 44.1],
    'P70':  [0.989711, 0.0106943, -0.00328671, 70.4],
    'P100': [0.99868757, -0.00512203, -0.00428818, 100.0],
    'P143': [1.0125835, 0.00767883, -0.00468418, 143.0],
    'P217': [0.98670965, -0.01582359, -0.00362294, 217.0],
    'P353': [0.98479307, -0.0174746, -0.00318638, 353.0],
}

for exp, bands in SYNCH_CODES.items():
    for band_label, code in bands.items():
        a0, a1, a2 = map(float, FREQ_V3[code][:3])
        synch_rows.append((exp, band_label, 'synch', 'alpha', a0, a1, a2, np.nan))

# Dust non-HFI: same coefficients as synch (requested)
dust_nonhfi_rows = []
for exp, bands in SYNCH_CODES.items():
    for band_label, code in bands.items():
        a0, a1, a2 = map(float, FREQ_V3[code][:3])
        dust_nonhfi_rows.append((exp, band_label, 'dust_nonHFI', 'alpha', a0, a1, a2, np.nan))

# HFI dust from table at Tdust
cc_td_hfi_path = os.path.join(path_cc, 'c_td_10-40_beta_hfi_pr3.fits')
# Read native grids (bands, Tdust, beta) and slice to Planck case
with fits.open(cc_td_hfi_path) as hdul:
    bands_arr = hdul[1].data[0][0]
    td_arr = np.asarray(hdul[1].data[0][1], dtype=float)
    beta_grid = np.asarray(hdul[1].data[0][2], dtype=float).ravel()
    # map_cc shape handling per functions.interpcc_setup
    # For Planck bands: dat[1].data[0][3][1][idx_band]
    full_maps = hdul[1].data[0][3]

hfi_dust_rows = []
for band_label in HFI_DUST_BANDS:
    # Locate band index in FITS table
    idx_band = [ii for ii, bb in enumerate(bands_arr) if str(band_label) == bb]
    if not idx_band:
        raise ValueError(f"Band {band_label} not found in HFI CC table")
    idx = idx_band[0]
    # Planck-specific slice
    with fits.open(cc_td_hfi_path) as hdul:
        full_maps = hdul[1].data[0][3]
        map_cc = np.asarray(full_maps[1][idx], dtype=float)  # shape ~ [len(beta), len(td)]
        td_arr = np.asarray(hdul[1].data[0][1], dtype=float)
        beta_grid = np.asarray(hdul[1].data[0][2], dtype=float).ravel()

    # Select temperature window up to 40 K (as in functions)
    sel_td = td_arr <= 40.0
    td_sel = td_arr[sel_td]
    Z = map_cc[:, sel_td].T  # shape (len(td_sel), len(beta))

    # Interpolate along temperature to get values at TDUST for all beta
    if TDUST <= td_sel.min() or TDUST >= td_sel.max():
        # Clamp to bounds if outside
        td_used = td_sel[np.argmin(np.abs(td_sel - TDUST))]
        row = Z[np.argmin(np.abs(td_sel - TDUST)), :]
    else:
        # Find bracketing indices
        i = np.searchsorted(td_sel, TDUST) - 1
        i = max(0, min(i, len(td_sel) - 2))
        t0, t1 = td_sel[i], td_sel[i+1]
        w = (TDUST - t0) / (t1 - t0)
        row = (1.0 - w) * Z[i, :] + w * Z[i+1, :]

    cc_vals = np.asarray(row, dtype=float)
    a0, a1, a2 = fit_quadratic(beta_grid, cc_vals)
    hfi_dust_rows.append(('Planck', band_label, 'dust_HFI', 'beta', float(a0), float(a1), float(a2), float(TDUST)))

# Build FITS binary table
rows = synch_rows + dust_nonhfi_rows + hfi_dust_rows
exp_col = fits.Column(name='EXPERIM', format='20A', array=np.array([r[0] for r in rows]))
band_col = fits.Column(name='BAND', format='8A', array=np.array([r[1] for r in rows]))
comp_col = fits.Column(name='COMP', format='12A', array=np.array([r[2] for r in rows]))
var_col = fits.Column(name='VAR', format='8A', array=np.array([r[3] for r in rows]))
a0_col = fits.Column(name='A0', format='D', array=np.array([r[4] for r in rows], dtype=float))
a1_col = fits.Column(name='A1', format='D', array=np.array([r[5] for r in rows], dtype=float))
a2_col = fits.Column(name='A2', format='D', array=np.array([r[6] for r in rows], dtype=float))
td_col = fits.Column(name='TDUST', format='D', array=np.array([r[7] for r in rows], dtype=float))

hdul = fits.HDUList([
    fits.PrimaryHDU(),
    fits.BinTableHDU.from_columns([exp_col, band_col, comp_col, var_col, a0_col, a1_col, a2_col, td_col], name='CC_POLYNOMS')
])

os.makedirs(path_cc, exist_ok=True)
out_path = os.path.join(path_cc, 'cc_polynoms.fits')
hdul.writeto(out_path, overwrite=True)
print(f"[OK] Saved polynomial coefficients to: {out_path}")


