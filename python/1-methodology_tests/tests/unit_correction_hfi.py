#%%
import numpy as np
from astropy.io import fits
import sys
sys.path.append('../') 
import functions as functions
import matplotlib.pyplot as plt

# Compare unit conversion KCMB->KRJ for Planck HFI bands using:
# - Generic thermodynamic conversion (cmb_unit_conversion)
# - HFI-specific conversion with bandpass shifts (planck_uc_hfi, use_bps=True)

# HFI polarization bands to compare
hfi_bands = np.array([100.0, 143.0, 217.0, 353.0], dtype=float)

# Generic KCMB->KRJ conversion (per frequency)
uc_generic = np.array([functions.cmb_unit_conversion(nu, option='KCMB2KRJ') for nu in hfi_bands], dtype=float)

# HFI-specific KCMB->KRJ conversion (returned in fixed order [100,143,217,353,545,857])
uc_hfi_all = functions.planck_uc_hfi(use_bps=True)
uc_hfi_sel = uc_hfi_all[:4]  # select 100, 143, 217, 353

# Print comparison table
print("[Unit conversion KCMB->KRJ comparison for Planck HFI (use_bps=True)]")
print("band  generic    hfi_bps   ratio(hfi/generic)  diff(%)")
for band, gen, hfi in zip(hfi_bands, uc_generic, uc_hfi_sel):
	ratio = hfi / gen if gen != 0 else np.nan
	diff_pct = 100.0 * (hfi - gen) / gen if gen != 0 else np.nan
	print(f"{int(band):>4d}  {gen:10.6f}  {hfi:10.6f}  {ratio:17.6f}  {diff_pct:8.3f}")


