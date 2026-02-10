#%%
import numpy as np
import healpy as hp
import os
from tqdm import tqdm
import pymaster as nmt
import sys
import matplotlib.pyplot as plt
try:
	from astropy.io import fits
except Exception:
	fits = None

full_spn = '/home/pablo/Desktop/master/tfm/spectra/spectra_full_quijote_galcut10_5_skyplusnoise_full_bin_20-199.fits'
full_n = '/home/pablo/Desktop/master/tfm/spectra/spectra_full_quijote_galcut10_5_noise_full_bin_20-199.fits'


def describe_fits(path: str):
	"""Open a FITS file and print a concise summary.

	Returns a dictionary with any spectra arrays that can be detected.
	Tries common layouts:
	- Binary table with columns like ell/ELL + TT/EE/BB/TE/TB/EB
	- Image HDUs with arrays per spectrum component
	"""
	if not os.path.exists(path):
		print(f"[missing] {path}")
		return None

	spectra = {
		"ell": None,
		"TT": None,
		"EE": None,
		"BB": None,
		"TE": None,
		"TB": None,
		"EB": None,
		"_tables": [],
		"_images": [],
	}

	if fits is None:
		print("[warning] astropy not available; cannot read FITS. Install with 'pip install astropy'.")
		return None

	try:
		with fits.open(path, memmap=True) as hdul:
			print(f"\nFile: {path}")
			print(f"HDU count: {len(hdul)}")
			# Print a compact info line per HDU
			for i, h in enumerate(hdul):
				kind = h.__class__.__name__
				shape = None
				if hasattr(h, "data") and h.data is not None:
					try:
						shape = tuple(h.data.shape)
					except Exception:
						shape = "?"
				name = h.name if hasattr(h, "name") else ""
				print(f"  [{i:02d}] {kind:<15} name='{name}' shape={shape}")

			# Attempt to extract spectra from common layouts
			# 1) Binary tables with spectral columns
			for h in hdul:
				if isinstance(h, fits.BinTableHDU):
					cols = [c.name for c in h.columns]
					table_dict = {c: h.data[c] for c in cols}
					spectra["_tables"].append({"name": h.name, "columns": cols})

					# Locate ell column (case-insensitive)
					ell_key = next((c for c in cols if c.lower() in ("ell", "l", "ells")), None)
					if ell_key is not None and spectra["ell"] is None:
						spectra["ell"] = np.asarray(h.data[ell_key], dtype=float)

					# Map common spectra names
					for key in ["TT", "EE", "BB", "TE", "TB", "EB"]:
						# Look for exact or case-insensitive matches; also allow CL_TT, C_TT, etc.
						cand = next(
							(
								c
								for c in cols
								if c.lower() in (key.lower(), f"cl_{key.lower()}", f"c_{key.lower()}")
							),
							None,
						)
						if cand is not None and spectra[key] is None:
							try:
								spectra[key] = np.asarray(h.data[cand], dtype=float)
							except Exception:
								pass

			# 2) Image HDUs: try to infer by name or index
			# Common patterns: one HDU per spectrum component, with name TT/EE/...
			image_components = {}
			for h in hdul:
				if isinstance(h, (fits.ImageHDU, fits.PrimaryHDU)) and h.data is not None:
					name = (h.name or "").upper()
					arr = np.asarray(h.data)
					spectra["_images"].append({"name": h.name, "shape": tuple(arr.shape)})
					if arr.ndim == 1:
						# If 1D, could be ell or a spectrum
						if name in {"ELL", "L", "ELLS"} and spectra["ell"] is None:
							spectra["ell"] = arr.astype(float)
						elif name in {"TT", "EE", "BB", "TE", "TB", "EB"} and spectra.get(name) is None:
							spectra[name] = arr.astype(float)

			# If no ell found but spectra exist, create a default ell
			if spectra["ell"] is None:
				# find a representative length
				for k in ["TT", "EE", "BB", "TE", "TB", "EB"]:
					v = spectra.get(k)
					if isinstance(v, np.ndarray):
						spectra["ell"] = np.arange(v.size)
						break

			# Brief report of what we found
			found = [k for k in ["TT", "EE", "BB", "TE", "TB", "EB"] if isinstance(spectra.get(k), np.ndarray)]
			print(f"Found columns: ell={'yes' if spectra['ell'] is not None else 'no'}, spectra={found}")
			for k in ["TT", "EE", "BB", "TE", "TB", "EB"]:
				v = spectra.get(k)
				if isinstance(v, np.ndarray):
					print(f"  {k}: shape={v.shape}, finite={np.isfinite(v).sum()}/{v.size}")

			return spectra
	except Exception as e:
		print(f"[error] Failed to read {path}: {e}")
		return None


if __name__ == "__main__":
	paths = [
		("sky+noise", full_spn),
		("noise-only", full_n),
	]
	for label, p in paths:
		print(f"\n=== Reading {label} spectra ===")
		result = describe_fits(p)
		if result is None:
			continue
		# Example: access arrays if needed
		ell = result.get("ell")
		EE = result.get("EE")
		BB = result.get("BB")
		TE = result.get("TE")
		# Keep it read-only/summary by default; plotting can be enabled if desired.

