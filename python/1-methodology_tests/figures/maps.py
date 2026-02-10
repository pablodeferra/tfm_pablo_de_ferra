#%%
from __future__ import annotations

from pathlib import Path
from typing import Optional
import sys

sys.path.append('../')
import numpy as np


def get_planck_cmb_cmap(
	use_planck_cmap: bool = True,
	txt_path: str = "/home/pablo/Desktop/master/tfm/Planck_Parchment_RGB.txt",
):
	"""Return the Planck CMB colormap (or None if disabled/unavailable)."""

	if not use_planck_cmap:
		return None

	try:
		from matplotlib.colors import ListedColormap

		rgb = np.loadtxt(txt_path) / 255.0
		planck_cmap = ListedColormap(rgb)
		planck_cmap.set_bad("gray")
		planck_cmap.set_under("white")
		return planck_cmap
	except Exception:
		return None


def smooth_to_fwhm(
	healpix_map: np.ndarray,
	fwhm_out_deg: float,
	fwhm_in_deg: Optional[float] = None,
):
	"""Smooth a HEALPix map so the *final* FWHM is fwhm_out_deg.

	If fwhm_in_deg is provided, the additional smoothing is
	$\sqrt{\mathrm{FWHM}_{out}^2 - \mathrm{FWHM}_{in}^2}$.
	If not provided, fwhm_out_deg is used directly as the smoothing kernel.
	"""

	import healpy as hp

	fwhm_out_rad = np.deg2rad(fwhm_out_deg)
	if fwhm_in_deg is None:
		fwhm_extra_rad = fwhm_out_rad
	else:
		fwhm_in_rad = np.deg2rad(fwhm_in_deg)
		fwhm_extra_rad = float(np.sqrt(max(fwhm_out_rad**2 - fwhm_in_rad**2, 0.0)))

	return hp.smoothing(healpix_map, fwhm=fwhm_extra_rad, verbose=False)


def plot_planck_30_intensity_mollview(
	fwhm_deg: float = 1.0,
	use_planck_cmap: bool = True,
	planck_cmap_txt: str = "/home/pablo/Desktop/Fisica/TFG/txts/Planck_Parchment_RGB.txt",
	norm: str = "hist",
	output_path: Optional[str] = None,
	show: bool = True,
	return_map: bool = False,
):
	"""Read Planck 30 GHz Intensity, smooth to 1 degree, and display with mollview."""

	import healpy as hp
	import matplotlib.pyplot as plt

	from data import data

	map_path = data["Planck"]["30"]["path"]
	map_path = str(map_path)
	if not Path(map_path).exists():
		raise FileNotFoundError(
			f"Planck 30 GHz map not found at: {map_path} (check `data.py:path_map`)"
		)

	# Planck LFI SkyMap FITS uses I,Q,U in fields 0,1,2.
	map_i = hp.read_map(map_path, field=0, verbose=False)

	# Native beam from `data.py` is 32.29 arcmin for 30 GHz.
	fwhm_in_deg = float((data["Planck"]["30"]["fwhm"]).to("deg").value)
	map_i_sm = smooth_to_fwhm(map_i, fwhm_out_deg=fwhm_deg, fwhm_in_deg=fwhm_in_deg)

	cmap = get_planck_cmb_cmap(use_planck_cmap=use_planck_cmap, txt_path=planck_cmap_txt)

	fig = plt.figure(figsize=(10, 6))
	hp.mollview(
		map_i_sm,
		title="",
		cmap=cmap,
		norm=norm,
		unit="K",
		fig=fig.number,
	)

	if output_path:
		Path(output_path).parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(output_path, dpi=300, bbox_inches="tight")

	if show:
		plt.show()

	if return_map:
		return fig, map_i_sm
	return fig


if __name__ == "__main__":
	fig = plot_planck_30_intensity_mollview(
		fwhm_deg=1.0,
		use_planck_cmap=True,
		planck_cmap_txt="/home/pablo/Desktop/master/tfm/Planck_Parchment_RGB.txt",
		show=True,
	)
	fig.savefig(
		"/home/pablo/Desktop/master/tfm/figures/maps/planck_30_intensity_mollview.pdf",
		dpi=300,
		bbox_inches="tight",
	)