#%%
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

	# Note: monopole removal handled by `hp.mollview(remove_mono=True)` below

	cmap = get_planck_cmb_cmap(use_planck_cmap=use_planck_cmap, txt_path=planck_cmap_txt)

	fig = plt.figure(figsize=(10, 6))
	hp.mollview(
		map_i_sm,
		title="",
		cmap=cmap,
		norm=norm,
		unit="K",
		fig=fig.number,
		remove_mono=True,
		remove_dip=False,
	)

	if output_path:
		Path(output_path).parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(output_path, dpi=300, bbox_inches="tight")

	if show:
		plt.show()

	if return_map:
		return fig, map_i_sm
	return fig
def plot_band_mollview(
	band,
	experiment: Optional[str] = None,
	fwhm_deg: float = 1.0,
	component: str = "I",  # 'I', 'Q', 'U', or 'P' (=sqrt(Q^2+U^2))
	use_planck_cmap: bool = True,
	planck_cmap_txt: str = "/home/pablo/Desktop/master/tfm/Planck_Parchment_RGB.txt",
	norm: str = "hist",
	save_path: Optional[str] = None,
	png_transparent: bool = False,
	show: bool = True,
	return_map: bool = False,
):
	"""Generic HEALPix mollview plotter for any band in the `data` dict.

	Parameters
	- band: frequency band as str or int (e.g. '11' or 353)
	- experiment: optional experiment name (e.g. 'Planck', 'WMAP', 'QUIJOTE'). If None,
	  the function will search `data` for the first experiment containing the band.
	- fwhm_deg: output smoothing FWHM in degrees
	- component: one of 'I','Q','U','P' (P = sqrt(Q^2+U^2))
	- save_path: path where to save the final PDF. If None a default path under
	  'figures/maps' will be used. Always saved as PDF.
	- show: whether to plt.show()
	- return_map: if True, returns (fig, smoothed_map)
	"""

	import healpy as hp
	import matplotlib.pyplot as plt
	from data import data

	b_str = str(band)

	# Find experiment if not provided
	if experiment is None:
		candidates = [exp for exp in data.keys() if b_str in data[exp]]
		if not candidates:
			raise ValueError(f"Band {b_str} not found in data dictionary for any experiment")
		# Prefer Planck if available
		if "Planck" in candidates:
			experiment = "Planck"
		else:
			experiment = candidates[0]

	if experiment not in data or b_str not in data[experiment]:
		raise ValueError(f"Band {b_str} not found for experiment '{experiment}' in data")

	entry = data[experiment][b_str]
	map_path = str(entry["path"]) if "path" in entry else str(entry.get("map_path", ""))
	if not Path(map_path).exists():
		raise FileNotFoundError(f"Map not found at: {map_path}")

	# Read requested component
	if component in ("I", "Q", "U"):
		field_map = {"I": 0, "Q": 1, "U": 2}
		hp_map = hp.read_map(map_path, field=field_map[component], verbose=False)
	elif component == "P":
		# Read Q and U, clean them (remove monopole/dipole), smooth them, then
		# compute polarization amplitude P = sqrt(Q^2 + U^2). Doing the
		# monopole/dipole removal on Q and U (and smoothing them) avoids
		# introducing dipole-like artifacts in the nonlinear P map.
		res = hp.read_map(map_path, field=[1, 2], verbose=False)
		# hp.read_map may return a tuple/list (q,u) or a 2xN array; handle both.
		if isinstance(res, (list, tuple)) and len(res) == 2:
			q, u = res
		elif isinstance(res, np.ndarray):
			if res.ndim == 1:
				raise ValueError("read_map returned a 1-D array when two fields were requested")
			# common shapes: (2, npix) or (npix, 2)
			if res.shape[0] == 2:
				q, u = res[0], res[1]
			elif res.shape[1] == 2:
				q, u = res[:, 0], res[:, 1]
			else:
				raise ValueError(f"Unexpected array shape returned by read_map: {res.shape}")
		else:
			raise ValueError("Unexpected return type from hp.read_map when requesting Q/U fields")
		# Build a validity mask: pixels are valid only where both Q and U are finite
		# and not equal to healpy's UNSEEN sentinel. We'll smooth only using
		# finite/non-UNSEEN data (replacing invalid pixels with 0 for the
		# smoothing kernel) and then reapply the mask so the final P map keeps
		# hp.UNSEEN in originally-unseen pixels.
		try:
			UNSEEN = hp.UNSEEN
		except Exception:
			# fallback numeric value if hp.UNSEEN is not available for some reason
			UNSEEN = -1.6375e30

		mask_q = np.isfinite(q) & (q != UNSEEN)
		mask_u = np.isfinite(u) & (u != UNSEEN)
		mask_valid = mask_q & mask_u

		# Prepare arrays for smoothing: replace invalid pixels with 0.0 so the
		# smoothing kernel does not get NaNs or UNSEEN values.
		q_for_smooth = q.copy()
		u_for_smooth = u.copy()
		q_for_smooth[~mask_q] = 0.0
		u_for_smooth[~mask_u] = 0.0

		# Safe helpers for monopole/dipole removal that handle different return
		# signatures across healpy versions (some return (map, coeffs)). If the
		# functions aren't available, just keep the original map.
		def _safe_remove_mono(m):
			try:
				return hp.remove_mono(m)
			except Exception:
				return m

		def _safe_remove_dip(m):
			try:
				out = hp.remove_dip(m)
				# remove_dip may return (map, dipole) or map directly
				if isinstance(out, (list, tuple)):
					return out[0]
				return out
			except Exception:
				return m

		# Apply monopole/dipole removal to the arrays prepared for smoothing
		# (this preserves the original masks in `mask_valid`).
		q_for_smooth = _safe_remove_mono(q_for_smooth)
		q_for_smooth = _safe_remove_dip(q_for_smooth)
		u_for_smooth = _safe_remove_mono(u_for_smooth)
		u_for_smooth = _safe_remove_dip(u_for_smooth)

		# Smooth Q and U to the requested output beam before combining.
		# Determine input FWHM if available (data entries often store an astropy Quantity)
		try:
			fwhm_in_deg = float((entry["fwhm"]).to("deg").value)
		except Exception:
			fwhm_in_deg = None

		q_sm = smooth_to_fwhm(q_for_smooth, fwhm_out_deg=fwhm_deg, fwhm_in_deg=fwhm_in_deg)
		u_sm = smooth_to_fwhm(u_for_smooth, fwhm_out_deg=fwhm_deg, fwhm_in_deg=fwhm_in_deg)

		# Build final P map: default to hp.UNSEEN and compute sqrt(Q^2+U^2)
		# only on originally-valid pixels so we preserve the unseen sentinel.
		hp_map = np.full_like(q_sm, UNSEEN, dtype=q_sm.dtype)
		if np.any(mask_valid):
			hp_map[mask_valid] = np.sqrt(q_sm[mask_valid] * q_sm[mask_valid] + u_sm[mask_valid] * u_sm[mask_valid])
	else:
		raise ValueError("component must be one of 'I','Q','U','P'")

	# Determine input FWHM if available (data entries often store an astropy Quantity)
	try:
		fwhm_in_deg = float((entry["fwhm"]).to("deg").value)
	except Exception:
		fwhm_in_deg = None

	# Smooth to requested FWHM
	hp_map_sm = smooth_to_fwhm(hp_map, fwhm_out_deg=fwhm_deg, fwhm_in_deg=fwhm_in_deg)

	cmap = get_planck_cmb_cmap(use_planck_cmap=use_planck_cmap, txt_path=planck_cmap_txt)

	fig = plt.figure(figsize=(10, 6))
	# Build keyword args for mollview and set a transparent background when
	# saving PNGs with `png_transparent=True`. Using bgcolor='None' instructs
	# healpy to draw a transparent background.
	moll_kwargs = dict(
		title=f"{experiment} {b_str} {component}",
		cmap=cmap,
		norm=norm,
		unit="K",
		fig=fig.number,
		remove_mono=True,
		remove_dip=False,
	)
	if png_transparent:
		# Transparent PNG: no title, no unit label, no colorbar, and a
		# transparent background.
		moll_kwargs["bgcolor"] = "None"
		moll_kwargs["title"] = ""
		moll_kwargs["unit"] = ""
		moll_kwargs["cbar"] = False
	hp.mollview(hp_map_sm, **moll_kwargs)
	# Draw graticule: white for transparent PNGs, black for PDFs.
	# try:
	# 	graticule_color = "k" if png_transparent else "k"
	# 	hp.graticule(dmer=40, color=graticule_color)
	# except Exception:
	# 	# If healpy version doesn't support these args, ignore the graticule
	# 	pass
	

	# Prepare save path and save file. By default we save PDF, but if
	# `png_transparent` is True we save a transparent PNG and adjust text
	# colors to white so labels/ticks/title are visible against a dark
	# background.
	if save_path is None:
		default = Path(__file__).resolve().parents[1] / "figures" / "maps"
		default.mkdir(parents=True, exist_ok=True)
		suffix = ".png" if png_transparent else ".pdf"
		save_path = default / f"{experiment}_{b_str}_{component}{suffix}"
	else:
		save_path = Path(save_path)
		save_path.parent.mkdir(parents=True, exist_ok=True)
		if png_transparent:
			if save_path.suffix.lower() != ".png":
				save_path = save_path.with_suffix(".png")
		else:
			if save_path.suffix.lower() != ".pdf":
				save_path = save_path.with_suffix(".pdf")

	# If saving a transparent PNG, make text and tick labels white so they
	# are visible on dark backgrounds when overlaid. We change text colors
	# on the figure objects directly to avoid globally mutating rcParams.
	if png_transparent:
		import matplotlib as mpl

		for text_obj in fig.findobj(mpl.text.Text):
			try:
				text_obj.set_color("white")
			except Exception:
				pass

		# Save with transparent background
		fig.savefig(str(save_path), dpi=300, bbox_inches="tight", transparent=True)
	else:
		fig.savefig(str(save_path), dpi=300, bbox_inches="tight")

	if show:
		plt.show()

	if return_map:
		return fig, hp_map_sm
	return fig

band = 23
experiment="WMAP"
component="P"
fwhm_deg=1.2

fig = plot_band_mollview(
	band=band,
	experiment=experiment,
	fwhm_deg=fwhm_deg,
	component=component,
	use_planck_cmap=True,
	planck_cmap_txt="/home/pablo/Desktop/master/tfm/Planck_Parchment_RGB.txt",
	save_path=f"/home/pablo/Desktop/master/tfm/figures_ppt/maps/map_{experiment}{band}_{component}_fwhm{fwhm_deg}.pdf",
    png_transparent=True,
    show=False,
)

#%%
# Overlaid masks on WMAP K-band P map
from data import masks
import healpy as hp
import matplotlib.pyplot as plt

mask_10_path = masks['QUIJOTE_galcut']['galcut10']['path']
mask_15_path = masks['QUIJOTE_galcut']['galcut15']['path']
mask_20_path = masks['QUIJOTE_galcut']['galcut20']['path']

mask_10 = hp.read_map(mask_10_path, verbose=False)
mask_15 = hp.read_map(mask_15_path, verbose=False)
mask_20 = hp.read_map(mask_20_path, verbose=False)

# Obtain the 23 P map using the program's function instead of custom reading
_, wmap_s = plot_band_mollview(
    band=23,
    experiment='WMAP',
    component='P',
    fwhm_deg=1.2,
    use_planck_cmap=False,
    save_path='/tmp/wmap_tmp.pdf',
    show=False,
    return_map=True
)

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

save_path_ppt = '/home/pablo/Desktop/master/tfm/figures_ppt/maps/'
Path(save_path_ppt).mkdir(parents=True, exist_ok=True)

fig_10 = plt.figure(figsize=(10, 6))
hp.mollview(wmap_s, fig = fig_10.number, cmap=cmap, norm='hist', cbar=False, title=None, alpha = np.ones(len(wmap_s)), bgcolor='None',
        remove_mono=True,
        remove_dip=False)
hp.mollview(mask_10, fig = fig_10.number, cmap='gray', cbar=False, title=None, min = -0.25, max = 0.8, alpha = masks_10_inv, bgcolor='None')
hp.graticule(dmer=40)
plt.savefig(save_path_ppt + 'mask_galcut10_wmap_k.png', dpi=800, transparent=True)
plt.close(fig_10)

fig_15 = plt.figure(figsize=(10, 6))
hp.mollview(wmap_s, fig = fig_15.number, cmap=cmap, norm='hist', cbar=False, title=None, alpha = np.ones(len(wmap_s)), bgcolor='None')
hp.mollview(mask_15, fig = fig_15.number, cmap='gray', cbar=False, title=None, min = -0.25, max = 0.8, alpha = masks_15_inv, bgcolor='None')
hp.graticule(dmer=40)
plt.savefig(save_path_ppt + 'mask_galcut15_wmap_k.png', dpi=800, transparent=True)
plt.close(fig_15)

fig_20 = plt.figure(figsize=(10, 6))
hp.mollview(wmap_s, fig = fig_20.number, cmap=cmap, norm='hist', cbar=False, title=None, alpha = np.ones(len(wmap_s)), bgcolor='None')
hp.mollview(mask_20, fig = fig_20.number, cmap='gray', cbar=False, title=None, min = -0.25, max = 0.8, alpha = masks_20_inv, bgcolor='None')
hp.graticule(dmer=40)
plt.savefig(save_path_ppt + 'mask_galcut20_wmap_k.png', dpi=800, transparent=True)
plt.close(fig_20)