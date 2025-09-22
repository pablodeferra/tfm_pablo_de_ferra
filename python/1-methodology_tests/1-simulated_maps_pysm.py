#%%
import pysm3
import healpy as hp
from astropy import units as u
import os
from data import data

nside = 512
path_save = '/home/pablo/Desktop/Paper/maps/PYSM/'

def generate_sky_maps(nside, path_save, experiment_select="all", band_select="all"):
    """Generate IQU maps for all experiments and bands in data.

    Parameters
    ----------
    nside : int
        Healpy nside for the maps.
    path_save : str
        Folder path to save the maps.
    experiment_select : str, default "all"
        If "all", generate maps for all experiments. Otherwise, only for the given experiment.
    band_select : str, list of str, or "all", default "all"
        If "all", generate maps for all bands. Otherwise, only for the given band(s).
    """
    # Sky model
    sky = pysm3.Sky(nside=nside, preset_strings=['s1', 'd1', 'a1', 'f1', 'c1'])

    # Loop over experiments
    for experiment, bands in data.items():
        if experiment_select != "all" and experiment != experiment_select:
            continue  # skip other experiments

        print(f"\nGenerating simulated maps for {experiment}:")
        for band_name, band_info in bands.items():
            # --- filtro de bandas ---
            if band_select != "all":
                if isinstance(band_select, (list, tuple, set)):
                    if band_name not in band_select:
                        continue
                else:  # caso string
                    if band_name != band_select:
                        continue

            print(f"\n[{experiment}] Band {band_name} GHz:")
            freq_band = band_info['freq'].to(u.GHz)   # frequency in GHz
            fwhm_band = band_info['fwhm'].to(u.rad)   # FWHM in radians

            # Generate map using PySM
            map_IQU = sky.get_emission(freq=freq_band)

            # Smoothing
            map_IQU_fwhm = pysm3.apply_smoothing_and_coord_transform(
                map_IQU, fwhm=fwhm_band
            )

            # Name and save
            freq_str = str(band_name)
            fwhm_str = f"{int(round(band_info['fwhm'].value * 100)):04d}"
            filename = f"total_{freq_str}GHz_n{nside}_fwhm_{fwhm_str}.fits"

            hp.write_map(os.path.join(path_save, filename), map_IQU_fwhm, overwrite=True)
            print(f"Saved: {filename}")


# generate_sky_maps(nside, path_save)
generate_sky_maps(nside, path_save, experiment_select='WMAP', band_select=['23', '33', '41', '61', '94'])
generate_sky_maps(nside, path_save, experiment_select='Planck', band_select=['30', '44', '70', '100', '143', '217', '353'])