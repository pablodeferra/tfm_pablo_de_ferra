#%%
import os
import healpy as hp
import numpy as np
from astropy.io import fits
import functions
from data import data, masks
from scipy.constants import c, h, k

mask_select = masks['quijote_galcut']['galcut10']
mask_name = mask_select['name']

def get_beam_for_band(band_name, data, ell_eff):
    """
    Return the interpolated beam transfer functions for a given frequency band.

    Parameters
    ----------
    band_name : str
        Name of the frequency band, e.g., '11', '30', '100'.
    data : dict
        Dictionary containing experiment and band information, including beam file paths.
    ell_eff : array_like
        Array of effective multipoles at which to interpolate the beam.

    Returns
    -------
    beam_interp : dict of numpy.ndarray
        Dictionary with keys 'T','E','B'. Each is the interpolated beam
        transfer function at the effective multipoles `ell_eff`.
    """


    # QUIJOTE
    if band_name in data.get('QUIJOTE', {}):
        with fits.open(data['QUIJOTE'][band_name]['beam']) as hdul:
            beam_hdu = hdul[1]
            col_map = {
                "11": "Bl_311",
                "13": "Bl_313",
                "17": "Bl_417",  # or "Bl_217"
                "19": "Bl_419",  # or "Bl_219"
            }
            colname = col_map[band_name]
            beam_arr = beam_hdu.data[colname][0]
            beam_interp = np.interp(ell_eff, np.arange(len(beam_arr)), beam_arr)
        return {"T": beam_interp, "E": beam_interp, "B": beam_interp}

    # WMAP
    elif band_name in data.get('WMAP', {}):
        beam_arr = np.loadtxt(data['WMAP'][band_name]['beam']).T[1]  # second column
        beam_interp = np.interp(ell_eff, np.arange(len(beam_arr)), beam_arr)
        return {"T": beam_interp, "E": beam_interp, "B": beam_interp}

    # Planck
    elif band_name in data.get('Planck', {}):
        if int(band_name) <= 70:  # LFI
            hdul = fits.open(data['Planck'][band_name]['beam'])
            beam_hdu = hdul[f'BEAMWF_0{band_name}X0{band_name}']
            Bl = beam_hdu.data['BL']
            beam_interp = np.interp(ell_eff, np.arange(len(Bl)), Bl)
            hdul.close()
            return {"T": beam_interp, "E": beam_interp, "B": beam_interp}
        else:  # HFI
            hdul = fits.open(data['Planck'][band_name]['beam'])
            window_hdu = hdul['WINDOW FUNCTIONS']
            Bl_T = np.interp(ell_eff, np.arange(len(window_hdu.data['T'])), window_hdu.data['T'])
            Bl_E = np.interp(ell_eff, np.arange(len(window_hdu.data['E'])), window_hdu.data['E'])
            Bl_B = np.interp(ell_eff, np.arange(len(window_hdu.data['B'])), window_hdu.data['B'])
            hdul.close()
            return {"T": Bl_T, "E": Bl_E, "B": Bl_B}
    else:
        raise ValueError(f"Band '{band_name}' not found in data.")

def cmb_unit_conversion(nuGHz, option='KCMB2KRJ', help=False):
    """
    Compute conversion factors between CMB thermodynamic temperature units,
    Rayleigh-Jeans temperature, and surface brightness in Jy/sr.

    Parameters
    ----------
    nuGHz : float
        Frequency in GHz.
    option : str, optional
        Conversion type. One of:
            'KCMB2KRJ'   : From Kelvin_CMB to Kelvin_RJ
            'KRJ2KCMB'   : From Kelvin_RJ to Kelvin_CMB
            'KCMB2Jysr'  : From Kelvin_CMB to Jy/sr
            'Jysr2KCMB'  : From Jy/sr to Kelvin_CMB
            'KRJ2Jysr'   : From Kelvin_RJ to Jy/sr
            'Jysr2KRJ'   : From Jy/sr to Kelvin_RJ
    help : bool, optional
        If True, prints available options. Default is False.

    Returns
    -------
    fac : float
        Conversion factor for the chosen option.
    """

    Tcmb = 2.72548

    cases = ['KCMB2KRJ', 'KRJ2KCMB', 'KCMB2Jysr', 'Jysr2KCMB', 'KRJ2Jysr', 'Jysr2KRJ']
    if help:
        print('  Syntax -- cmb_unit_conversion(nuGHz, option=)')
        print('  Possible options are', cases)

    nu = nuGHz * 1e9
    x = h * nu / (k * Tcmb)
    thermo = x**2 * np.exp(x) / (np.exp(x) - 1.)**2
    rj = (2.0 * k * nu**2 / c**2) * 1e26

    if option == 'KCMB2KRJ':
        fac = thermo
    elif option == 'KRJ2KCMB':
        fac = 1 / thermo
    elif option == 'KCMB2Jysr':
        fac = thermo * rj
    elif option == 'Jysr2KCMB':
        fac = 1 / (thermo * rj)
    elif option == 'KRJ2Jysr':
        fac = rj
    elif option == 'Jysr2KRJ':
        fac = 1 / rj
    else:
        print("Units not identified. Returning -1")
        fac = -1

    return fac

def correct_power_spectra(path_spectra, path_avg_std_skyplusnoise, path_avg_std_noise,
                          band_list, data, nside,
                          correct_beam=True, correct_unit=True, correct_pixel=True,
                          save=False, path_out_file=None):
    """
    Correct power spectra for specified band pairs using beam, unit, and pixel windows.
    Subtract noise spectra, compute corrected error bars, and optionally save to FITS.

    Parameters
    ----------
    path_spectra : str
        Path to FITS file with the original sky+noise spectra (contains ell_eff).
    path_avg_std_skyplusnoise : str
        Path to FITS file with avg+std sky+noise spectra (for error calculation).
    path_avg_std_noise : str
        Path to FITS file with avg+std noise spectra.
    band_list : list of str
        List of frequency bands to include (e.g., ['11','30']).
    data : dict
        Experiment/band information including frequencies and beam paths.
    nside : int
        HEALPix nside for pixel window computation.
    correct_beam : bool, optional
        Whether to apply beam window correction. Default is True.
    correct_unit : bool, optional
        Whether to apply unit conversion. Default is True.
    correct_pixel : bool, optional
        Whether to apply pixel window correction. Default is True.
    save : bool, optional
        Whether to save the output FITS file. Default is False.
    path_out_file : str, optional
        Path to output FITS file. If None and save=True, defaults to "corrected_cls.fits".

    Returns
    -------
    corrected_cls : dict
        Dictionary with corrected spectra and errors, e.g.:
            corrected_cls['band1_band2']['EE']['SPECTRUM']
            corrected_cls['band1_band2']['EE']['ERROR']
            corrected_cls['band1_band2']['ell_eff']
    out_file : str or None
        Path to saved FITS file if save=True, else None.
    """

    if save and path_out_file is None:
        path_out_file = "corrected_cls.fits"

    # Load spectra
    spectra = functions.read_spectra_from_fits(path_spectra, band_list)
    avg_std_skyplusnoise = functions.read_spectra_from_fits(path_avg_std_skyplusnoise, band_list)
    avg_std_noise = functions.read_spectra_from_fits(path_avg_std_noise, band_list)

    ell_eff = next(iter(spectra.values()))['ell_eff']

    # Pixel window
    if correct_pixel:
        wpix = hp.pixwin(nside)
        wp_interp = np.interp(ell_eff, np.arange(len(wpix)), wpix)
    else:
        wp_interp = np.ones_like(ell_eff)

    # Precompute factors for each band
    all_bands = set()
    for key in spectra.keys():
        band1, band2 = key.split('_')
        all_bands.update([band1, band2])

    beam_dict, unit_dict, wp_dict = {}, {}, {}
    for band in all_bands:
        for exp in data:
            if band in data[exp]:
                beam_dict[band] = get_beam_for_band(band, data, ell_eff) if correct_beam else {"T": np.ones_like(ell_eff), "E": np.ones_like(ell_eff), "B": np.ones_like(ell_eff)}
                unit_dict[band] = cmb_unit_conversion(data[exp][band]['freq'].to('GHz').value, 'KCMB2KRJ') if correct_unit else 1.
                wp_dict[band] = wp_interp if correct_pixel else np.ones_like(ell_eff)
                break

    # Apply corrections and subtract noise
    corr_spectra = {}
    for key, spec in spectra.items():
        band1, band2 = key.split('_')

        corr_spectra[key] = {}
        for cl_key in ['TT', 'EE', 'BB', 'TE', 'TB', 'EB']:
            # Beam factor depending on spectrum type
            if cl_key == 'TT':
                beam_factor = beam_dict[band1]['T'] * beam_dict[band2]['T']
            elif cl_key == 'EE':
                beam_factor = beam_dict[band1]['E'] * beam_dict[band2]['E']
            elif cl_key == 'BB':
                beam_factor = beam_dict[band1]['B'] * beam_dict[band2]['B']
            elif cl_key == 'TE':
                beam_factor = beam_dict[band1]['T'] * beam_dict[band2]['E']
            elif cl_key == 'TB':
                beam_factor = beam_dict[band1]['T'] * beam_dict[band2]['B']
            elif cl_key == 'EB':
                beam_factor = beam_dict[band1]['E'] * beam_dict[band2]['B']

            factor = beam_factor * unit_dict[band1] * unit_dict[band2] * wp_dict[band1] * wp_dict[band2]

            Nl = avg_std_noise[key][cl_key]['MEAN']
            Cl = spec[cl_key] - Nl
            spectrum_corr = np.abs(Cl / factor)
            errbar = np.abs(np.sqrt(avg_std_skyplusnoise[key][cl_key]['STD']**2 +
                             avg_std_noise[key][cl_key]['STD']**2) / factor)

            corr_spectra[key][cl_key] = {'SPECTRUM': spectrum_corr, 'ERROR': errbar}

        # Keep multipole info
        corr_spectra[key]['ell1'] = spec['ell1']
        corr_spectra[key]['ell2'] = spec['ell2']
        corr_spectra[key]['ell_eff'] = spec['ell_eff']

    # Save to FITS if requested
    out_file = None
    if save:
        out_file = path_out_file
        hdu_list = fits.HDUList([fits.PrimaryHDU()])

        for band_i in band_list:
            for band_j in band_list:
                key = f"{band_i}_{band_j}"
                spec_dict = corr_spectra[key]

                cols = []
                for cl_key in ['ell1', 'ell2', 'ell_eff', 'TT', 'EE', 'BB', 'TE', 'TB', 'EB']:
                    if cl_key in ['ell1', 'ell2', 'ell_eff']:
                        cols.append(fits.Column(name=cl_key, format='D', array=spec_dict[cl_key]))
                    else:
                        cols.append(fits.Column(name=f"{cl_key}_SPECTRUM", format='D', array=spec_dict[cl_key]['SPECTRUM']))
                        cols.append(fits.Column(name=f"{cl_key}_ERROR",  format='D', array=spec_dict[cl_key]['ERROR']))

                hdu = fits.BinTableHDU.from_columns(cols)
                hdu.header['BAND_I'] = band_i
                hdu.header['BAND_J'] = band_j
                hdu.header['COMMENT'] = "Corrected spectra with noise subtraction, beam/unit/pixel correction applied"
                hdu.name = key
                hdu_list.append(hdu)

        hdu_list.writeto(out_file, overwrite=True)
        print(f"Saved corrected spectra with errors to {out_file}")

    return corr_spectra, out_file


# Example execution parameters
n_sim = 100
nside = 512

mask_select = masks['quijote_galcut']['galcut10']
mask_name = mask_select['name']
use_simulated_maps = True
use_white_noise = True
out_path = '/home/pablo/Desktop/master/tfm/spectra/'
path_spectra = os.path.join(out_path, f'power_spectra_{mask_name}.fits')
path_avg_std_skyplusnoise = os.path.join(out_path, f'spectra_avg_std_{mask_name}_avg_std{n_sim}_skyplusnoise.fits')
path_avg_std_noise = os.path.join(out_path, f'spectra_avg_std_{mask_name}_avg_std{n_sim}_noise.fits')

path_corrected_spectra = os.path.join(out_path, f'corrected_power_spectra_{mask_name}.fits')

# Bands to use
quijote_bands = ['11']
wmap_bands = ['23', '33', '41', '61', '94']
planck_bands = ['30', '44', '70', '100', '143', '217', '353']

band_list = quijote_bands + wmap_bands + planck_bands

corr_spectra, out_file = correct_power_spectra(
    path_spectra, path_avg_std_skyplusnoise, path_avg_std_noise,
    band_list, data, nside, correct_beam=True, correct_unit=True,
    correct_pixel=True, save=True, path_out_file=path_corrected_spectra
)
