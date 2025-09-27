#%%
import os
import healpy as hp
import numpy as np
from astropy.io import fits
import functions
from data import data, masks
from scipy.constants import c,h,k

mask_select = masks['quijote_galcut']['galcut10']
mask_name = mask_select['name']

def get_beam_for_band(band_name, data, ell_eff):
    """
    Return the interpolated beam transfer function for a given frequency band.

    Parameters
    ----------
    band_name : str
        Name of the frequency band, e.g., '11', '30', '100'.
    data : dict
        Dictionary containing experiment and band information, including beam file paths.
        Must contain sub-dictionaries for each experiment (e.g., 'QUIJOTE', 'WMAP', 'Planck').
    ell_eff : array_like
        Array of effective multipoles at which to interpolate the beam.

    Returns
    -------
    beam_interp : numpy.ndarray
        Beam transfer function interpolated at the effective multipoles `ell_eff`.

    Notes
    -----
    The function supports different formats for different experiments:
      - QUIJOTE: beam stored as HEALPix Cl (TT) in FITS.
      - WMAP: beam stored in a text file, column 2 used.
      - Planck:
          - LFI channels (<=70 GHz): beam in HDU 'BEAMWF_{band}X{band}'.
          - HFI channels (>70 GHz): beam in HDU 'WINDOW FUNCTIONS', column 'E'.
    """

    # QUIJOTE
    if band_name in data.get('QUIJOTE', {}):
        beam_cl = hp.read_cl(data['QUIJOTE'][band_name]['beam'])[1, 0]  # TT
        beam_interp = np.interp(ell_eff, np.arange(len(beam_cl)), beam_cl)

    # WMAP
    elif band_name in data.get('WMAP', {}):
        beam_arr = np.loadtxt(data['WMAP'][band_name]['beam']).T[1]  # second column
        beam_interp = np.interp(ell_eff, np.arange(len(beam_arr)), beam_arr)

    # Planck
    elif band_name in data.get('Planck', {}):
        if int(band_name) <= 70:  # LFI
            hdul = fits.open(data['Planck'][band_name]['beam'])
            beam_hdu = hdul[f'BEAMWF_{band_name}X{band_name}']
            Bl = beam_hdu.data['BL']
            beam_interp = np.interp(ell_eff, np.arange(len(Bl)), Bl)
            hdul.close()
        else:  # HFI
            hdul = fits.open(data['Planck'][band_name]['beam'])
            window_hdu = hdul['WINDOW FUNCTIONS']
            Bl = window_hdu.data['E']  # E-mode column
            beam_interp = np.interp(ell_eff, np.arange(len(Bl)), Bl)
            hdul.close()
    else:
        raise ValueError(f"Band '{band_name}' not found in data.")

    return beam_interp

def cmb_unit_conversion(nuGHz,option='KCMB2KRJ',help=False):

    Tcmb = 2.72548 

    casos = ['KCMB2KRJ', 'KRJ2KCMB', 'KCMB2Jysr', 'Jysr2KCMB', 'KRJ2Jysr', 'Jysr2KRJ']
    if help==True:
       print('  Syntax -- cmb_unit_conversion(nuGHz,option=)')
       print('  Possible options are',casos)

    # Basic computation
    nu  = nuGHz*1e9
    x   = h * nu/ (k*Tcmb)
    thermo = x**2 * np.exp(x)/(np.exp(x)-1.)**2
    rj     = ( 2.0 * k * nu**2 / c**2 ) * 1e26

    # Identify case
    if option == 'KCMB2KRJ':
       fac = thermo
    elif option == 'KRJ2KCMB':
       fac = 1/thermo
    elif option == 'KCMB2Jysr':
       fac = thermo * rj
    elif option == 'Jysr2KCMB':
       fac = 1 / (thermo*rj)
    elif option == 'KRJ2Jysr':
       fac = rj
    elif option == 'Jysr2KRJ':
       fac = 1/rj
    else:
        print("Units not identified. Returning -1")
        fac = -1

    return fac


def correct_power_spectra_with_noise(path_spectra, path_avg_std_skyplusnoise, path_avg_std_noise,
                                     band_list, data, nside,
                                     correct_beam=True, correct_unit=True, correct_pixel=True,
                                     save=False, mask_name=None):
    """
    Correct power spectra for specified band pairs using beam/unit/pixel window,
    subtract noise spectra, compute corrected error bars, and optionally save as FITS.

    Formula:
        corrected_spectrum = (C_l - N_l) / factor
        error_bar          = sqrt(STD_skyplusnoise^2 + STD_noise^2) / factor

    Parameters
    ----------
    path_spectra : str
        FITS file with the original sky+noise spectra (contains ell_eff).
    path_avg_std_skyplusnoise : str
        FITS file containing avg+std sky+noise spectra (for error calculation).
    path_avg_std_noise : str
        FITS file containing avg+std noise spectra.
    band_list : list of str
        List of frequency bands to include (e.g., ['11','30']).
    data : dict
        Experiment/band information including frequencies and beam paths.
    nside : int
        HEALPix nside for pixel window computation.
    correct_beam, correct_unit, correct_pixel : bool
        Whether to apply corresponding corrections.
    save : bool, default False
        Whether to save the output FITS file.
    mask_name : str, optional
        Required if save=True; used in filename.

    Returns
    -------
    corrected_cls : dict
        Dictionary with corrected spectra and errors:
            corrected_cls['band1_band2']['EE']['spectrum']
            corrected_cls['band1_band2']['EE']['error']
            corrected_cls['band1_band2']['ell_eff']
    out_file : str or None
        Path to saved FITS file if save=True, else None.
    """

    if save and mask_name is None:
        raise ValueError("mask_name must be provided if save=True")

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
                beam_dict[band] = get_beam_for_band(band, data, ell_eff) if correct_beam else np.ones_like(ell_eff)
                unit_dict[band] = cmb_unit_conversion(data[exp][band]['freq'].to('GHz').value, 'KCMB2KRJ') if correct_unit else 1.
                wp_dict[band] = wp_interp if correct_pixel else np.ones_like(ell_eff)
                break

    # Apply corrections and subtract noise
    corr_spectra = {}
    for key, spec in spectra.items():
        band1, band2 = key.split('_')
        factor = (beam_dict[band1] * beam_dict[band2] *
                  unit_dict[band1] * unit_dict[band2] *
                  wp_dict[band1] * wp_dict[band2])

        corr_spectra[key] = {}
        for cl_key in ['TT','EE','BB','TE','TB','EB']:
            Nl = avg_std_noise[key][cl_key]['MEAN']
            Cl = spec[cl_key] - Nl
            spectrum_corr = Cl / factor
            errbar = np.sqrt(avg_std_skyplusnoise[key][cl_key]['STD']**2 +
                             avg_std_noise[key][cl_key]['STD']**2) / factor

            corr_spectra[key][cl_key] = {'SPECTRUM': spectrum_corr, 'ERROR': errbar}

        # Keep multipole info
        corr_spectra[key]['ell1'] = spec['ell1']
        corr_spectra[key]['ell2'] = spec['ell2']
        corr_spectra[key]['ell_eff'] = spec['ell_eff']

    # Save to FITS if requested
    out_file = None
    if save:
        out_path = os.path.dirname(path_spectra)
        file_name = f"corrected_cls_{mask_name}.fits"
        hdu_list = fits.HDUList([fits.PrimaryHDU()])

        for band_i in band_list:
            for band_j in band_list:
                key = f"{band_i}_{band_j}"
                spec_dict = corr_spectra[key]

                cols = []
                for cl_key in ['ell1','ell2','ell_eff','TT','EE','BB','TE','TB','EB']:
                    if cl_key in ['ell1','ell2','ell_eff']:
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

        out_file = os.path.join(out_path, file_name)
        hdu_list.writeto(out_file, overwrite=True)
        print(f"Saved corrected spectra with errors to {out_file}")

    return corr_spectra, out_file

n_sim = 10
nside = 512

mask_select = masks['quijote_galcut']['galcut10']
mask_name = mask_select['name']
use_simulated_maps = True
use_white_noise = True
out_path = '/home/pablo/Desktop/master/tfm/spectra/'
path_spectra = os.path.join(out_path, f'power_spectra_{mask_name}.fits')
path_avg_std_skyplusnoise = os.path.join(out_path, f'spectra_avg_std_{mask_name}_avg_std{n_sim}_skyplusnoise.fits')
path_avg_std_noise = os.path.join(out_path, f'spectra_avg_std_{mask_name}_avg_std{n_sim}_noise.fits')

save_path = '/home/pablo/Desktop/master/tfm/figures/spectra_auto_cross_test/'

# Bands to use
quijote_bands = ['11']
wmap_bands = ['23', '33', '41', '61', '94']
planck_bands = ['30', '44', '70', '100', '143', '217', '353']

band_list = quijote_bands + wmap_bands + planck_bands


corr_spectra, out_file = correct_power_spectra_with_noise(
    path_spectra, path_avg_std_skyplusnoise, path_avg_std_noise,
    band_list, data, nside, correct_beam=True, correct_unit=True,
    correct_pixel=True, save=True, mask_name=mask_name
)


#%%
spectra_matrix = functions.read_spectra_from_fits(path_spectra, band_list)

ell_eff = spectra_matrix['11_11']['ell_eff']

beam_23 = get_beam_for_band('23', data, ell_eff)