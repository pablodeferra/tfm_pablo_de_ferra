#%%
import os
import healpy as hp
import numpy as np
from astropy.io import fits
import functions
from data import data, path_map, masks, path_mask
from scipy.constants import c,h,k

mask_select = masks['quijote_galcut']['galcut10']
mask_name = mask_select['name']

def get_beam_for_band(band_name, data, b):
    """
    Returns the interpolated beam transfer function for a given frequency band.

    Parameters
    ----------
    band_name : str
        Name of the band, e.g., '11', '23_1', '30', '100', etc.
    data : dict
        Dictionary containing experiment and band information, including beam file paths.
    b : nmt.NmtBin
        NaMaster binning object used to get effective multipoles.

    Returns
    -------
    beam_interp : numpy.ndarray
        Beam transfer function interpolated at the effective multipoles of b.
    """
    ell = b.get_effective_ells()
    
    if band_name in data['QUIJOTE']:
        # QUIJOTE experiment
        beam_cl = hp.read_cl(data['QUIJOTE'][band_name]['beam'])[1, 0]
        beam_interp = np.interp(ell, np.arange(beam_cl.size), beam_cl)
    
    elif band_name in data['WMAP']:
        # WMAP experiment
        beam_arr = np.loadtxt(data['WMAP'][band_name]['beam']).T[1]
        beam_interp = np.interp(ell, np.arange(beam_arr.size), beam_arr)
    
    elif band_name in data['Planck']:
        # Planck experiment
        if int(band_name) <= 70:  # LFI channels: 30, 44, 70 GHz
            hdul = fits.open(data['Planck'][band_name]['beam'])
            beam_ext = hdul['BEAMWF_{}X{}'.format(band_name, band_name)]
            Bl = beam_ext.data['BL']
            beam_interp = np.interp(ell, np.arange(Bl.size), Bl)
        else:
            # Planck HFI channels
            hdul = fits.open(data['Planck'][band_name]['beam'])
            window_hdu = hdul['WINDOW FUNCTIONS']
            # Use the T or E column depending on the desired correction
            Bl = window_hdu.data['E']
            beam_interp = np.interp(ell, np.arange(Bl.size), Bl)
    else:
        raise ValueError(f"Band {band_name} not found in data.")
    
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

def correct_power_spectra(path_fits, data, b, nside, correct_beam=True, 
                          correct_unit=True, correct_pixel=True, 
                          save=False, save_path=None, mask_name=None):
    """
    Read power spectra from a FITS file, apply corrections (beam, unit, pixel window)
    for all spectra found, and optionally save the corrected spectra to a new FITS file 
    with '_corr' suffix.

    Corrections are applied as:
        C_l_corrected = C_l / (B1 * B2 * U1 * U2 * W1 * W2)^2

    Parameters
    ----------
    path_fits : str
        Path to the FITS file containing the original spectra.
    data : dict
        Dictionary containing all experiment and band information, including central frequencies 
        and beam paths for all bands.
    b : nmt.NmtBin
        NaMaster binning object to get effective multipoles.
    nside : int
        HEALPix nside for pixel window computation.
    correct_beam : bool, default True
        Whether to correct spectra by the beam.
    correct_unit : bool, default True
        Whether to correct spectra from mK_CMB^2 to mK_RJ^2.
    correct_pixel : bool, default True
        Whether to correct spectra by pixel window function.
    save : bool, default False
        Whether to save the corrected spectra to a FITS file.
    save_path : str, optional
        Directory to save corrected FITS. If None, saves in the same folder as input.
    mask_name : str, optional
        Mask name to include in output filename. Required if save=True.

    Returns
    -------
    corrected_spectra : dict
        Dictionary containing the corrected spectra.
    out_file : str or None
        Path to the saved FITS file if save=True, otherwise None.
    """

    if save and mask_name is None:
        raise ValueError("mask_name must be provided if save=True")

    # Load spectra
    spectra = functions.read_spectra_from_fits(path_fits, band_list=None)
    ell_eff = next(iter(spectra.values()))['ell_eff']

    # Precompute pixel window function
    if correct_pixel:
        wpix = hp.pixwin(nside)
        wp_interp = np.interp(ell_eff, np.arange(len(wpix)), wpix)
    else:
        wp_interp = np.ones_like(ell_eff)

    # Precompute beam and unit corrections for all bands
    all_bands = set()
    for key in spectra.keys():
        band1, band2 = key.split('_')
        all_bands.update([band1, band2])

    beam_dict = {}
    unit_dict = {}
    wp_dict = {}

    for band in all_bands:
        for exp in data:
            if band in data[exp]:
                # Beam
                beam_dict[band] = get_beam_for_band(band, data[exp], b) if correct_beam else np.ones_like(ell_eff)
                # Unit
                unit_dict[band] = cmb_unit_conversion(data[exp][band]['freq'].to('GHz').value, 'KCMB2KRJ') if correct_unit else 1.0
                # Pixel
                wp_dict[band] = wp_interp if correct_pixel else np.ones_like(ell_eff)
                break

    # Apply corrections
    corrected_spectra = {}
    for key, spec in spectra.items():
        band1, band2 = key.split('_')
        factor = (beam_dict[band1] * beam_dict[band2] *
                  unit_dict[band1] * unit_dict[band2] *
                  wp_dict[band1] * wp_dict[band2])**2

        corr_spec = {}
        if isinstance(spec['TT'], dict):
            for cl_key in ['TT','EE','BB','TE','TB','EB']:
                corr_spec[cl_key] = {'MEAN': spec[cl_key]['MEAN'] / factor,
                                     'STD':  spec[cl_key]['STD']  / factor}
            for cl_key in ['ell1','ell2','ell_eff']:
                corr_spec[cl_key] = spec[cl_key]
        else:
            for cl_key in ['ell1','ell2','ell_eff']:
                corr_spec[cl_key] = spec[cl_key].copy()
            for cl_key in ['TT','EE','BB','TE','TB','EB']:
                corr_spec[cl_key] = spec[cl_key] / factor

        corrected_spectra[key] = corr_spec

    out_file = None
    if save:
        if save_path is None:
            save_path = os.path.dirname(path_fits)
        out_file = os.path.join(save_path, f'power_spectra_corr_{mask_name}.fits')

        hdus = [fits.PrimaryHDU()]
        for key, spec in corrected_spectra.items():
            cols = []
            if isinstance(spec['TT'], dict):
                for cl_key in ['ell1','ell2','ell_eff','TT','EE','BB','TE','TB','EB']:
                    cols.append(fits.Column(name=f"{cl_key}_MEAN", array=spec[cl_key]['MEAN'], format='D'))
                    cols.append(fits.Column(name=f"{cl_key}_STD",  array=spec[cl_key]['STD'],  format='D'))
            else:
                for cl_key in ['ell1','ell2','ell_eff','TT','EE','BB','TE','TB','EB']:
                    cols.append(fits.Column(name=cl_key, array=spec[cl_key], format='D'))
            hdu = fits.BinTableHDU.from_columns(cols, name=key)
            hdus.append(hdu)

        hdul = fits.HDUList(hdus)
        hdul.writeto(out_file, overwrite=True)
        print(f"Saved corrected spectra to {out_file}")

    return corrected_spectra, out_file
