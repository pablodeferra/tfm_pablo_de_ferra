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
                "17": "Bl_417",
                "19": "Bl_419",
            }
            colname = col_map.get(band_name)
            if colname is None:
                # fallback: take first column-like Bl_*
                for name in beam_hdu.columns.names:
                    if name.lower().startswith('bl_'):
                        colname = name
                        break
            beam_arr = beam_hdu.data[colname][0]
            beam_interp = np.interp(ell_eff, np.arange(len(beam_arr)), beam_arr)
        return {"T": beam_interp, "E": beam_interp, "B": beam_interp}

    # WMAP
    elif band_name in data.get('WMAP', {}):
        beam_arr = np.loadtxt(data['WMAP'][band_name]['beam']).T[1]
        beam_interp = np.interp(ell_eff, np.arange(len(beam_arr)), beam_arr)
        return {"T": beam_interp, "E": beam_interp, "B": beam_interp}

    # Planck
    elif band_name in data.get('Planck', {}):
        if int(band_name) <= 70:  # LFI
            hdul = fits.open(data['Planck'][band_name]['beam'])
            # try to find correct extension name
            extname = f'BEAMWF_0{band_name}X0{band_name}'
            if extname in hdul:
                beam_hdu = hdul[extname]
                Bl = beam_hdu.data['BL']
            else:
                # fallback: take first extension with 'BL' column
                beam_hdu = hdul[1]
                Bl = beam_hdu.data[beam_hdu.columns.names[0]]
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
    Compute unit conversion factors between CMB thermodynamic temperature (K_CMB),
    Rayleigh-Jeans temperature (K_RJ), and surface brightness (Jy/sr).

    Parameters
    ----------
    nuGHz : float or array-like
        Frequency in GHz.
    option : str, optional
        Type of conversion to perform. Available options:
            - 'KCMB2KRJ' : Convert from K_CMB to K_RJ
            - 'KRJ2KCMB' : Convert from K_RJ to K_CMB
            - 'KCMB2Jysr' : Convert from K_CMB to Jy/sr
            - 'Jysr2KCMB' : Convert from Jy/sr to K_CMB
            - 'KRJ2Jysr' : Convert from K_RJ to Jy/sr
            - 'Jysr2KRJ' : Convert from Jy/sr to K_RJ
    help : bool, optional
        If True, prints the available conversion options and syntax.

    Returns
    -------
    float or ndarray
        Conversion factor for the selected unit transformation.

    Notes
    -----
    - Based on standard CMB thermodynamic relations.
    - Tcmb = 2.72548 K (Planck 2018).
    - h, k, c are Planck's constant, Boltzmann constant, and speed of light.

    Examples
    --------
    >>> cmb_unit_conversion(30, 'KCMB2KRJ')
    1.0289...
    >>> cmb_unit_conversion(143, 'KCMB2Jysr')
    269.3...
    """
    # Physical constants
    h = 6.62607015e-34  # Planck constant [J s]
    k = 1.380649e-23    # Boltzmann constant [J/K]
    c = 2.99792458e8    # Speed of light [m/s]
    Tcmb = 2.72548       # CMB temperature [K]

    cases = ['KCMB2KRJ', 'KRJ2KCMB', 'KCMB2Jysr', 'Jysr2KCMB', 'KRJ2Jysr', 'Jysr2KRJ']
    if help:
        print("Syntax: cmb_unit_conversion(nuGHz, option=)")
        print("Available options:", cases)

    nu = np.asarray(nuGHz) * 1e9  # [Hz]
    x = h * nu / (k * Tcmb)
    thermo = x**2 * np.exp(x) / (np.exp(x) - 1.0)**2
    rj = (2.0 * k * nu**2 / c**2) * 1e26  # [Jy/sr per K_RJ]

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


# ---------------------------------------------------------------------
def correct_power_spectra(path_spectra, path_avg_std_skyplusnoise, path_avg_std_noise,
                          band_list, data, nside,
                          correct_beam=True, correct_unit=True, correct_pixel=True,
                          save=False, path_out_file=None):
    """
    Correct power spectra by removing noise bias and applying beam, pixel,
    and unit corrections. Optionally saves corrected spectra to a FITS file.

    Parameters
    ----------
    path_spectra : str
        Path to input FITS file containing raw (sky + noise) spectra.
    path_avg_std_skyplusnoise : str
        Path to FITS file with average and standard deviation for sky + noise simulations.
    path_avg_std_noise : str
        Path to FITS file with average and standard deviation for pure noise simulations.
    band_list : list of str
        List of band names or pairs (e.g., ['30_44', '44_70']).
    data : dict
        Metadata dictionary with beam file paths and band frequencies.
    nside : int
        HEALPix NSIDE resolution of the input maps.
    correct_beam : bool, optional
        If True, deconvolve beam window functions.
    correct_unit : bool, optional
        If True, convert from K_CMB to K_RJ units.
    correct_pixel : bool, optional
        If True, deconvolve the HEALPix pixel window function.
    save : bool, optional
        If True, save the corrected spectra to a FITS file.
    path_out_file : str, optional
        Output FITS file path. Defaults to "corrected_cls.fits" if not provided.

    Returns
    -------
    tuple
        (corr_spectra, out_file)
        corr_spectra : dict
            Dictionary with corrected power spectra and errors.
        out_file : str or None
            Path to the output FITS file if saved, otherwise None.

    Notes
    -----
    - Subtracts noise power spectra from total sky+noise spectra.
    - Deconvolves beam and pixel window functions.
    - Converts units from K_CMB² to K_RJ² if `correct_unit=True`.
    - Propagates errors from input standard deviations.
    """
    if save and path_out_file is None:
        path_out_file = "corrected_cls.fits"

    # Load input spectra
    spectra = functions.read_spectra_from_fits(path_spectra, band_list)
    avg_std_skyplusnoise = functions.read_spectra_from_fits(
        path_avg_std_skyplusnoise, band_list, use_white_noise=use_white_noise
    )
    avg_std_noise = functions.read_spectra_from_fits(
        path_avg_std_noise, band_list, use_white_noise=use_white_noise
    )

    # Effective multipoles from first entry
    first_entry = next(iter(spectra.values()))
    ell_eff = np.array(first_entry['ell_eff'])

    # Pixel window function
    if correct_pixel:
        wpix = hp.pixwin(nside)
        wp_interp = np.interp(ell_eff, np.arange(len(wpix)), wpix)
    else:
        wp_interp = np.ones_like(ell_eff)

    # Precompute correction factors for each band
    all_bands = set()
    for key in spectra.keys():
        if "_" in key:
            band1, band2 = key.split('_', 1)
            all_bands.update([band1, band2])

    beam_dict, unit_dict, wp_dict = {}, {}, {}
    for band in all_bands:
        for exp in data:
            if band in data[exp]:
                # Beam correction
                if correct_beam:
                    beam_dict[band] = get_beam_for_band(band, data, ell_eff)
                else:
                    beam_dict[band] = {"T": np.ones_like(ell_eff),
                                       "E": np.ones_like(ell_eff),
                                       "B": np.ones_like(ell_eff)}

                # Frequency in GHz
                freq = data[exp][band].get('freq')
                try:
                    nuGHz = freq.to('GHz').value
                except Exception:
                    nuGHz = float(freq)

                # Unit conversion factor (K_CMB → K_RJ)
                unit_dict[band] = cmb_unit_conversion(nuGHz, 'KCMB2KRJ') if correct_unit else 1.0
                wp_dict[band] = wp_interp if correct_pixel else np.ones_like(ell_eff)
                break

    # Apply corrections and noise subtraction
    corr_spectra = {}
    for key, spec in spectra.items():
        if "_" not in key:
            continue
        band1, band2 = key.split('_', 1)
        corr_spectra[key] = {}

        for cl_key in ['TT', 'EE', 'BB', 'TE', 'TB', 'EB']:
            # Beam factor for each spectrum type
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

            # Physical deconvolution factor (divide) and unit factor (multiply)
            phys_factor = beam_factor * wp_dict[band1] * wp_dict[band2]
            unit_factor = unit_dict[band1] * unit_dict[band2]

            # Retrieve raw Cl
            spec_val = spec.get(cl_key)
            if spec_val is None:
                continue
            if isinstance(spec_val, dict):
                if 'MEAN' in spec_val:
                    Cl_raw = np.array(spec_val['MEAN'])
                elif 'SPECTRUM' in spec_val:
                    Cl_raw = np.array(spec_val['SPECTRUM'])
                else:
                    raise ValueError(f"Unexpected dict format for spectrum {key} {cl_key}")
            else:
                Cl_raw = np.array(spec_val)

            Nl = np.array(avg_std_noise[key][cl_key]['MEAN'])
            Cl = Cl_raw - Nl

            safe_phys = np.array(phys_factor, dtype=float)
            safe_phys[safe_phys == 0] = np.nan

            # 1) Deconvolve beam and pixel windows
            Cl_deconv = Cl / safe_phys

            # 2) Convert units (K_CMB² → K_RJ²)
            spectrum_corr = Cl_deconv * unit_factor

            # Propagate errors
            err_num = np.sqrt(
                np.array(avg_std_skyplusnoise[key][cl_key]['STD'])**2 +
                np.array(avg_std_noise[key][cl_key]['STD'])**2
            )
            errbar = (err_num / safe_phys) * unit_factor

            corr_spectra[key][cl_key] = {'SPECTRUM': spectrum_corr, 'ERROR': errbar}

        # Copy multipole info
        corr_spectra[key]['ell1'] = np.array(spec['ell1'])
        corr_spectra[key]['ell2'] = np.array(spec['ell2'])
        corr_spectra[key]['ell_eff'] = np.array(spec['ell_eff'])

    # Save to FITS file if requested
    out_file = None
    if save:
        out_file = path_out_file
        hdu_list = fits.HDUList([fits.PrimaryHDU()])

        for key, spec_dict in sorted(corr_spectra.items()):
            band_i, band_j = key.split('_', 1)
            cols = []
            for cl_key in ['ell1', 'ell2', 'ell_eff', 'TT', 'EE', 'BB', 'TE', 'TB', 'EB']:
                if cl_key in ['ell1', 'ell2', 'ell_eff']:
                    cols.append(fits.Column(name=cl_key, format='D', array=spec_dict[cl_key]))
                else:
                    if cl_key not in spec_dict:
                        continue
                    cols.append(fits.Column(name=f"{cl_key}_SPECTRUM", format='D', array=spec_dict[cl_key]['SPECTRUM']))
                    cols.append(fits.Column(name=f"{cl_key}_ERROR", format='D', array=spec_dict[cl_key]['ERROR']))

            hdu = fits.BinTableHDU.from_columns(cols)
            hdu.header['BAND_I'] = band_i
            hdu.header['BAND_J'] = band_j
            hdu.header['COMMENT'] = (
                "Corrected spectra: noise subtracted, beam/unit/pixel corrections applied"
            )
            hdu.name = key
            hdu_list.append(hdu)

        hdu_list.writeto(out_file, overwrite=True)
        print(f"[OK] Saved corrected spectra with errors to {out_file}")

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