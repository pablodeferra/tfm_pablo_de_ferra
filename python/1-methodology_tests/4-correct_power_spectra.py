#%%
import numpy as np
import healpy as hp
import os
from tqdm import tqdm
import pymaster as nmt
from data import data, path_map, masks, path_mask
from astropy.io import fits
import re



def get_beam_for_band(band_name, data, b):
    """
    Returns the interpolated beam for a given band

    Parameters
    ----------
    band_name : str
        Nombre de la banda, p.ej. '11', '23_1', '30', '100', ...
    data : dict
        Diccionario de información de experimentos y bandas.
    b : nmt.NmtBin
        Objeto de binning de NaMaster.
    Returns
    -------
    beam_interp : array
        Beam interpolado a los ells efectivos de b.
    """
    ell = b.get_effective_ells()
    
    if band_name in data['QUIJOTE']:
        # QUIJOTE
        beam_cl = hp.read_cl(data['QUIJOTE'][band_name]['beam'])[1,0]
        beam_interp = np.interp(ell, np.arange(beam_cl.size), beam_cl)
    
    elif band_name in data['WMAP']:
        # WMAP
        beam_arr = np.loadtxt(data['WMAP'][band_name]['beam']).T[1]
        beam_interp = np.interp(ell, np.arange(beam_arr.size), beam_arr)
    
    elif band_name in data['Planck']:
        # Planck LFI
        if int(band_name) <= 70:  # LFI 30, 44, 70
            hdul = fits.open(data['Planck'][band_name]['beam'])
            beam_ext = hdul['BEAMWF_{}X{}'.format(band_name, band_name)]
            Bl = beam_ext.data['BL']
            beam_interp = np.interp(ell, np.arange(Bl.size), Bl)
        else:
            # Planck HFI
            hdul = fits.open(data['Planck'][band_name]['beam'])
            window_hdu = hdul['WINDOW FUNCTIONS']
            # Se puede usar la columna T o E dependiendo de la corrección
            Bl = window_hdu.data['E']
            beam_interp = np.interp(ell, np.arange(Bl.size), Bl)
    else:
        raise ValueError(f"Banda {band_name} no encontrada en data.")
    
    return beam_interp

def apply_beam_correction(spectra_matrix, band_list, data, b):
    """
    Corrects the spectra matrix from beams
    
    Parameters
    ----------
    spectra_matrix : ndarray
        Matriz de espectros (N_band, N_band) con diccionarios de ['TT','EE',...].
    band_list : list of str
        Lista de bandas usadas en spectra_matrix.
    data : dict
        Diccionario de información de experimentos y bandas.
    b : nmt.NmtBin
        Objeto de binning.
    
    Returns
    -------
    corrected_matrix : ndarray
        Matriz de espectros corregida.
    """
    N_band = len(band_list)
    corrected_matrix = np.empty_like(spectra_matrix, dtype=object)
    
    beams = {band: get_beam_for_band(band, data, b) for band in band_list}
    
    for i in range(N_band):
        for j in range(N_band):
            cl_dict = spectra_matrix[i,j].copy()
            beam_i = beams[band_list[i]]
            beam_j = beams[band_list[j]]
            for key in ['TT','EE','BB','TE','TB','EB']:
                cl_dict[key] = cl_dict[key] / (beam_i * beam_j)
            corrected_matrix[i,j] = cl_dict
    return corrected_matrix

def apply_beam_correction_avg_std(avg_matrix, std_matrix, band_list, data, b):
    """
    Corrects the mean and std of the 100 simulations from beam

    Parameters
    ----------
    avg_matrix : ndarray
        Matriz (N_band, N_band) con diccionarios de medias de espectros ['TT','EE',...].
    std_matrix : ndarray
        Matriz (N_band, N_band) con diccionarios de desviaciones estándar.
    band_list : list of str
        Lista de bandas correspondientes a las filas/columnas.
    data : dict
        Diccionario con info de experimentos y bandas.
    b : nmt.NmtBin
        Objeto de binning para obtener ells efectivos.

    Returns
    -------
    avg_matrix_corr, std_matrix_corr : ndarray
        Matrices corregidas.
    """
    N_band = len(band_list)
    avg_matrix_corr = np.empty_like(avg_matrix, dtype=object)
    std_matrix_corr = np.empty_like(std_matrix, dtype=object)

    beams = {band: get_beam_for_band(band, data, b) for band in band_list}

    for i in range(N_band):
        for j in range(N_band):
            avg_dict = avg_matrix[i,j].copy()
            std_dict = std_matrix[i,j].copy()
            beam_i = beams[band_list[i]]
            beam_j = beams[band_list[j]]

            for key in ['TT','EE','BB','TE','TB','EB']:
                avg_dict[key] = avg_dict[key] / (beam_i * beam_j)
                std_dict[key] = std_dict[key] / (beam_i * beam_j)

            avg_matrix_corr[i,j] = avg_dict
            std_matrix_corr[i,j] = std_dict

    return avg_matrix_corr, std_matrix_corr



# 3. Aplicar corrección de beams
avg_matrix_corr, std_matrix_corr = apply_beam_correction_avg_std(
    avg_matrix, std_matrix, band_list, data, b
)



# 5. Apply beam correction
spectra_matrix_corrected = apply_beam_correction(spectra_matrix, band_list, data, b)

