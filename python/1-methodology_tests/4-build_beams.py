#%%
import os
import numpy as np
from data import data

# Differential Assemblies per band
BANDS = {
    "K": ["K1"],
    "Ka": ["Ka1"],
    "Q": ["Q1", "Q2"],
    "V": ["V1", "V2"],
    "W": ["W1", "W2", "W3", "W4"],
}

def load_beam_file(file_path):
    """
    Load B_l data from a WMAP beam file, keeping all columns.

    Returns
    -------
    header_lines : list of str
        List of header lines (starting with '#').
    data : numpy.ndarray
        Array with shape (N,3) containing l, B_l, fractional error.
    """
    header_lines = []
    data_lines = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                header_lines.append(line.rstrip())
            else:
                data_lines.append([float(x) for x in line.split()])
    data = np.array(data_lines)
    return header_lines, data


def save_band_beam(band_name, header_template, data, save_path):
    """
    Save the averaged beam for a frequency band to a txt file,
    preserving the original WMAP header format and modifying the
    description to indicate the frequency band instead of the DA.

    Parameters
    ----------
    band_name : str
        Name of the frequency band (e.g., "K", "Ka", "Q", "V", "W").
    header_template : list of str
        List of header lines from the original DA beam file (lines starting with '#').
        The function will replace the line mentioning "differencing assembly"
        with a line indicating the frequency band.
    data : numpy.ndarray
        Array with shape (N,3), where:
            - column 0: multipole moment l
            - column 1: averaged beam transfer function B_l
            - column 2: averaged fractional error (delta B_l / B_l)
    save_path : str
        Path to the directory where the output file will be saved.

    Returns
    -------
    None
        The function writes a txt file to disk and prints a confirmation message.
    """
    new_header = []
    for line in header_template:
        # Replace DA description with frequency band
        if "differencing assembly" in line:
            line = f"# Beam Transfer Function (amplitude) for frequency band {band_name},"
        new_header.append(line)
    
    file_name = f"wmap_ampl_bl_{band_name}_9yr_v5p1.txt"
    file_path = os.path.join(save_path, file_name)
    
    with open(file_path, 'w') as f:
        for line in new_header:
            f.write(line + '\n')
        for row in data:
            f.write(f"{int(row[0]):>5} {row[1]:>14.8f} {row[2]:>14.8f}\n")
    
    print(f"Saved band beam file: {file_path}")


def generate_band_beams(BANDS, beam_path, save_path):
    """
    Generate averaged beams for each frequency band from DA beams.
    
    Parameters
    ----------
    BANDS : dict
        Dictionary of bands and their DAs.
    beam_path : str
        Path to the directory containing original DA beam txt files.
    save_path : str
        Path where the new band beams will be saved.

    Returns
    -------
    None
        The function writes a text file for each band to disk and prints a confirmation message.
    """
    
    for band, das in BANDS.items():
        all_data = []
        header_template = None
        for da in das:
            file_name = f"wmap_ampl_bl_{da}_9yr_v5p1.txt"
            file_path = os.path.join(beam_path, file_name)
            if not os.path.exists(file_path):
                print(f"WARNING: Beam file not found: {file_path}")
                continue
            header, data = load_beam_file(file_path)
            if header_template is None:
                header_template = header
            all_data.append(data)
        if len(all_data) == 0:
            print(f"No beams found for band {band}")
            continue
        # Average B_l (column 2) and fractional error (column 3) across DAs
        avg_data = np.copy(all_data[0])
        if len(all_data) > 1:
            for i in range(1, len(all_data)):
                avg_data[:,1] += all_data[i][:,1]
                avg_data[:,2] += all_data[i][:,2]
            avg_data[:,1] /= len(all_data)
            avg_data[:,2] /= len(all_data)
        save_band_beam(band, header_template, avg_data, save_path)


beam_path = os.path.dirname(data['WMAP']['23']['beam'])
save_path = os.path.dirname(data['WMAP']['23']['beam'])

generate_band_beams(BANDS, beam_path=beam_path, save_path=save_path)
