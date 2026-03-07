#%%
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import pymaster as nmt
import sys
from pathlib import Path

sys.path.append('../')
from data import data, path_masks, masks


class MaskBuilder:
    """Class for building galactic cut masks."""
    
    def __init__(self, nside=512):
        """
        Initialize the MaskBuilder.
        
        Parameters:
        -----------
        nside : int
            HEALPix nside parameter
        """
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        self.galactic_latitude = hp.pix2ang(nside, np.arange(self.npix), lonlat=True)[1]
    
    def create_galactic_cut_mask(self, latitude_threshold=10.0):
        """
        Create a galactic latitude cut mask.
        
        Parameters:
        -----------
        latitude_threshold : float
            Galactic latitude threshold in degrees
            
        Returns:
        --------
        numpy.ndarray : Binary mask (0 = masked, 1 = unmasked)
        """
        return np.where(np.abs(self.galactic_latitude) < latitude_threshold, 0., 1.)
    
    def create_intensity_cut_mask(self, intensity_map, intensity_threshold=10.0):
        """
        Create an intensity cut mask.
        
        Parameters:
        -----------
        intensity_map : numpy.ndarray
            Intensity map (e.g., QUIJOTE 11 GHz I map)
        intensity_threshold : float
            Intensity threshold in mK
            
        Returns:
        --------
        numpy.ndarray : Binary mask (0 = masked, 1 = unmasked)
        """
        # Mask pixels above threshold
        return np.where(np.abs(intensity_map) > intensity_threshold, 0., 1.)
    
    def load_quijote_mask(self, mask_name='lowdec_satband'):
        """
        Load QUIJOTE mask.
        
        Parameters:
        -----------
        mask_name : str
            Name of the QUIJOTE mask
            
        Returns:
        --------
        numpy.ndarray : QUIJOTE mask
        """
        mask_path = masks['QUIJOTE'][mask_name]['path']
        return hp.read_map(mask_path)
    
    def combine_masks(self, *mask_list):
        """
        Combine multiple masks (logical AND).
        
        Parameters:
        -----------
        *mask_list : numpy.ndarray
            Masks to combine
            
        Returns:
        --------
        numpy.ndarray : Combined mask
        """
        combined = np.ones_like(mask_list[0])
        for mask in mask_list:
            combined *= mask
        return combined
    
    def save_mask(self, mask, filename, output_dir=None):
        """
        Save mask to FITS file.
        
        Parameters:
        -----------
        mask : numpy.ndarray
            Mask to save
        filename : str
            Output filename
        output_dir : str, optional
            Output directory (default: path_masks)
        """
        if output_dir is None:
            output_dir = Path(path_masks)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / filename
        hp.write_map(str(output_path), mask, dtype='float64', overwrite=True)
        print(f"Mask saved to: {output_path}")
    
    def visualize_mask(self, mask, title="", save_path=None):
        """
        Visualize mask using Mollweide projection.
        
        Parameters:
        -----------
        mask : numpy.ndarray
            Mask to visualize
        title : str
            Plot title
        save_path : str, optional
            Path to save the figure
        """
        hp.mollview(mask, title=title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()


def create_galactic_cut_masks():
    """Create galactic cut masks with different latitude thresholds."""
    
    # Initialize mask builder
    mask_builder = MaskBuilder(nside=512)
    
    # ========================================================================
    # CONFIGURATION PARAMETERS
    # ========================================================================
    galactic_cut_threshold = 20.      # Galactic latitude cut in degrees
    intensity_threshold_mk = None       # Intensity threshold in mK (None = no threshold)
    apod_scale_deg = 5.              # Apodization scale in degrees
    
    # ========================================================================
    # LOAD QUIJOTE DATA
    # ========================================================================
    # Load QUIJOTE 11 GHz intensity map
    quijote_11_path = data['QUIJOTE']['11']['path']
    quijote_11_I = hp.read_map(quijote_11_path, field=0)  
    print(f"Loaded QUIJOTE 11 GHz I map from: {quijote_11_path}")
    
    # Load QUIJOTE base mask
    quijote_mask = mask_builder.load_quijote_mask('lowdec_satband')
    print(f"Loaded QUIJOTE lowdec_satband mask")
    
    # ========================================================================
    # CREATE MASKS
    # ========================================================================
    # Create galactic cut mask
    galcut_mask = mask_builder.create_galactic_cut_mask(
        latitude_threshold=galactic_cut_threshold
    )
    f_sky_galcut = np.sum(galcut_mask) / len(galcut_mask)
    print(f"Galactic cut mask f_sky: {f_sky_galcut:.4f}")
    
    # Create intensity cut mask (if threshold is specified)
    if intensity_threshold_mk is not None:
        print(f"\nCreating intensity cut mask (|I| < {intensity_threshold_mk} mK)...")
        intensity_mask = mask_builder.create_intensity_cut_mask(
            quijote_11_I, 
            intensity_threshold=intensity_threshold_mk
        )
        f_sky_intensity = np.sum(intensity_mask) / len(intensity_mask)
        print(f"Intensity cut mask f_sky: {f_sky_intensity:.4f}")
        
        # Combine all masks (galactic cut + QUIJOTE mask + intensity cut)
        print(f"\nCombining masks (galcut + QUIJOTE + intensity cut)...")
        combined_mask = mask_builder.combine_masks(
            galcut_mask, 
            quijote_mask, 
            intensity_mask
        )
        threshold_str = f"{int(intensity_threshold_mk)}mk"
    else:
        print("\nNo intensity threshold applied.")
        # Combine masks (galactic cut + QUIJOTE mask only)
        print(f"Combining masks (galcut + QUIJOTE)...")
        combined_mask = mask_builder.combine_masks(
            galcut_mask, 
            quijote_mask
        )
        threshold_str = "0mk"
    
    f_sky_combined = np.sum(combined_mask) / len(combined_mask)
    print(f"Combined mask f_sky (before apodization): {f_sky_combined:.4f}")
    
    # ========================================================================
    # APODIZE MASK
    # ========================================================================
    print(f"\nApplying C2 apodization ({apod_scale_deg}°)...")
    combined_mask_apodized = nmt.mask_apodization(
        combined_mask, 
        apod_scale_deg, 
        apotype="C2"
    )
    f_sky_apodized = np.sum(combined_mask_apodized) / len(combined_mask_apodized)
    print(f"Combined mask f_sky (after apodization): {f_sky_apodized:.4f}")
    
    # ========================================================================
    # SAVE APODIZED MASK
    # ========================================================================
    output_dir = '/media/pablo/cmb_ssd/masks/quijote_galcut'
    filename_apod = f"mask_lowdec_satband_galcut{int(galactic_cut_threshold)}_{threshold_str}_apodC2_{int(apod_scale_deg)}.fits"
    mask_builder.save_mask(combined_mask_apodized, filename_apod, output_dir=output_dir)
    
    # ========================================================================
    # VISUALIZE MASKS
    # ========================================================================
    
    # Visualize apodized mask
    mask_builder.visualize_mask(
        combined_mask_apodized,
        title=f"Combined Mask - Apodized C2 {int(apod_scale_deg)}°"
    )


create_galactic_cut_masks()
