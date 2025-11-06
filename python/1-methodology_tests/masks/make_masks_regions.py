#%%
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import sys
import pymaster as nmt
from matplotlib.colors import ListedColormap

sys.path.append('../') 
from data import data, path_map, masks, path_masks


# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================
NSIDE_MASK_LOW = 8          # Low resolution for mask definition
NSIDE_MASK_HIGH = 256       # High resolution for smoothing
NSIDE_FINAL = 512           # Final resolution for output masks

SMOOTH_MASKS = True         # Enable/disable mask smoothing
SMOOTH_FWHM = np.deg2rad(10.0)  # FWHM for smoothing (in radians)

USE_PLANCK_CMAP = True      # Use Planck colormap for polarization maps


# ============================================================================
# LOAD DATA
# ============================================================================
# Load WMAP data and compute polarization intensity
wmap_path = data['WMAP']['23']['path']
wmap = hp.read_map(wmap_path, field=[0, 1, 2])
p = np.sqrt(wmap[1]**2 + wmap[2]**2)
p = hp.smoothing(p, fwhm=np.deg2rad(1))

# Load original mask
mask_path = masks['QUIJOTE']['lowdec_satband']['path']
mask_original = hp.read_map(mask_path)

npix_mask = hp.nside2npix(NSIDE_MASK_LOW)


# ============================================================================
# DEFINE MASK PIXELS AT NSIDE=8
# ============================================================================
# NPS (North Polar Spur) region
nps_pixels = [
    275, 242, 210, 178, 146, 113, 84, 60, 59, 
    39, 24, 25, 42, 62, 86, 115, 147, 179, 212, 
    243, 211, 114, 86, 61, 85, 41, 40, 148, 87,
    12, 23, 83, 276
]

# Cygnus region
cygnus_pixels = [
    307, 340, 339, 371, 404, 436, 435, 437, 438, 407, 
    375, 374, 373, 372, 344, 343, 342, 341, 308, 309, 
    310, 311, 376, 406, 405, 408, 345, 312, 439, 440, 
    409, 
]

# Fan region
fan_pixels = [
    378, 410, 411, 442, 443, 444, 445, 446, 447, 415, 
    383, 351, 319, 318, 317, 316, 315, 347, 346, 313,
    378, 379, 380, 381, 382, 412, 413, 414, 348, 349,
    350, 284, 377, 441, 314, 416, 352
]

# South region
south_pixels = [
    468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 
    478, 479, 480, 512, 544, 576, 608, 545, 513, 482, 
    481, 449, 448, 511, 510, 509, 508, 507, 506, 505,
    504, 503, 502, 501, 500, 533, 534, 535, 536, 537,
    538, 539, 540, 541, 542, 543, 565, 566, 567, 568, 
    569, 570, 571, 572, 573, 574, 575, 607, 606, 605, 
    604, 603, 602, 601, 600, 599, 598, 630, 631, 632, 
    633, 634, 635, 636, 637, 638, 639, 662, 663, 664, 
    665, 666, 667, 668, 693, 692, 691, 690, 714, 715, 
    716, 717, 669, 694
]

# North region: all pixels NOT in other masks
all_pixels = set(range(npix_mask))
used_pixels = set(nps_pixels) | set(cygnus_pixels) | set(fan_pixels) | set(south_pixels)
north_pixels = sorted(list(all_pixels - used_pixels))


# ============================================================================
# CREATE BINARY MASKS AT NSIDE=8
# ============================================================================
mask_nps_nside8 = np.zeros(npix_mask)
mask_cygnus_nside8 = np.zeros(npix_mask)
mask_fan_nside8 = np.zeros(npix_mask)
mask_south_nside8 = np.zeros(npix_mask)
mask_north_nside8 = np.zeros(npix_mask)

mask_nps_nside8[nps_pixels] = 1
mask_cygnus_nside8[cygnus_pixels] = 1
mask_fan_nside8[fan_pixels] = 1
mask_south_nside8[south_pixels] = 1
mask_north_nside8[north_pixels] = 1


# ============================================================================
# SMOOTH MASKS AND UPGRADE TO FINAL RESOLUTION
# ============================================================================

# ============================================================================
# SMOOTH MASKS AND UPGRADE TO FINAL RESOLUTION
# ============================================================================
if SMOOTH_MASKS:
    # Upgrade to high resolution for better smoothing
    mask_nps_high = hp.ud_grade(mask_nps_nside8, NSIDE_MASK_HIGH, order_in='RING', order_out='RING')
    mask_cygnus_high = hp.ud_grade(mask_cygnus_nside8, NSIDE_MASK_HIGH, order_in='RING', order_out='RING')
    mask_fan_high = hp.ud_grade(mask_fan_nside8, NSIDE_MASK_HIGH, order_in='RING', order_out='RING')
    mask_south_high = hp.ud_grade(mask_south_nside8, NSIDE_MASK_HIGH, order_in='RING', order_out='RING')
    mask_north_high = hp.ud_grade(mask_north_nside8, NSIDE_MASK_HIGH, order_in='RING', order_out='RING')
    
    # Smooth each mask independently with Gaussian kernel
    mask_nps_smooth = hp.smoothing(mask_nps_high, fwhm=SMOOTH_FWHM)
    mask_cygnus_smooth = hp.smoothing(mask_cygnus_high, fwhm=SMOOTH_FWHM)
    mask_fan_smooth = hp.smoothing(mask_fan_high, fwhm=SMOOTH_FWHM)
    mask_south_smooth = hp.smoothing(mask_south_high, fwhm=SMOOTH_FWHM)
    mask_north_smooth = hp.smoothing(mask_north_high, fwhm=SMOOTH_FWHM)
    
    # Stack all smoothed masks
    all_masks_smooth = np.array([
        mask_nps_smooth,
        mask_cygnus_smooth,
        mask_fan_smooth,
        mask_south_smooth,
        mask_north_smooth
    ])
    
    # Winner-takes-all: assign each pixel to mask with highest value
    # This guarantees no overlap between masks
    max_mask_idx = np.argmax(all_masks_smooth, axis=0)
    
    # Create binary masks based on winner-takes-all assignment
    mask_nps_high_final = (max_mask_idx == 0).astype(float)
    mask_cygnus_high_final = (max_mask_idx == 1).astype(float)
    mask_fan_high_final = (max_mask_idx == 2).astype(float)
    mask_south_high_final = (max_mask_idx == 3).astype(float)
    mask_north_high_final = (max_mask_idx == 4).astype(float)
    
    # Downgrade to final resolution
    mask_nps_nside512 = hp.ud_grade(mask_nps_high_final, NSIDE_FINAL, order_in='RING', order_out='RING')
    mask_cygnus_nside512 = hp.ud_grade(mask_cygnus_high_final, NSIDE_FINAL, order_in='RING', order_out='RING')
    mask_fan_nside512 = hp.ud_grade(mask_fan_high_final, NSIDE_FINAL, order_in='RING', order_out='RING')
    mask_south_nside512 = hp.ud_grade(mask_south_high_final, NSIDE_FINAL, order_in='RING', order_out='RING')
    mask_north_nside512 = hp.ud_grade(mask_north_high_final, NSIDE_FINAL, order_in='RING', order_out='RING')
    
    # ========================================================================
    # VERIFICATION: Check for mask overlap
    # ========================================================================
    print("\n" + "="*60)
    print("OVERLAP VERIFICATION (nside=%d)" % NSIDE_MASK_HIGH)
    print("="*60)
    
    overlap_check = (mask_nps_high_final + mask_cygnus_high_final + mask_fan_high_final + 
                     mask_south_high_final + mask_north_high_final)
    
    unique_values = np.unique(overlap_check)
    max_overlap = np.max(overlap_check)
    pixels_with_overlap = np.sum(overlap_check > 1)
    
    print(f"Unique values in mask sum: {unique_values}")
    print(f"Maximum overlap value: {max_overlap}")
    print(f"Pixels with overlap (>1): {pixels_with_overlap}")
    
    if max_overlap <= 1:
        print("ONFIRMED: No overlap between masks at nside=%d" % NSIDE_MASK_HIGH)
    else:
        print("WARNING: Overlap detected!")
    
    print("\nOVERLAP VERIFICATION (nside=%d)" % NSIDE_FINAL)
    print("="*60)
    
    # Binarize masks at final resolution
    mask_nps_512_bin = (mask_nps_nside512 > 0.5).astype(int)
    mask_cygnus_512_bin = (mask_cygnus_nside512 > 0.5).astype(int)
    mask_fan_512_bin = (mask_fan_nside512 > 0.5).astype(int)
    mask_south_512_bin = (mask_south_nside512 > 0.5).astype(int)
    mask_north_512_bin = (mask_north_nside512 > 0.5).astype(int)
    
    overlap_check_512 = (mask_nps_512_bin + mask_cygnus_512_bin + mask_fan_512_bin + 
                         mask_south_512_bin + mask_north_512_bin)
    
    unique_values_512 = np.unique(overlap_check_512)
    max_overlap_512 = np.max(overlap_check_512)
    pixels_with_overlap_512 = np.sum(overlap_check_512 > 1)
    
    print(f"Unique values in mask sum: {unique_values_512}")
    print(f"Maximum overlap value: {max_overlap_512}")
    print(f"Pixels with overlap (>1): {pixels_with_overlap_512}")
    
    if max_overlap_512 <= 1:
        print("CONFIRMED: No overlap between masks at nside=%d" % NSIDE_FINAL)
    else:
        print("WARNING: Overlap detected!")
    
    print("="*60 + "\n")
    
else:
    mask_nps_nside512 = hp.ud_grade(mask_nps_nside8, NSIDE_FINAL, order_in='RING', order_out='RING')
    mask_cygnus_nside512 = hp.ud_grade(mask_cygnus_nside8, NSIDE_FINAL, order_in='RING', order_out='RING')
    mask_fan_nside512 = hp.ud_grade(mask_fan_nside8, NSIDE_FINAL, order_in='RING', order_out='RING')
    mask_south_nside512 = hp.ud_grade(mask_south_nside8, NSIDE_FINAL, order_in='RING', order_out='RING')
    mask_north_nside512 = hp.ud_grade(mask_north_nside8, NSIDE_FINAL, order_in='RING', order_out='RING')


# Combine each mask with original mask
mask_nps_combined = mask_original * mask_nps_nside512
mask_cygnus_combined = mask_original * mask_cygnus_nside512
mask_fan_combined = mask_original * mask_fan_nside512
mask_south_combined = mask_original * mask_south_nside512
mask_north_combined = mask_original * mask_north_nside512


# ============================================================================
# APODIZE AND SAVE MASKS
# ============================================================================

# Output directory for masks
mask_output_dir = '/media/pablo/cmb_ssd/masks/specific_regions'

# Apodization scale in degrees
apod_scale_deg = 5.0

# Dictionary of masks to save
masks_to_save = {
    'nps': mask_nps_combined,
    'cygnus': mask_cygnus_combined,
    'fan': mask_fan_combined,
    'south': mask_south_combined,
    'north': mask_north_combined
}

# Apply apodization and save each mask
for region_name, mask in masks_to_save.items():
    print(f"\nProcessing {region_name.upper()} mask...")
    
    # Apply C2 apodization using NaMaster
    mask_apodized = nmt.mask_apodization(mask, apod_scale_deg, apotype="C2")
    
    # Calculate f_sky
    f_sky = np.sum(mask_apodized) / len(mask_apodized)
    print(f"  f_sky (apodized): {f_sky:.4f}")
    
    # Generate filename
    filename = f'mask_lowdec_satband_galcut0_0mk_{region_name}_apodC2_{int(apod_scale_deg)}.fits'
    output_path = f'{mask_output_dir}/{filename}'
    
    # Save mask
    hp.write_map(output_path, mask_apodized, dtype='float64', overwrite=True)
    print(f"  Saved to: {output_path}")


#%%

# Create masked polarization maps for each region
p_nps = np.where(mask_nps_combined == 0, hp.UNSEEN, p)
p_cygnus = np.where(mask_cygnus_combined == 0, hp.UNSEEN, p)
p_fan = np.where(mask_fan_combined == 0, hp.UNSEEN, p)
p_south = np.where(mask_south_combined == 0, hp.UNSEEN, p)
p_north = np.where(mask_north_combined == 0, hp.UNSEEN, p)


mask_rgb = np.zeros(len(mask_original))
mask_count = (mask_nps_combined + mask_cygnus_combined + mask_fan_combined + 
              mask_south_combined + mask_north_combined)

# Assign values for visualization (check for overlap first)
mask_rgb = np.where(mask_count > 1, 10, mask_rgb)  # Overlap (should be 0)
mask_rgb = np.where((mask_count == 1) & (mask_nps_combined > 0), 1, mask_rgb)
mask_rgb = np.where((mask_count == 1) & (mask_cygnus_combined > 0), 2, mask_rgb) 
mask_rgb = np.where((mask_count == 1) & (mask_fan_combined > 0), 3, mask_rgb) 
mask_rgb = np.where((mask_count == 1) & (mask_south_combined > 0), 4, mask_rgb) 
mask_rgb = np.where((mask_count == 1) & (mask_north_combined > 0), 5, mask_rgb)
mask_rgb = np.where((mask_count == 0), hp.UNSEEN, mask_rgb)

cmap = None
if USE_PLANCK_CMAP:
    # Load Planck CMB colormap
    colombi1_cmap = ListedColormap(
        np.loadtxt('/home/pablo/Desktop/Fisica/TFG/txts/Planck_Parchment_RGB.txt') / 255.
    )
    colombi1_cmap.set_bad('gray')  # Color for missing pixels
    colombi1_cmap.set_under("white")  # Background color
    cmap = colombi1_cmap


output_dir = '/home/pablo/Desktop/master/tfm/figures/masks'

# Figure 1: NPS Region
fig1 = plt.figure(figsize=(10, 8))
hp.mollview(p_nps, title='', cmap=cmap, norm='hist', cbar=False)
plt.tight_layout()
plt.savefig(f'{output_dir}/mask_lowdec_satband_galcut0_0mk_nps_apodC2_0.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Figure 2: Cygnus Region
fig2 = plt.figure(figsize=(10, 8))
hp.mollview(p_cygnus, title='', cmap=cmap, norm='hist', cbar=False)
plt.tight_layout()
plt.savefig(f'{output_dir}/mask_lowdec_satband_galcut0_0mk_cygnus_apodC2_0.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Figure 3: Fan Region
fig3 = plt.figure(figsize=(10, 8))
hp.mollview(p_fan, title='', cmap=cmap, norm='hist', cbar=False)
plt.tight_layout()
plt.savefig(f'{output_dir}/mask_lowdec_satband_galcut0_0mk_fan_apodC2_0.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Figure 4: South Region
fig4 = plt.figure(figsize=(10, 8))
hp.mollview(p_south, title='', cmap=cmap, norm='hist', cbar=False)
plt.tight_layout()
plt.savefig(f'{output_dir}/mask_lowdec_satband_galcut0_0mk_south_apodC2_0.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Figure 5: North Region
fig5 = plt.figure(figsize=(10, 8))
hp.mollview(p_north, title='', cmap=cmap, norm='hist', cbar=False)
plt.tight_layout()
plt.savefig(f'{output_dir}/mask_lowdec_satband_galcut0_0mk_north_apodC2_0.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Figure 6: Combined Masks
fig6 = plt.figure(figsize=(10, 8))
hp.mollview(mask_rgb, title='', cmap='bone_r', cbar=False, min=-1, max=5)
hp.graticule(dmer=40, color='white')

# Add labels for each region
hp.projtext(150, -2.5, 'Fan', lonlat=True, fontsize=16, color='w')
hp.projtext(90, -2.5, 'Cygnus', lonlat=True, fontsize=16, color='w')
hp.projtext(40, 40, 'NPS', lonlat=True, fontsize=16, color='w')
hp.projtext(90, 30, 'North', lonlat=True, fontsize=16, color='w')
hp.projtext(140, -32, 'South', lonlat=True, fontsize=16, color='w')

plt.tight_layout()
plt.savefig(f'{output_dir}/mask_lowdec_satband_galcut0_0mk_combined_apodC2_0.pdf', dpi=300, bbox_inches='tight')
plt.show()

